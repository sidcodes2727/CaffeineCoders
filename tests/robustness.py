"""
Robustness suite for the Decide service (run against a live endpoint).

    python tests/robustness.py --url http://localhost:8000/v1/decide [--eval data/dev.jsonl] [--cold]

Sections
  1. invariance   - label-preserving perturbations; pass bar: < 3% of committed values flip
  2. directional  - edits that must move p in a known direction
  3. slices       - F1 / ECE per slice on a labelled split
  4. edge cases   - malformed / extreme inputs must still give schema-valid responses (or a clean 4xx
                    for requests that are not valid API calls at all)
  5. latency      - 1,000 sequential calls per decision, worst-case inputs, determinism, cold start
Writes tests/robustness_report.json.
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from perturb import PERTURBATIONS, apply, _PRIMARY_RX, _SENT  # noqa: E402
from selfcheck import Client, validate, ece, macro_f1, percentile, QUEUES  # noqa: E402

DECISIONS = ["route", "approve", "grounded"]


def ask(client, state, decision, rid="t"):
    resp, ms, err = client.call({"request_id": rid, "decision": decision, "state": state})
    return resp, ms, err


def value_p(resp, decision):
    if resp is None:
        return None, None
    if decision == "route":
        return resp["value"], resp["probs"]
    return resp["value"], resp["p_true"]


# ------------------------------------------------------------------------------------------
def invariance(client, rows, rng, n=200):
    out = {}
    sample = rng.sample(rows, min(n, len(rows)))
    base = {d: [value_p(ask(client, r["state"], d)[0], d) for r in sample] for d in DECISIONS}
    for name in PERTURBATIONS:
        res = {}
        for d in DECISIONS:
            flips = tot = 0
            dp = []
            for r, (v0, p0) in zip(sample, base[d]):
                s = apply(r["state"], name, random.Random(hash((name, r["request_id"])) & 0xffff), decision=d)
                if s == r["state"]:
                    continue                                   # perturbation does not touch this decision
                v1, p1 = value_p(ask(client, s, d)[0], d)
                tot += 1
                flips += int(v1 != v0)
                dp.append(abs((p1[v0] if d == "route" else p1) - (p0[v0] if d == "route" else p0)))
            if tot:
                res[d] = {"n": tot, "flip_rate": flips / tot, "mean_abs_dp": sum(dp) / len(dp),
                          "pass": flips / tot < 0.03}
        out[name] = res
    return out


# ------------------------------------------------------------------------------------------
CANON = {"billing": "I was charged twice for the same subscription this month.",
         "account": "I can't log in to my account, the password reset does nothing.",
         "shipping": "My parcel was marked delivered but never arrived.",
         "technical": "The app crashes every time I open the export page.",
         "general": "Do you offer a discount for registered charities?"}


def _replace_request(body, new):
    m = _PRIMARY_RX.search(body)
    if m:
        rest = body[m.end():]
        parts = _SENT.split(rest.strip(), maxsplit=1)
        tail = parts[1] if len(parts) > 1 else ""
        return body[:m.end()] + " " + new + " " + tail
    return new + " Thanks."


def _change_number(text):
    swaps = [(r"\b14\b", "30"), (r"\bfourteen\b", "thirty"), (r"\btwo weeks\b", "five weeks"), (r"\b30\b", "45"),
             (r"\bthirty\b", "ninety"), (r"\b7\b", "21"), (r"\bseven\b", "twenty-one"), (r"\b5\b", "9"),
             (r"\bfive\b", "nine"), (r"\b(\d+)\b", None)]
    for rx, rep in swaps:
        if re.search(rx, text):
            if rep is None:
                return re.sub(rx, lambda m: str(int(m.group(1)) * 3 + 1), text, count=1)
            return re.sub(rx, rep, text, count=1)
    return None


def _drop_support(draft, kb):
    dw = set(re.findall(r"[a-z0-9]+", draft.lower()))
    sents = [s for s in _SENT.split(kb) if s.strip()]
    if len(sents) < 2:
        return None
    scores = [len(dw & set(re.findall(r"[a-z0-9]+", s.lower()))) for s in sents]
    i = max(range(len(sents)), key=scores.__getitem__)
    return " ".join(s for j, s in enumerate(sents) if j != i)


def _rephrase(draft):
    t = draft
    for a, b in [("you will", "you'll"), ("I have", "I've"), ("There is", "There's"), ("there is", "there's"),
                 ("I will", "I'll"), ("It is", "It's"), ("That is", "That's")]:
        t = t.replace(a, b)
    return "Hi there. " + t + " Hope that helps."


def directional(client, rows, rng, caps, n=200):
    tests = {}

    def run(name, decision, pairs, check):
        ok = tot = 0
        for s0, s1 in pairs:
            r0, r1 = ask(client, s0, decision)[0], ask(client, s1, decision)[0]
            if r0 is None or r1 is None:
                continue
            tot += 1
            ok += int(check(r0, r1))
        tests[name] = {"n": tot, "pass_rate": ok / tot if tot else None}

    sample = rng.sample(rows, min(n, len(rows)))
    tiers = ["free", "pro", "enterprise"]
    # approve: amount above cap must lower p
    pairs = []
    for r in sample:
        s = r["state"]
        t = tiers.index(s["customer_tier"]) if s["customer_tier"] in tiers else 0
        if 0 < s["amount"] < caps[t]:
            new = round(caps[t] * 3, 2)
            s1 = dict(s, amount=new, body=re.sub(r"\$[\d,.]+\d", "$%.2f" % new, s["body"]))
            pairs.append((s, s1))
    run("approve: amount pushed above tier cap -> p drops", "approve", pairs,
        lambda a, b: b["p_true"] < a["p_true"] or b["p_true"] <= 0.05)
    # approve: free -> enterprise must not lower p
    pairs = [(r["state"], dict(r["state"], customer_tier="enterprise",
                               body=r["state"]["body"].replace("Free plan", "Enterprise plan")))
             for r in sample if r["state"]["customer_tier"] == "free"]
    run("approve: tier free -> enterprise -> p does not drop", "approve", pairs,
        lambda a, b: b["p_true"] >= a["p_true"] - 0.02)
    # approve: history phrase (training wording and an unseen paraphrase)
    for label, phrase in [("seen wording", "I raised something similar not long ago."),
                          ("paraphrase", "I already complained about this exact thing a few weeks back.")]:
        pairs = [(r["state"], dict(r["state"], body=r["state"]["body"] + " " + phrase)) for r in sample]
        run("approve: history phrase added (%s) -> p drops" % label, "approve", pairs,
            lambda a, b: b["p_true"] < a["p_true"] or b["p_true"] <= 0.05)
    # approve: red-flag request replaces the core request
    for label, req in [("seen wording", "Please delete my account and erase all my data."),
                       ("paraphrase", "I need you to switch off two-step verification on my login.")]:
        pairs = [(r["state"], dict(r["state"], body=_replace_request(r["state"]["body"], req))) for r in sample]
        run("approve: red-flag request (%s) -> p drops" % label, "approve", pairs,
            lambda a, b: b["p_true"] < a["p_true"] or b["p_true"] <= 0.05)

    # grounded, on rows labelled grounded
    gs = [r for r in sample if r["labels"]["grounded"]]
    pairs = [(r["state"], dict(r["state"], kb_article=_drop_support(r["state"]["draft_reply"], r["state"]["kb_article"])))
             for r in gs if _drop_support(r["state"]["draft_reply"], r["state"]["kb_article"])]
    run("grounded: supporting KB sentence deleted -> p drops", "grounded", pairs,
        lambda a, b: b["p_true"] < a["p_true"])
    pairs = [(r["state"], dict(r["state"], draft_reply=_change_number(r["state"]["draft_reply"])))
             for r in gs if _change_number(r["state"]["draft_reply"])]
    run("grounded: number in draft changed -> p drops", "grounded", pairs,
        lambda a, b: b["p_true"] < a["p_true"])
    pairs = [(r["state"], dict(r["state"], draft_reply=_rephrase(r["state"]["draft_reply"]))) for r in gs]
    run("grounded: draft rephrased, same facts -> stays grounded", "grounded", pairs,
        lambda a, b: b["p_true"] > 0.5 or b["p_true"] >= a["p_true"] - 0.1)

    # route: swap the core request
    pairs, targets = [], []
    for r in sample:
        new_q = rng.choice([q for q in QUEUES if q != r["labels"]["route"]])
        pairs.append((r["state"], dict(r["state"], body=_replace_request(r["state"]["body"], CANON[new_q]))))
        targets.append(new_q)
    it = iter(targets)
    run("route: core request swapped -> new queue predicted", "route", pairs, lambda a, b: b["value"] == next(it))
    return tests


# ------------------------------------------------------------------------------------------
def slices(client, rows, caps):
    recs = []
    for r in rows:
        out = {d: ask(client, r["state"], d)[0] for d in DECISIONS}
        recs.append((r, out))
    tiers = ["free", "pro", "enterprise"]

    def route_stats(sub):
        if not sub:
            return None
        y = [r["labels"]["route"] for r, _ in sub]
        p = [o["route"]["value"] for _, o in sub]
        conf = [max(o["route"]["probs"].values()) for _, o in sub]
        top = [max(o["route"]["probs"], key=o["route"]["probs"].get) for _, o in sub]
        return {"n": len(sub), "acc": sum(a == b for a, b in zip(y, p)) / len(sub),
                "ece": ece(conf, [1.0 if a == b else 0.0 for a, b in zip(y, top)])}

    def bin_stats(sub, d):
        if not sub:
            return None
        y = [bool(r["labels"][d]) for r, _ in sub]
        v = [o[d]["value"] for _, o in sub]
        p = [o[d]["p_true"] for _, o in sub]
        st = {"n": len(sub), "acc": sum(a == b for a, b in zip(y, v)) / len(sub),
              "macro_f1": macro_f1(list(zip(y, v)), [True, False]), "ece": ece(p, [float(a) for a in y])}
        if d == "approve":
            st["cost"] = sum(3.0 if (b and not a) else (1.0 if (a and not b) else 0.0) for a, b in zip(y, v)) / len(sub)
        return st

    res = {"route_by_queue": {q: route_stats([x for x in recs if x[0]["labels"]["route"] == q]) for q in QUEUES}}
    res["approve_by_tier"] = {t: bin_stats([x for x in recs if x[0]["state"]["customer_tier"] == t], "approve")
                              for t in tiers}
    near = [x for x in recs if x[0]["state"]["amount"] > 0 and
            abs(x[0]["state"]["amount"] - caps[tiers.index(x[0]["state"]["customer_tier"])]) <=
            0.1 * caps[tiers.index(x[0]["state"]["customer_tier"])]]
    res["approve_within_10pct_of_cap"] = bin_stats(near, "approve")
    kb_len = sorted(len(x[0]["state"]["kb_article"]) for x in recs)
    med = kb_len[len(kb_len) // 2]
    res["grounded_short_kb"] = bin_stats([x for x in recs if len(x[0]["state"]["kb_article"]) <= med], "grounded")
    res["grounded_long_kb"] = bin_stats([x for x in recs if len(x[0]["state"]["kb_article"]) > med], "grounded")
    mk = [x for x in recs if _PRIMARY_RX.search(x[0]["state"]["body"])]
    nm = [x for x in recs if not _PRIMARY_RX.search(x[0]["state"]["body"])]
    res["route_with_marker"], res["route_without_marker"] = route_stats(mk), route_stats(nm)
    ntok = lambda x: len(x[0]["state"]["body"].split())
    res["route_body_short_lt10_tokens"] = route_stats([x for x in recs if ntok(x) < 10])
    res["route_body_long_gt60_tokens"] = route_stats([x for x in recs if ntok(x) > 60])
    return res


# ------------------------------------------------------------------------------------------
def edge_cases(url):
    long_txt = "The export fails. " * 700
    states = {
        "empty state": {},
        "all null": {k: None for k in ["subject", "body", "customer_tier", "amount", "draft_reply", "kb_article"]},
        "empty strings": {k: "" for k in ["subject", "body", "customer_tier", "draft_reply", "kb_article"]},
        "very long text (12k chars)": {"subject": "x", "body": long_txt, "draft_reply": long_txt, "kb_article": long_txt},
        "non-English": {"subject": "Problema con la factura", "body": "Me cobraron dos veces este mes. 请帮忙。"},
        "only emojis": {"subject": "😡😡", "body": "🙏🙏🙏 🔥"},
        "only numbers": {"subject": "123", "body": "4567 89 0", "amount": 12},
        "amount as string '1,200'": {"body": "refund please", "amount": "1,200", "customer_tier": "pro"},
        "amount as string '$1.2k'": {"body": "refund please", "amount": "$1.2k", "customer_tier": "pro"},
        "amount in words": {"body": "refund please", "amount": "twelve hundred", "customer_tier": "pro"},
        "amount garbage string": {"body": "refund please", "amount": "lots", "customer_tier": "pro"},
        "amount negative": {"body": "refund please", "amount": -50},
        "amount zero": {"body": "refund please", "amount": 0},
        "amount huge": {"body": "refund please", "amount": 1e18},
        "amount NaN-ish": {"body": "refund please", "amount": "NaN"},
        "unknown tier": {"body": "refund please", "amount": 10, "customer_tier": "platinum"},
        "tier wrong type": {"body": "refund please", "amount": 10, "customer_tier": 3},
        "extra fields": {"body": "hi", "foo": [1, 2], "bar": {"x": 1}},
        "body wrong type": {"body": 12345, "subject": ["a", "b"]},
    }
    client = Client(url, 10.0)
    results = {}
    for name, st in states.items():
        for d in DECISIONS:
            resp, ms, err = client.call({"request_id": "edge", "decision": d, "state": st})
            errs = [err] if err else validate(resp, d, "edge")
            results["%s / %s" % (name, d)] = {"ok": not errs, "errors": errs, "ms": round(ms, 2)}
    # requests that are not valid API calls: must fail cleanly (4xx JSON), never 5xx / crash
    bad = {"wrong decision name": {"request_id": "e", "decision": "refund", "state": {}},
           "missing request_id": {"decision": "route", "state": {"body": "hi"}},
           "missing state": {"request_id": "e", "decision": "route"},
           "not JSON object": [1, 2, 3]}
    for name, payload in bad.items():
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                body = json.loads(r.read())
                code = r.status
        except urllib.error.HTTPError as e:
            code, body = e.code, e.read()[:200]
        except Exception as e:  # connection reset = crash
            code, body = None, str(e)
        if name == "missing request_id" and code == 200:
            ok = not validate(body, "route", body.get("request_id"))
        else:
            ok = code is not None and 400 <= code < 500
        results["invalid call: " + name] = {"ok": ok, "status": code}
    return results


# ------------------------------------------------------------------------------------------
def latency(client, rows, n=1000):
    out = {}
    for d in DECISIONS:
        for i in range(30):
            ask(client, rows[i % len(rows)]["state"], d)
        ms = [ask(client, rows[i % len(rows)]["state"], d)[1] for i in range(n)]
        out[d] = {"n": n, "p50": percentile(ms, .5), "p95": percentile(ms, .95)}
    longest = sorted(rows, key=lambda r: -(len(r["state"]["body"]) + len(r["state"]["kb_article"])))[:50]
    worst = {"subject": "x", "body": "The export fails. " * 700, "draft_reply": "We reply within 14 days. " * 300,
             "kb_article": "Refunds take fourteen days. " * 300, "amount": 10, "customer_tier": "pro"}
    for d in DECISIONS:
        ms = [ask(client, r["state"], d)[1] for r in longest]
        msw = [ask(client, worst, d)[1] for _ in range(20)]
        out[d]["longest_50_dataset_rows_p95"] = percentile(ms, .95)
        out[d]["synthetic_12k_char_p95"] = percentile(msw, .95)
    det = True
    for r in rows[:50]:
        for d in DECISIONS:
            a, b = ask(client, r["state"], d)[0], ask(client, r["state"], d)[0]
            a = {k: v for k, v in a.items() if k != "latency_ms"}
            b = {k: v for k, v in b.items() if k != "latency_ms"}
            det &= a == b
    out["deterministic"] = det
    return out


def cold_start(port=8011):
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port),
                             "--log-level", "warning"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    up = None
    while time.time() - t0 < 120:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=1) as r:
                if json.loads(r.read()).get("loaded"):
                    up = time.time() - t0
                    break
        except Exception:
            time.sleep(0.2)
    proc.terminate()
    proc.wait()
    return {"seconds_to_ready": up}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--eval", default="data/dev.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cold", action="store_true", help="also time a cold start of a second server")
    ap.add_argument("--out", default="tests/robustness_report.json")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.eval) if l.strip()]
    caps = json.load(open(os.path.join(ROOT, "artifacts", "calibration.json")))["caps"]
    client = Client(args.url, 10.0)
    rng = random.Random(7)
    rep = {}

    print("1. invariance ...", flush=True)
    rep["invariance"] = invariance(client, rows, rng, args.n)
    for name, res in rep["invariance"].items():
        print("   %-24s " % name + "  ".join("%s flip %5.1f%% %s" % (d, 100 * v["flip_rate"], "ok" if v["pass"] else "FAIL")
                                             for d, v in res.items()))
    print("2. directional ...", flush=True)
    rep["directional"] = directional(client, rows, rng, caps, args.n)
    for name, v in rep["directional"].items():
        print("   %-62s pass %5.1f%%  (n=%d)" % (name, 100 * (v["pass_rate"] or 0), v["n"]))
    print("3. slices ...", flush=True)
    rep["slices"] = slices(client, rows, caps)
    for name, v in rep["slices"].items():
        if isinstance(v, dict) and v and all(isinstance(x, dict) or x is None for x in v.values()):
            for k, s in v.items():
                print("   %-28s %-11s %s" % (name, k, {a: round(b, 3) for a, b in (s or {}).items()}))
        else:
            print("   %-40s %s" % (name, {a: round(b, 3) for a, b in (v or {}).items()}))
    print("4. edge cases ...", flush=True)
    rep["edge_cases"] = edge_cases(args.url)
    bad = {k: v for k, v in rep["edge_cases"].items() if not v["ok"]}
    print("   %d / %d edge cases OK" % (len(rep["edge_cases"]) - len(bad), len(rep["edge_cases"])))
    for k, v in bad.items():
        print("   FAIL", k, v)
    print("5. latency & stability ...", flush=True)
    rep["latency"] = latency(client, rows)
    for d in DECISIONS:
        print("   %-9s %s" % (d, {a: round(b, 2) for a, b in rep["latency"][d].items()}))
    print("   deterministic:", rep["latency"]["deterministic"])
    if args.cold:
        rep["cold_start"] = cold_start()
        print("   cold start:", rep["cold_start"])
    json.dump(rep, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
