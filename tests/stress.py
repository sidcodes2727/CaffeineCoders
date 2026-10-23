"""
Stress run: the official selfcheck.py over dev, the hard set and every complex-query variant.

    python tests/make_hard.py && python tests/make_complex.py
    python tests/stress.py --url http://localhost:8000/v1/decide

Prints one row per set (score, the three quality metrics, ECE, coverage, p95) and writes
tests/stress_report.json. Every number comes from selfcheck.py itself - the judges' scorer.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "stress_report.json"))
    args = ap.parse_args()
    sets = [os.path.join(ROOT, "data", "dev.jsonl"), os.path.join(ROOT, "data", "hard_dev.jsonl")]
    sets += sorted(glob.glob(os.path.join(ROOT, "data", "complex", "*.jsonl")))
    report = {}
    print("%-18s %6s | %6s %6s %6s | %8s %8s | %5s %5s %5s | %5s %5s %5s | %6s" % (
        "set", "SCORE", "q", "cal", "abst", "routeF1", "apprCost", "ECEr", "ECEa", "ECEg", "covR", "covA", "covG",
        "p95ms"))
    for path in sets:
        if not os.path.exists(path):
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out = tmp.name
        subprocess.run([sys.executable, os.path.join(ROOT, "selfcheck.py"), "--url", args.url, "--eval", path,
                        "--json", out], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        r = json.load(open(out))
        os.unlink(out)
        q, c, p, cv = r["quality"], r["calibration"], r["score_parts"], r["quality"]["coverage"]
        report[name] = {"score": r["score"], "gates": r["gates"], "score_parts": p, "quality": q, "calibration": c,
                        "latency_ms": r["latency_ms"]}
        print("%-18s %6.2f | %.3f  %.3f  %.3f  | %.4f   %.4f   | %.3f %.3f %.3f | %.2f  %.2f  %.2f  | %5.2f%s" % (
            name, r["score"], p["quality"], p["calibration"], p["abstention"], q["route_macro_f1"],
            q["approve_cost"], c["route_ece"], c["approve_ece"], c["grounded_ece"], cv["route"], cv["approve"],
            cv["grounded"], r["latency_ms"]["mean_p95"], "" if all(r["gates"].values()) else "  GATE FAIL"), flush=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
