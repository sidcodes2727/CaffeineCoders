"""
Compare model versions with the official scorer, each served from its own snapshot (code + artifacts).

    python tests/compare_versions.py --python <venv python> \
        v5c=/path/to/snapshot  v5c-nocascade=/path/to/snapshot:DECIDE_NO_CASCADE=1  v6=/path/to/other

Every version is started with `uvicorn app:app` from its directory, warmed up, scored with
tests/stress.py (selfcheck.py over dev, the hard set and every complex variant), then stopped.
Writes tests/compare_report.json and prints one table: rows = sets, columns = versions.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def wait_ready(port, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=1) as r:
                if json.loads(r.read()).get("loaded"):
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("versions", nargs="+", help="name=dir[:ENV=VAL,...]")
    ap.add_argument("--python", default=sys.executable, help="interpreter of a serving-only venv")
    ap.add_argument("--port", type=int, default=8031)
    ap.add_argument("--out", default=os.path.join(HERE, "compare_report.json"))
    args = ap.parse_args()
    results = {}
    for spec in args.versions:
        name, rest = spec.split("=", 1)
        path, _, envs = rest.partition(":")
        env = dict(os.environ)
        for kv in filter(None, envs.split(",")):
            k, v = kv.split("=", 1)
            env[k] = v
        print("== %s (%s %s)" % (name, path, envs), flush=True)
        proc = subprocess.Popen([args.python, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(args.port),
                                 "--log-level", "warning"], cwd=path, env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            if not wait_ready(args.port):
                print("   did not start")
                continue
            out = os.path.join(HERE, "_stress_%s.json" % name)
            subprocess.run([args.python, os.path.join(HERE, "stress.py"), "--url",
                            "http://127.0.0.1:%d/v1/decide" % args.port, "--out", out], check=False)
            results[name] = json.load(open(out))
            os.remove(out)
        finally:
            proc.terminate()
            proc.wait()
            time.sleep(1)
    json.dump(results, open(args.out, "w"), indent=1)
    names = list(results)
    sets = list(next(iter(results.values())).keys()) if results else []
    print("\n%-18s" % "SCORE" + "".join("%16s" % n for n in names))
    for s in sets:
        print("%-18s" % s + "".join("%16.2f" % results[n][s]["score"] if s in results[n] else "%16s" % "-" for n in names))
    print("%-18s" % "mean" + "".join("%16.2f" % (sum(v["score"] for v in results[n].values()) / len(results[n])) for n in names))
    print("\n%-18s" % "mean p95 ms (dev)" + "".join("%16.2f" % results[n]["dev"]["latency_ms"]["mean_p95"] for n in names))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
