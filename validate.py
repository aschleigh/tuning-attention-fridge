"""Check segment.py against simulate.py's ground truth.

    python validate.py events.csv truth.csv

Use this to pick thresholds before the real capture, and to sanity-check that a
threshold change improved things rather than just changing the count. On real
data you have no truth file -- that is what the --cut clips are for, listening.
"""

import argparse
import csv
import sys

import numpy as np


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("truth")
    ap.add_argument("--tol", type=float, default=2.0, help="match window, seconds")
    args = ap.parse_args()

    ev = sorted(read(args.events), key=lambda r: float(r["t_start"]))
    tr = sorted(read(args.truth), key=lambda r: float(r["t_start"]))
    et = np.array([float(r["t_start"]) for r in ev])
    tt = np.array([float(r["t_start"]) for r in tr])

    used = set()
    pairs = []
    for i, t in enumerate(tt):
        d = np.abs(et - t)
        d[list(used)] = np.inf
        j = int(np.argmin(d))
        if d[j] <= args.tol:
            used.add(j)
            pairs.append((i, j))

    hit = len(pairs)
    print(f"truth {len(tt)}   detected {len(et)}   matched {hit}")
    print(f"recall    {hit/len(tt):6.1%}   (missed {len(tt)-hit})")
    print(f"precision {hit/max(len(et),1):6.1%}   (spurious {len(et)-hit})")

    if not pairs:
        sys.exit(1)

    for label, tkey, ekey in [("peak vel (dps)", "vel", "peak_open_dps"),
                              ("extent (deg)", "ext", "extent_deg"),
                              ("dwell (s)", "dwell", "dwell_s")]:
        a = np.array([float(tr[i][tkey]) for i, _ in pairs])
        b = np.array([float(ev[j][ekey]) for _, j in pairs])
        err = b - a
        r = np.corrcoef(a, b)[0, 1]
        print(f"{label:16s} bias {np.mean(err):+7.2f}  "
              f"abs err {np.mean(np.abs(err)):6.2f}  r={r:.3f}")

    missed = [i for i in range(len(tt)) if i not in {p[0] for p in pairs}]
    if missed:
        print("\nmissed openings by person:")
        tally = {}
        for i in missed:
            tally[tr[i]["person"]] = tally.get(tr[i]["person"], 0) + 1
        for k, v in sorted(tally.items(), key=lambda x: -x[1]):
            print(f"  {k:9s} {v}")
        print("  reopens among missed:",
              sum(1 for i in missed if tr[i].get("reopen") == "True"))
        ext = [float(tr[i]["ext"]) for i in missed]
        print(f"  their extent: median {np.median(ext):.0f} deg, "
              f"min {min(ext):.0f}, max {max(ext):.0f}")


if __name__ == "__main__":
    main()
