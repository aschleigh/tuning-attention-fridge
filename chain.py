"""Roteness at the level of the ROUTINE, not the gesture.

A single object tells you whether someone's elbow repeats itself. Several
objects used in sequence tell you whether their morning does. Those are
different claims and the second one is harder to dismiss.

    python chain.py ev_faucet.csv ev_kettle.csv ev_fridge.csv \
        --names faucet kettle fridge --anchor faucet --window 600 --collapse

Reports:
  * which orderings actually occur, and how often
  * the lag on each transition, and its coefficient of variation
  * per person, if you have labels

The CV column is the number to care about. A median lag says what the routine
is; the CV says whether it is a routine at all. Low CV on a transition means
the second action is triggered by the first rather than decided each time.
"""

import argparse
import csv
from collections import Counter, defaultdict

import numpy as np


def load_events(path, name):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        out.append({"obj": name,
                    "t": float(r["t_start"]),
                    "t_end": float(r.get("t_end", r["t_start"]))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="+")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--window", type=float, default=600.0,
                    help="seconds after the anchor to look for followers")
    ap.add_argument("--lookback", type=float, default=180.0,
                    help="seconds before the anchor to include")
    ap.add_argument("--collapse", action="store_true",
                    help="merge consecutive events of the same object")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--label-col", default="person")
    ap.add_argument("--label-time", default="t_start")
    ap.add_argument("--label-tol", type=float, default=300.0)
    ap.add_argument("--min-share", type=float, default=0.03)
    args = ap.parse_args()

    if len(args.events) != len(args.names):
        raise SystemExit("--names must match the number of event files")

    evs = []
    for path, name in zip(args.events, args.names):
        e = load_events(path, name)
        evs.extend(e)
        print(f"[chain] {name:10s} {len(e):5d} events")
    evs.sort(key=lambda e: e["t"])

    labels, ltimes = None, None
    if args.labels:
        with open(args.labels) as fh:
            rows = list(csv.DictReader(fh))
        ltimes = np.array([float(r[args.label_time]) for r in rows])
        labels = [r[args.label_col] for r in rows]

    anchors = [e for e in evs if e["obj"] == args.anchor]
    if not anchors:
        raise SystemExit(f"no events for anchor '{args.anchor}'")

    chains = []
    for a in anchors:
        members = [e for e in evs
                   if -args.lookback <= (e["t"] - a["t"]) <= args.window]
        members.sort(key=lambda e: e["t"])
        seq = [(e["obj"], e["t"]) for e in members]
        if args.collapse:
            keep = []
            for o, t in seq:
                if keep and keep[-1][0] == o:
                    continue
                keep.append((o, t))
            seq = keep
        who = None
        if labels is not None:
            j = int(np.argmin(np.abs(ltimes - a["t"])))
            if abs(ltimes[j] - a["t"]) <= args.label_tol:
                who = labels[j]
        chains.append({"anchor_t": a["t"], "seq": seq, "who": who})

    # ---- which orderings occur -------------------------------------------
    pat = Counter(" > ".join(o for o, _ in c["seq"]) for c in chains)
    total = sum(pat.values())
    print(f"\norderings around '{args.anchor}' "
          f"(-{args.lookback:.0f}s .. +{args.window:.0f}s), n={total}")
    print(f"  {'sequence':48s} {'n':>4s} {'share':>7s}")
    modal = None
    for s, n in pat.most_common():
        if n / total < args.min_share:
            continue
        if modal is None:
            modal = s
        print(f"  {s[:48]:48s} {n:4d} {n/total:6.0%}")
    print(f"\n  order consistency: {pat[modal]/total:.0%} of mornings "
          f"follow the modal order")

    # ---- transition lags --------------------------------------------------
    lags = defaultdict(list)
    for c in chains:
        for (o1, t1), (o2, t2) in zip(c["seq"], c["seq"][1:]):
            lags[(o1, o2)].append(t2 - t1)

    print(f"\n{'transition':28s} {'n':>4s} {'median':>9s} {'CV':>7s}   routine?")
    for (o1, o2), v in sorted(lags.items(), key=lambda kv: -len(kv[1])):
        if len(v) < 3:
            continue
        v = np.array(v)
        cv = float(np.std(v) / (abs(np.mean(v)) + 1e-9))
        verdict = "tight" if cv < 0.3 else ("loose" if cv < 0.7 else "no")
        print(f"{o1 + ' -> ' + o2:28s} {len(v):4d} {np.median(v):8.1f}s "
              f"{cv:7.2f}   {verdict}")

    # ---- per person -------------------------------------------------------
    if labels is not None:
        print(f"\n{'person':10s} {'n':>4s} {'order':>7s} {'mean lag CV':>12s}")
        by = defaultdict(list)
        for c in chains:
            if c["who"]:
                by[c["who"]].append(c)
        for who, cs in sorted(by.items()):
            p = Counter(" > ".join(o for o, _ in c["seq"]) for c in cs)
            order = p.most_common(1)[0][1] / len(cs)
            pl = defaultdict(list)
            for c in cs:
                for (o1, t1), (o2, t2) in zip(c["seq"], c["seq"][1:]):
                    pl[(o1, o2)].append(t2 - t1)
            cvs = [np.std(v) / (abs(np.mean(v)) + 1e-9)
                   for v in pl.values() if len(v) >= 3]
            mcv = float(np.mean(cvs)) if cvs else float("nan")
            print(f"{who:10s} {len(cs):4d} {order:6.0%} {mcv:12.2f}")
        print("\nOrder near 100% with a low lag CV is a person running a script.")
        print("High order consistency with a high lag CV is a person who does")
        print("the same things in the same sequence but is not on rails.")


if __name__ == "__main__":
    main()
