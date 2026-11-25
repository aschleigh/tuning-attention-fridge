"""Fake a week of fridge-door IMU data, so you can build the piece before the
hardware is mounted -- and so you have ground truth to check segment.py against.

Four synthetic roommates with different habits:

  quiet   - gentle, slow, consistent        (low effort, LOW variance)
  slammer - hard pull, hard shut, consistent (high effort, low variance)
  browser - long dwells, frequent reopens
  erratic - moderate effort, HIGH variance   (the un-rote one)

Note the deliberate dissociation between effort and roteness: 'quiet' and
'slammer' differ hugely in force and barely at all in consistency. If your
analysis cannot tell those two things apart, it is measuring the wrong thing.

    python simulate.py --days 7 --out data/ --truth truth.csv
"""

import argparse
import csv
import gzip
import os

import numpy as np

RNG = np.random.default_rng(11)

PEOPLE = {
    #            n/day  peak dps   extent deg  dwell s    jitter  slam
    "quiet":   dict(n=9,  vel=(70, 8),   ext=(62, 5),  dwell=(3.0, 1.0), jit=0.04, slam=0.25),
    "slammer": dict(n=11, vel=(260, 22), ext=(95, 7),  dwell=(2.0, 0.8), jit=0.05, slam=1.9),
    "browser": dict(n=14, vel=(130, 18), ext=(80, 9),  dwell=(11.0, 7.0), jit=0.10, slam=0.7),
    "erratic": dict(n=10, vel=(150, 70), ext=(75, 22), dwell=(5.0, 4.5), jit=0.22, slam=1.0),
}

# rough hourly weighting of fridge use across a day
HOURS = np.array([.2, .1, .1, .1, .2, .5, 2.0, 3.0, 2.0, 1.0, 1.0, 1.5,
                  2.5, 1.5, 1.0, 1.0, 1.5, 3.0, 4.0, 3.0, 2.0, 1.5, 1.0, .5])


def swing(peak_dps, dur, fs, jitter):
    """A minimum-jerk-ish velocity bump, which is what a real pull looks like."""
    n = max(int(dur * fs), 4)
    u = np.linspace(0, 1, n)
    v = peak_dps * (u ** 2) * ((1 - u) ** 2) * 16.0
    v *= 1.0 + jitter * RNG.standard_normal(n) * 0.3
    return v


def gesture(p, fs):
    """One opening: open swing, dwell, close swing. Returns gyro and accel."""
    vel = max(RNG.normal(*p["vel"]), 25.0)
    ext = max(RNG.normal(*p["ext"]), 20.0)
    dwell = max(RNG.normal(*p["dwell"]), 0.4)

    open_dur = max(2.0 * np.deg2rad(ext) * 57.3 / max(vel, 1.0), 0.35)
    g_open = swing(vel, open_dur, fs, p["jit"])
    g_open *= ext / max(np.trapezoid(g_open, dx=1 / fs), 1e-6)   # hit the extent

    close_vel = vel * RNG.normal(0.85, 0.12)
    close_dur = max(2.0 * np.deg2rad(ext) * 57.3 / max(close_vel, 1.0), 0.3)
    g_close = -swing(close_vel, close_dur, fs, p["jit"])
    g_close *= ext / max(abs(np.trapezoid(g_close, dx=1 / fs)), 1e-6)

    g_dwell = np.zeros(int(dwell * fs))
    g = np.concatenate([g_open, g_dwell, g_close])

    a = np.ones(len(g)) + 0.01 * RNG.standard_normal(len(g))
    a[-3:] += p["slam"] * RNG.normal(1.0, 0.25)                   # impact spike
    if RNG.random() < 0.4:                                        # key/handle fumble
        a[: 2] += 0.05 * RNG.standard_normal(2)
    return g, a, dict(vel=vel, ext=ext, dwell=dwell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--fs", type=float, default=50.0,
                    help="output rate; match whatever your IMU actually streams")
    ap.add_argument("--out", default="data")
    ap.add_argument("--truth", default="truth.csv")
    ap.add_argument("--t0", type=float, default=1763600000.0)
    ap.add_argument("--drift", type=float, default=0.6,
                    help="gyro bias drift amplitude in deg/s (fridge is a thermal cycler)")
    args = ap.parse_args()

    fs = args.fs
    os.makedirs(args.out, exist_ok=True)
    truth = []
    plan = []

    for day in range(args.days):
        for name, p in PEOPLE.items():
            n = RNG.poisson(p["n"])
            w = HOURS / HOURS.sum()
            hours = RNG.choice(24, size=n, p=w)
            for h in hours:
                t = args.t0 + day * 86400 + h * 3600 + RNG.uniform(0, 3600)
                plan.append([t, name, p, False])
                if RNG.random() < (0.45 if name == "browser" else 0.12):
                    plan.append([t, name, p, True])      # reopen, timed below

    plan.sort(key=lambda x: (x[0], x[3]))
    print(f"[sim] {len(plan)} openings over {args.days} days")

    by_hour = {}
    last_end = -np.inf
    for t_ev, name, p, is_reopen in plan:
        # a door cannot be opened while it is already open: a reopen starts
        # after the previous close, and nothing is allowed to overlap.
        if is_reopen:
            t_ev = last_end + RNG.uniform(3.0, 25.0)
        t_ev = max(t_ev, last_end + 1.0)

        g, a, meta = gesture(p, fs)
        n = len(g)
        tt = t_ev + np.arange(n) / fs
        last_end = tt[-1]
        key = int(t_ev // 3600)
        by_hour.setdefault(key, []).append((tt, g, a))
        truth.append(dict(t_start=t_ev, t_end=last_end, person=name,
                          duration=n / fs, reopen=bool(is_reopen), **meta))

    # Write a uniform stream, like a real logger produces. Idle is mostly zeros,
    # which gzip crushes, so the files stay small anyway.
    for key in sorted(by_hour):
        hour_start = key * 3600
        n = int(3600 * fs)
        t = hour_start + np.arange(n) / fs
        gz = np.zeros(n)
        am = np.ones(n)
        for tt, g, a in by_hour[key]:
            i0 = int(round((tt[0] - hour_start) * fs))
            i1 = min(i0 + len(g), n)
            if i1 <= 0 or i0 >= n:
                continue
            j0 = max(i0, 0)
            gz[j0:i1] += g[j0 - i0: i1 - i0]
            am[j0:i1] = a[j0 - i0: i1 - i0]

        gz += args.drift * np.sin(2 * np.pi * (t % 2400) / 2400)   # thermal drift
        gz += 0.12 * RNG.standard_normal(n)
        am += 0.004 * RNG.standard_normal(n)
        zeros = np.zeros(n)

        block = np.column_stack([t, zeros, zeros, am,
                                 0.05 * RNG.standard_normal(n),
                                 0.05 * RNG.standard_normal(n),
                                 gz, zeros, zeros, zeros])
        path = os.path.join(args.out, f"imu_sim_{key}.csv.gz")
        with gzip.open(path, "wt", newline="") as fh:
            fh.write("t,AX,AY,AZ,GX,GY,GZ,angX,angY,angZ\n")
            np.savetxt(fh, block, fmt="%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f,%.1f")

    with open(args.truth, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(truth[0].keys()))
        w.writeheader()
        w.writerows(truth)
    print(f"[sim] wrote {len(by_hour)} hourly files to {args.out}/ and {args.truth}")


if __name__ == "__main__":
    main()
