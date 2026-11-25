"""Three objects, causally linked: faucet -> kettle -> fridge.

The morning routine is the point. Each person fills the kettle at the faucet,
boils it, pours, and goes to the fridge for milk, with person-specific lags and
a person-specific probability of doing it out of order. That gives chain.py
something with known answers to recover.

    python simulate_routine.py --days 7 --out routine/
    python segment.py "routine/faucet_*.csv.gz" --profile faucet --out ev_faucet.csv
    python segment.py "routine/kettle_*.csv.gz" --profile kettle --out ev_kettle.csv
    python segment.py "routine/fridge_*.csv.gz" --profile door   --out ev_fridge.csv
    python chain.py ev_faucet.csv ev_kettle.csv ev_fridge.csv \
        --names faucet kettle fridge --anchor faucet

Faucet and kettle are written as rotating GRAVITY VECTORS, because that is what
an accelerometer on a horizontally-hinged lever or a tipped kettle actually
measures. The fridge door is written as a gyro burst, because a vertical hinge
is invisible to gravity.
"""

import argparse
import csv
import gzip
import os

import numpy as np

RNG = np.random.default_rng(5)

#                 hour   lag faucet->kettle   boil->fridge    order noise
PEOPLE = {
    "ana":   dict(hour=6.6,  lag=(8, 1.5),  milk=(6, 2),   scramble=0.03, fill=70),
    "ben":   dict(hour=7.4,  lag=(25, 12),  milk=(30, 18), scramble=0.25, fill=95),
    "cass":  dict(hour=8.1,  lag=(11, 3),   milk=(9, 4),   scramble=0.08, fill=55),
    "dev":   dict(hour=9.3,  lag=(15, 9),   milk=(14, 10), scramble=0.18, fill=85),
}
HDR = "t,AX,AY,AZ,GX,GY,GZ,angX,angY,angZ\n"


def ramp(n, peak, rise, hold):
    """rise -> plateau -> fall, in samples."""
    x = np.zeros(n)
    r, h = int(rise), int(hold)
    r = max(min(r, n // 3), 2)
    h = max(min(h, n - 2 * r - 1), 1)
    u = np.linspace(0, 1, r)
    s = u * u * (3 - 2 * u)                    # smoothstep
    x[:r] = peak * s
    x[r:r + h] = peak
    f = n - r - h
    if f > 1:
        v = np.linspace(0, 1, f)
        x[r + h:] = peak * (1 - v * v * (3 - 2 * v))
    return x


def tilt_block(fs, peak_deg, rise_s, hold_s, fall_s, yaw_deg=0.0, impact=0.0):
    """A tilt gesture as a rotating gravity vector plus matching gyro."""
    rise_s, hold_s, fall_s = max(rise_s, .15), max(hold_s, .3), max(fall_s, .15)
    peak_deg = max(peak_deg, 8.0)
    n = int((rise_s + hold_s + fall_s) * fs)
    th = np.deg2rad(ramp(n, peak_deg, rise_s * fs, hold_s * fs))
    A = np.column_stack([np.sin(th), np.zeros(n), np.cos(th)])
    A += 0.004 * RNG.standard_normal(A.shape)
    rate = np.degrees(np.gradient(th, 1 / fs))
    G = np.column_stack([np.zeros(n), rate, np.zeros(n)])
    if yaw_deg:                                # temperature swivel: gravity-blind
        k = max(int(0.4 * fs), 2)
        G[:k, 2] += yaw_deg / (k / fs)
    G += 0.08 * RNG.standard_normal(G.shape)
    if impact:
        A[-3:, 2] += impact * RNG.normal(1.0, 0.2)
    return A, G


def door_block(fs, peak_dps, extent_deg, dwell_s, slam):
    peak_dps = max(peak_dps, 30.0)
    extent_deg = max(extent_deg, 20.0)
    dwell_s = max(dwell_s, 0.2)
    slam = max(slam, 0.05)
    def swing(peak, dur):
        m = max(int(dur * fs), 4)
        u = np.linspace(0, 1, m)
        return peak * (u ** 2) * ((1 - u) ** 2) * 16.0
    od = max(2.0 * extent_deg / max(peak_dps, 1.0), 0.35)
    go = swing(peak_dps, od)
    go *= extent_deg / max(np.trapezoid(go, dx=1 / fs), 1e-6)
    cv = peak_dps * RNG.normal(0.85, 0.1)
    cd = max(2.0 * extent_deg / max(cv, 1.0), 0.3)
    gc = -swing(cv, cd)
    gc *= extent_deg / max(abs(np.trapezoid(gc, dx=1 / fs)), 1e-6)
    g = np.concatenate([go, np.zeros(int(dwell_s * fs)), gc])
    n = len(g)
    A = np.column_stack([np.zeros(n), np.zeros(n), np.ones(n)])
    A[-3:, 2] += slam * RNG.normal(1.0, 0.25)
    A += 0.004 * RNG.standard_normal(A.shape)
    G = np.column_stack([np.zeros(n), np.zeros(n), g])
    G += 0.1 * RNG.standard_normal(G.shape)
    return A, G


def write_obj(name, blocks, fs, outdir, drift=0.0):
    """Lay blocks onto an idle timeline and write hourly gzipped CSVs."""
    by_hour = {}
    for t0, A, G in blocks:
        by_hour.setdefault(int(t0 // 3600), []).append((t0, A, G))
    for key, items in sorted(by_hour.items()):
        hs = key * 3600
        n = int(3600 * fs)
        t = hs + np.arange(n) / fs
        Aa = np.zeros((n, 3)); Aa[:, 2] = 1.0
        Gg = np.zeros((n, 3))
        for t0, A, G in items:
            i0 = int(round((t0 - hs) * fs))
            i1 = min(i0 + len(A), n)
            j0 = max(i0, 0)
            if i1 <= j0:
                continue
            Aa[j0:i1] = A[j0 - i0:i1 - i0]
            Gg[j0:i1] = G[j0 - i0:i1 - i0]
        Aa += 0.004 * RNG.standard_normal(Aa.shape)
        Gg += 0.1 * RNG.standard_normal(Gg.shape)
        if drift:
            Gg[:, 2] += drift * np.sin(2 * np.pi * (t % 2400) / 2400)
        block = np.column_stack([t, Aa, Gg, np.zeros((n, 3))])
        path = os.path.join(outdir, f"{name}_{key}.csv.gz")
        with gzip.open(path, "wt", newline="") as fh:
            fh.write(HDR)
            np.savetxt(fh, block, fmt="%.3f," + ",".join(["%.4f"] * 6) + ",%.1f,%.1f,%.1f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--fs", type=float, default=50.0)
    ap.add_argument("--out", default="routine")
    ap.add_argument("--truth", default="truth_routine.csv")
    ap.add_argument("--t0", type=float, default=1763600000.0)
    args = ap.parse_args()

    fs = args.fs
    os.makedirs(args.out, exist_ok=True)
    faucet, kettle, fridge, truth = [], [], [], []

    for day in range(args.days):
        base = args.t0 - (args.t0 % 86400) + day * 86400
        for who, p in PEOPLE.items():
            t = base + (p["hour"] + RNG.normal(0, 0.25)) * 3600
            scrambled = RNG.random() < p["scramble"]

            # 1. kettle lifted off its base and carried to the sink
            A, G = tilt_block(fs, RNG.normal(30, 5), 0.5, RNG.normal(1.5, .4), 0.5)
            kettle.append((t, A, G))

            # 2. faucet: lever up, water runs while the kettle fills
            fill_s = p["fill"] / 60.0 * RNG.normal(1.0, 0.08) * 6
            lag = max(RNG.normal(*p["lag"]), 2.0)
            t_f = t + max(RNG.normal(3, 1), 0.5)
            yaw = abs(RNG.normal(0, 12)) if RNG.random() < 0.25 else 0.0
            A, G = tilt_block(fs, RNG.normal(42, 6), RNG.normal(.5, .15),
                              fill_s, RNG.normal(.4, .1), yaw_deg=yaw)
            faucet.append((t_f, A, G))

            # 3. kettle set back on its base, then boils (no motion), then pours
            t_back = t_f + fill_s + RNG.normal(3, 1)
            A, G = tilt_block(fs, RNG.normal(28, 5), .5, RNG.normal(1.2, .3), .5,
                              impact=RNG.normal(0.6, 0.25))
            kettle.append((t_back, A, G))

            boil_s = 25.0 + p["fill"] * 1.6 * RNG.normal(1.0, 0.05)   # ∝ volume
            t_pour = t_back + boil_s + max(RNG.normal(lag, 3), 1.0)
            A, G = tilt_block(fs, RNG.normal(75, 9), RNG.normal(1.1, .3),
                              RNG.normal(6, 2), RNG.normal(1.0, .3),
                              impact=RNG.normal(0.4, 0.2))
            kettle.append((t_pour, A, G))

            # 4. fridge for milk -- unless this is a scrambled morning
            t_milk = t_pour + max(RNG.normal(*p["milk"]), 2.0)
            if scrambled:
                t_milk = t_f - RNG.uniform(20, 90)        # milk out first
            A, G = door_block(fs, RNG.normal(150, 40), RNG.normal(75, 10),
                              RNG.normal(4, 2), RNG.normal(0.9, 0.4))
            fridge.append((t_milk, A, G))

            truth.append(dict(person=who, day=day, t_faucet=t_f, t_pour=t_pour,
                              t_fridge=t_milk, boil_s=round(boil_s, 1),
                              scrambled=scrambled))

        # a few unrelated fridge openings through the day
        for _ in range(RNG.poisson(14)):
            t = base + RNG.uniform(10, 23) * 3600
            A, G = door_block(fs, RNG.normal(150, 50), RNG.normal(75, 12),
                              RNG.normal(4, 3), RNG.normal(0.9, 0.5))
            fridge.append((t, A, G))

    for name, blocks, drift in [("faucet", faucet, 0.0), ("kettle", kettle, 0.0),
                                ("fridge", fridge, 0.6)]:
        blocks.sort(key=lambda b: b[0])
        write_obj(name, blocks, fs, args.out, drift)
        print(f"[routine] {name}: {len(blocks)} gestures")

    with open(args.truth, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(truth[0].keys()))
        w.writeheader(); w.writerows(truth)
    print(f"[routine] wrote {args.out}/ and {args.truth}")


if __name__ == "__main__":
    main()
