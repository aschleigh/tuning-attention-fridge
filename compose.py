"""From events.csv to sound.

    python compose.py map      events.csv                     # see the curve
    python compose.py manifest events.csv --clips clips/ -o manifest.jsonl
    python compose.py day      events.csv --day 0 -o day0.wav
    python compose.py week     events.csv -o week.wav
    python compose.py stack    events.csv -o stack/ --by person --labels people.csv

The renderer here is a PLACEHOLDER synth, not the piece. It exists so you can
hear your segmentation and your mapping on day one, before any model has run.
Once FlowEdit/RAVE output exists, point --clips at it and the same day/week/
stack code arranges the real material.

THE MAPPING
-----------
effort (peak opening velocity) -> morph (0 = this fridge, 1 = bowed string)

    morph = m_max * tanh(v / v_knee)

v_knee defaults to the MEDIAN peak velocity across the capture, which puts the
saturation knee right in the middle of the household's habits. Gentle openings
land on the steep part of the curve and spread out across timbre; forceful ones
pile up at the top and become indistinguishable. That is the Feldenkrais claim
rendered as a transfer function rather than stated in a programme note.
"""

import argparse
import csv
import json
import os

import numpy as np

SR = 48000


# --------------------------------------------------------------------------
# events + mapping
# --------------------------------------------------------------------------

def read_events(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        e = {}
        for k, v in r.items():
            if v in ("True", "False"):
                e[k] = (v == "True")
            else:
                try:
                    e[k] = float(v)
                except ValueError:
                    e[k] = v
        out.append(e)
    return sorted(out, key=lambda e: e["t_start"])


def morph_curve(vel, v_knee, m_max=1.0):
    return m_max * np.tanh(np.asarray(vel, dtype=float) / max(v_knee, 1e-6))


def attach_morph(events, v_knee=None, m_max=1.0):
    vel = np.array([e["peak_open_dps"] for e in events])
    if v_knee is None:
        v_knee = float(np.median(vel))
    m = morph_curve(vel, v_knee, m_max)
    for e, mm in zip(events, m):
        e["morph"] = float(mm)
    return v_knee


def cmd_map(args):
    events = read_events(args.events)
    knee = attach_morph(events, args.knee, args.m_max)
    vel = np.array([e["peak_open_dps"] for e in events])
    m = np.array([e["morph"] for e in events])
    print(f"v_knee = {knee:.1f} dps   m_max = {args.m_max}")
    print(f"velocity  p10 {np.percentile(vel,10):.0f}  median {np.median(vel):.0f} "
          f" p90 {np.percentile(vel,90):.0f}  max {vel.max():.0f}")

    # how much morph-range each velocity quartile actually gets: the whole point
    qs = np.percentile(vel, [0, 25, 50, 75, 100])
    print("\nquartile        velocity        morph span   <- discrimination")
    for i in range(4):
        sel = (vel >= qs[i]) & (vel <= qs[i + 1])
        if sel.sum() < 2:
            continue
        span = m[sel].max() - m[sel].min()
        bar = "#" * max(int(span * 60), 1)
        print(f"  Q{i+1}  {qs[i]:6.0f}-{qs[i+1]:<6.0f} dps   {span:5.3f}  {bar}")
    print("\nThe top quartile should be visibly starved of range. If it is not,")
    print("lower --knee until it is.")


# --------------------------------------------------------------------------
# placeholder synth
# --------------------------------------------------------------------------

def _env(n, attack, release, sr=SR):
    a = max(int(attack * sr), 2)
    r = max(int(release * sr), 2)
    e = np.ones(n)
    a, r = min(a, n // 2), min(r, n // 2)
    e[:a] = np.linspace(0, 1, a) ** 2
    e[n - r:] = np.linspace(1, 0, r) ** 1.5
    return e


def render_event(e, sr=SR, f0=None):
    """Fridge-ish noise at morph=0, bowed-string-ish at morph=1.

    Deliberately crude. The only thing it needs to get right is that the
    PERCEPTUAL DISTANCE between two renders tracks the difference in morph.
    """
    dur = float(np.clip(e.get("duration", 3.0), 0.4, 12.0))
    n = int(dur * sr)
    t = np.arange(n) / sr
    m = float(e.get("morph", 0.5))
    if f0 is None:
        f0 = 55.0 * 2 ** (np.clip(e.get("extent_deg", 70) / 70.0, 0.5, 2.0) - 1.0)

    # fridge pole: compressor hum + broadband rattle
    hum = 0.35 * (np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 240 * t))
    noise = np.random.default_rng(int(e["t_start"]) % 2**31).standard_normal(n)
    b = np.exp(-np.arange(64) / 8.0)
    rattle = np.convolve(noise, b, mode="same") * 0.25
    fridge = hum * 0.6 + rattle

    # string pole: harmonic stack, slight vibrato, bow-noise floor
    string = np.zeros(n)
    nh = 14
    for k in range(1, nh + 1):
        amp = 1.0 / (k ** 1.3)
        vib = 1.0 + 0.004 * np.sin(2 * np.pi * 5.2 * t + k)
        string += amp * np.sin(2 * np.pi * f0 * k * t * vib)
    string /= np.max(np.abs(string)) + 1e-9
    string = string * 0.8 + rattle * 0.08

    x = (1 - m) * fridge + m * string

    attack = float(np.clip(e.get("rise_s", 0.3), 0.02, 1.5)) * (1.2 - 0.6 * m)
    x *= _env(n, attack, 0.25 * dur, sr)

    # the shut: an impact burst scaled by measured deceleration
    imp_n = int(0.12 * sr)
    if n > imp_n:
        rng = np.random.default_rng((int(e["t_start"]) + 7) % 2**31)
        burst = rng.standard_normal(imp_n) * np.exp(-np.arange(imp_n) / (0.012 * sr))
        gain = float(np.clip(e.get("close_decel", 0.5), 0, 4)) * 0.5 * (1 - 0.6 * m)
        x[n - imp_n:] += burst * gain

    peak = np.max(np.abs(x)) + 1e-9
    return (x / peak * 0.85).astype(np.float32)


def load_or_render(e, clips, idx, sr=SR):
    """Use a rendered clip if one exists, else synthesize a placeholder."""
    if clips:
        for name in (e.get("clip"), f"open_{idx:05d}.wav"):
            if not name:
                continue
            p = os.path.join(clips, name)
            if os.path.exists(p):
                import soundfile as sf
                x, fsr = sf.read(p, dtype="float32", always_2d=False)
                if x.ndim > 1:
                    x = x.mean(axis=1)
                if fsr != sr:
                    k = int(round(len(x) * sr / fsr))
                    x = np.interp(np.linspace(0, len(x) - 1, k),
                                  np.arange(len(x)), x).astype(np.float32)
                return x
    return render_event(e, sr)


# --------------------------------------------------------------------------
# arrangement
# --------------------------------------------------------------------------

def place(events, clips, t_of, total_s, sr=SR, tail=4.0):
    """Mix each event into a timeline. t_of(e) -> seconds on that timeline."""
    out = np.zeros(int((total_s + tail) * sr), dtype=np.float32)
    for i, e in enumerate(events):
        x = load_or_render(e, clips, i, sr)
        a = int(t_of(e) * sr)
        b = min(a + len(x), len(out))
        if a >= len(out) or b <= a:
            continue
        out[a:b] += x[: b - a]
    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak * 0.9).astype(np.float32)


def day_index(events):
    t0 = min(e["t_start"] for e in events)
    day0 = t0 - (t0 % 86400)
    for e in events:
        e["day"] = int((e["t_start"] - day0) // 86400)
        e["sec_of_day"] = (e["t_start"] - day0) % 86400
    return day0


def cmd_day(args):
    import soundfile as sf
    events = read_events(args.events)
    attach_morph(events, args.knee, args.m_max)
    day_index(events)
    sel = [e for e in events if e["day"] == args.day]
    if not sel:
        raise SystemExit(f"no events on day {args.day}")
    print(f"[day {args.day}] {len(sel)} openings -> {args.length:.0f} s")
    # 00:00..24:00 maps to 0..length, so days are directly comparable
    y = place(sel, args.clips, lambda e: e["sec_of_day"] / 86400 * args.length,
              args.length)
    sf.write(args.out, y, SR)
    print(f"[day {args.day}] wrote {args.out} ({len(y)/SR:.1f} s)")


def cmd_week(args):
    import soundfile as sf
    events = read_events(args.events)
    attach_morph(events, args.knee, args.m_max)
    day0 = day_index(events)
    ndays = max(e["day"] for e in events) + 1
    per = args.length / ndays
    print(f"[week] {len(events)} openings, {ndays} days, {per:.0f} s/day")
    y = place(events, args.clips,
              lambda e: e["day"] * per + e["sec_of_day"] / 86400 * per,
              args.length)
    sf.write(args.out, y, SR)
    print(f"[week] wrote {args.out} ({len(y)/SR:.1f} s)")


# --------------------------------------------------------------------------
# stacking and residual: the roteness device
# --------------------------------------------------------------------------

def envelope(x, hop=0.01, sr=SR):
    n = max(int(hop * sr), 1)
    k = len(x) // n
    if k < 2:
        return np.zeros(2)
    e = np.sqrt(np.mean(x[: k * n].reshape(k, n) ** 2, axis=1) + 1e-12)
    return e


def envelope_coherence(M):
    """Mean pairwise correlation between the events' amplitude envelopes."""
    E = np.array([envelope(row) for row in M])
    E = (E - E.mean(axis=1, keepdims=True)) / (E.std(axis=1, keepdims=True) + 1e-9)
    C = (E @ E.T) / E.shape[1]
    iu = np.triu_indices(len(E), k=1)
    return float(np.mean(C[iu]))


ROTE_KEYS = ["peak_open_dps", "extent_deg", "dwell_s", "rise_s", "close_decel"]


def cmd_rote(args):
    """Roteness as dispersion in feature space -- independent of how hard
    anyone pulls. A gentle person and a violent person can be equally rote."""
    events = read_events(args.events)
    groups = {"all": events}
    if args.labels:
        with open(args.labels) as fh:
            rows = list(csv.DictReader(fh))
        lt = np.array([float(r["t_start"]) for r in rows])
        names = [r[args.by] for r in rows]
        groups = {}
        for e in events:
            j = int(np.argmin(np.abs(lt - e["t_start"])))
            key = names[j] if abs(lt[j] - e["t_start"]) <= args.label_tol else "unlabelled"
            groups.setdefault(key, []).append(e)

    print(f"{'group':12s} {'n':>4s} " +
          " ".join(f"{k.split('_')[0][:6]:>7s}" for k in ROTE_KEYS) +
          f" {'mean CV':>9s} {'effort':>8s}")
    print("-" * (12 + 5 + 8 * len(ROTE_KEYS) + 19))
    for name, evs in sorted(groups.items()):
        if len(evs) < 3:
            continue
        cvs = []
        for k in ROTE_KEYS:
            v = np.array([e[k] for e in evs], dtype=float)
            cvs.append(np.std(v) / (np.abs(np.mean(v)) + 1e-9))
        effort = np.median([e["peak_open_dps"] for e in evs])
        print(f"{name:12s} {len(evs):4d} " +
              " ".join(f"{c:7.3f}" for c in cvs) +
              f" {np.mean(cvs):9.3f} {effort:8.0f}")
    print("\nLow mean CV = rote. Compare it against the effort column: if the")
    print("two do not track each other, roteness and force are independent,")
    print("which is the whole claim the piece rests on.")


def cmd_stack(args):
    import soundfile as sf
    events = read_events(args.events)
    attach_morph(events, args.knee, args.m_max)

    groups = {"all": events}
    if args.labels:
        with open(args.labels) as fh:
            rows = [r for r in csv.DictReader(fh)]
        lt = np.array([float(r["t_start"]) for r in rows])
        names = [r[args.by] for r in rows]
        groups = {}
        for e in events:
            # nearest label within tol: detected onset never matches a label
            # timestamp exactly, and on real data labels come from a diary
            j = int(np.argmin(np.abs(lt - e["t_start"])))
            key = names[j] if abs(lt[j] - e["t_start"]) <= args.label_tol else "unlabelled"
            e[args.by] = key
            groups.setdefault(key, []).append(e)

    os.makedirs(args.out, exist_ok=True)
    win = int(args.window * SR)
    print(f"{'group':12s} {'n':>4s} {'coherence':>10s}  (1.0 = perfectly rote)")

    for name, evs in sorted(groups.items()):
        if len(evs) < 3:
            continue
        M = np.zeros((len(evs), win), dtype=np.float32)
        for i, e in enumerate(evs):
            x = load_or_render(e, args.clips, i)
            M[i, : min(win, len(x))] = x[:win]       # aligned at pull onset

        mean = M.mean(axis=0)
        resid = M - mean

        # Coherence is measured on ENVELOPES, not waveforms. Two identical
        # gestures recorded minutes apart have uncorrelated fine phase, so raw
        # waveform averaging mostly measures pitch agreement and cancels
        # everything else. The envelope is what "same gesture" actually means.
        coh = envelope_coherence(M)
        print(f"{name:12s} {len(evs):4d} {coh:10.3f}")

        stack = M.sum(axis=0)
        sf.write(os.path.join(args.out, f"{name}_stack.wav"),
                 (stack / (np.max(np.abs(stack)) + 1e-9) * 0.9), SR)
        sf.write(os.path.join(args.out, f"{name}_mean.wav"),
                 (mean / (np.max(np.abs(mean)) + 1e-9) * 0.9), SR)
        flat = resid.reshape(-1)
        sf.write(os.path.join(args.out, f"{name}_residual.wav"),
                 (flat / (np.max(np.abs(flat)) + 1e-9) * 0.9), SR)

    print(f"\n[stack] wrote to {args.out}/")
    print("  *_stack.wav     every opening on top of each other")
    print("  *_mean.wav      the habit itself")
    print("  *_residual.wav  each opening minus the habit, in sequence")


# --------------------------------------------------------------------------
# manifest for FlowEdit / RAVE
# --------------------------------------------------------------------------

def cmd_manifest(args):
    events = read_events(args.events)
    knee = attach_morph(events, args.knee, args.m_max)
    day_index(events)
    with open(args.out, "w") as fh:
        for i, e in enumerate(events):
            fh.write(json.dumps({
                "id": f"open_{i:05d}",
                "source_wav": os.path.join(args.clips or "clips", f"open_{i:05d}.wav"),
                "t_start": e["t_start"],
                "day": e["day"],
                "sec_of_day": round(e["sec_of_day"], 1),
                "morph": round(e["morph"], 4),
                "target_prompt": args.prompt,
                "flowedit": {
                    # map morph onto however many steps your FlowEdit edits over
                    "n_max": int(round(e["morph"] * args.steps)),
                    "n_min": 0,
                    "src_guidance": args.src_guidance,
                    "tar_guidance": args.tar_guidance,
                },
                "rave": {
                    # drive latent dims from the gesture; dim 0 = the morph axis
                    "latent_bias": [round(e["morph"] * 2 - 1, 4),
                                    round(np.tanh(e["extent_deg"] / 90.0), 4),
                                    round(np.tanh(e["dwell_s"] / 8.0), 4)],
                },
                "features": {k: e[k] for k in
                             ("peak_open_dps", "extent_deg", "dwell_s",
                              "close_decel", "rise_s", "n_vel_peaks",
                              "smoothness", "reopen")},
            }) + "\n")
    print(f"[manifest] {len(events)} entries -> {args.out} (v_knee={knee:.1f} dps)")
    print(f"[manifest] n_max spans {min(int(round(e['morph']*args.steps)) for e in events)}"
          f"..{max(int(round(e['morph']*args.steps)) for e in events)} of {args.steps} steps")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("events")
        p.add_argument("--knee", type=float, default=None,
                       help="velocity at the saturation knee (default: median)")
        p.add_argument("--m-max", dest="m_max", type=float, default=1.0)
        p.add_argument("--clips", default=None, help="dir of per-opening WAVs")

    p = sub.add_parser("map"); common(p); p.set_defaults(fn=cmd_map)

    p = sub.add_parser("day"); common(p)
    p.add_argument("--day", type=int, default=0)
    p.add_argument("--length", type=float, default=90.0)
    p.add_argument("-o", "--out", default="day.wav"); p.set_defaults(fn=cmd_day)

    p = sub.add_parser("week"); common(p)
    p.add_argument("--length", type=float, default=630.0)
    p.add_argument("-o", "--out", default="week.wav"); p.set_defaults(fn=cmd_week)

    p = sub.add_parser("stack"); common(p)
    p.add_argument("--window", type=float, default=6.0)
    p.add_argument("--by", default="person")
    p.add_argument("--labels", default=None,
                   help="csv with t_start + a label column")
    p.add_argument("--label-tol", type=float, default=5.0,
                   help="seconds a label may be off by and still match")
    p.add_argument("-o", "--out", default="stack"); p.set_defaults(fn=cmd_stack)

    p = sub.add_parser("rote"); common(p)
    p.add_argument("--by", default="person")
    p.add_argument("--labels", default=None)
    p.add_argument("--label-tol", type=float, default=5.0)
    p.set_defaults(fn=cmd_rote)

    p = sub.add_parser("manifest"); common(p)
    p.add_argument("--prompt", default="bowed double bass, sustained, close-miked")
    p.add_argument("--steps", type=int, default=28,
                   help="total diffusion steps your FlowEdit runs")
    p.add_argument("--src-guidance", type=float, default=1.5)
    p.add_argument("--tar-guidance", type=float, default=5.5)
    p.add_argument("-o", "--out", default="manifest.jsonl")
    p.set_defaults(fn=cmd_manifest)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
