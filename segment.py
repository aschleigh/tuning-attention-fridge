"""Turn raw logs into one row per door opening.

    python segment.py "data/*.csv.gz" --out events.csv
    python segment.py "data/*.csv.gz" --out events.csv --set hi_dps=35 lo_dps=8
    python segment.py "data/*.csv.gz" --out events.csv \
        --audio kitchen.wav --audio-start 1763600000 --cut clips/

--cut writes one WAV per opening from a continuously-recorded contact mic.
--audio-start is the unix timestamp of sample 0 of that WAV. Get it by
clapping/slamming once at the start of the recording with the logger already
running, then line up that transient; or just note time.time() when you hit
record and accept a few hundred ms of error.
"""

import argparse
import csv
import os

import numpy as np

import fridgelib as fl


def parse_sets(pairs):
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        if k not in fl.Config.__dataclass_fields__:
            raise SystemExit(f"unknown config key: {k}")
        typ = fl.Config.__dataclass_fields__[k].type
        out[k] = int(v) if typ is int else float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", help="path or glob, e.g. 'data/*.csv.gz'")
    ap.add_argument("--out", default="events.csv")
    ap.add_argument("--set", nargs="*", dest="sets", help="cfg overrides k=v")
    ap.add_argument("--audio", help="continuous contact-mic WAV")
    ap.add_argument("--audio-start", type=float, help="unix time of audio sample 0")
    ap.add_argument("--cut", help="directory to write per-opening WAVs into")
    ap.add_argument("--pad", type=float, default=0.5, help="seconds around each cut")
    ap.add_argument("--profile", default="door", choices=["door", "faucet", "kettle"],
                    help="door=integrate+reset gyro; faucet/kettle=absolute tilt from gravity")
    ap.add_argument("--chunk", type=int, default=24,
                    help="log files processed at once (24 hourly files = 1 day)")
    args = ap.parse_args()

    tilt_mode = args.profile in fl.TILT_PROFILES
    cfg = fl.PROFILES[args.profile].replace(**parse_sets(args.sets))
    files = fl.expand(args.logs)

    head = fl.load(files[: min(3, len(files))])
    probe = fl.prepare_tilt(head, cfg) if tilt_mode else fl.prepare(head, cfg)
    print(f"[seg] profile={args.profile}  {len(files)} files, {probe['fs']:.1f} Hz")
    if tilt_mode:
        print(f"[seg] rest pose g0 = {np.round(probe['g0'], 3).tolist()}")
        print(f"[seg] tilt range {probe['tilt'].min():.1f} .. {probe['tilt'].max():.1f} deg")
    else:
        print(f"[seg] hinge axis = {'XYZ'[probe['axis']]} "
              f"(gyro std {np.std(probe['G'], axis=0).round(2).tolist()})")
        print(f"[seg] gyro bias range {probe['bias'].min():.2f} .. "
              f"{probe['bias'].max():.2f} deg/s")
        if cfg.hinge_axis < 0:
            cfg = cfg.replace(hinge_axis=probe["axis"])   # lock it for every chunk
    for a, b in fl.gaps(probe["t"])[:5]:
        print(f"[seg] dropout {b-a:.1f}s at t={a:.0f}")
    del probe, head

    events, _ = fl.segment_all(files, cfg, per_chunk=args.chunk, profile=args.profile)
    if not events:
        raise SystemExit("[seg] no events -- lower hi_dps/open_deg and retry")

    span_h = (events[-1].t_end - events[0].t_start) / 3600
    per_day = len(events) / max(span_h / 24, 1e-9)
    print(f"[seg] {len(events)} events ({per_day:.0f}/day)")
    if tilt_mode:
        pk = [e.peak_tilt_deg for e in events]
        pl = [e.plateau_s for e in events]
        yw = [e.yaw_deg for e in events]
        print(f"[seg] peak tilt median {np.median(pk):.0f} deg  range {min(pk):.0f}-{max(pk):.0f}")
        print(f"[seg] plateau  median {np.median(pl):.1f} s    range {min(pl):.1f}-{max(pl):.1f}")
        print(f"[seg] yaw/use  median {np.median(yw):.0f} deg  "
              f"(0 = never adjusts the temperature)")
    else:
        ext = [e.extent_deg for e in events]
        vel = [e.peak_open_dps for e in events]
        print(f"[seg] extent  median {np.median(ext):.0f} deg   range {min(ext):.0f}-{max(ext):.0f}")
        print(f"[seg] peak vel median {np.median(vel):.0f} dps  range {min(vel):.0f}-{max(vel):.0f}")
    print(f"[seg] reopen<{cfg.reopen_s:.0f}s: {sum(e.reopen for e in events)}   "
          f"clipped: {sum(e.clipped for e in events)}")

    keys = list((fl.TiltEvent if tilt_mode else fl.Event).__dataclass_fields__.keys())
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for e in events:
            w.writerow(e.as_row())
    print(f"[seg] wrote {args.out}")

    if args.cut:
        cut_audio(events, args, cfg)


def cut_audio(events, args, cfg):
    import soundfile as sf
    if args.audio_start is None:
        raise SystemExit("--cut needs --audio-start")
    os.makedirs(args.cut, exist_ok=True)
    info = sf.info(args.audio)
    sr = info.samplerate
    n_written = 0
    for i, e in enumerate(events):
        a = (e.t_start - args.audio_start - args.pad) * sr
        b = (e.t_end - args.audio_start + args.pad) * sr
        if b <= 0 or a >= info.frames:
            continue
        a, b = int(max(a, 0)), int(min(b, info.frames))
        x, _ = sf.read(args.audio, start=a, stop=b, dtype="float32")
        sf.write(os.path.join(args.cut, f"open_{i:05d}.wav"), x, sr)
        n_written += 1
    print(f"[seg] cut {n_written} clips into {args.cut}/")


if __name__ == "__main__":
    main()
