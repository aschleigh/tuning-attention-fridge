"""Log the imu2osc OSC stream to disk.

Run alongside the repo, in a second terminal:

    uv run run.py --last          # terminal 1: IMU -> OSC on localhost:45531
    python logger.py --out data/  # terminal 2: OSC -> gzipped CSV

Writes one file per hour so a crash costs you an hour, not a week.
Columns: t (unix seconds), AX AY AZ (g), GX GY GZ (deg/s), angX angY angZ (deg).

The ang* channels are logged for reference only. They drift and they are
referenced to wherever the IMU was when the script started, so segment.py
re-integrates door angle from the gyro instead. See README.
"""

import argparse
import csv
import gzip
import os
import time

from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server

FIELDS = ["AX", "AY", "AZ", "GX", "GY", "GZ", "angX", "angY", "angZ"]


class Rotator:
    """Append rows to an hourly gzipped CSV, opening new files as time passes."""

    def __init__(self, outdir, prefix="imu"):
        self.outdir = outdir
        self.prefix = prefix
        self.hour = None
        self.fh = None
        self.writer = None
        os.makedirs(outdir, exist_ok=True)

    def _roll(self, now):
        hour = time.strftime("%Y%m%d_%H", time.localtime(now))
        if hour == self.hour:
            return
        if self.fh:
            self.fh.close()
        path = os.path.join(self.outdir, f"{self.prefix}_{hour}.csv.gz")
        fresh = not os.path.exists(path)
        self.fh = gzip.open(path, "at", newline="")
        self.writer = csv.writer(self.fh)
        if fresh:
            self.writer.writerow(["t"] + FIELDS)
        self.hour = hour
        print(f"[logger] writing {path}")

    def row(self, now, values):
        self._roll(now)
        self.writer.writerow([f"{now:.3f}"] + [f"{v:.4f}" for v in values])

    def flush(self):
        if self.fh:
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=45531)
    ap.add_argument("--prefix", default="imu", help="filename prefix, e.g. fridge / frontdoor")
    ap.add_argument("--flush-every", type=float, default=2.0,
                    help="seconds between fsync-ish flushes")
    args = ap.parse_args()

    rot = Rotator(args.out, args.prefix)
    state = {k: 0.0 for k in FIELDS}
    seen = set()
    last_flush = [0.0]
    count = [0]

    def handle(addr, *vals):
        if not vals:
            return
        key = addr.rsplit("/", 1)[-1]
        if key not in state:
            return
        state[key] = float(vals[0])
        seen.add(key)
        # imu2osc sends the nine values as one burst per frame; emit a row once
        # every channel of a frame has arrived, then start the next frame.
        if len(seen) == len(FIELDS):
            now = time.time()
            rot.row(now, [state[k] for k in FIELDS])
            seen.clear()
            count[0] += 1
            if now - last_flush[0] > args.flush_every:
                rot.flush()
                last_flush[0] = now
                print(f"\r[logger] {count[0]} frames", end="", flush=True)

    disp = osc_dispatcher.Dispatcher()
    disp.set_default_handler(handle)
    server = osc_server.BlockingOSCUDPServer((args.host, args.port), disp)
    print(f"[logger] listening on {args.host}:{args.port}, writing to {args.out}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        rot.close()
        print(f"\n[logger] closed after {count[0]} frames")


if __name__ == "__main__":
    main()
