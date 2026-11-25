"""Shared analysis for the refrigerator-door piece.

Why this file re-derives angle instead of using /imu/ang*:

  1. A fridge door turns about a VERTICAL hinge. Gravity is identical at 0 deg
     and at 90 deg, so the accelerometer cannot see door angle at all. The gyro
     is the only channel that knows.
  2. imu2osc's ang* is integrated from wherever the IMU sat at startup and
     drifts over hours. Over a week that is useless.

What saves us is the magnetic gasket: the door has exactly one resting
position and snaps to it. So we integrate the gyro during motion and force the
angle back to zero whenever the door comes to rest closed. Drift inside a
single 2-10 s opening is negligible.
"""

from dataclasses import dataclass, asdict, field
import glob
import gzip
import os

import numpy as np

FIELDS = ["AX", "AY", "AZ", "GX", "GY", "GZ", "angX", "angY", "angZ"]


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class Config:
    """Everything you will want to tune by listening. See README 'Tuning'."""

    hi_dps: float = 25.0        # motion starts above this |gyro|, deg/s
    lo_dps: float = 6.0         # motion ends below this |gyro|, deg/s
    quiet_s: float = 0.35       # how long under lo_dps counts as "stopped"
    open_deg: float = 8.0       # angle that makes a motion count as an opening
    close_deg: float = 5.0      # angle under which the door is back shut
    force_close_s: float = 120.0  # give up and close the event after this long
    min_event_s: float = 0.3    # shorter than this is a bump, not an opening
    reopen_s: float = 30.0      # your "reopened within 30 s" flag
    smooth_s: float = 0.05      # moving-average window on the gyro
    bias_window_s: float = 2.0  # quiet runs at least this long update gyro bias
    hinge_axis: int = -1        # -1 = autodetect, else 0/1/2 for GX/GY/GZ
    # tilt-mode (faucet, kettle): absolute angle from gravity, no integration
    tilt_on_deg: float = 10.0   # tilt above this starts an event
    tilt_off_deg: float = 5.0   # ...and below this ends it
    tilt_peak_deg: float = 15.0 # events never reaching this are discarded
    tilt_gap_s: float = 1.5     # dips below threshold shorter than this bridge

    def replace(self, **kw):
        d = asdict(self)
        d.update(kw)
        return Config(**d)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def expand(paths):
    if isinstance(paths, str):
        paths = sorted(glob.glob(paths)) if any(c in paths for c in "*?[") else [paths]
    if not paths:
        raise FileNotFoundError("no log files matched")
    return sorted(paths)


def _read_one(p):
    try:
        import pandas as pd
        df = pd.read_csv(p, dtype=np.float64)
        if list(df.columns)[:1] != ["t"]:
            raise ValueError(f"{p}: first column must be 't'")
        return df.to_numpy()
    except ImportError:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt") as fh:
            fh.readline()
            return np.loadtxt(fh, delimiter=",", ndmin=2)


def load(paths):
    """Load logger CSVs (plain or .gz) into a dict of arrays.

    Fine for a day. For a whole week use stream() instead -- 7 days at 50 Hz is
    ~30M rows, which is several GB once it is float64 in RAM.
    """
    chunks = [a for a in (_read_one(p) for p in expand(paths)) if a.size]
    arr = np.vstack(chunks)
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    arr = arr[np.diff(arr[:, 0], prepend=-np.inf) > 0]   # drop dup timestamps

    out = {"t": arr[:, 0]}
    for i, name in enumerate(FIELDS):
        out[name] = arr[:, i + 1]
    out["A"] = np.stack([out["AX"], out["AY"], out["AZ"]], axis=1)
    out["G"] = np.stack([out["GX"], out["GY"], out["GZ"]], axis=1)
    return out


def stream(paths, per_chunk=24, overlap_files=1):
    """Yield the logs a chunk of files at a time, with one file of overlap.

    The overlap means an opening that straddles a file boundary is seen whole
    by at least one chunk; segment_all() then drops the duplicate.
    """
    files = expand(paths)
    i = 0
    while i < len(files):
        lo = max(i - overlap_files, 0)
        hi = min(i + per_chunk, len(files))
        yield load(files[lo:hi]), (lo, hi, len(files))
        i = hi


def sample_rate(t):
    """Median rate, robust to the BLE dropouts you will definitely get."""
    dt = np.diff(t)
    dt = dt[(dt > 0) & (dt < 1.0)]
    if dt.size == 0:
        raise ValueError("cannot estimate sample rate")
    return float(1.0 / np.median(dt))


def gaps(t, tol=3.0):
    """Return (start, end) of dropouts longer than tol x the median period."""
    dt = np.diff(t)
    med = np.median(dt[(dt > 0) & (dt < 1.0)])
    idx = np.flatnonzero(dt > tol * med)
    return [(float(t[i]), float(t[i + 1])) for i in idx]


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

def detect_hinge_axis(G):
    """The hinge axis is whichever gyro channel actually swings."""
    return int(np.argmax(np.std(G, axis=0)))


def _movavg(x, n):
    if n < 2:
        return x.copy()
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def track_bias(t, g, cfg, fs):
    """Gyro zero-offset, tracked over time.

    A fridge is a temperature-cycling box and the IMU sits on it, so the gyro
    offset wanders. Take the median of g over every long quiet run, then
    interpolate between those anchors.
    """
    quiet = np.abs(g - np.median(g)) < cfg.lo_dps
    runs = _runs(quiet)
    min_len = max(int(cfg.bias_window_s * fs), 5)
    centres, values = [], []
    for a, b in runs:
        if b - a >= min_len:
            centres.append(0.5 * (t[a] + t[b - 1]))
            values.append(np.median(g[a:b]))
    if len(centres) < 2:
        return np.full_like(g, np.median(g[quiet]) if quiet.any() else 0.0)
    return np.interp(t, np.array(centres), np.array(values))


def _runs(mask):
    """[(start, stop)] index pairs for each run of True in a boolean array."""
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def prepare(data, cfg):
    """Attach fs, hinge-axis gyro (bias-corrected and smoothed), accel magnitude."""
    t = data["t"]
    fs = sample_rate(t)
    axis = cfg.hinge_axis if cfg.hinge_axis >= 0 else detect_hinge_axis(data["G"])
    g_raw = data["G"][:, axis]
    bias = track_bias(t, g_raw, cfg, fs)
    g = _movavg(g_raw - bias, max(int(cfg.smooth_s * fs), 1))
    amag = np.linalg.norm(data["A"], axis=1)
    data = dict(data)
    data.update(fs=fs, axis=axis, g=g, g_raw=g_raw, bias=bias, amag=amag)
    return data


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------

@dataclass
class Event:
    i0: int
    i1: int
    t_start: float
    t_end: float
    duration: float = 0.0
    extent_deg: float = 0.0
    peak_open_dps: float = 0.0
    peak_close_dps: float = 0.0
    rise_s: float = 0.0         # motion onset -> peak opening velocity
    dwell_s: float = 0.0        # door held open, not moving
    close_decel: float = 0.0    # impact magnitude in g at the shut
    n_vel_peaks: int = 0        # 1 = one smooth pull; more = fumbling
    smoothness: float = 0.0     # log dimensionless jerk; lower = smoother
    reopen: bool = False
    clipped: bool = False       # hit force_close_s, treat as suspect

    def as_row(self):
        return asdict(self)


def segment(data, cfg):
    """State machine over the hinge-axis gyro. One Event per door opening.

    CLOSED -> SWING (integrating) -> DWELL (angle held) -> ... -> CLOSED

    The event ends when the integrated angle comes back under close_deg and the
    door stops moving, NOT when motion stops -- otherwise every long browse
    would split into two events.
    """
    t, g, fs = data["t"], data["g"], data["fs"]
    n = len(t)
    quiet_n = max(int(cfg.quiet_s * fs), 1)

    events = []
    state = "CLOSED"
    angle = 0.0
    quiet_count = 0
    i_start = 0
    t_rest = t[0]

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0 or dt > 1.0:          # dropout: do not integrate across it
            dt = 0.0
        moving = abs(g[i]) > (cfg.hi_dps if state == "CLOSED" else cfg.lo_dps)

        if moving:
            quiet_count = 0
            angle += g[i] * dt
            if state == "CLOSED":
                state = "SWING"
                i_start = i
                t_rest = t[i]
        else:
            quiet_count += 1
            if state != "CLOSED" and quiet_count == quiet_n:
                state = "DWELL"
                t_rest = t[i]

        if state != "CLOSED":
            shut = abs(angle) < cfg.close_deg and quiet_count >= quiet_n
            stuck = (t[i] - t_rest) > cfg.force_close_s and quiet_count >= quiet_n
            if shut or stuck:
                if abs(angle) >= cfg.close_deg:
                    pass                  # left open / drifted; flagged below
                ev = Event(i0=i_start, i1=i, t_start=float(t[i_start]),
                           t_end=float(t[i]), clipped=bool(stuck))
                if ev.t_end - ev.t_start >= cfg.min_event_s:
                    events.append(ev)
                state, angle, quiet_count = "CLOSED", 0.0, 0

    events = [e for e in events if _measure(e, data, cfg)]
    for a, b in zip(events, events[1:]):
        a.reopen = (b.t_start - a.t_end) <= cfg.reopen_s
    return events


def _measure(ev, data, cfg):
    """Fill in an event's features. Returns False if it fails the open_deg test."""
    t, g, fs = data["t"], data["g"], data["fs"]
    sl = slice(ev.i0, ev.i1 + 1)
    tt, gg = t[sl], g[sl]
    if len(tt) < 3:
        return False

    dt = np.diff(tt, prepend=tt[0])
    dt[(dt <= 0) | (dt > 1.0)] = 0.0
    ang = np.cumsum(gg * dt)

    sign = 1.0 if ang[np.argmax(np.abs(ang))] >= 0 else -1.0
    ang = ang * sign
    gg = gg * sign                       # now positive = opening direction

    ev.extent_deg = float(np.max(ang))
    if ev.extent_deg < cfg.open_deg:
        return False

    k_ext = int(np.argmax(ang))
    ev.duration = float(tt[-1] - tt[0])
    ev.peak_open_dps = float(np.max(gg[: k_ext + 1])) if k_ext > 0 else float(np.max(gg))
    ev.peak_close_dps = float(-np.min(gg[k_ext:])) if k_ext < len(gg) - 1 else 0.0
    ev.rise_s = float(tt[int(np.argmax(gg[: k_ext + 1]))] - tt[0]) if k_ext > 0 else 0.0

    # dwell: time inside the event where the door is open but not moving
    still = (np.abs(gg) < cfg.lo_dps) & (ang > cfg.close_deg)
    ev.dwell_s = float(np.sum(dt[still]))

    # impact at the shut: accel excursion away from 1 g in the last 0.5 s
    tail = max(ev.i1 - int(0.5 * fs), ev.i0)
    ev.close_decel = float(np.max(np.abs(data["amag"][tail: ev.i1 + 1] - 1.0)))

    # how many times velocity re-peaks: 1 = one committed pull, >1 = fumbling
    opening = gg[: k_ext + 1]
    if opening.size > 3:
        thr = 0.25 * max(opening.max(), 1e-6)
        above = opening > thr
        ev.n_vel_peaks = max(len(_runs(above)), 1)
    else:
        ev.n_vel_peaks = 1

    # log dimensionless jerk over the opening swing (Balasubramanian et al.)
    if opening.size > 4 and ev.duration > 0:
        jerk = np.gradient(opening, tt[: k_ext + 1])
        peak = max(np.max(np.abs(opening)), 1e-6)
        dur = max(tt[k_ext] - tt[0], 1e-3)
        dj = (dur ** 3 / peak ** 2) * np.trapezoid(jerk ** 2, tt[: k_ext + 1])
        ev.smoothness = float(np.log(max(dj, 1e-12)))
    return True


# --------------------------------------------------------------------------
# tilt: for objects whose motion IS visible to gravity
# --------------------------------------------------------------------------
#
# The door was the hard case: a vertical hinge is invisible to the
# accelerometer, so angle had to be integrated and reset. A faucet lever and a
# kettle both rotate about HORIZONTAL axes, so gravity swings with them and the
# accelerometer reads absolute angle directly -- no integration, no drift, no
# reset trick, no bias tracking. This is the easier and more accurate path, and
# it is why these two objects are better instrumented than the door, not worse.

def resting_gravity(A, amag, still_tol=0.03, min_frac=0.02):
    """Median accel vector over the still periods = the object's rest pose."""
    still = np.abs(amag - 1.0) < still_tol
    if still.sum() < max(int(min_frac * len(A)), 20):
        still = np.abs(amag - 1.0) < (still_tol * 4)
    if still.sum() < 10:
        g0 = np.median(A, axis=0)
    else:
        g0 = np.median(A[still], axis=0)
    n = np.linalg.norm(g0)
    return g0 / (n + 1e-9)


def tilt_series(A, g0):
    """Angle in degrees between the current accel vector and the rest pose.

    Only meaningful while the object is roughly static -- during a fast lift the
    accelerometer reads gravity plus the motion. Smoothing plus the fact that
    pours and lever-holds have a plateau makes this work in practice; the peak
    is taken from the plateau, not the transient.
    """
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    c = np.clip((A / n) @ g0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


@dataclass
class TiltEvent:
    i0: int
    i1: int
    t_start: float
    t_end: float
    duration: float = 0.0
    peak_tilt_deg: float = 0.0     # pour angle / lever lift height
    plateau_s: float = 0.0         # time held near peak: run time, pour hold
    rise_s: float = 0.0
    peak_rate_dps: float = 0.0
    yaw_deg: float = 0.0           # rotation about the gravity axis during use
    settle_impact: float = 0.0     # accel excess at the end: setting it down
    smoothness: float = 0.0
    reopen: bool = False
    clipped: bool = False

    def as_row(self):
        return asdict(self)


def prepare_tilt(data, cfg):
    t = data["t"]
    fs = sample_rate(t)
    amag = np.linalg.norm(data["A"], axis=1)
    g0 = resting_gravity(data["A"], amag)
    w = max(int(cfg.smooth_s * 4 * fs), 1)
    tilt = _movavg(tilt_series(data["A"], g0), w)
    gmag = _movavg(np.linalg.norm(data["G"], axis=1), max(int(cfg.smooth_s * fs), 1))
    yaw_rate = data["G"] @ g0            # rotation about the gravity axis
    data = dict(data)
    data.update(fs=fs, axis=-1, g=gmag, g0=g0, tilt=tilt,
                yaw_rate=yaw_rate, amag=amag, bias=np.zeros(len(t)))
    return data


def segment_tilt(data, cfg):
    """Hysteresis on absolute tilt. One event per lift / pour / lever-raise."""
    t, tilt, fs = data["t"], data["tilt"], data["fs"]
    active = tilt > cfg.tilt_on_deg
    events = []
    gap_n = max(int(cfg.tilt_gap_s * fs), 1)

    spans = _runs(active)
    merged = []
    for a, b in spans:                   # bridge brief dips below threshold
        if merged and a - merged[-1][1] <= gap_n:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    for a, b in merged:
        # extend outward to where tilt falls under the release threshold
        while a > 0 and tilt[a - 1] > cfg.tilt_off_deg:
            a -= 1
        while b < len(tilt) - 1 and tilt[b] > cfg.tilt_off_deg:
            b += 1
        if t[b - 1] - t[a] < cfg.min_event_s:
            continue
        ev = TiltEvent(i0=a, i1=b - 1, t_start=float(t[a]), t_end=float(t[b - 1]))
        if _measure_tilt(ev, data, cfg):
            events.append(ev)

    for x, y in zip(events, events[1:]):
        x.reopen = (y.t_start - x.t_end) <= cfg.reopen_s
    return events


def _measure_tilt(ev, data, cfg):
    sl = slice(ev.i0, ev.i1 + 1)
    tt = data["t"][sl]
    th = data["tilt"][sl]
    if len(tt) < 4:
        return False
    dt = np.diff(tt, prepend=tt[0])
    dt[(dt <= 0) | (dt > 1.0)] = 0.0

    ev.duration = float(tt[-1] - tt[0])
    ev.peak_tilt_deg = float(np.max(th))
    if ev.peak_tilt_deg < cfg.tilt_peak_deg:
        return False

    k = int(np.argmax(th))
    ev.rise_s = float(tt[k] - tt[0])
    ev.plateau_s = float(np.sum(dt[th > 0.8 * ev.peak_tilt_deg]))
    rate = np.gradient(th, tt)
    ev.peak_rate_dps = float(np.max(np.abs(rate[: k + 1]))) if k > 0 else 0.0
    ev.yaw_deg = float(np.sum(np.abs(data["yaw_rate"][sl]) * dt))

    tail = max(ev.i1 - int(0.5 * data["fs"]), ev.i0)
    ev.settle_impact = float(np.max(np.abs(data["amag"][tail: ev.i1 + 1] - 1.0)))

    if k > 4:
        jerk = np.gradient(rate[: k + 1], tt[: k + 1])
        peak = max(np.max(np.abs(rate[: k + 1])), 1e-6)
        dur = max(tt[k] - tt[0], 1e-3)
        dj = (dur ** 3 / peak ** 2) * np.trapezoid(jerk ** 2, tt[: k + 1])
        ev.smoothness = float(np.log(max(dj, 1e-12)))
    return True


# object presets -----------------------------------------------------------

PROFILES = {
    "door":   Config(),
    "faucet": Config(tilt_on_deg=6.0, tilt_off_deg=3.0, tilt_peak_deg=10.0,
                     tilt_gap_s=1.5, min_event_s=0.8, reopen_s=20.0,
                     smooth_s=0.08),
    "kettle": Config(tilt_on_deg=12.0, tilt_off_deg=6.0, tilt_peak_deg=20.0,
                     tilt_gap_s=2.0, min_event_s=0.6, reopen_s=60.0,
                     smooth_s=0.10),
}

TILT_PROFILES = {"faucet", "kettle"}


def segment_all(paths, cfg, per_chunk=24, verbose=True, profile="door"):
    """Segment an arbitrarily long capture without loading it all at once."""
    tilt_mode = profile in TILT_PROFILES
    found, info = [], {}
    for data, (lo, hi, total) in stream(paths, per_chunk=per_chunk):
        data = prepare_tilt(data, cfg) if tilt_mode else prepare(data, cfg)
        info.setdefault("fs", data["fs"])
        info.setdefault("axis", data["axis"])
        evs = segment_tilt(data, cfg) if tilt_mode else segment(data, cfg)
        found.extend(evs)
        if verbose:
            print(f"[seg] files {lo}-{hi}/{total}: {len(evs)} openings", flush=True)

    found.sort(key=lambda e: e.t_start)
    deduped = []
    for e in found:                       # overlap can yield the same opening twice
        if deduped and abs(e.t_start - deduped[-1].t_start) < 0.5:
            continue
        deduped.append(e)
    for a, b in zip(deduped, deduped[1:]):
        a.reopen = (b.t_start - a.t_end) <= cfg.reopen_s
    return deduped, info


def events_to_array(events, keys):
    return np.array([[getattr(e, k) for k in keys] for e in events], dtype=float)
