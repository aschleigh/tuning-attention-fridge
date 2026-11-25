# Portrait of a Refrigerator — capture & composition pipeline

Wraps [`alextongue/imu2osc`](https://github.com/alextongue/imu2osc) without modifying it.
`run.py` streams OSC; everything here listens downstream.

```
run.py ──OSC:45531──▶ logger.py ──▶ data/*.csv.gz
                                          │
                                   segment.py ──▶ events.csv  (+ clips/*.wav)
                                          │
              ┌───────────────────────────┼───────────────────────┐
         compose.py rote            compose.py manifest      compose.py day/week/stack
        (is it rote?)            (→ FlowEdit / RAVE)         (arrangement)
```

## Install

```bash
pip install python-osc numpy pandas soundfile
```

## Run it on fake data first

No hardware needed. This is worth doing before you mount anything, because it
tells you whether your thresholds and your metrics work at all.

```bash
python simulate.py --days 7 --out data/ --truth truth.csv
python segment.py "data/*.csv.gz" --out events.csv
python validate.py events.csv truth.csv          # expect ~99% recall
python compose.py rote events.csv --labels truth.csv --by person
python compose.py map  events.csv
python compose.py day  events.csv --day 0 -o day0.wav
python compose.py stack events.csv --labels truth.csv -o stack/
```

The simulator has four roommates with deliberately dissociated effort and
consistency: `slammer` is forceful and rote, `quiet` is gentle and rote,
`erratic` is neither. If your analysis cannot separate those, it is measuring
force when you wanted habit.

## Capture

```bash
uv run run.py --last                        # terminal 1
python logger.py --out data/ --prefix fridge  # terminal 2
```

One gzipped CSV per hour, so a crash costs an hour and not a week. At 50 Hz
expect roughly 40 MB/day compressed.

### Mounting

- **Handle side, flat on the door face.** Angular velocity is identical
  anywhere on the door, but the slam shows up as *linear* accel and scales
  with distance from the hinge.
- **Push the output rate up** as far as BLE will carry it (50 Hz+). The slam
  transient is the shortest thing you are measuring. Check what you actually
  get — `segment.py` prints the measured rate; don't trust the spec sheet.
- **Ignore the magnetometer** if you're on a BWT901CL. You are strapping it to
  a steel box with a magnet strip around the door.
- **Power.** The battery will not last a week. Run USB power, or charge nightly
  and decide whether the gap is a hole in the piece or a rest in it.
- **Wrap `run.py` in a restart loop.** BLE will drop.
  `while true; do uv run run.py --last; sleep 5; done`

### Why angle is re-derived instead of using `/imu/ang*`

A fridge door turns about a **vertical** hinge, so gravity is identical at 0°
and 90° — the accelerometer cannot see door angle at all. And imu2osc's `ang*`
is integrated from wherever the IMU sat at startup and drifts over hours; the
README's own footnote warns about this. The gyro is the only channel that
knows, and the magnetic gasket gives it a reliable zero: integrate during
motion, force back to zero whenever the door comes to rest shut. Drift inside
one 2–10 s opening is negligible. `fridgelib.track_bias` also re-estimates the
gyro offset continuously, because a fridge is a thermal cycler and the offset
wanders.

## Segmenting

```bash
python segment.py "data/*.csv.gz" --out events.csv
python segment.py "data/*.csv.gz" --out events.csv --set hi_dps=35 open_deg=12
```

Streams a day at a time, so a week doesn't have to fit in RAM. One row per
opening with: `extent_deg`, `peak_open_dps`, `peak_close_dps`, `rise_s`,
`dwell_s`, `close_decel`, `n_vel_peaks`, `smoothness`, `reopen`, `clipped`.

The event ends when the integrated angle returns to zero, **not** when motion
stops — otherwise every long browse splits into two events.

`n_vel_peaks` is 1 for one committed pull and higher for fumbling.
`smoothness` is log dimensionless jerk; lower is smoother. Those two are your
best roteness channels, better than peak velocity.

### Tuning

Cut clips and listen — that was always your plan and it's the right one:

```bash
python segment.py "data/*.csv.gz" --out events.csv \
    --audio kitchen.wav --audio-start 1763600000 --cut clips/
```

Get `--audio-start` by slamming the door once at the top of the recording with
the logger already running, then lining up that transient.

| symptom | knob |
| --- | --- |
| one opening split in two | raise `quiet_s`, raise `close_deg` |
| two openings merged | lower `force_close_s`, lower `close_deg` |
| bumps counted as openings | raise `open_deg`, raise `hi_dps` |
| gentle openings missed | lower `hi_dps`, lower `open_deg` |
| many `clipped=True` | bias tracking is failing — check for a long dropout |

## Measuring roteness

```bash
python compose.py rote events.csv --labels diary.csv --by person
```

Dispersion in feature space (coefficient of variation per feature), reported
next to median effort. The point is to watch those two columns *fail* to track
each other. If roteness and force were the same thing you wouldn't need the
piece.

`--labels` wants a CSV with `t_start` and a label column, matched to the
nearest event within `--label-tol` seconds. On real data that's a diary, or
the cluster assignments if you go unsupervised — in which case check whether
your clusters are people or just times of day. If it's times of day, that is a
real finding and worth reporting as one.

## The mapping

```bash
python compose.py map events.csv
```

```
morph = m_max · tanh(peak_open_dps / v_knee)      v_knee defaults to the median
```

Prints how much morph range each velocity quartile receives. On the test data
the bottom quartile gets 0.315 of range and the top gets 0.038 — an 8× collapse
in discrimination. That is the Feldenkrais claim as a transfer function: the
instrument goes insensitive exactly where the body does. Lower `--knee` until
the top quartile is visibly starved.

## Rendering

```bash
python compose.py day   events.csv --day 2 -o day2.wav --length 90
python compose.py week  events.csv -o week.wav --length 630
python compose.py stack events.csv --labels diary.csv -o stack/
```

Days map 00:00–24:00 onto the same 90 s, so they're directly comparable.

`stack` writes three things per group:

- `*_stack.wav` — every opening on top of each other, aligned at pull onset.
  Rote gestures thicken; variable ones smear.
- `*_mean.wav` — the habit itself.
- `*_residual.wav` — each opening minus the habit, in sequence. In a rote
  household this is nearly silent, with a few events in it: the 3 a.m. visit,
  the day someone came in with both arms full. That near-silence is a better
  portrait than the raw week.

Coherence in the `stack` report is measured on **envelopes**, not waveforms.
Two identical gestures minutes apart have uncorrelated fine phase, so raw
waveform averaging mostly measures pitch agreement.

Without `--clips` everything renders through a crude placeholder synth
(fridge-ish noise at morph 0, bowed-string-ish at morph 1) so you can hear the
segmentation and the mapping on day one. Point `--clips` at your model output
and the same arrangement code handles the real material.

## Into FlowEdit / RAVE

```bash
python compose.py manifest events.csv --clips clips/ -o manifest.jsonl \
    --prompt "bowed double bass, sustained, close-miked" --steps 28
```

One JSON line per opening:

```json
{"id": "open_00000", "source_wav": "clips/open_00000.wav", "morph": 0.5582,
 "flowedit": {"n_max": 16, "n_min": 0, "src_guidance": 1.5, "tar_guidance": 5.5},
 "rave": {"latent_bias": [0.1163, 0.7136, 0.8119]},
 "features": {"peak_open_dps": 103.5, "extent_deg": 80.5, "dwell_s": 9.06, "...": 0}}
```

- **FlowEdit — the single opening.** `n_max` is morph mapped onto your edit step
  count; set `--steps` to whatever your implementation uses. 7 days × ~50
  openings is ~350 renders, an overnight batch.
- **RAVE — the day and the week.** Too many renders for diffusion at that scale.
  Train on the fridge corpus and drive the latent from `latent_bias`: dim 0 is
  the morph axis, then extent, then dwell. Also gives you the real-time engine
  if you build the breath-driven transport.
- **LatentFT — the endpoint.** Replaces a generic text prompt with a destination
  you made yourself. Worth fine-tuning on *the fridge played with a bow* — rosin
  on the door edge, the wire shelves, the condenser coils — so both ends of the
  morph stay inside the apartment.

## Second IMU

`logger.py --prefix frontdoor` into the same directory, then segment the two
globs separately. Front door needs its own thresholds: the latch gives a harder
zero than the gasket, dwell is bimodal (seconds for a person, 30 s+ when
someone props it with a foot carrying bags), and two people through one opening
is a single event, so expect to undercount.

## Known limits

- No identity channel. Unlabelled openings can only be clustered, and the
  clusters may be times of day rather than people.
- `clipped=True` means the event hit `force_close_s` — the angle never came
  back to zero. Usually a dropout mid-opening. Treat as suspect, don't delete.
- The placeholder synth is not the piece. It is a listening aid for tuning.

---

# Three objects: faucet, kettle, door

```bash
python simulate_routine.py --days 7 --out routine/
python segment.py "routine/faucet_*.csv.gz" --profile faucet --out ev_faucet.csv
python segment.py "routine/kettle_*.csv.gz" --profile kettle --out ev_kettle.csv
python segment.py "routine/fridge_*.csv.gz" --profile door   --out ev_fridge.csv
python chain.py ev_faucet.csv ev_kettle.csv ev_fridge.csv \
    --names faucet kettle fridge --anchor faucet --collapse \
    --labels truth_routine.csv --label-time t_faucet
```

## Why these two are *easier* than the door

The door was the hard case. A vertical hinge is invisible to the accelerometer,
so angle had to be integrated from the gyro, bias-tracked, and reset at the
gasket. A faucet lever and a tipped kettle both rotate about **horizontal**
axes, so gravity swings with them and the accelerometer reads absolute angle
directly — no integration, no drift, no reset, no bias tracking. Pour angle is
the cleanest measurement in the project.

`--profile faucet|kettle` switches to tilt mode. Features become
`peak_tilt_deg`, `plateau_s` (run time / pour hold), `rise_s`,
`peak_rate_dps`, `yaw_deg`, `settle_impact`, `smoothness`.

## Faucet

A single-lever mixer encodes two things at once: **lift = flow**, **swivel =
temperature**. Lift is gravity-visible and clean. Swivel is rotation about the
vertical axis, so it is gravity-blind *and* has no return-to-zero — the lever
stays where you left it, so absolute temperature position drifts without bound
over a week.

Don't fight that. `yaw_deg` reports rotation about the gravity axis *per use*,
which answers the better question: **does this person ever renegotiate the
temperature, or did they set it once and never touch it again?** A median
`yaw_deg` of 0 is a finding, not a missing measurement. If you do want absolute
position, the lever's hard stops at full-hot and full-cold give you an
occasional reference — detect them as a sudden stop against a wall.

Waterproof it. Heat-shrink or silicone, mounted on the lever's upper face away
from the spout.

## Kettle

Not a fixed object — it gets lifted, carried, filled, set down, tipped. That
means several events per use, which is correct and which `--collapse` handles
in `chain.py`.

- **Pour angle** reads straight off gravity with no integration at all.
- **`settle_impact`** is how hard it gets put down, the kettle's equivalent of
  your door slam.
- **Boil time is proportional to water volume.** The gap between the kettle
  returning to its base and the pour tells you how much water was boiled.
  Overfilling a kettle is the canonical domestic energy waste — so this object
  measures surplus in *joules*, not just in gesture. That is the same argument
  as the door's surplus effort, in a unit nobody can wave away.
- **Mount on the handle, not the body.** The body cooks. Expect thermal drift
  regardless.
- A fill and a pour are both tilts; distinguishing them from the kettle alone
  is unreliable. Cross-reference the faucet's timing instead — a tilt during a
  faucet run is a fill. That is what the chain analysis is for.

## Why three objects and not a survey

I argued earlier that variety dilutes the comparison. These three don't,
because they aren't a set — they're a **sequence**. The kettle is filled at the
faucet, boiled, poured, and the fridge supplies the milk. That is the morning,
with causal ordering and fixed lags, and it's the "apartment coming to life"
you wanted without instrumenting every hinge in the place.

That gives you three nested timescales of habit:

| scale | measured by | question |
| --- | --- | --- |
| joint | `compose.py rote` | does the gesture repeat itself? |
| task | `chain.py` lag CV | is the next action triggered or decided? |
| life | lag CV across days | is the whole morning on rails? |

Nobody needs convincing that we're creatures of habit. The interesting claim is
that habit is **fractal** — as fixed in the elbow as in the schedule. Three
objects in sequence is the smallest apparatus that can test it.

They also orchestrate. The door is a transient, the faucet is a noise drone,
the kettle is the only one with intrinsic pitch — nucleate boiling rises, then
famously quiets just before full boil. Attack, noise, pitch: a three-minute arc
with a built-in climax, repeated every single day.

## Two IMUs, three objects

Instrument **faucet + kettle** together first. They are adjacent, used in the
same gesture, and the faucet disambiguates the kettle's fills from its pours.
Run the door separately in a second week and use time-of-day to stitch. If you
buy a third, put it on the front door, not a fourth kitchen object.
