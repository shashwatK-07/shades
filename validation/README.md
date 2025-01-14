# Validation against photographed shadows

Tier 1: validate the **occluder mesh** and the **sun vector** against shadow
lines on the playing surface. The field is planar and surveyed, so four
landmarks give a homography and every field pixel becomes metres. This
deliberately does not depend on the seat positions being right.

Tier 2 (shadow lines on the bowl itself, via full `solvePnP` camera pose) is
harder and partly circular — it uses the bowl model to validate the bowl
model. Do it second, once Tier 1 passes.

## Layout

```
validation/
  manifest.json          [{id, venue, url, timestamp_utc, source}]
  annotations/0001.json   clicked landmarks + traced shadow polyline
```

**Images are not stored in git.** The manifest holds URLs; the annotations
hold your own clicks. Keep local copies wherever you like.

## Getting data

You need (image, timestamp to the minute, venue). That last requirement kills
most sources.

- **Statcast / Gameday** publish per-pitch UTC timestamps. Match a broadcast
  still to a pitch and the time is exact and free.
- **Wire photos** (Getty, AP) carry caption timestamps, often to the minute.
- **Stadium webcams / time-lapses** where they exist — continuous coverage,
  timestamps built in.

Check the timezone on every single one. A DST error is a one-hour error, a
~15° azimuth error, and a catastrophically wrong shadow line — and it looks
like a modelling bug rather than a data bug. `manifest.json` asks for both
UTC and local time so the two can be checked against each other.

## Annotating

```
python scripts/annotate.py --image ~/photos/0001.jpg --id 0001
```

Click the landmarks it prompts for, press `n`, trace along the shadow edge,
press `w`. The landmark world coordinates must be in the **same frame as the
stadium YAML** (x=east, y=north, z=up, metres) — the built-in diamond puts
home plate at the origin with +y toward second base, which is not
automatically your stadium's frame.

Before trusting an annotation, check its homography residuals
(`photo_validation.homography_residuals`). A landmark reprojecting several metres
off is a misclick, and it poisons every shadow point in that photo.

## The metric

Per photo, three numbers — all three matter:

| number | what it tells you |
|---|---|
| median abs error | the headline |
| **signed mean** | is the model systematically short (+) or long (−)? |
| spread | right *shape*, or just right on average? |

The signed mean is the diagnostic. Random scatter about zero means you're at
the noise floor. A consistent −3 m means a parameter is wrong — an overhang
depth or deck height off by a fixed amount — and the sign says which way to
correct it.

## Noise floor — compute this before reporting anything

Two physical limits bound how good this can get. For a 20 m overhang:

| sun elevation | penumbra | ±1 min timestamp | combined |
|---|---|---|---|
| 60° | 0.25 m | 0.07 m | **0.26 m** |
| 30° | 0.74 m | 0.21 m | **0.77 m** |
| 15° | 2.78 m | 0.78 m | **2.89 m** |

- **Penumbra**: the sun is a disc, so a shadow edge is a gradient of width
  `h · 0.0093 / sin²α`. The traced "line" is a choice about where in a fuzzy
  band to click.
- **Timestamp**: shadow depth is `d = h/tan α`, so `∂d/∂α = h/sin²α` ≈ 1.4
  m/deg at 30°. Elevation moves ~0.15°/min, so one minute costs ~0.2 m.

A median error of **1–2 m is an excellent result**. If you report 0.2 m you
have fit noise or made an arithmetic error, and someone will notice.

Note the steep degradation at low sun: below ~20° elevation the floor is
metres, not centimetres. Low-sun photos stress the model in useful ways but
they cannot be held to the same tolerance.

## The negative control

Re-run with every timestamp deliberately shifted, ±60 min:

```python
offsets, errors = photo_validation.time_offset_sweep(observed_xy, stadium, ts, range(-60, 61, 10))
```

You want a clean V with its minimum at zero. If the error does *not* get
substantially worse at ±30 min, the metric isn't sensitive to what it claims
to measure and the headline number is meaningless. A minimum at +12 min is a
systematic timestamp bug — which is a finding, and a good one.

The sweep uses `symmetric_line_distance`, not the per-photo median. The
one-way distance is **not monotonic** in the offset: a real bowl casts
several disjoint shadow edges, and at large shifts a traced line can drift
near a *different* edge and score well for the wrong reason. Measuring both
directions penalises predicting an edge that isn't there. Trust the curve
inside roughly ±30 min; past ±40 the shadow structure changes qualitatively
(edges merge, the field saturates) and it flattens regardless.

## Where it will go wrong

- **Lens distortion.** Broadcast wide-angle has real barrel distortion; a
  homography assumes none. Avoid extreme wide shots.
- **Too few photos.** Under ~10 you are reporting noise. Aim for 20+ spanning
  at least three times of day and two seasons.
- **One camera angle.** Then you have validated one shadow line, repeatedly.
  Vary the vantage point.
- **Ambiguous edges.** Cloud shadows, stadium lights at dusk, shadows from
  light towers that aren't modelled. Discard rather than guess, and log what
  you discarded and why.

## What it unlocks

Once the harness exists, validation becomes calibration: wrap the error
metric in `scipy.optimize.minimize` and fit the handful of uncertain YAML
parameters (deck heights, overhang depths, bowl rotation) to the photographs.
That reframes the project from "I modelled a stadium and hope it's right" to
"I recovered stadium geometry from photographs, and here are the residuals."
Report fitted-vs-published values for any dimension you can source
independently — that's the out-of-sample check.
