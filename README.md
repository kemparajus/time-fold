# Time-Fold: verified change detection from satellite history

Time-Fold compares two satellite views of the same place and turns them into
evidence about what changed. SAM 3.1 finds and counts objects, then Muse Spark
checks those detections against the images and writes a short, honest report.

Built for the Meta Global AI Developer Hackathon (Indie track, Push the Frontier theme).

## Why
Satellite history exists for almost everywhere, but reading it takes expertise.
Time-Fold gives a plain-language summary of what changed, and also says how far
the automatic measurements can be trusted.

## How it works
1. Two screenshots of the same place from different years go in.
2. Each image is split into tiles and enlarged, then **SAM 3.1** (Meta Model API)
   detects objects (building, airplane, car, tree) and returns boxes.
3. Detection boxes are drawn on both images and counted per year.
4. **Muse Spark** receives the original and annotated images plus the counts. It
   verifies which detections look right or wrong, describes what visibly changed,
   labels causes as hypotheses, gives a qualitative 2050 scenario (not a
   prediction), and states confidence and limits.

The key idea: **one AI measures, another AI audits the measurement.**

## Example: Kempegowda International Airport, Bengaluru (2015 vs 2026)
- Terminal visibly expanded westward, and previously bare ground was built up.
- Spark flagged false positives (jet bridges counted as airplanes), the car count
  of 0 as clearly wrong, and a terminal split into several building boxes.
- Outputs: `before_annotated.png`, `after_annotated.png`, `report.txt`, `stats.json`.

## Limitations (stated honestly)
- Counts and area shares are rough (bounding boxes, not masks) and not ground-truthed.
- Small objects such as cars and individual trees are often missed.
- Two images cannot prove causes or trends. Causes are hypotheses only.
- 2050 text is a scenario with stated assumptions, not a forecast.
- Imagery source and capture dates must be checked by the user: [add your imagery source here].

## Run it
```
pip install requests pillow
```
1. Put `before.png` and `after.png` next to the script.
2. Open `timefold.py`, set `API_KEY`, the two years, and the place.
3. Run it. Results are cached in `./cache` so repeat runs cost nothing.

Never commit your real API key.

## Models used
- SAM 3.1 via Meta Model API (object detection and segmentation)
- Muse Spark 1.2 via Meta Model API (verification and reporting)

## Roadmap
- Real mask-based area instead of boxes
- More places (solar parks, new housing layouts)
- Simple web interface

## Author
Kemparaju S, Chamarajanagar, Karnataka, India
