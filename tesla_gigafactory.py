"""
Time-Fold (full scene version)
SAM 3.1 finds the things in the scene, Muse Spark audits the detections and
writes the report.

Scene classes (from your image description)
  factory building  the huge elongated factory                       (counted)
  solar panel       the rooftop panels that spell TESLA              (area)
  parking lot       sprawling parking and paved staging lots         (area)
  road              roads and industrial access routes               (area)
  river             the Colorado River                               (area)
  forest            green vegetation and forested areas              (area)
  cleared land      clearing zones, bare soil, construction sites    (area)
  pond              ponds and retention water                        (counted)
  vehicle           cars and trucks                                  (counted)

What is new
  - REAL AREA: SAM returns a mask for every object. This script decodes the
    masks (Meta's meta-sam-parser package, installed automatically) and
    measures the true share of the image each class covers. Bounding boxes
    were far too rough for rivers, forests and parking lots.
  - Per-class scales: big objects use the whole image, small objects use tiles.
  - Colored mask overlays are drawn on both images, plus boxes for counted objects.

Run in Google Colab: upload the two images, paste this whole file into ONE
code cell, edit ONLY the settings block, press play.
Cost: roughly 100 small SAM calls plus 1 Spark call (trim CLASSES to save).
Results are cached in ./cache, so re-running the same images costs nothing.
NOT yet tested against the live API: if something fails, send me the error.
"""
import base64, hashlib, io, json, os, re, subprocess, sys
import requests
from PIL import Image, ImageDraw

try:
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "meta-sam-parser"],
                   check=False)
    try:
        from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    except ImportError:
        SegmentationMask = decode_mask_to_raster = None
        print("WARNING: mask decoder not available, falling back to rough box areas.")

# ================= SETTINGS: edit only this block =================
API_KEY = "PASTE_YOUR_KEY_HERE"   # Colab only. NEVER put your real key in GitHub.

BEFORE_IMAGE = "before.png"       # run  !ls  in Colab and match the real file names
BEFORE_YEAR = 2015                # REAL capture year from the map label
AFTER_IMAGE = "after.png"
AFTER_YEAR = 2025                 # REAL capture year from the map label

PLACE = "Tesla Gigafactory Texas, Austin, USA (Colorado River to the right of the factory)"
OUT_PREFIX = "gigafactory_texas3"

WHOLE = (1, 1)                    # one pass over the whole image
TALL = (1, 4)                     # four stacked tiles (the images are tall and narrow)
HALF = (1, 2)
CLASSES = {
    "factory building": {"kind": "object", "passes": [WHOLE, TALL],
                         "prompts": ["building", "warehouse", "factory roof"]},
    "solar panel":      {"kind": "area", "passes": [WHOLE, HALF],
                         "prompts": ["solar panel"]},
    "parking lot":      {"kind": "area", "passes": [WHOLE, HALF],
                         "prompts": ["parking lot"]},
    "road":             {"kind": "area", "passes": [WHOLE, HALF],
                         "prompts": ["road"]},
    "river":            {"kind": "area", "passes": [WHOLE, HALF],
                         "prompts": ["river"]},
    "forest":           {"kind": "area", "passes": [WHOLE, HALF],
                         "prompts": ["forest", "tree"]},
    "cleared land":     {"kind": "area", "passes": [WHOLE, HALF],
                         "prompts": ["bare soil", "construction site"]},
    "pond":             {"kind": "object", "passes": [WHOLE, TALL],
                         "prompts": ["pond"]},
    "vehicle":          {"kind": "object", "passes": [TALL],
                         "prompts": ["car", "truck"]},
}
DEDUPE_IOU = 0.5          # counted objects: merge boxes overlapping more than this
CONTAIN = 0.8             # counted objects: drop a box mostly inside a bigger one
OVERLAY_ALPHA = 90        # 0-255, strength of the colored mask overlay
WHOLE_LONG_SIDE = 1024    # size SAM sees for whole-image passes
DOWNLOAD_AT_END = True
# ==================================================================

COLORS = {"factory building": "red", "solar panel": "yellow", "parking lot": "deepskyblue",
          "road": "silver", "river": "blue", "forest": "lime", "cleared land": "orange",
          "pond": "magenta", "vehicle": "cyan"}
BASE = "https://api.meta.ai/v1"
HEAD = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
SPARK = "muse-spark-1.2"
SAM = "sam-3.1"
TO255 = bytes([0, 255] + [0] * 254)     # maps raster values 0/1 to 0/255
os.makedirs("cache", exist_ok=True)


# ---------- image helpers ----------
def load(path, max_side=2048):
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}\nFolder: {os.getcwd()}\n"
                 "Run  !ls  in a Colab cell and copy the exact file name.")
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def to_url(img, long_side=1024, upscale=True):
    im = img.copy()
    scale = long_side / max(im.size)
    if scale > 1 and not upscale:
        scale = 1
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                   Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------- cache + API ----------
def cached(key, fn):
    f = "cache/" + hashlib.sha256(key.encode()).hexdigest()[:16] + ".json"
    if os.path.exists(f):
        return json.load(open(f))
    out = fn()
    json.dump(out, open(f, "w"))
    return out


def post(path, body):
    r = requests.post(f"{BASE}{path}", headers=HEAD, json=body, timeout=180)
    if not r.ok:
        sys.exit(f"API error {r.status_code}: {r.text[:500]}")
    return r


def sam_call(data_url, label):
    def call():
        body = {"model": SAM, "input": [{"role": "user", "content": [
            {"type": "input_text", "text": label},
            {"type": "input_image", "image_url": data_url}]}]}
        return {"raw": post("/responses", body).text}
    return cached(f"sam|{label}|{data_url}", call)["raw"]


# ---------- parsing SAM output ----------
REC = re.compile(r"<\|box;([^|]*)\|>\s*<\|mask;([^|]*)\|>")


def sam_text(raw):
    """Pull the output_text out of the Responses JSON (fall back to raw text)."""
    try:
        j = json.loads(raw)
        parts = []
        for item in j.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
        return "\n".join(parts)
    except Exception:
        return raw


def fields(s):
    return dict(kv.split("=", 1) for kv in s.split(";") if "=" in kv)


def parse_records(text):
    """Return [((x1, y1, x2, y2, w, h), mask_data_string), ...]; x2/y2 become half-open."""
    out = []
    for bs, ms in REC.findall(text):
        b, m = fields(bs), fields(ms)
        try:
            box = (float(b["x1"]), float(b["y1"]), float(b["x2"]) + 1, float(b["y2"]) + 1,
                   int(b["w"]), int(b["h"]))
        except (KeyError, ValueError):
            continue
        out.append((box, m.get("data", "")))
    return out


def mask_image(data):
    """Decode SAM mask data 'H,W,payload' into a PIL 'L' image (0/255), or None."""
    if decode_mask_to_raster is None:
        return None
    try:
        h, w, payload = data.split(",", 2)
        h, w = int(h), int(w)
        enc = "lossless" if payload[:1] == "~" else "one_bit"
        raster = decode_mask_to_raster(
            SegmentationMask(encoding=enc, payload=payload, width=w, height=h))
        return Image.frombytes("L", (w, h), raster.translate(TO255))
    except Exception:
        return None


# ---------- detection ----------
def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def dedupe(dets):
    """Counted objects only: biggest box first, drop heavy overlaps and boxes inside others."""
    keep = []
    for b in sorted(dets, key=lambda d: -area(d)):
        ok = True
        for k in keep:
            i = inter(b, k)
            union = area(b) + area(k) - i
            if (union > 0 and i / union >= DEDUPE_IOU) or (area(b) > 0 and i / area(b) >= CONTAIN):
                ok = False
                break
        if ok:
            keep.append(b)
    return keep


def detect(img, cfg):
    """Detections as (x1, y1, x2, y2, mask_image_or_None) in full-image pixels."""
    W, H = img.size
    found = []
    for label in cfg["prompts"]:
        for cols, rows in cfg["passes"]:
            ls = WHOLE_LONG_SIDE if (cols, rows) == (1, 1) else 1024
            for i in range(rows):
                for j in range(cols):
                    x0, y0 = j * W // cols, i * H // rows
                    x1c, y1c = (j + 1) * W // cols, (i + 1) * H // rows
                    tile = img.crop((x0, y0, x1c, y1c))
                    text = sam_text(sam_call(to_url(tile, ls), label))
                    for (bx1, by1, bx2, by2, w, h), data in parse_records(text):
                        sx, sy = tile.width / float(w), tile.height / float(h)
                        found.append((x0 + bx1 * sx, y0 + by1 * sy,
                                      x0 + bx2 * sx, y0 + by2 * sy, mask_image(data)))
    return dedupe(found) if cfg["kind"] == "object" else found


def coverage(dets, size):
    """Union of all masks (or boxes if a mask is missing). Returns (share, 'L' canvas)."""
    W, H = size
    canvas = Image.new("L", (W + 8, H + 8), 0)
    for x1, y1, x2, y2, m in dets:
        X1, Y1 = int(round(x1)), int(round(y1))
        tw, th = max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1)))
        layer = m.resize((tw, th), Image.NEAREST) if m is not None else Image.new("L", (tw, th), 255)
        canvas.paste(255, (X1, Y1, X1 + tw, Y1 + th), layer)
    canvas = canvas.crop((0, 0, W, H))
    covered = sum(canvas.histogram()[1:])
    return covered / float(W * H), canvas


# ---------- drawing ----------
def annotate(img, results, out_path):
    im = img.copy()
    for cls, (dets, canvas) in results.items():
        layer = Image.new("RGB", im.size, COLORS.get(cls, "white"))
        im.paste(layer, (0, 0), canvas.point(lambda p: OVERLAY_ALPHA if p else 0))
    d = ImageDraw.Draw(im)
    for cls, (dets, canvas) in results.items():
        if CLASSES[cls]["kind"] == "object":
            for x1, y1, x2, y2, _ in dets:
                d.rectangle([x1, y1, x2, y2], outline=COLORS.get(cls, "white"), width=3)
    im.save(out_path)
    return im


def side_by_side(a_ann, b_ann, ya, yb, out_path):
    h = max(a_ann.height, b_ann.height)
    gap, bar = 20, 40
    canvas = Image.new("RGB", (a_ann.width + b_ann.width + gap, h + bar), "white")
    canvas.paste(a_ann, (0, bar))
    canvas.paste(b_ann, (a_ann.width + gap, bar))
    d = ImageDraw.Draw(canvas)
    d.text((10, 10), f"BEFORE ({ya})", fill="black")
    d.text((a_ann.width + gap + 10, 10), f"AFTER ({yb})", fill="black")
    canvas.save(out_path)


# ---------- Spark ----------
def explain(urls, ya, yb, place, stats):
    legend = ", ".join(f"{COLORS.get(c, 'white')}={c}" for c in CLASSES)
    method = ("mask-based (true share of the image covered)" if decode_mask_to_raster
              else "bounding-box based (rough)")
    prompt = (
        f"You are a careful remote-sensing analyst. Place: {place}. "
        f"Image 1 = {ya} original, image 2 = {yb} original, "
        f"image 3 = {ya} with detector overlay, image 4 = {yb} with detector overlay. "
        f"Overlay colors: {legend}. Colored translucent areas are SAM 3.1 masks; "
        "thin outlines mark individually counted objects. "
        f"Detector results: {json.dumps(stats)}. 'area_share' is {method}; "
        "'count' is given only for counted objects. The detector can produce false "
        "positives and misses. Check especially whether any LARGE structure was missed "
        "or only partly covered, and whether area classes bleed onto the wrong terrain. "
        "Parts of the scene may be unchanged: use them as a sanity check. "
        "Images may be stitched from imagery of different dates: mention any visible seams. "
        "Write a SHORT report (about 250 words, plain text) with exactly these headings: "
        "VERIFICATION (3 bullets: which results look right or wrong, judged from the "
        "overlays), WHAT CHANGED (4 bullets, trustworthy observations only), "
        "2050 SCENARIO (2 sentences, qualitative, state assumptions, not a prediction), "
        "CONFIDENCE AND LIMITS (2 bullets). Hypotheses must be labelled as hypotheses. "
        "Do not accuse anyone."
    )

    def call():
        content = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": u}} for u in urls]
        body = {"model": SPARK, "messages": [{"role": "user", "content": content}]}
        j = post("/chat/completions", body).json()
        return {"text": j["choices"][0]["message"]["content"]}
    return cached(f"spark|{prompt}|{''.join(u[-200:] for u in urls)}", call)["text"]


# ---------- main ----------
def main():
    if API_KEY.startswith("PASTE"):
        sys.exit("Put your API key in the settings block at the top.")
    a_img, b_img = load(BEFORE_IMAGE), load(AFTER_IMAGE)
    calls = sum(len(c["prompts"]) * sum(x * y for x, y in c["passes"]) for c in CLASSES.values()) * 2
    print(f"before {a_img.size}, after {b_img.size}; about {calls} SAM calls")

    res_a, res_b, stats = {}, {}, {}
    for cls, cfg in CLASSES.items():
        print("detecting:", cls, cfg["prompts"])
        dets_a, dets_b = detect(a_img, cfg), detect(b_img, cfg)
        sa, ca = coverage(dets_a, a_img.size)
        sb, cb = coverage(dets_b, b_img.size)
        res_a[cls], res_b[cls] = (dets_a, ca), (dets_b, cb)
        s = {"kind": cfg["kind"],
             f"area_share_{BEFORE_YEAR}": round(sa, 3),
             f"area_share_{AFTER_YEAR}": round(sb, 3),
             "area_change": round(sb - sa, 3)}
        if cfg["kind"] == "object":
            s[f"count_{BEFORE_YEAR}"] = len(dets_a)
            s[f"count_{AFTER_YEAR}"] = len(dets_b)
            s["count_change"] = len(dets_b) - len(dets_a)
        stats[cls] = s
    print(json.dumps(stats, indent=2))

    a_path = f"{OUT_PREFIX}_before_annotated.png"
    b_path = f"{OUT_PREFIX}_after_annotated.png"
    s_path = f"{OUT_PREFIX}_side_by_side.png"
    a_ann = annotate(a_img, res_a, a_path)
    b_ann = annotate(b_img, res_b, b_path)
    side_by_side(a_ann, b_ann, BEFORE_YEAR, AFTER_YEAR, s_path)

    urls = [to_url(x, 1536, upscale=False) for x in (a_img, b_img, a_ann, b_ann)]
    report = explain(urls, BEFORE_YEAR, AFTER_YEAR, PLACE, stats)
    print("\n" + report)

    j_path, r_path = f"{OUT_PREFIX}_stats.json", f"{OUT_PREFIX}_report.txt"
    json.dump(stats, open(j_path, "w"), indent=2)
    open(r_path, "w").write(report)
    outputs = [r_path, j_path, a_path, b_path, s_path]
    print("\nSaved:", ", ".join(outputs))

    try:
        from IPython.display import display
        display(Image.open(s_path))
    except Exception:
        pass
    if DOWNLOAD_AT_END:
        try:
            from google.colab import files
            for f in outputs:
                files.download(f)
        except Exception:
            print("Auto-download skipped. Use the folder panel to download the files.")


main()
