"""
Time-Fold v5: SAM 3.1 finds objects tile by tile, Muse Spark verifies and explains.
Edit ONLY the settings block, then run the cell.

New in v5:
  - Each image is cut into TILES x TILES pieces, and each piece is enlarged
    before SAM sees it (helps with small objects like cars and trees).
  - Spark also sees the annotated images, so it can check SAM's boxes directly.
  - Short report (about one screen).
  - Annotated images are displayed at the end (works in Colab).

Cost note: calls = classes x 2 images x tiles^2 (default 4 x 2 x 4 = 32 small SAM calls).
Results are cached in ./cache, so re-runs cost nothing.
"""
import base64, hashlib, io, json, os, re, sys
import requests
from PIL import Image, ImageDraw

# ============ SETTINGS: edit only this block ============
API_KEY = "YOUR_API_KEY"      # keep private, never share this
BEFORE_IMAGE = "before.png"
BEFORE_YEAR = 2015                   # real capture year shown on the map
AFTER_IMAGE = "after.png"
AFTER_YEAR = 2026                    # real capture year shown on the map
PLACE = "Kempegowda International Airport, Bengaluru, India"
CLASSES = ["building", "airplane", "car", "tree"]
TILES = 2                            # 2 = 2x2 pieces; 1 = old behaviour
# ========================================================

COLORS = {"building": "red", "airplane": "yellow", "car": "cyan", "tree": "lime"}
BASE = "https://api.meta.ai/v1"
HEAD = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
SPARK = "muse-spark-1.2"
SAM = "sam-3.1"
os.makedirs("cache", exist_ok=True)


def load(path, max_side=2048):
    if not os.path.exists(path):
        sys.exit(f"File not found: {path} (folder: {os.getcwd()})")
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def to_url(img, long_side=1024):
    """Resize so the long side is long_side (enlarges small tiles), return data URL."""
    im = img.copy()
    scale = long_side / max(im.size)
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                   Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def cached(key, fn):
    f = "cache/" + hashlib.sha256(key.encode()).hexdigest()[:16] + ".json"
    if os.path.exists(f):
        return json.load(open(f))
    out = fn()
    json.dump(out, open(f, "w"))
    return out


def post(path, body):
    r = requests.post(f"{BASE}{path}", headers=HEAD, json=body, timeout=120)
    if not r.ok:
        sys.exit(f"API error {r.status_code}: {r.text[:500]}")
    return r


BOX = re.compile(r"x1=([\d.]+);y1=([\d.]+);x2=([\d.]+);y2=([\d.]+);w=(\d+);h=(\d+)")


def sam_call(data_url, label):
    def call():
        body = {"model": SAM, "input": [{"role": "user", "content": [
            {"type": "input_text", "text": label},
            {"type": "input_image", "image_url": data_url}]}]}
        return {"raw": post("/responses", body).text}
    return cached(f"sam|{label}|{data_url}", call)["raw"]


def segment_tiled(img, label):
    """Run SAM on each tile and map the boxes back to full-image coordinates."""
    W, H = img.size
    found = []
    for i in range(TILES):
        for j in range(TILES):
            x0, y0 = j * W // TILES, i * H // TILES
            x1c, y1c = (j + 1) * W // TILES, (i + 1) * H // TILES
            tile = img.crop((x0, y0, x1c, y1c))
            raw = sam_call(to_url(tile), label)
            for a, b, c, d, w, h in BOX.findall(raw):
                sx, sy = tile.width / float(w), tile.height / float(h)
                found.append((x0 + float(a) * sx, y0 + float(b) * sy,
                              x0 + float(c) * sx, y0 + float(d) * sy, W, H))
    return found


def box_area_share(boxes):
    """ROUGH: sum of box areas / image area. Large boxes overcount."""
    if not boxes:
        return 0.0
    W, H = boxes[0][4], boxes[0][5]
    total = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2, _, _ in boxes)
    return min(1.0, total / (W * H))


def annotate(img, boxes_by_class, out_path):
    im = img.copy()
    d = ImageDraw.Draw(im)
    for c, boxes in boxes_by_class.items():
        for x1, y1, x2, y2, _, _ in boxes:
            d.rectangle([x1, y1, x2, y2], outline=COLORS.get(c, "white"), width=2)
    im.save(out_path)
    return im


def explain(urls, ya, yb, place, stats):
    prompt = (
        f"You are a careful remote-sensing analyst. Place: {place}. "
        f"Image 1 = {ya} original, image 2 = {yb} original, "
        f"image 3 = {ya} with detector boxes drawn, image 4 = {yb} with detector boxes drawn. "
        "Colors: red=building, yellow=airplane, cyan=car, lime=tree. "
        f"Detector (SAM 3.1) counts: {json.dumps(stats)}. 'area_share' is a rough "
        "bounding-box estimate. The detector can produce false positives and misses. "
        "Write a SHORT report (about 200 words, plain text) with exactly these headings: "
        "VERIFICATION (2-3 bullets: which counts look right or wrong, judged from the "
        "annotated images), WHAT CHANGED (3 bullets, trustworthy observations only), "
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


def main():
    if API_KEY.startswith("PASTE"):
        sys.exit("Put your API key in the settings block at the top.")
    a_img, b_img = load(BEFORE_IMAGE), load(AFTER_IMAGE)
    boxes_a, boxes_b, stats = {}, {}, {}
    for c in CLASSES:
        print("segmenting:", c)
        boxes_a[c] = segment_tiled(a_img, c)
        boxes_b[c] = segment_tiled(b_img, c)
        stats[c] = {
            f"count_{BEFORE_YEAR}": len(boxes_a[c]),
            f"count_{AFTER_YEAR}": len(boxes_b[c]),
            "count_change": len(boxes_b[c]) - len(boxes_a[c]),
            f"area_share_{BEFORE_YEAR}": round(box_area_share(boxes_a[c]), 3),
            f"area_share_{AFTER_YEAR}": round(box_area_share(boxes_b[c]), 3),
        }
    print(json.dumps(stats, indent=2))
    a_ann = annotate(a_img, boxes_a, "before_annotated.png")
    b_ann = annotate(b_img, boxes_b, "after_annotated.png")
    urls = [to_url(a_img), to_url(b_img), to_url(a_ann), to_url(b_ann)]
    report = explain(urls, BEFORE_YEAR, AFTER_YEAR, PLACE, stats)
    print("\n" + report)
    json.dump(stats, open("stats.json", "w"), indent=2)
    open("report.txt", "w").write(report)
    print("\nSaved stats.json, report.txt, before_annotated.png, after_annotated.png")
    try:
        from IPython.display import display
        display(a_ann)
        display(b_ann)
    except Exception:
        pass


main()
