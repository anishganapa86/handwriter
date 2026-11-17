import re

import cv2
import numpy as np
from PIL import Image

from . import worksheet
from .sheet import trace

PAGE_W_MM = worksheet.PAGE_W_MM
_reader = None
_trocr = None
PUNCT = re.compile(r"\s+([.,;:!?)])")


def load_image(data, name, page=0):
    if name.lower().endswith(".pdf") or data[:5] == b"%PDF-":
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        page = max(0, min(page, len(doc) - 1))
        pix = doc[page].get_pixmap(dpi=170, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), len(doc), True
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image, 1, False


def flatten(image, is_pdf):
    if is_pdf:
        scale = worksheet.OUT_W / image.shape[1]
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), "pdf page"
    return worksheet.deskew(image)


def ink_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    block = min(255, max(31, (min(gray.shape) // 12) | 1))
    diff = cv2.medianBlur(gray, block).astype(np.int32) - gray.astype(np.int32)
    level = max(30.0, 0.5 * float(np.percentile(diff, 99.5)))
    mask = (diff > level).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def reader():
    global _reader
    if _reader is None:
        import easyocr
        import torch
        _reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    return _reader


def line_reader():
    global _trocr
    if _trocr is None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        gpu = torch.cuda.is_available()
        name = "microsoft/trocr-large-handwritten" if gpu else "microsoft/trocr-base-handwritten"
        processor = TrOCRProcessor.from_pretrained(name)
        model = VisionEncoderDecoderModel.from_pretrained(name).to("cuda" if gpu else "cpu").eval()

        def recognise(images):
            with torch.no_grad():
                pixels = processor(images=images, return_tensors="pt").pixel_values.to(model.device)
                ids = model.generate(pixels, max_new_tokens=64)
            return processor.batch_decode(ids, skip_special_tokens=True)

        _trocr = recognise
    return _trocr


def read_text(image):
    found = reader().readtext(image, paragraph=False, width_ths=0.6, height_ths=0.6, slope_ths=0.3)
    h, w = image.shape[:2]
    boxes = []
    for quad, text, confidence in found:
        quad = np.array(quad)
        quad = quad.astype(np.float64)
        tall = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
        middle = quad[:, 1].mean()
        x0, x1 = int(np.floor(quad[:, 0].min())), int(np.ceil(quad[:, 0].max()))
        y0, y1 = int(np.floor(middle - tall / 2)), int(np.ceil(middle + tall / 2))
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        boxes.append({"px": [int(x0), int(y0), int(x1), int(y1)], "text": text.strip(), "confidence": float(confidence)})
    wide = [b for b in boxes if (b["px"][2] - b["px"][0]) > 1.6 * (b["px"][3] - b["px"][1])]
    if wide:
        recognise = line_reader()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crops = [Image.fromarray(rgb[max(0, b["px"][1] - 4):b["px"][3] + 4, max(0, b["px"][0] - 4):b["px"][2] + 4]) for b in wide]
        for start in range(0, len(crops), 16):
            for box, text in zip(wide[start:start + 16], recognise(crops[start:start + 16])):
                text = PUNCT.sub(lambda m: m.group(1), " ".join(text.split()))
                if text.endswith(" .") or (text.endswith(".") and not box["text"].rstrip().endswith(".")):
                    text = text.rstrip(". ")
                if text:
                    box["alt"], box["text"] = box["text"], text
    boxes.sort(key=lambda b: (round(b["px"][1] / 18), b["px"][0]))
    return boxes


def to_mm(boxes, width):
    mm = PAGE_W_MM / width
    out = []
    for b in boxes:
        x0, y0, x1, y1 = b["px"]
        height = (y1 - y0) * mm
        out.append({"x": round(x0 * mm, 2), "y": round(y0 * mm, 2), "w": round((x1 - x0) * mm, 2), "h": round(height, 2),
                    "kind": "line", "text": b["text"], "size": round(float(np.clip(height / 3.0, 1.2, 3.2)), 2)})
    return out


def diagram(mask, boxes, width_mm=PAGE_W_MM):
    mask = mask.copy()
    px = mask.shape[1] / width_mm
    for b in boxes:
        x0, y0 = int(b["x"] * px) - 2, int(b["y"] * px) - 2
        x1, y1 = int((b["x"] + b["w"]) * px) + 2, int((b["y"] + b["h"]) * px) + 2
        mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    small = [i for i in range(1, count) if stats[i][4] < 14]
    mask[np.isin(labels, small)] = 0
    strokes = trace(mask, 5, 7)
    return [(s / px).round(2).tolist() for s in strokes if len(s) > 1]


def analyse(data, name, page=0):
    image, pages, is_pdf = load_image(data, name, page)
    if image is None:
        return None
    flat, method = flatten(image, is_pdf)
    mask = ink_mask(flat)
    try:
        boxes, can_read = to_mm(read_text(flat), flat.shape[1]), True
    except ImportError:          # EasyOCR/TrOCR are installed on the PC only; elsewhere the page is traced and text typed in
        boxes, can_read = [], False
    size = {"w": PAGE_W_MM, "h": round(flat.shape[0] * PAGE_W_MM / flat.shape[1], 2)}
    return {"image": flat, "mask": mask, "boxes": boxes, "page": size, "pages": pages, "method": method, "can_read": can_read}
