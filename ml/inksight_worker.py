import json
import os
import re
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=custom-kernel-fusion-rewriter,custom_kernel-fusion-autotuner",
)

import numpy as np
import tensorflow as tf
import tensorflow_text
from huggingface_hub import snapshot_download
from PIL import Image

SIZE = 224
START = (SIZE + 1) * 2


def decode(text):
    tokens = [int(t) for t in re.findall(r"<ink_token_(\d+)>", text)]
    strokes, current, i = [], [], 0
    while i < len(tokens):
        if tokens[i] == START:
            if current:
                strokes.append(current)
            current = []
            i += 1
        elif i + 1 < len(tokens) and tokens[i + 1] != START:
            x, y = tokens[i], tokens[i + 1] - (SIZE + 1)
            if 0 <= x <= SIZE and 0 <= y <= SIZE:
                current.append([x, y])
            i += 2
        else:
            i += 1
    if current:
        strokes.append(current)
    return re.sub(r"<[^>]+>", "", text).strip(), strokes


def main(in_path, out_path):
    jobs = json.load(open(in_path, encoding="utf-8"))
    source = os.environ.get("HW_INKSIGHT_MODEL", "Derendering/InkSight-Small-p")
    model = tf.saved_model.load(source if os.path.isdir(source) else snapshot_download(source))
    fn = model.signatures["serving_default"]
    results = []
    for start in range(0, len(jobs), 16):
        batch = jobs[start:start + 16]
        images = [tf.io.encode_jpeg(np.array(Image.open(j["image"]).convert("RGB"))) for j in batch]
        encoded = tf.reshape(tf.stack(images), (len(batch), 1))
        out = fn(**{"input_text": tf.constant(["Recognize and derender."] * len(batch)), "image/encoded": encoded})
        for k, job in enumerate(batch):
            text, strokes = decode(out["output_0"].numpy()[k][0].decode())
            if not strokes:
                retry = fn(**{"input_text": tf.constant(["Derender the ink."]), "image/encoded": encoded[k:k + 1]})
                _, strokes = decode(retry["output_0"].numpy()[0][0].decode())
            results.append({"id": job["id"], "text": text, "strokes": strokes})
    json.dump(results, open(out_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
