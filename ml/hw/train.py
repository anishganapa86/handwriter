import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import PHOTO_WEIGHT, augment, collate, encode, load_samples, new_symbols, split, writer_profile
from .evaluate import better, evaluate, write_report
from .generate import BASE_DIR, Writer, base_stats
from .model import Charset, extend_alphabet, load_checkpoint, pick_device, save_checkpoint


def registry(data_dir):
    root = Path(data_dir) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "champion.json"
    champion = json.loads(path.read_text()) if path.exists() else None
    versions = sorted(p.name for p in root.glob("v[0-9]*") if (p / "meta.json").exists())
    return root, champion, versions


def champion_path(data_dir):
    root, champion, _ = registry(data_dir)
    return root / champion["version"] if champion else BASE_DIR


def count_lines(data_dir):
    return len(load_samples(data_dir))


def fit(model, charset, train_items, val_items, mu, std, device, epochs, lr, batch_size, log):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(0)
    best, best_state, patience, history = float("inf"), None, 0, []

    def batches(items, shuffle):
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        groups = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        if shuffle:
            rng.shuffle(groups)
        return [[items[i] for i in g] for g in groups]

    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for group in batches(train_items, True):
            group = [(augment(o, mu, std, rng), t, w) for o, t, w in group]
            loss = model.loss(*collate(group, charset, device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total += loss.item() * len(group)
            seen += len(group)
        model.eval()
        with torch.no_grad():
            val = float(np.mean([model.loss(*collate(g, charset, device)).item() for g in batches(val_items, False)]))
        history.append({"epoch": epoch, "train": total / seen, "val": val})
        log(f"epoch {epoch:3d}  train {total / seen:.4f}  val {val:.4f}")
        if val < best - 1e-3:
            best, patience = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 6:
                break
    model.load_state_dict(best_state)
    return history


def pick_prime(train, items):
    timed = [(len(s["text"]), i) for i, s in enumerate(train) if s["timed"] and 14 <= len(s["text"]) <= 40]
    if not timed:
        timed = [(len(s["text"]), i) for i, s in enumerate(train) if 12 <= len(s["text"]) <= 34 and " " in s["text"]]
    if not timed:
        return None
    _, i = sorted(timed)[len(timed) // 2]
    return {"text": train[i]["text"], "offsets": items[i][0].tolist()}


def run(data_dir, epochs=40, lr=1e-4, batch_size=16, device=None, use_recognizer=True, log=print, style_only=None):
    data_dir = Path(data_dir)
    root, champion, versions = registry(data_dir)
    samples = load_samples(data_dir)
    train, holdout = split(samples)
    if len(train) < 10 or not holdout:
        raise SystemExit(f"need at least 10 training lines and 1 held-out line, have {len(train)} and {len(holdout)}")
    stats = base_stats()
    device = pick_device(device)
    model, charset, base_meta = load_checkpoint(BASE_DIR, device)
    added = new_symbols(train, charset)
    if added:
        model, chars = extend_alphabet(model, charset.chars, added)
        charset = Charset(chars)
        model.to(device)
        log(f"extended character set with: {added}")
    mu, std = base_meta["mu"], base_meta["std"]
    profile = writer_profile(samples, stats)
    items = [(encode(s, profile, stats, mu, std), s["text"], 1.0 if s["timed"] else PHOTO_WEIGHT) for s in train]
    val_idx = set(range(3, len(items), 8))
    train_items = [it for i, it in enumerate(items) if i not in val_idx]
    val_items = [it for i, it in enumerate(items) if i in val_idx]
    if style_only is None:
        style_only = not any(s["timed"] for s in train)
    if style_only and not added:
        for part in (model.lstm1, model.window):
            for param in part.parameters():
                param.requires_grad_(False)
    log(f"{len(train_items)} train, {len(val_items)} validation, {len(holdout)} held-out lines on {device}"
        f"{', text alignment layers frozen' if style_only and not added else ''}")
    started = time.time()
    history = fit(model, charset, train_items, val_items, mu, std, device, epochs, lr, batch_size, log)
    version = f"v{(int(versions[-1][1:]) + 1 if versions else 1):04d}"
    meta = {"mu": mu, "std": std, "charset": charset.chars, "core_units": stats["core"], "version": version,
            "writer": profile, "prime": pick_prime(train, items), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "lines": {"train": len(train_items), "val": len(val_items), "holdout": len(holdout), "total": len(samples)},
            "added_symbols": added, "style_only": bool(style_only and not added), "history": history, "seconds": round(time.time() - started, 1)}
    save_checkpoint(root / version, model, meta)
    candidate = Writer(root / version)
    metrics, rows = evaluate(candidate, holdout, profile, stats, use_recognizer=use_recognizer)
    previous, previous_rows = None, None
    reference = Writer(root / champion["version"]) if champion else Writer(BASE_DIR)
    ref_metrics, previous_rows = evaluate(reference, holdout, profile, stats, use_recognizer=use_recognizer)
    previous = {"version": champion["version"] if champion else "base", "metrics": ref_metrics}
    promoted, reason = better(metrics, ref_metrics)
    (root / version / "metrics.json").write_text(json.dumps({"metrics": metrics, "against": previous, "promoted": promoted,
                                                              "reason": reason}, indent=1))
    reports = data_dir / "reports"
    reports.mkdir(exist_ok=True)
    write_report(reports / f"{version}.html", version, metrics, rows, previous, previous_rows, promoted, reason, meta["lines"])
    if promoted:
        (root / "champion.json").write_text(json.dumps({"version": version, "metrics": metrics}, indent=1))
    (root / "state.json").write_text(json.dumps({"lines_at_last_train": len(samples), "last_version": version}))
    log(f"{version}: dtw {metrics['dtw_dir']:.4f} vs {ref_metrics['dtw_dir']:.4f}, cer {metrics['cer']} vs {ref_metrics['cer']} -> "
        f"{'promoted' if promoted else 'kept previous'} ({reason})")
    return version, promoted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "data"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--no-recognizer", action="store_true")
    args = parser.parse_args()
    run(args.data, args.epochs, args.lr, args.batch_size, args.device, not args.no_recognizer)
