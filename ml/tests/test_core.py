import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hw import layout
from hw.dataset import norm_text, split
from hw.evaluate import better, cer, dtw
from hw.generate import chunk_words, split_words
from hw.model import Charset, SynthesisNetwork, extend_alphabet
from hw.strokes import core_metrics, from_offsets, read_iam_xml, to_offsets, write_iam_xml


def strokes():
    t = np.linspace(0, 1, 30)
    return [np.stack([t * 40, 5 * np.sin(t * 9), t], axis=1), np.array([[45.0, 1.0, 1.2]])]


def test_offsets_round_trip():
    src = [s[:, :2] for s in strokes()]
    back = from_offsets(to_offsets(src))
    assert len(back) == 2 and len(back[1]) == 1
    assert np.allclose(back[0] + src[0][0], src[0], atol=1e-4)


def test_iam_xml_round_trip(tmp_path):
    path = tmp_path / "line.xml"
    write_iam_xml(strokes(), 'a < b & "c"', path)
    back, text = read_iam_xml(path)
    assert text == 'a < b & "c"'
    assert len(back) == 2 and np.allclose(back[0][:, :2], strokes()[0][:, :2], atol=0.006)


def test_holdout_never_trains():
    samples = [{"text": "Hold  me out", "holdout": True, "source": "capture"},
               {"text": "hold me out", "holdout": False, "source": "capture"},
               {"text": "train me", "holdout": False, "source": "capture"},
               {"text": "hold me out", "holdout": False, "source": "photo"}]
    train, held = split(samples)
    assert [s["text"] for s in train] == ["train me", "hold me out"] and len(held) == 1
    assert norm_text(" A  b ") == "a b"


def test_alphabet_extension_keeps_old_behaviour():
    torch.manual_seed(0)
    old = Charset("ab-")
    model = SynthesisNetwork(old.size, hidden_size=16)
    wider, chars = extend_alphabet(model, old.chars, "=b")
    assert chars == "ab-="
    x = torch.randn(1, 5, 3)
    c_old = torch.zeros(1, 2, old.size)
    c_old[0, 0, 1] = c_old[0, 1, 2] = 1
    c_new = torch.zeros(1, 2, len(chars) + 1)
    c_new[0, 0, 1] = c_new[0, 1, 2] = 1
    mask = torch.ones(1, 2)
    for a, b in zip(model(x, c_old, mask), wider(x, c_new, mask)):
        assert torch.allclose(a, b, atol=1e-5)


def test_markup_parsing():
    assert layout.parse_atom("x^2") == [("text", "x"), ("sup", "2")]
    assert layout.parse_atom("\\frac{a+1}{b}") == [("frac", "a+1", "b")]
    assert layout.parse_atom("a_{n}^{10}") == [("text", "a"), ("sub", "n"), ("sup", "10")]
    assert layout.split_tokens("so \\frac{1 + x}{2} is") == ["so", "\\frac{1 + x}{2}", "is"]
    assert layout.is_math("m^2") and not layout.is_math("plain") and not layout.is_math("end^")


def test_dtw_and_cer():
    a = np.random.default_rng(0).normal(size=(40, 3))
    assert dtw(a, a) == 0.0 and dtw(a, a + 1) > 0
    assert cer(["helo world"], ["hello world"]) == 1 / 11


def test_promotion_rule():
    champion = {"dtw_dir": 1.0, "cer": 0.10, "complete": 1.0}
    assert better({"dtw_dir": 0.9, "cer": 0.10, "complete": 1.0}, champion)[0]
    assert not better({"dtw_dir": 0.9, "cer": 0.20, "complete": 1.0}, champion)[0]
    assert not better({"dtw_dir": 1.0, "cer": 0.10, "complete": 1.0}, champion)[0]
    assert better({"dtw_dir": 5.0, "cer": None, "complete": 1.0}, None)[0]


def test_word_split_and_chunks():
    assert [" ".join(c) for c in chunk_words("aaaa bbbb cccc dddd".split(), 10)] == ["aaaa bbbb", "cccc dddd"]
    s1 = np.stack([np.linspace(0, 10, 12), np.linspace(0, 8, 12)], axis=1)
    s2 = s1 + [30, 0]
    words = split_words([s1, s2], [np.zeros(12, int), np.full(12, 3)], "ab cd")
    assert [w["text"] for w in words] == ["ab", "cd"] and all(w["strokes"] for w in words)
    assert abs(words[1]["gap"] - 20) < 1e-6
    assert core_metrics([s1, s2])["core"] > 0


def test_glyph_library(tmp_path):
    from hw.glyphs import GlyphWriter, save_glyph
    path = tmp_path / "glyphs.json"
    box = [[0.0, 20.0], [2.0, 17.0], [4.0, 20.0]]
    for ch in "ox":
        save_glyph(path, ch, [[[[p[0] + 30, p[1]] for p in box]], [box], []], 20.0)
    writer = GlyphWriter(path)
    assert abs(writer.core_units - 3.0) < 1e-6 and writer.unsupported("ox q") == ["q"]
    words = writer.words(["ox xo q"], 0.9, 1)[0]
    assert [w["text"] for w in words] == ["ox", "xo"] and words[1]["gap"] > 0
    assert all(len(w["strokes"]) == 2 and w["width"] > 4 for w in words)
    page = layout.layout_page(writer, "ox xo^{o}", None, 3)
    assert len(page["pages"][0]) == 5


def test_photo_lines_become_samples(tmp_path):
    import json
    from hw.dataset import load_samples, writer_profile
    words = tmp_path / "photo_words"
    words.mkdir()
    stroke = [[0, 0], [10, 20], [20, 0], [30, 20]]
    for order, (text, status) in enumerate([("the", "accepted"), ("old", "accepted"), ("x", "rejected"), ("house", "accepted")]):
        (words / f"p_{order}.json").write_text(json.dumps({"id": f"p_{order}", "photo": "p", "line": 0, "order": order, "text": text,
                                                             "status": status, "strokes": [stroke], "box": [order * 50, 100, order * 50 + 40, 130]}))
    samples = load_samples(tmp_path)
    texts = sorted(s["text"] for s in samples)
    assert texts == ["house", "old", "the", "the old"]
    line = next(s for s in samples if s["text"] == "the old")
    assert line["strokes"][1][0][0] == 50 and not line["timed"]
    assert writer_profile(samples, {"core": 100.0, "step": 10.0, "ink": 400.0})["photo_units"]["p"] > 0


def test_fragments_assemble_new_words():
    from hw.fragments import Fragments

    def word(n):
        x = np.linspace(0, 2.0 * n, 40 * n)
        return [np.stack([x, -1.0 + np.cos(x * np.pi)], axis=1)]

    frags = Fragments({"cat": [word(3)], "dog": [word(3)], "at": [word(2)]})
    assert all(s.cuts and len(s.cuts) == len(s.text) + 1 for s in frags.samples)
    cuts = frags.samples[0].cuts
    assert abs(frags.samples[0].points[cuts[1], 0] - 2.0) < 0.6
    rng = np.random.default_rng(0)
    strokes, units = frags.compose("cog", rng)
    assert units == 2 and len(strokes) == 1 and abs(strokes[0][:, 0].max() - 6.0) < 1.2
    assert frags.compose("cab", rng) is None


def test_letters_join_when_habit_says_so(tmp_path):
    from hw.glyphs import GlyphWriter, save_glyph
    path = tmp_path / "glyphs.json"
    hump = [[x / 10, 20 - 1.5 * abs(np.sin(x / 10 * np.pi))] for x in range(0, 21)]
    for ch in "nu":
        save_glyph(path, ch, [[hump]], 20.0)
    writer = GlyphWriter(path)
    rng = np.random.default_rng(0)
    apart = writer.word("nun", rng, 0.4, {})
    writer.style.update({"join": 1.0, "join_out": {}, "join_in": {}})
    joined = writer.word("nun", rng, 0.4, {})
    assert len(apart["strokes"]) == 3 and len(joined["strokes"]) == 1
    assert joined["width"] < apart["width"] + 1.0 and len(joined["strokes"][0]) > 63
    writer.joins = 0.0
    assert len(writer.word("nun", rng, 0.4, {})["strokes"]) == 3


def test_plotter_gcode_is_safe():
    from hw import plotter
    pen = plotter.Pen(offset_x=-28, offset_y=-35, holder_front=100)
    page = [np.array([[30.0 + 10 * k, y], [34.0 + 10 * k, y + 3], [38.0 + 10 * k, y]]) for y in range(40, 240, 9) for k in range(15)]
    page.append(np.array([[100.0, 60.0]]))
    parts, problems = plotter.place_page(page, 215.9, 279.4, pen)
    assert not problems and len(parts) == 2 and parts[0]["turned"] is False and parts[1]["turned"] is True
    assert sum(len(part["strokes"]) for part in parts) == len(page)
    for part in parts:
        job, issues = plotter.build_gcode(part["strokes"], pen)
        assert not issues
        faults, drawn = plotter.simulate(job["gcode"], pen)
        assert faults == [] and drawn > 100
        code = job["gcode"]
        assert code.index("G28") < code.index("M400 U1") < code.index("; PEN_FITTED")
        assert "M109" not in code and "M190" not in code and " E" not in code.replace("EXECUTABLE", "")
        blob = plotter.package(job, part["strokes"])
        import io, zipfile
        names = zipfile.ZipFile(io.BytesIO(blob)).namelist()
        assert "Metadata/plate_1.gcode" in names and "Metadata/project_settings.config" in names
    one, none = plotter.place_page(page, 215.9, 279.4, plotter.Pen(offset_x=-28, offset_y=-35, holder_front=40))
    assert not none and len(one) == 1
    bad, why = plotter.build_gcode([np.array([[300.0, 10.0], [310.0, 10.0]])], pen)
    assert bad is None and why
    high = plotter.build_gcode(parts[0]["strokes"], pen, dry_run=True)[0]["gcode"]
    assert f"G1 Z{plotter.num(pen.z_draw + 5)}" in high and f"G1 Z{plotter.num(pen.z_draw)} " not in high
    cal, issues = plotter.calibration(pen)
    assert not issues and plotter.simulate(cal["gcode"], pen)[0] == [] and cal["pauses"] == 2
