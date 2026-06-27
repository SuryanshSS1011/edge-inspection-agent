"""Tests for the MVTec loader against a synthetic directory tree (no real dataset)."""

import pytest

from eval.datasets import load_mvtec


def _make_tree(tmp_path):
    cat = tmp_path / "bottle"
    (cat / "test" / "good").mkdir(parents=True)
    (cat / "test" / "broken_large").mkdir(parents=True)
    (cat / "train" / "good").mkdir(parents=True)
    # good test images
    (cat / "test" / "good" / "000.png").write_bytes(b"x")
    (cat / "test" / "good" / "001.png").write_bytes(b"x")
    # defective test images
    (cat / "test" / "broken_large" / "000.png").write_bytes(b"x")
    # a non-image file that must be ignored
    (cat / "test" / "good" / "notes.txt").write_text("ignore")
    # train images (all good)
    (cat / "train" / "good" / "000.png").write_bytes(b"x")
    return tmp_path


def test_test_split_labels(tmp_path):
    root = _make_tree(tmp_path)
    items = list(load_mvtec(str(root), "bottle", "test"))
    labels = {path: label for path, label, _ in items}
    assert len(items) == 3  # 2 good + 1 defective; txt ignored
    goods = [l for l in labels.values() if l == 0]
    defects = [l for l in labels.values() if l == 1]
    assert len(goods) == 2
    assert len(defects) == 1


def test_train_split_all_good(tmp_path):
    root = _make_tree(tmp_path)
    items = list(load_mvtec(str(root), "bottle", "train"))
    assert items and all(label == 0 for _, label, _ in items)


def test_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(load_mvtec(str(tmp_path), "bottle", "test"))
