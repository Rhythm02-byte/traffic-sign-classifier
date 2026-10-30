import csv

import pytest
from PIL import Image

from tsc.prepare_data import convert_test, convert_training


def make_kaggle_copy(root):
    # A tiny stand-in for the Kaggle layout: two training images, one test image
    for class_id, name in [(0, "a"), (1, "b")]:
        folder = root / "Train" / str(class_id)
        folder.mkdir(parents=True)
        Image.new("RGB", (30, 30), color=(class_id * 100, 0, 0)).save(folder / f"{name}.png")
    (root / "Test").mkdir()
    Image.new("RGB", (30, 30)).save(root / "Test" / "00000.png")
    with open(root / "Test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Width", "Height", "ClassId", "Path"])
        writer.writerow([30, 30, 1, "Test/00000.png"])
    return root


def test_convert_training_writes_zero_padded_ppm_folders(tmp_path):
    kaggle = make_kaggle_copy(tmp_path / "kaggle")
    out = tmp_path / "out" / "Training"

    count = convert_training(kaggle / "Train", out)

    assert count == 2
    assert (out / "00000" / "a.ppm").exists()
    assert (out / "00001" / "b.ppm").exists()


def test_convert_test_matches_each_image_to_its_label(tmp_path):
    kaggle = make_kaggle_copy(tmp_path / "kaggle")
    images = tmp_path / "out" / "Final_Test" / "Images"
    gt_csv = tmp_path / "out" / "GT-final_test.csv"

    count = convert_test(kaggle, images, gt_csv)

    assert count == 1
    assert (images / "00000.ppm").exists()
    lines = gt_csv.read_text().splitlines()
    assert lines == ["Filename;ClassId", "00000.ppm;1"]


def test_conversion_refuses_to_overwrite_existing_output(tmp_path):
    kaggle = make_kaggle_copy(tmp_path / "kaggle")
    out = tmp_path / "out" / "Training"
    convert_training(kaggle / "Train", out)

    with pytest.raises(SystemExit):
        convert_training(kaggle / "Train", out)
