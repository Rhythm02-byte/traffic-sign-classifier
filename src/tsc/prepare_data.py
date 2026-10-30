"""Convert the Kaggle copy of GTSRB into the layout torchvision's GTSRB class reads.

Kaggle copy (default location: data/):
    Train/<class>/<file>.png        class folders named 0..42
    Test/<file>.png and Test.csv    Test.csv has ClassId and Path columns

Output (default location: data/gtsrb/):
    GTSRB/Training/<00000>/<file>.ppm      zero-padded class folders
    GTSRB/Final_Test/Images/<file>.ppm
    GT-final_test.csv                      Filename;ClassId

The Kaggle files are only read. Existing output is never overwritten.
"""
import argparse
import csv
from pathlib import Path

from PIL import Image

from tsc.data import DATA_DIR


def convert_training(kaggle_train, out_dir):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} already exists and is not empty, refusing to overwrite")

    count = 0
    for class_dir in sorted(kaggle_train.iterdir(), key=lambda p: int(p.name)):
        # Zero-padded names make the folder order match the class numbers
        class_out = out_dir / f"{int(class_dir.name):05d}"
        class_out.mkdir(parents=True, exist_ok=True)
        for png in sorted(class_dir.glob("*.png")):
            Image.open(png).convert("RGB").save(class_out / f"{png.stem}.ppm")
            count += 1
    return count


def convert_test(kaggle_dir, images_dir, gt_csv):
    if gt_csv.exists():
        raise SystemExit(f"{gt_csv} already exists, refusing to overwrite")

    with open(kaggle_dir / "Test.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    images_dir.mkdir(parents=True, exist_ok=True)
    with open(gt_csv, "w", newline="") as out:
        writer = csv.writer(out, delimiter=";")
        writer.writerow(["Filename", "ClassId"])
        for row in rows:
            # The same row gives the image and its label, so they always match
            src = kaggle_dir / row["Path"]
            filename = Path(row["Path"]).stem + ".ppm"
            Image.open(src).convert("RGB").save(images_dir / filename)
            writer.writerow([filename, row["ClassId"]])
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert the Kaggle GTSRB copy for torchvision.")
    parser.add_argument("--kaggle-dir", type=Path, default=DATA_DIR, help="folder holding Train/, Test/, Test.csv")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "gtsrb", help="folder to write the torchvision layout to")
    args = parser.parse_args()

    n_train = convert_training(args.kaggle_dir / "Train", args.out / "GTSRB" / "Training")
    print(f"Converted {n_train} training images")
    n_test = convert_test(args.kaggle_dir, args.out / "GTSRB" / "Final_Test" / "Images", args.out / "GT-final_test.csv")
    print(f"Converted {n_test} test images")


if __name__ == "__main__":
    main()
