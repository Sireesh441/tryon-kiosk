"""
filter_person_annotations.py

Filters a COCO instances_*.json annotation file down to person-only
annotations, and writes:
  1. A slimmed annotation JSON (same COCO format, person-only)
  2. A plain text list of image filenames that contain at least one person

Usage:
    py filter_person_annotations.py                # val2017 (default)
    py filter_person_annotations.py --split train2017

Expects the COCO layout:
    data/coco/annotations/instances_{split}.json
    data/coco/{split}/*.jpg
"""

import argparse
import json
from pathlib import Path

# --- Config ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
PERSON_CATEGORY_NAME = "person"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="val2017", choices=["val2017", "train2017"],
        help="Which COCO split's annotations to filter (default: val2017)",
    )
    args = parser.parse_args()

    annotations_dir = PROJECT_ROOT / "data" / "coco" / "annotations"
    annotations_path = annotations_dir / f"instances_{args.split}.json"
    output_annotations_path = annotations_dir / f"instances_{args.split}_person.json"
    output_image_list_path = annotations_dir / f"{args.split}_person_image_filenames.txt"

    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Could not find {annotations_path}. "
            f"Make sure {args.split} annotations are extracted to data/coco/annotations/"
        )

    print(f"Loading {annotations_path.name} ...")
    with open(annotations_path, "r") as f:
        coco = json.load(f)

    # Find the person category ID (should be 1 in COCO, but don't hardcode it)
    person_category = next(
        (cat for cat in coco["categories"] if cat["name"] == PERSON_CATEGORY_NAME),
        None,
    )
    if person_category is None:
        raise ValueError("No 'person' category found in this annotation file.")
    person_category_id = person_category["id"]
    print(f"Found 'person' category with id={person_category_id}")

    # Filter annotations to person-only
    person_annotations = [
        ann for ann in coco["annotations"] if ann["category_id"] == person_category_id
    ]
    print(f"Kept {len(person_annotations)} / {len(coco['annotations'])} annotations (person-only)")

    # Find which images actually have at least one person annotation
    image_ids_with_person = {ann["image_id"] for ann in person_annotations}
    person_images = [img for img in coco["images"] if img["id"] in image_ids_with_person]
    print(f"{len(person_images)} / {len(coco['images'])} images contain at least one person")

    # Build the slimmed COCO-format output
    filtered_coco = {
        "images": person_images,
        "annotations": person_annotations,
        "categories": [person_category],
    }

    output_annotations_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_annotations_path, "w") as f:
        json.dump(filtered_coco, f)
    print(f"Wrote filtered annotations to {output_annotations_path}")

    # Write plain filename list for quick reference / sanity checks
    filenames = sorted(img["file_name"] for img in person_images)
    with open(output_image_list_path, "w") as f:
        f.write("\n".join(filenames))
    print(f"Wrote {len(filenames)} filenames to {output_image_list_path}")

    print("\nDone. Sanity check a few entries:")
    for img in person_images[:3]:
        print(f"  - {img['file_name']} (id={img['id']})")


if __name__ == "__main__":
    main()
