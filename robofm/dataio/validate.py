"""Dataset integrity validator and CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .reader import UnifiedMMapDataset
from .schema import AtomType, record_checksum


@dataclass(frozen=True)
class ValidationReport:
    records: int
    atoms: int
    targets: int
    images: int


def validate_dataset(root: str, *, verify_checksums: bool = True) -> ValidationReport:
    root_path = Path(root)
    if not (root_path / "COMMITTED").is_file():
        children = sorted(
            path for path in root_path.iterdir()
            if path.is_dir() and (path / "COMMITTED").is_file()
        )
        if not children:
            raise ValueError(f"no committed V1 datasets found under: {root_path}")
        reports = [
            validate_dataset(str(path), verify_checksums=verify_checksums)
            for path in children
        ]
        return ValidationReport(
            records=sum(report.records for report in reports),
            atoms=sum(report.atoms for report in reports),
            targets=sum(report.targets for report in reports),
            images=sum(report.images for report in reports),
        )
    dataset = UnifiedMMapDataset(root)
    atom_count = 0
    target_count = 0
    referenced_images: set[tuple[int, int]] = set()
    for record_index in range(len(dataset)):
        record = dataset[record_index]
        atom_count += len(record.atom_values)
        target_count += len(record.targets)
        if not bool(np.all((record.loss_mask == 0) | (record.loss_mask == 1))):
            raise ValueError(f"record {record_index} has a non-binary loss mask")
        if not bool(np.all((record.reset_mask == 0) | (record.reset_mask == 1))):
            raise ValueError(f"record {record_index} has a non-binary reset mask")
        if not bool(np.all(
            (record.memory_update_mask == 0) | (record.memory_update_mask == 1)
        )):
            raise ValueError(f"record {record_index} has a non-binary memory update mask")
        if not bool(np.all(np.isin(record.atom_types, [
            int(AtomType.LANGUAGE_TOKEN), int(AtomType.IMAGE)
        ]))):
            raise ValueError(f"record {record_index} has an unsupported atom type")
        image_positions = record.atom_types == int(AtomType.IMAGE)
        for image_index in record.atom_values[image_positions]:
            image_id = int(image_index)
            dataset.get_image(record.shard_index, image_id)
            referenced_images.add((record.shard_index, image_id))
        if len(record.targets):
            if int(record.targets["source_position"].max()) >= len(record.atom_values):
                raise ValueError(f"record {record_index} has an out-of-range target source")
            image_targets = record.targets["atom_type"] == int(AtomType.IMAGE)
            for image_index in record.targets["target_value"][image_targets]:
                image_id = int(image_index)
                dataset.get_image(record.shard_index, image_id)
                referenced_images.add((record.shard_index, image_id))
        if verify_checksums:
            shard = dataset.shards[record.shard_index]
            expected = int(shard.records[record.shard_record_index]["checksum"])
            actual = record_checksum(
                record.atom_values, record.atom_types,
                record.loss_mask, record.memory_update_mask,
                record.reset_mask, record.targets,
            )
            if expected != actual:
                raise ValueError(f"record {record_index} checksum mismatch: {expected} != {actual}")
    return ValidationReport(
        records=len(dataset),
        atoms=atom_count,
        targets=target_count,
        images=len(referenced_images),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args(argv)
    report = validate_dataset(args.dataset, verify_checksums=not args.skip_checksums)
    print(
        f"valid dataset: records={report.records}, atoms={report.atoms}, "
        f"targets={report.targets}, referenced_images={report.images}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
