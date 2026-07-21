import tempfile
from pathlib import Path
import unittest

import numpy as np

from airsoul.dataio import (
    AtomType,
    LaneScheduler,
    TargetEntry,
    UnifiedDatasetWriter,
    UnifiedMMapDataset,
    collate_chunks,
)
from airsoul.dataio.validate import validate_dataset


class UnifiedDataIOTest(unittest.TestCase):
    def _write_dataset(self, root: Path) -> Path:
        output = root / "dataset"
        with UnifiedDatasetWriter(
            output,
            tokenizer={"name": "test", "vocab_size": 1024, "hash": "test-hash"},
            special_tokens={"pad": 0, "bos": 1, "eos": 2, "image": 3},
            producer={"name": "unit-test", "version": "1"},
            max_atoms_per_shard=4,
        ) as writer:
            writer.append_record(
                atom_values=[1, 0, 10, 2],
                atom_types=[AtomType.LANGUAGE_TOKEN, AtomType.IMAGE,
                            AtomType.LANGUAGE_TOKEN, AtomType.LANGUAGE_TOKEN],
                images=[np.arange(18, dtype=np.uint8).reshape(2, 3, 3)],
                loss_mask=[0, 0, 1, 1],
                memory_update_mask=[1, 0, 1, 1],
                reset_mask=[1, 0, 0, 0],
                targets=[
                    TargetEntry(2, AtomType.LANGUAGE_TOKEN, 99, weight=0.5),
                    TargetEntry(3, AtomType.IMAGE, 0),
                ],
            )
            writer.append_record(
                atom_values=[1, 2],
                atom_types=[AtomType.LANGUAGE_TOKEN, AtomType.LANGUAGE_TOKEN],
                loss_mask=[0, 1],
            )
        return output

    def test_round_trip_mmap_images_targets_and_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            output = self._write_dataset(Path(temp))
            with UnifiedMMapDataset(output) as dataset:
                self.assertEqual(len(dataset), 2)
                self.assertEqual(len(dataset.shards), 2)
                first = dataset[0]
                self.assertEqual(first.atom_values.tolist(), [1, 0, 10, 2])
                self.assertEqual(first.loss_mask.tolist(), [0, 0, 1, 1])
                self.assertEqual(first.memory_update_mask.tolist(), [1, 0, 1, 1])
                self.assertEqual(first.targets["target_value"].tolist(), [99, 0])
                image = dataset.get_image(first.shard_index, int(first.atom_values[1]))
                np.testing.assert_array_equal(image, np.arange(18, dtype=np.uint8).reshape(2, 3, 3))

                chunk = dataset.read_chunk(0, 2, 2)
                self.assertTrue(chunk.end_of_record)
                self.assertEqual(chunk.targets["source_position"].tolist(), [0, 1])

                second_chunk = dataset.read_chunk(1, 0, 2)
                batch = collate_chunks(
                    [dataset.read_chunk(0, 0, 3), second_chunk],
                    pad_token_id=0,
                    pad_to_multiple=4,
                    dataset=dataset,
                    load_images=True,
                )
                self.assertEqual(batch.atom_values.shape, (2, 4))
                self.assertEqual(batch.lengths.tolist(), [3, 2])
                self.assertEqual(batch.padding_mask[0].tolist(), [False, False, False, True])
                self.assertEqual(batch.loss_mask[1].tolist(), [False, True, False, False])
                self.assertEqual(batch.memory_update_mask[0].tolist(), [True, False, True, False])
                self.assertIn(1, batch.image_payloads[0])

                fixed = collate_chunks(
                    [None, second_chunk], pad_token_id=0, pad_to_length=8
                )
                self.assertEqual(fixed.atom_values.shape, (2, 8))
                self.assertEqual(fixed.lengths.tolist(), [0, 2])

            report = validate_dataset(str(output))
            self.assertEqual((report.records, report.atoms, report.targets, report.images), (2, 6, 2, 1))

    def test_lane_cursor_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            output = self._write_dataset(Path(temp))
            with UnifiedMMapDataset(output) as dataset:
                scheduler = LaneScheduler(
                    dataset, lane_count=1, chunk_length=2,
                    shuffle=False, repeat=False,
                )
                first = scheduler.peek()[0]
                self.assertEqual((first.record_index, first.atom_offset), (0, 0))
                scheduler.commit()
                state = scheduler.state_dict()

                restored = LaneScheduler(
                    dataset, lane_count=1, chunk_length=2,
                    shuffle=False, repeat=False,
                )
                restored.load_state_dict(state)
                next_request = restored.peek()[0]
                self.assertEqual((next_request.record_index, next_request.atom_offset), (0, 2))
                restored.commit()
                following = restored.peek()[0]
                self.assertEqual((following.record_index, following.atom_offset), (1, 0))

    def test_abort_does_not_publish_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dataset"
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with UnifiedDatasetWriter(
                    output,
                    tokenizer={"name": "test"},
                    special_tokens={"pad": 0},
                ) as writer:
                    writer.append_record(
                        [1], [AtomType.LANGUAGE_TOKEN], loss_mask=[1]
                    )
                    raise RuntimeError("stop")
            self.assertFalse(output.exists())

    def test_streaming_record_is_chunk_bounded_and_checksum_compatible(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "streamed"
            with UnifiedDatasetWriter(
                output,
                tokenizer={"name": "test"},
                special_tokens={"pad": 0},
            ) as writer:
                with writer.stream_record(expected_atoms=5) as record:
                    record.append(
                        [1, 4],
                        [AtomType.LANGUAGE_TOKEN, AtomType.LANGUAGE_TOKEN],
                        loss_mask=[0, 1],
                        reset_mask=[1, 0],
                    )
                    record.append(
                        [0, 5, 6],
                        [AtomType.IMAGE, AtomType.LANGUAGE_TOKEN, AtomType.LANGUAGE_TOKEN],
                        images=[np.ones((2, 2, 3), dtype=np.uint8)],
                        memory_update_mask=[1, 0, 1],
                        targets=[TargetEntry(2, AtomType.LANGUAGE_TOKEN, 7)],
                    )
            with UnifiedMMapDataset(output) as dataset:
                streamed = dataset[0]
                self.assertEqual(streamed.atom_values.tolist(), [1, 4, 0, 5, 6])
                self.assertEqual(streamed.targets["source_position"].tolist(), [4])
                self.assertEqual(streamed.memory_update_mask.tolist(), [1, 1, 1, 0, 1])
            report = validate_dataset(str(output))
            self.assertEqual((report.records, report.atoms), (1, 5))


if __name__ == "__main__":
    unittest.main()
