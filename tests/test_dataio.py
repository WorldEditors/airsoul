import tempfile
from pathlib import Path
import unittest

import numpy as np

from robofm.dataio import (
    AtomType,
    LaneScheduler,
    TargetEntry,
    UnifiedDatasetWriter,
    UnifiedMMapDataset,
    collate_chunks,
    open_unified_dataset,
    write_unified_messages,
)
from robofm.dataio.validate import validate_dataset
from robofm.dataio import write_unified_record
from robofm.dataio.producer import (
    BYTE_TOKEN_BASE,
    DEFAULT_SPECIAL_TOKENS,
    TOKENIZER_VOCAB_SIZE,
)


class UnifiedDataIOTest(unittest.TestCase):
    def test_standard_vlm_and_function_messages(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "messages"
            write_unified_messages(
                output,
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "inspect this frame"},
                            {"type": "image", "image": np.zeros((2, 2, 3), dtype=np.uint8)},
                        ],
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "set_action",
                                "arguments": {"action": 2},
                            },
                        }],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "tool_result": {"ok": True},
                    },
                ],
                producer={"name": "test", "protocol": "messages-v1"},
            )
            with UnifiedMMapDataset(output) as dataset:
                self.assertEqual(dataset.manifest["producer"]["protocol"], "messages-v1")
                record = dataset[0]
                self.assertEqual(
                    int(np.count_nonzero(record.atom_types == int(AtomType.IMAGE))),
                    1,
                )
                self.assertIn(DEFAULT_SPECIAL_TOKENS["<|tool_call|>"], record.atom_values)

    def test_benchmark_producer_writes_committed_v1_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "record-000000"
            write_unified_record(
                output,
                {
                    "observations": np.arange(8, dtype=np.float32).reshape(2, 4),
                    "actions": np.array([1, 2], dtype=np.int32),
                    "images": np.arange(18, dtype=np.uint8).reshape(2, 3, 3, 1),
                },
                producer={"name": "test"},
            )
            self.assertTrue((output / "COMMITTED").is_file())
            dataset = UnifiedMMapDataset(output)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.manifest["producer"]["name"], "test")
            self.assertEqual(dataset.manifest["totals"]["images"], 2)
            record = dataset[0]
            language_values = record.atom_values[
                record.atom_types == int(AtomType.LANGUAGE_TOKEN)
            ]
            self.assertEqual(int(language_values[0]), DEFAULT_SPECIAL_TOKENS["<bos>"])
            self.assertEqual(int(language_values[-1]), DEFAULT_SPECIAL_TOKENS["<eos>"])
            self.assertEqual(
                int(np.count_nonzero(language_values == DEFAULT_SPECIAL_TOKENS["<field>"])),
                6,
            )
            self.assertLess(int(language_values.max()), TOKENIZER_VOCAB_SIZE)
            dataset.close()

    def test_record_collection_uses_global_record_and_image_indexes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(2):
                write_unified_record(
                    root / f"record-{index:06d}",
                    {"images": np.full((1, 2, 2, 1), index, dtype=np.uint8)},
                    producer={"name": "test"},
                )
            dataset = open_unified_dataset(root)
            self.assertEqual(len(dataset), 2)
            first = dataset.read_chunk(0, 0, dataset.record_length(0))
            second = dataset.read_chunk(1, 0, dataset.record_length(1))
            self.assertEqual(first.record_index, 0)
            self.assertEqual(second.record_index, 1)
            batch = collate_chunks(
                [first, second], pad_token_id=0, dataset=dataset, load_images=True
            )
            self.assertEqual(int(next(iter(batch.image_payloads[0].values())).max()), 0)
            self.assertEqual(int(next(iter(batch.image_payloads[1].values())).max()), 1)
            dataset.close()

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
