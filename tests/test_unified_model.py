import unittest


try:
    import torch
except ImportError:  # Local Data I/O development can be NumPy-only.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TransformerSWAStateTest(unittest.TestCase):
    def _model(self):
        from robofm.backbones.transformer_swa import TransformerSWABackbone

        torch.manual_seed(7)
        return TransformerSWABackbone(
            hidden_size=16,
            num_layers=2,
            num_heads=4,
            intermediate_size=32,
            window_size=8,
            dropout=0.0,
        ).eval()

    def test_chunked_matches_full_sequence(self):
        model = self._model()
        inputs = torch.randn(2, 9, 16)
        positions = torch.arange(9).expand(2, -1)
        valid = torch.ones(2, 9, dtype=torch.bool)
        update = valid.clone()
        update[1, 3] = False
        reset = torch.zeros_like(valid)
        reset[0, 5] = True

        full = model.forward_chunk(
            inputs,
            attention_mask=valid,
            memory_update_mask=update,
            reset_mask=reset,
            position_ids=positions,
        )
        first = model.forward_chunk(
            inputs[:, :4],
            attention_mask=valid[:, :4],
            memory_update_mask=update[:, :4],
            reset_mask=reset[:, :4],
            position_ids=positions[:, :4],
        )
        second = model.forward_chunk(
            inputs[:, 4:],
            state=first.state,
            attention_mask=valid[:, 4:],
            memory_update_mask=update[:, 4:],
            reset_mask=reset[:, 4:],
            position_ids=positions[:, 4:],
        )
        chunked = torch.cat((first.hidden_states, second.hidden_states), dim=1)
        torch.testing.assert_close(full.hidden_states, chunked, atol=1e-5, rtol=1e-5)

    def test_all_padding_preserves_state(self):
        model = self._model()
        first = model.forward_chunk(torch.randn(1, 2, 16))
        empty = model.forward_chunk(
            torch.empty(1, 0, 16), state=first.state,
            attention_mask=torch.empty(1, 0, dtype=torch.bool),
        )
        self.assertIs(empty.state, first.state)
        self.assertEqual(tuple(empty.hidden_states.shape), (1, 0, 16))

    def test_state_lane_select_scatter_and_detach(self):
        from robofm.backbones import (
            detach_state, scatter_state_lanes, select_state_lanes, state_nbytes
        )

        state = {"memory": torch.arange(24.0).reshape(3, 2, 4).requires_grad_()}
        indices = torch.tensor([2, 0])
        selected = select_state_lanes(state, indices, batch_size=3)
        torch.testing.assert_close(selected["memory"], state["memory"][[2, 0]])
        updated = {"memory": selected["memory"] + 100}
        scattered = scatter_state_lanes(state, updated, indices, batch_size=3)
        torch.testing.assert_close(scattered["memory"][1], state["memory"][1])
        detached = detach_state(scattered)
        self.assertFalse(detached["memory"].requires_grad)
        self.assertEqual(state_nbytes(detached), detached["memory"].numel() * 4)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class UnifiedRuntimeTest(unittest.TestCase):
    def test_tbptt_checkpoint_restores_next_lane_cursor(self):
        import tempfile
        from pathlib import Path

        from robofm.backbones import BackboneConfig
        from robofm.dataio import AtomType, LaneScheduler, UnifiedDatasetWriter, UnifiedMMapDataset
        from robofm.models.unified_sequence import UnifiedModelConfig, UnifiedSequenceModel
        from robofm.runtime.config import (
            CheckpointConfig, LoggingConfig, RuntimeConfig, TBPTTConfig
        )
        from robofm.runtime.distributed import DistributedContext
        from robofm.runtime.trainer import UnifiedTrainer

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_root = root / "data"
            with UnifiedDatasetWriter(
                data_root, tokenizer={"name": "test"}, special_tokens={"pad": 0}
            ) as writer:
                writer.append_record(
                    [1, 2, 3, 4, 5],
                    [AtomType.LANGUAGE_TOKEN] * 5,
                    loss_mask=[0, 1, 1, 1, 1],
                )
            model_config = UnifiedModelConfig(
                vocab_size=16,
                hidden_size=16,
                backbone=BackboneConfig(
                    name="transformer_swa", hidden_size=16, num_layers=1,
                    num_heads=4, intermediate_size=32, window_size=4,
                ),
            )
            context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
            tbptt = TBPTTConfig(
                chunk_length=2, lane_count=1, tbptt_chunks=2,
                optimizer_step_chunks=2,
            )
            runtime = RuntimeConfig(dtype="float32", fsdp2=False)
            logging = LoggingConfig(directory=str(root / "logs"), tensorboard=False)
            checkpoint = CheckpointConfig(
                directory=str(root / "checkpoints"), every_optimizer_steps=1
            )
            with UnifiedMMapDataset(data_root) as dataset:
                lanes = LaneScheduler(
                    dataset, lane_count=1, chunk_length=2,
                    shuffle=False, repeat=False,
                )
                model = UnifiedSequenceModel(model_config)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                trainer = UnifiedTrainer(
                    model=model, optimizer=optimizer, lane_scheduler=lanes,
                    context=context, tbptt=tbptt, runtime=runtime,
                    logging=logging, checkpoint=checkpoint,
                )
                progress = trainer.run(max_optimizer_steps=1)
                trainer.close()
                self.assertEqual(progress.optimizer_step, 1)
                self.assertEqual(lanes.peek()[0].atom_offset, 4)
                lanes.rollback()

                restored_lanes = LaneScheduler(
                    dataset, lane_count=1, chunk_length=2,
                    shuffle=False, repeat=False,
                )
                restored_model = UnifiedSequenceModel(model_config)
                restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
                restored = UnifiedTrainer(
                    model=restored_model, optimizer=restored_optimizer,
                    lane_scheduler=restored_lanes, context=context, tbptt=tbptt,
                    runtime=runtime, logging=LoggingConfig(
                        directory=str(root / "logs-restored"), tensorboard=False
                    ), checkpoint=checkpoint,
                )
                restored.resume()
                self.assertEqual(restored.progress.optimizer_step, 1)
                self.assertEqual(restored_lanes.peek()[0].atom_offset, 4)
                restored_lanes.rollback()
                restored.close()


if __name__ == "__main__":
    unittest.main()
