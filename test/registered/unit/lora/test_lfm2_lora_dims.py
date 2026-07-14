"""Unit tests for LFM2 wider-surface LoRA (out_proj + SwiGLU MLP).

Guards two regressions from enabling LoRA beyond q/k/v on LFM2:

1. `Lfm2ForCausalLM.get_hidden_dim` shapes. Without this hook the LoRA memory
   pool fell back to a llama-name table that only knew `o_proj`/`gate_up_proj`
   and raised `NotImplementedError` for LFM2's `out_proj`/`w1`/`w2`/`w3`. The
   w1/w2/w3 dims must match `Lfm2MLP`'s block-adjusted intermediate size or the
   buffer shape assert in mem_pool fires; out_proj differs between attention and
   conv layers.

2. `_rename_expert_w_to_proj` scope. It unconditionally rewrote any `.w1/.w2/.w3`
   key to `.gate_proj/.up_proj/.down_proj`, silently repacking LFM2's dense
   `feed_forward.w1/w2/w3` and breaking the MLP LoRA. It must fire on routed
   experts only.

Hermetic: methods are invoked unbound on a stub, so no CUDA / linears / dist.

Usage:
    python -m pytest test/registered/unit/lora/test_lfm2_lora_dims.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

from sglang.srt.configs.lfm2 import Lfm2Config
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.models.lfm2 import Lfm2ForCausalLM, lfm2_mlp_intermediate_size


def _make_config() -> Lfm2Config:
    # multiplier + multiple_of chosen so the rounding actually rounds:
    # int(2*768/3)=512, int(1.3*512)=665, 64*ceil(665/64)=704.
    return Lfm2Config(
        hidden_size=512,
        intermediate_size=768,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        block_auto_adjust_ff_dim=True,
        block_ffn_dim_multiplier=1.3,
        block_multiple_of=64,
        # layer 0 attention, layer 1 conv
        layer_types=["full_attention", "conv"],
    )


class TestLfm2GetHiddenDim(unittest.TestCase):
    def setUp(self):
        self.config = _make_config()
        self.stub = types.SimpleNamespace(config=self.config)

    def dim(self, module_name, layer_idx):
        return Lfm2ForCausalLM.get_hidden_dim(self.stub, module_name, layer_idx)

    def test_ffn_dim_block_adjust(self):
        # Derived property: pins the block_auto_adjust_ff_dim arithmetic.
        self.assertEqual(lfm2_mlp_intermediate_size(self.config), 704)

    def test_mlp_dims_track_intermediate_size(self):
        interm = lfm2_mlp_intermediate_size(self.config)
        self.assertEqual(self.dim("w1", 0), (512, interm))
        self.assertEqual(self.dim("w3", 0), (512, interm))
        self.assertEqual(self.dim("w2", 0), (interm, 512))

    def test_out_proj_differs_by_layer_type(self):
        # Attention layer: heads*head_dim -> hidden. Conv layer: hidden -> hidden.
        self.assertEqual(self.dim("out_proj", 0), (8 * 64, 512))
        self.assertEqual(self.dim("out_proj", 1), (512, 512))

    def test_qkv_proj(self):
        self.assertEqual(self.dim("qkv_proj", 0), (512, 64 * (8 + 4 * 2)))

    def test_unknown_module_raises(self):
        with self.assertRaises(NotImplementedError):
            self.dim("gate_up_proj", 0)


class TestRenameExpertGuard(unittest.TestCase):
    def rename(self, weights):
        # Unbound; the method only touches `weights` + the module-level regex.
        LoRAAdapter._rename_expert_w_to_proj(None, weights)
        return weights

    def test_dense_mlp_names_pass_through(self):
        weights = {
            "base_model.model.layers.1.feed_forward.w1.lora_A.weight": 0,
            "base_model.model.layers.1.feed_forward.w2.lora_B.weight": 0,
            "base_model.model.layers.1.feed_forward.w3.lora_A.weight": 0,
        }
        out = self.rename(dict(weights))
        self.assertEqual(set(out), set(weights))  # untouched

    def test_routed_expert_names_are_renamed(self):
        out = self.rename(
            {"base_model.model.layers.3.mlp.experts.0.w1.lora_A.weight": 0}
        )
        self.assertIn(
            "base_model.model.layers.3.mlp.experts.0.gate_proj.lora_A.weight", out
        )

    def test_shared_expert_names_are_renamed(self):
        # Shared experts still contain "experts" and must keep being renamed;
        # only dense (non-expert) MLPs are exempted by the guard.
        out = self.rename(
            {"base_model.model.layers.3.mlp.shared_experts.w2.lora_B.weight": 0}
        )
        self.assertIn(
            "base_model.model.layers.3.mlp.shared_experts.down_proj.lora_B.weight", out
        )


if __name__ == "__main__":
    unittest.main()
