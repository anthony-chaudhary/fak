#!/usr/bin/env python
"""Build a tiny random text-only qwen3_5 (Qwen3.6 / Qwen3-Next) fixture for the fak
Gated-DeltaNet oracle.

No public tiny `qwen3_5` checkpoint exists (Qwen3.6-27B is the only published size,
15.4 GB in q4), so — exactly as the minimax_m3 fixture does (à la
yujiepan/glm-5-tiny-random) — this constructs a small, CPU-instantiable
`Qwen3_5ForCausalLM` text decoder with random weights. It exercises the whole
hybrid stack the in-kernel engine must reproduce:

  - 3 `linear_attention` Gated-DeltaNet layers (short causal conv k=3 + delta-rule
    recurrence + swish output gate, fp32 state) and
  - 1 `full_attention` layer with the architectural output gate (q/gate chunk +
    sigmoid), per-head qk-norm, and partial RoPE (0.25),

i.e. the exact `internal/model/qwen35.go` + gated-full-attn path, at f32 on a plain
CPU box (no GPU / 27B artifact node needed). The witness is HF transformers (which
we did NOT author): for fixed token IDs it emits embedding + per-layer hidden states
+ logits, and the Go core must reproduce them.

Requires `transformers>=5.10` (the first release shipping native `qwen3_5` modeling)
and torch. Usage:

    python internal/model/make_qwen35_tiny.py .cache/qwen35-tiny
    python internal/model/export_oracle.py --online \
      --model .cache/qwen35-tiny --out internal/model/.cache/oracle-qwen35 \
      --prompt-ids-json '[[785,6722,315,9621,374],[16,11,220,17,11,220,18,11,220,19,11],[750,912,2877,11,293,982,262,470]]'

then `go test ./internal/model/ -run TestOptionalQwen35 -count=1`.
"""
import sys

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

out = sys.argv[1] if len(sys.argv) > 1 else ".cache/qwen35-tiny"

# GPT-2 BPE tokenizer (vocab 50257): the exporter calls AutoTokenizer on the fixture
# dir, but the forward-parity proof feeds token IDS only, so the tokenizer identity is
# irrelevant — the standard oracle prompt ids (max 9621) stay in range.
tok = AutoTokenizer.from_pretrained("gpt2")

cfg = Qwen3_5TextConfig(
    vocab_size=tok.vocab_size,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    # Gated-DeltaNet linear-attention axes (16 key / 48 value heads x 128 at full size).
    linear_conv_kernel_dim=3,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    # full_attention_interval=4 -> layer_types = [linear, linear, linear, full]: every
    # fourth layer is gated full attention, the rest are Gated DeltaNet.
    full_attention_interval=4,
    max_position_embeddings=512,
    # partial RoPE 0.25 (the real model's value): only the leading head_dim*0.25 lanes
    # rotate; the rest pass through. theta default 1e4 (the 27B uses 1e7).
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
    bos_token_id=tok.bos_token_id,
    eos_token_id=tok.eos_token_id,
)

torch.manual_seed(0)
model = Qwen3_5ForCausalLM(cfg).to(torch.float32).eval()

# De-trivialize the zero-init RMSNorm weights so the (1+w) gain on the ordinary norms
# (input/post/q/k/final) actually bends the norm — otherwise every gain is exactly 1.0
# and the (1+w) vs plain-weight distinction (the gated DeltaNet norm uses plain weight)
# would go untested. The Gated-DeltaNet norm.weight is ones-init plain weight; perturbing
# it stays a valid plain weight on both sides since both read the same exported tensor.
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name and p.dim() >= 1:
            p.normal_(0.0, 0.1)

model.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print(f"saved tiny qwen3_5 fixture -> {out}")
print(f"  model_type={cfg.model_type} layers={cfg.num_hidden_layers} layer_types={cfg.layer_types}")
print(f"  params={sum(p.numel() for p in model.parameters()):,}")
