#!/usr/bin/env python
"""Build a tiny random text-only minimax_m3 fixture for the fak MSA oracle.

No public tiny `minimax_m3` checkpoint exists (the family post-dates the dev
cutoff, and only the full 60-layer MiniMaxAI/MiniMax-M3 is published), so —
exactly as the ticket suggests "create one (à la yujiepan/glm-5-tiny-random)"
— this constructs a small, CPU-instantiable MiniMaxM3VL text decoder with
random weights: MiniMax Sparse Attention (a lightning indexer on the sparse
layers), per-head Gemma qk-norm, partial RoPE, and a SwiGLU-OAI MoE with one
always-on shared expert. It runs on a plain CPU box (no GPU/artifact node),
which is what makes the oracle reproducible here.

Requires `transformers>=5.12` (the first release that ships native minimax_m3_vl
modeling) and torch. Usage:

    python internal/model/make_minimax_m3_tiny.py .cache/minimax-m3-tiny
    python internal/model/export_oracle.py --online \
      --model .cache/minimax-m3-tiny --out internal/model/.cache/oracle-minimax \
      --prompt-ids-json '[[785,6722,315,9621,374],[16,11,220,17,11,220,18,11,220,19,11],[750,912,2877,11,293,982,262,470]]'

then `go test ./internal/model/ -run TestOptionalMiniMaxM3Oracle -count=1`.
"""
import sys

import torch
from transformers import AutoTokenizer
from transformers.models.minimax_m3_vl import (
    MiniMaxM3VLForCausalLM,
    MiniMaxM3VLTextConfig,
)

out = sys.argv[1] if len(sys.argv) > 1 else ".cache/minimax-m3-tiny"

# GPT-2 BPE tokenizer (vocab 50257): the ticket's prompt ids (max 9621) and the
# exporter's eviction-fixture text both stay in range; the model is random, so
# the tokenizer identity is irrelevant to the id-only forward-parity proof.
tok = AutoTokenizer.from_pretrained("gpt2")

cfg = MiniMaxM3VLTextConfig(
    vocab_size=50257,
    hidden_size=64,
    intermediate_size=32,            # routed-expert width
    dense_intermediate_size=64,
    shared_intermediate_size=32,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    index_n_heads=2,                 # == num_key_value_heads (one index head per GQA group)
    index_head_dim=32,
    index_block_size=4,              # small so the short prompts span several blocks
    index_topk_blocks=2,
    index_local_blocks=1,
    num_local_experts=4,
    num_experts_per_tok=2,
    rotary_dim=16,
    partial_rotary_factor=0.5,
    swiglu_alpha=1.702,
    swiglu_limit=7.0,
    routed_scaling_factor=2.0,
    rms_norm_eps=1e-6,
    max_position_embeddings=512,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
    # mix dense + sparse attention so both the GQA backbone and the MSA indexer
    # path are exercised; every layer is MoE.
    layer_types=["full_attention", "full_attention", "minimax_m3_sparse", "minimax_m3_sparse"],
    mlp_layer_types=["sparse", "sparse", "sparse", "sparse"],
    tie_word_embeddings=False,
    bos_token_id=tok.bos_token_id,
    eos_token_id=tok.eos_token_id,
)
cfg.use_gemma_norm = True  # M3 uses (1+w) RMSNorm; recorded for completeness

torch.manual_seed(0)
model = MiniMaxM3VLForCausalLM(cfg).to(torch.float32).eval()

# De-trivialize the zero-init RMSNorm weights so (1+w) actually bends the norm —
# otherwise every gain is exactly 1.0 and the gemma-norm path is untested.
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name and p.dim() >= 1:
            p.normal_(0.0, 0.1)

model.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print(f"saved tiny minimax_m3 fixture -> {out}")
print(f"  model_type={cfg.model_type} layers={cfg.num_hidden_layers} layer_types={cfg.layer_types}")
print(f"  params={sum(p.numel() for p in model.parameters()):,}")
