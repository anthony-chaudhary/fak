#!/usr/bin/env python
"""Build a tiny random OLMo 2 fixture for the fak per-family HF oracle (#474).

OLMo 2 (allenai/OLMo-2-*) is a POST-norm decoder: unlike the Llama PreNorm
backbone, the RMSNorm runs AFTER each sub-layer on the raw residual
(x = x + norm(attn(x)); x = x + norm(mlp(x))), and it always carries
qk-norms applied over the WHOLE q/k projection (not per-head). fak's loader
derives both axes from the family — BlockTopology=PostNorm (weights.go) and
Config.QKNorm=true (full-projection branch in arch.go applyQKNormCfg) — but
until now the OLMo 2 row was config-derived + topology-tested only, with no
numeric HF oracle. This builds a small CPU-instantiable random OLMo 2 decoder
so the row can move from synthetic-witnessed to witnessed.

The public allenai checkpoints are 1B+; a tiny random one is built here (à la
yujiepan/glm-5-tiny-random) so the oracle reproduces on a plain CPU box with
no GPU/artifact node. Requires torch + transformers (Olmo2ForCausalLM, shipped
since transformers 4.47). Usage:

    python internal/model/make_olmo2_tiny.py .cache/olmo2-tiny
    python internal/model/export_oracle.py --online \
      --model .cache/olmo2-tiny --out internal/model/.cache/oracle-olmo2 \
      --prompt-ids-json '[[785,6722,315,9621,374],[16,11,220,17,11,220,18,11,220,19,11],[750,912,2877,11,293,982,262,470]]'

then `go test ./internal/model/ -run TestOptionalOLMo2Oracle -count=1`.
"""
import sys

import torch
from transformers import AutoTokenizer, Olmo2Config, Olmo2ForCausalLM

out = sys.argv[1] if len(sys.argv) > 1 else ".cache/olmo2-tiny"

# GPT-2 BPE tokenizer (vocab 50257): the prompt ids (max 9621) stay in range; the
# model is random so the tokenizer identity is irrelevant to the id-only forward.
tok = AutoTokenizer.from_pretrained("gpt2")

cfg = Olmo2Config(
    vocab_size=50257,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,          # GQA, so the kv projection (and k_norm) is narrower
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    max_position_embeddings=512,
    tie_word_embeddings=False,
    bos_token_id=tok.bos_token_id,
    eos_token_id=tok.eos_token_id,
)

torch.manual_seed(0)
model = Olmo2ForCausalLM(cfg).to(torch.float32).eval()

# De-trivialize the RMSNorm gains so the POST-norm placement and the
# full-projection qk-norm actually bend the output — otherwise every gain is
# 1.0 and a missing/misplaced norm would be indistinguishable from a no-op.
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name and p.dim() >= 1:
            p.normal_(0.0, 0.1)

model.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print(f"saved tiny olmo2 fixture -> {out}")
print(f"  model_type={cfg.model_type} layers={cfg.num_hidden_layers} "
      f"heads={cfg.num_attention_heads} kv={cfg.num_key_value_heads}")
print(f"  params={sum(p.numel() for p in model.parameters()):,}")
