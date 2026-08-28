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
and torch to build weights. The dependency-free `--describe` mode prints the exact
configuration contract before an export is attempted. Usage:

    python3 internal/model/make_qwen35_tiny.py .cache/qwen35-tiny
    python3 internal/model/make_qwen35_tiny.py .cache/ornith-tiny --ornith
    python3 internal/model/export_oracle.py --online \
      --model .cache/qwen35-tiny --out internal/model/.cache/oracle-qwen35 \
      --prompt-ids-json '[[785,6722,315,9621,374],[16,11,220,17,11,220,18,11,220,19,11],[750,912,2877,11,293,982,262,470]]'

then `go test ./internal/model/ -run TestOptionalQwen35 -count=1`.
"""
import argparse
import json


def ornith_config_spec(vocab_size=50257, bos_token_id=50256, eos_token_id=50256):
    """Return the dependency-free spec and exact kwargs used to build Ornith."""
    config_kwargs = {
        "vocab_size": vocab_size,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "full_attention_interval": 4,
        "max_position_embeddings": 512,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000.0,
            "rope_type": "default",
        },
        "attn_output_gate": True,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
    }
    return {
        "schema": "fak.model.ornith-tiny-builder/1",
        "scope": "dense-9b-text-oracle-builder-not-parity-evidence",
        # Published Ornith is wrapped by Qwen3_5ForConditionalGeneration, whose
        # outer config is qwen3_5. This builder intentionally instantiates only
        # the text decoder, whose Qwen3_5TextConfig identity is qwen3_5_text.
        "published_wrapper_model_type": "qwen3_5",
        "builder_config_model_type": "qwen3_5_text",
        "text_only": True,
        "rotary_dim": int(config_kwargs["head_dim"] *
                          config_kwargs["rope_parameters"]["partial_rotary_factor"]),
        "config_kwargs": config_kwargs,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", default=".cache/qwen35-tiny")
    parser.add_argument("--ornith", action="store_true",
                        help="pin the dense Ornith-9B text axes required by issue #1031")
    parser.add_argument("--describe", action="store_true",
                        help="print the selected builder contract without importing transformers/torch")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.describe:
        if not args.ornith:
            raise SystemExit("--describe currently requires --ornith")
        print(json.dumps(ornith_config_spec(), sort_keys=True))
        return

    # Keep the heavyweight optional dependencies below the describe seam so the
    # regeneration hint can be audited on a clean control host before installing them.
    import torch
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    # GPT-2 BPE tokenizer (vocab 50257): the exporter feeds token IDs, so tokenizer
    # identity is irrelevant; its vocabulary simply contains the fixed oracle IDs.
    tok = AutoTokenizer.from_pretrained("gpt2")
    if args.ornith:
        spec = ornith_config_spec(tok.vocab_size, tok.bos_token_id, tok.eos_token_id)
        cfg = Qwen3_5TextConfig(**spec["config_kwargs"])
        if cfg.model_type != spec["builder_config_model_type"]:
            raise RuntimeError(
                f"Qwen3_5TextConfig model_type={cfg.model_type!r}, "
                f"want {spec['builder_config_model_type']!r} from Ornith builder spec"
            )
    else:
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
            linear_conv_kernel_dim=3,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            full_attention_interval=4,
            max_position_embeddings=512,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
            },
            bos_token_id=tok.bos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(cfg).to(torch.float32).eval()

    # De-trivialize zero-init RMSNorm weights so the (1+w) gain on ordinary norms
    # bends the norm and the oracle exercises that family-specific distinction.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.dim() >= 1:
                p.normal_(0.0, 0.1)

    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"saved tiny qwen3_5 fixture -> {args.out}")
    print(f"  model_type={cfg.model_type} layers={cfg.num_hidden_layers} layer_types={cfg.layer_types}")
    if args.ornith:
        print("  contract=" + json.dumps(spec, sort_keys=True))
    print(f"  params={sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
