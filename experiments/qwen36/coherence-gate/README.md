# Qwen3.6 long-prompt coherence A/B gate (#4627)

This is the standing, fail-closed gate for changing Q4_K decode from the lean-Q8 activation path
(`FAK_KQ_INT8=0`) to the int8 Q4_K path (`FAK_KQ_INT8=1`). It exists because first-token/argmax
parity from #4624 does not prove that a 512-token completion stays coherent. The gate uses the
same symptom and sampling profile as P1 #4273.

## Contract

For every prompt bucket (`short`, `~1.3k`, and `longer`) and both decode profiles:

- greedy;
- Qwen recommended non-thinking sampling: temperature 0.7, top-p 0.8, top-k 20,
  presence penalty 1.5;

run the exact same prompt and deterministic in-kernel seed against both decode arms. The gate
passes only when:

1. `int8-q4k.trigram_repeat <= q8.trigram_repeat` in **every** bucket/profile pair; and
2. required-section-label presence is recorded identically for both arms as a quality diagnostic.

A missing arm, failed generation, or any positive repetition delta is a hard failure. Required-label
absence remains visible in the artifact but does not by itself fail this decode-regression gate: #4273
already establishes that the Q8 baseline can miss every label, and C4 asks whether int8 gets worse. The emitted `fak-qwen36-coherence-gate/1` JSON records commands, environment,
prompt byte counts, output paths, elapsed time, raw metrics, comparisons, and the final verdict.
Trigram repeat is `repeated trigram windows / all trigram windows`, where a window is repeated
when its normalized word triple has appeared earlier in the same completion.

## Model-free proof

```sh
go run ./experiments/qwen36/coherence-gate -selfcheck
# SELF_CHECK PASS ... red_fixture=REJECT green_fixture=PASS
```

The red fixture deliberately makes the int8 arm more repetitive and proves the gate rejects it;
the green fixture proves equality is allowed (`<=`, not `<`).

## Real-weights run

Create a private manifest next to the captured prompts (prompt text can contain private report
content and must not be committed):

```json
{
  "schema": "fak-qwen36-coherence-manifest/1",
  "model": "/models/Qwen_Qwen3.6-27B-Q4_K_M.gguf",
  "max_tokens": 512,
  "prompts": [
    {"bucket":"short", "path":"short.txt", "required_labels":["Executive summary","Risks","Next steps"]},
    {"bucket":"~1.3k", "path":"q36safe9.txt", "required_labels":["Executive summary","Risks","Next steps"]},
    {"bucket":"longer", "path":"longer.txt", "required_labels":["Executive summary","Risks","Next steps"]}
  ]
}
```

Then run on a sanctioned node holding the real 27B GGUF:

```sh
go build -o /tmp/fak-coherence ./cmd/fak
go run ./experiments/qwen36/coherence-gate \
  -fak /tmp/fak-coherence \
  -manifest /private/qwen36/coherence-manifest.json \
  -out /tmp/qwen36-coherence-4627 \
  -seed 0
```

Preserve `coherence-gate.json` and the six arm pairs as the observed artifact. Do not flip the
int8 default if the process exits 1. The exact rebuilt 5,371-byte `q36safe9` prompt from #4273 is
the authoritative `~1.3k` input; do not replace it with synthetic prose when closing #4627.

The first-token parity prerequisite remains witnessed separately by #4624 (token id 248068).
This gate does not close #4273; it only proves that the decode-path change does not amplify it.
