---
title: "Qwen3.8 native-Metal startup predicate audit"
description: "Reference documentation for Qwen3.8 native-Metal startup predicate audit, preserving the page's implementation details, evidence, and operating context."
---

# Qwen3.8 native-Metal startup predicate audit

Issue [#8964](https://github.com/anthony-chaudhary/fak/issues/8964) asked for a
first-parent bisect between `8145dc0bea` and `2a7cbe0c5d`. The requested
boundary does not exist under the accepted launch recipe.

On the same Apple M3 Pro, 36 GiB, 18-GPU-core machine and the exact
17,106,775,008-byte Qwen3.8 Q4_K_M artifact, both endpoints reached
`/v1/models` after the same 64,915,847,712-byte file-cache displacement:

- accepted `8145dc0beac8396db08d75dee1a969faf0e80bf9`: ready in 31 seconds;
- alleged bad `2a7cbe0c5d1a909df8f1353e6477bb7344734856`: ready in 28 seconds.

Both runs used fak's native Metal path and the documented resident-Q4_K
recipe:

```console
FAK_METAL_STREAM_Q4K=1 FAK_Q4K_FREE_CPU=1 FAK_Q4K=1 fak serve \
  --addr 127.0.0.1:18964 --gguf <exact-gguf> --model qwen38:27b \
  --metal --context-budget-tokens 4096
```

No llama.cpp process or backend participated. The listener identified
`engine=inkernel`; the launch printed the Apple-Silicon Metal GPU route; no
CPU, model, backend, or quant fallback marker appeared.

## Why the reported command cannot be bisected

The command frozen in #8964 omitted `FAK_Q4K=1`. That omission selects the
lean Q8 staging arm rather than the accepted resident-Q4_K arm. It fails at
the supposed good revision too: `8145dc0bea` exited 137 before readiness after
91 seconds with 22,863,968 KiB sampled RSS. At the issue-era endpoint,
`2a7cbe0c5d` missed readiness for 180 seconds once and exited 137 before
readiness under a repeated pressure run. A predicate whose good endpoint is
bad cannot identify a first-bad commit.

The bounded invariant is load-arm selection in
`cmd/fak/serve_load_helpers.go`: `FAK_Q4K` selects the resident Q4_K loader;
`FAK_Q4K_FREE_CPU` only controls whether CPU backing is released after a
Metal upload. It does not select that loader. There is therefore no changed
source-path boundary to hand to #8655 from this interval.

Two later-base observations at `2fbb2961fa530b6567fed18736f5bc576efeeaea`
also reached readiness under the documented recipe (146 and 131 seconds).
Those observations are slower than the accepted cold bound but contradict the
claimed alive-and-unbound three-minute startup failure.

## Read back

```console
go test ./docs/_witnesses/issue-CHILD-qwen38-startup-bisect \
  -run '^TestStartupBisectReadback$' -count=1 -v
```

`bisect.json` is the machine-readable disposition. The scrubbed logs contain
no model path, username, host name, serial number, credential, or private
control transcript.
