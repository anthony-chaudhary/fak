---
title: "LLMLingua-2 compressor adapter"
description: "The lingua Compressor is a first-class, off-by-default adapter selected with --via lingua or FAKCOMPRESSOR=lingua. Configure its model service with:"
---
# LLMLingua-2 compressor adapter

The `lingua` Compressor is a first-class, off-by-default adapter selected with `--via lingua` or `FAK_COMPRESSOR=lingua`. Configure its model service with:

```text
FAK_LINGUA_URL=http://127.0.0.1:PORT
```

fak sends `POST /v1/compress` with `{"text":"..."}` and expects a JSON response containing `text`, plus optional `model`, `original_tokens`, and `compressed_tokens` provenance. The adapter has no root Go dependency and does not bundle model weights or claim that a mock service is LLMLingua-2.

The transform is lossy and never becomes the default. In the live Admit path, the pre-transform bytes are persisted through the shared CAS before the compressed model view is exposed. `fak headroom compare --via none,native,lingua` provides the common-corpus local arm; overall completion remains false until #6064's live quality, provider-token, TTFT, regrowth, and total-cost metrics are attached.
