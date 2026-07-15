# Kernel-served verbalizable-direction witness

This directory is the public, scrubbed artifact target for #4437. The sanctioned GCP L4 node currently serves `qwen2.5-0.5b-instruct-q8_0.gguf` through the CUDA backend, but its deployed image predates the bidirectional hidden hook. `meta.json` records that state as `not-yet-live`; it is not a model-effect claim.

After the hook commit is present in the CUDA image, capture matched positive/negative hidden dumps at the same absolute position, derive the normalized diff-of-means `direction.f32`, and run coefficients `-2,-1,0,1,2`. Check in scrubbed per-layer f32 dumps and a JSONL table with verbalized-concept shift and downstream projections. Replace `status` only after those files exist and can be re-read independently.
