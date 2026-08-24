# Caveman pairwise protocol v2 witnesses

`diagnosis.json` is a deterministic extraction from the immutable protocol-v1 receipt at `../caveman-pairwise-judge/receipt.json` (SHA-256 `7b7f3801daf3f4be5e94aa0af50666eec9c25ed1cd265a7bb178db39aefbef87`). It accounts for all 14 order-unstable pairs without assigning a preferred outcome: one empty provider response is `output_truncation`; the remaining 13 are winner-versus-tie disagreements at the tie boundary.

No protocol-v2 application receipt is committed. A fresh application is permitted only after the separately reported held-out calibration and same-order repeatability gates pass. Until such a run exists and all semantic, safety, quality, and provenance gates pass, token eligibility remains suppressed.
