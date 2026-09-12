# Request feature activation

Chat completions, Messages, and fak syscall responses expose request-local
observations through three comma-separated, closed-identifier lists:

| Field | Meaning |
| --- | --- |
| `X-Fak-Features-Enabled` | Active or standby configuration captured at ingress. |
| `X-Fak-Features-Used` | Accepted execution observed before the first final response header commit. |
| `X-Fak-Features-Used-Final` | Declared HTTP trailer containing the final observed set after request work finishes. |

Enabled does not imply used. Initial headers are immutable after commit. A
streaming response can therefore have an empty initial Used header and report
`compact_history` in its final trailer. Read the body to completion before
inspecting trailers. Intermediaries that discard trailers also discard this late
evidence; the initial header cannot reconstruct it. Streaming bytes are forwarded
without buffering. An incomplete worker suppresses the final-use trailer value.

Native typed history compaction records use only when an actual cut is consumed
by successful native generation. Prompt-only `EncodePrompt`, disabled or idle
compaction, refused cuts, and failed generation do not publish native use. The
observer preserves only a bounded summary of the immediate typed compaction
input and output. Its JSON byte counts and hashes describe serialized typed
messages; its token counts are explicitly estimates. Shared whole-message
regions do not certify a model KV cache anchor. Proof collection failure does
not erase the fact that an actual cut was used.

Accepted real vDSO results, result elision, and manifest route selection also
record use at their execution seams. Repeated uses of one identifier collapse to
one request entry. Access logs expose `feature_used`, and
`fak_gateway_feature_used_requests_total{feature}` counts requests using each
feature. These observations provide neither authorization nor measured savings.

Native software coverage and hardware qualification are separate. Issue
[#12835](https://github.com/anthony-chaudhary/fak/issues/12835) tracks the remaining
real Qwen3.8 active/idle execution witness; source implementation alone does not
establish that hardware evidence.
