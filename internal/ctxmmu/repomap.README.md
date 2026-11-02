# Explicit selected-file repository map

This optional FAK agent subsystem adapts Aider's definition/reference graph,
weighted PageRank and budgeted symbol selection. It is a bounded **Go AST**
adaptation, not all-language Aider equivalence, a workspace scanner, persistent
index or an authorization mechanism.

## Normal consumer

`fak agent --native --task "inspect Target" --repo-map-file main.go --repo-map-file target.go --repo-map-tokens 1024`

A qualified build is required; this example is not evidence that the installed
controller exposes the flags. The option is off unless explicit files are named.
Only a fresh native one-shot arm with already-armed coding tools and a declared
Read tool is supported. Raw/dual, interactive, workflow, resumed checkpoint and
preexisting-conversation forms are refused rather than silently reinterpreted.
Children do not automatically inherit the option.

After the ordinary first-turn session gate, the caller feeds one explicit Read
at a time through the **existing normal tool dispatcher**. Native route/policy,
path confinement, tool-call budgets, audit/trace, result admission and observed
file versions remain authoritative. A refusal, changed arming, unexpected engine,
transformed/quarantined result, partial/range/truncated result, excessive input
or malformed Go input stops before the first model request. There is no raw
filesystem fallback or hidden retry. No glob, directory traversal, git scan,
private-file discovery, network endpoint or permission change is introduced.
Explicit selection is not an exemption from read policy; native symlink and
workspace boundaries remain in force.

Only these newly introduced preload tool results are compacted in the model
transcript to explicit `fak.repomap.read/1` status, path, returned-content digest,
native file-version and size receipts. Every synthetic assistant call has its
corresponding tool result. Original native results still cross audit/trace and
feed the existing read-version observations. Ordinary model-requested tool
results are unchanged, including a later full Read of a mapped file.

The derived map arrives before the first model request as ordinary, labeled
**untrusted repository data**, never a system instruction, skill, memory grant
or access authority. It renders declaration names/signatures and approximate
cross-file reference importance instead of entire bodies. It uses the existing
prompt assembly, checkpoint and compactor paths without a retention exemption.
One request-local snapshot is frozen; no background refresh or disk cache exists.
Subsequent edits can make it stale: the map is navigation evidence, never an
authority for a mutation. The native version/read-before-write rules still apply.

## Bounds and evidence

- At most 8 distinct canonical relative Go paths, each ≤256 UTF-8 bytes. Absolute,
  traversal, backslash/drive, hidden-component and non-Go paths are refused.
- Already-armed native retained-body limit must be≤1 MiB. Only complete bodies
  ≤64 KiB/file and≤512 KiB total are accepted. Serial dispatch stops at the first
  oversized/refused body, so at most one oversized native result follows the
  accepted prefix. JSON escaping can cost up to 6 bytes per body byte plus bounded
  path/version fields; this is not a 512 KiB JSON transport claim.
- **Native Read hashes the whole regular file** for its identity-bound version.
  Its retained-body limit is not a physical-I/O byte cap. A 5s cooperative preload
  context checks cancellation between native read chunks and ranking work; this
  is not a hard deadline for an uninterruptible filesystem operation.
- Pure parsing accepts at most 32,768 AST nodes/file, 1,024 definitions, 8,192
  identifier references and 8,192 graph edges. Ranking runs exactly 32 bounded
  PageRank iterations (damping 0.85), in canonical deterministic order.
- Explicit map budget 256..4096, default 2048 estimated tokens, uses
  `ceil(UTF-8 bytes/4)`. Complete map including labels/counts/digest is ≤4×budget
  bytes. It is **not** a model-tokenizer guarantee. Synthetic calls and compact
  receipts remain separately visible context; total context planning remains
  governed by the existing assembler.
- Content SHA binds the native returned UTF-8 content; `fv1` is a different
  identity/domain-bound native version, not a plain content digest.
- Rank weights borrow Aider's square-root frequency, mentioned-identifier and
  descriptive-name multipliers, common-definition downweight and unreferenced
  definition self-edges. FAK uses Go AST tags, explicit selected files and a
  bounded task-identifier personalization instead of tree-sitter, automatic repo
  traversal, chat-file exclusions, SQLite caches or NetworkX.
  After distributing PageRank into definition scores, exact names in the bounded
  task mentions receive a final x10 score prior. Edge weighting alone can cancel
  during outgoing normalization and let an unreferenced self-loop mask an
  explicitly named target. This final prior is an intentional FAK adaptation,
  not a claim of unchanged Aider ranking.

Source baseline: current inspected agent/ctxmmu/cmd tree has no producer for a
ranked repo map. This source slice aims to place a bounded cross-file navigation
view before the first model request without sending full selected-file bodies.
Synthetic fixtures check actual native Read mediation, refusal-before-model,
ordinary transcript preservation, deterministic graph ordering and output bound.
Coverage/recall, task success, latency, actual tokenizer cost and installed-client
behavior are **unmeasured**. This is not automatic new-port acceptance or a claim
of end-to-end gain.

Focused qualification commands (UNRUN, for a qualified isolated native build):
`go test ./internal/ctxmmu -run '^TestRepoMapBoundedGraph$' -count=1`
`go test ./internal/agent -run '^TestRepoMapPreloadNativeBoundary$' -count=1`
`go test ./cmd/fak -run '^TestAgentRepoMapOptIn$' -count=1`

## Upstream and changes

Donor: Aider-AI/aider@5dc9490bb35f9729ef2c95d00a19ccd30c26339c,
`aider/repomap.py` Git blob 541bba6ef4a5385b8cf032201ec6e3f3e32a6ea6.
Apache-2.0 license read at `LICENSE.txt`, Git blob
d645695673349e3947e8e5ae42332d0ac3164cd7, retained verbatim in
`REPOMAP-LICENSE.txt`. Attribution: Aider contributors.
The Go implementation and native selected-read integration are FAK adaptations;
the above bounds, permission seam and immutable request-local snapshot are not
claimed as upstream guarantees. No Empryo/BUSL implementation is copied.
