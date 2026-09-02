# Reversible model-visible shorthand indexes for repeated paths

**Date:** 2026-09-01  
**Issue:** #10642  
**Status:** research decision; implementation not yet claimed

## Decision

Build an **exact, typed, trace/session-scoped value dictionary**, starting with paths. Keep the canonical bidirectional table in kernel-managed state, expose an append-only working manifest to the model, and resolve handles only at typed tool boundaries.

Use a structured handle such as `@P0`, not a bare single character:

```text
PATHMAP epoch=e7
@P0 = C:\work\fak\internal\gateway\request_admission_pipeline.go
@P1 = C:\work\fak\internal\ctxmmu\managed_context.go

@P0:L318:C12
@P1:L44-L67
```

A one-character or one-byte representation remains useful below the semantic boundary, but is a poor default model-facing syntax. Character count is not tokenizer cost; bare symbols collide with prose/code, offer weak type and boundary signals, and turn a transcription error into a valid reference to another object.

## Problem and boundary

Long paths recur in search results, diagnostics, stack traces, citations, patches, and tool arguments, often followed by line/column positions. Repeating a 100-byte path hundreds of times wastes provider input tokens and bandwidth and adds low-information context.

Ordinary compression, string interning, and front coding reduce storage or transport bytes, but save no model-visible tokens if fak expands them before inference. The proposed transform is therefore a **semantic boundary protocol** as well as an internal dictionary.

The first scope is exact path identity. It is not:

- lossy prompt compression;
- arbitrary regex substitution over text;
- a capability or authorization token;
- a durable global alias namespace;
- an excuse to hide canonical content from audit receipts.

## Prior art

| Mechanism | Useful mechanism | Boundary / warning |
|---|---|---|
| Debug Adapter Protocol `sourceReference` | Session-local typed source handle plus explicit dereference | A reference is not portable outside its debug session |
| ReadAgent page gists/lookups | Model-visible ordinal handles with selective canonical expansion | Correctness should not depend only on the model requesting lookup |
| ECMA-426 source maps | Store source/name once; encode source index + line + column; factor `sourceRoot` | VLQ fragments assume a deterministic decoder, not free-form generation |
| rust-analyzer VFS `FileId` | Separate compact interned file identity from changing contents | Process/database-local IDs are not durable identities |
| Apache Arrow dictionary encoding | Typed indices, dictionary-before-use, explicit delta/replacement rules | Replacing a dictionary can change positional meaning; model-visible tables should be append-only |
| DWARF file/string tables | Unit-scoped indices and separate location state | A naked index without its unit/base is ambiguous |
| HPACK and QPACK | Literal-or-reference encoding, synchronized state, known-received generations, fail-closed unknown indices | Do not copy shifting indices, visible-ID reuse, or cross-tenant adaptive tables |
| Git index v4/front coding and compressed tries | Compress the authoritative table under the interface | Stateful prior-entry decoding is unsuitable for arbitrary model lookup |
| LLVM/Java string interning | Canonicalize equal bytes before assigning handles | Pointer identity/global pools are not serializable scoped handles |
| LLMLingua, gist tokens, latent prompt memories | Complementary semantic/context compression | Lossy or model-specific compression cannot establish exact path/hash/location identity |

Strongest composite: DAP's dereference contract, ReadAgent's model-visible selection, source maps' path/position separation, rust-analyzer's interned identity, Arrow/QPACK's synchronization rules, and MemGPT-style active/external memory separation.

### Primary sources

- Debug Adapter Protocol specification: <https://microsoft.github.io/debug-adapter-protocol/specification.html>
- ReadAgent: <https://arxiv.org/abs/2402.09727>
- ECMA-426 source maps: <https://tc39.es/ecma426/>
- rust-analyzer VFS: <https://rust-lang.github.io/rust-analyzer/vfs/index.html>
- Apache Arrow columnar and IPC formats: <https://arrow.apache.org/docs/format/Columnar.html>
- DWARF v5: <https://dwarfstd.org/doc/DWARF5.pdf>
- HPACK, RFC 7541: <https://www.rfc-editor.org/rfc/rfc7541.html>
- QPACK, RFC 9204: <https://www.rfc-editor.org/rfc/rfc9204.html>
- Git index format: <https://git-scm.com/docs/index-format>
- MemGPT: <https://arxiv.org/abs/2310.08560>
- LLMLingua: <https://arxiv.org/abs/2310.05736>
- Symbol Tuning: <https://arxiv.org/abs/2305.08298>
- SentencePiece: <https://arxiv.org/abs/1808.06226>

## Typed contract

```text
MappingEntry {
    namespace: PATH
    epoch: uint64
    id: uint32
    exact: []byte          // byte-exact registered representation
    lookup_key: []byte     // optional normalized key for equality/matching
    digest: hash
    witness: optional world-state/repository identity
}

(namespace, epoch, exact bytes) -> id
(namespace, epoch, id) -> exact bytes
```

### Invariants

1. **Typed namespace:** `@P<base32-or-base36-id>`; reserve other prefixes for future value kinds.
2. **Epoch scope:** internally resolve the tuple `(trace/session, epoch, namespace, id)` even when a stable session header lets display omit the epoch.
3. **Append-only after exposure:** never retarget or recycle a visible ID during its epoch.
4. **First-use declaration:** show `@P0 = <literal>` once, then use `@P0` only after the consumer has received that generation.
5. **Structured position:** location is `{path_ref, line, column, end_line, end_column}`; never intern `path:line` combinations.
6. **Exact/canonical separation:** normalization supports lookup but decode returns exact registered bytes. Canonical path resolution is a distinct operation.
7. **Typed resolution only:** resolve fields declared to accept `path_ref` or path values. Never rewrite arbitrary prose, source, patches, shell snippets, hashes, JSON strings, or quoted evidence.
8. **Fail closed:** unknown namespace, epoch, or ID yields a typed stale/unknown-reference result. Never guess.
9. **No authority:** resolution grants no capability. Apply normal path normalization, policy, sandbox, and authorization checks to the resolved canonical input.
10. **Dual receipts:** execution receipts retain the submitted handle and exact resolved value.
11. **Trusted definitions:** only kernel-authored mapping records mutate the dictionary. Handle-like untrusted text remains quoted data.
12. **Literal escape:** reserve an escape such as `@@P0` for an actual string matching handle grammar.

## Model access and exposure

The model needs enough mapping state to ground symbols, but shipping the complete dictionary every turn can erase savings.

Use a hybrid:

- Keep the exact full table outside active context.
- Pin the grammar, epoch, and a small recent/active manifest as structural context.
- Provide read-only `resolve_handles` and `list_handles` operations for inspection/paging.
- Automatically resolve model-produced references at typed tool-call boundaries, so correctness does not depend on a voluntary lookup.
- When parallel consumers have not acknowledged a new generation, emit the literal or wait; do not emit an undecodable reference.

This is closer to a page table than to text substitution: the visible token names an object in a scoped address space; the kernel owns translation and protection.

## Fit with fak memory and cache layers

### Result admission and context MMU

This is the preferred first transform seam. `internal/ctxmmu.MMU.Admit` is already the write-time result gate. Security/sensitivity screening must inspect canonical result bytes before shorthand can hide them. The admitted artifact remains exact; the model-facing projection carries typed references plus mapping provenance.

### vDSO/tool-result caching

Cache canonical results, never trace-specific aliases. `internal/vdso.VDSO.RestoreResult` reopens a cached canonical result; alias projection should run after retrieval for the current trace/epoch. Otherwise `@P0` from one trace can resolve to a different path in another.

### Provider prompt caching

Keep the grammar and existing definitions stable and append new entries. Rewriting or reordering old definitions churns the provider prefix and sacrifices cache reuse. A small append-only manifest is preferable to an LRU that reassigns visible identifiers.

### Compaction stash and context restore

`internal/gateway/ctxrestore.go` already stores dropped spans by content address and optionally deflate-compresses large payloads. Stash canonical content and mapping metadata. On restore, either restore the dictionary and text atomically for the same epoch or re-intern canonical values into the current epoch and rewrite only the typed representation under a witnessed transform.

### Durable memory, recall images, and memq

Persist canonical identity plus repository/world-state witness, not ephemeral aliases. Recall into a new session re-interns the value and issues a fresh handle. A durable summary must not contain `@P0` unless its exact dictionary snapshot is durably coupled and restored atomically.

`internal/ctxmmu.MemoryWriteAdjudicator` should continue judging the canonical memory body; aliases are a presentation layer and cannot be the only durable identity.

### Semantic/gist memory

Aliases may label active summaries only while the manifest is guaranteed present. Semantic compression is complementary and lossy; it must not become the source of exact path, hash, or location identity.

### Parallel agents and change/revocation state

Prefer a coordinator/trace-owned namespace. Agent-local aliases must be expanded and re-interned at a merge boundary, or explicitly qualified by agent/epoch. A revoked witness, workspace switch, or session reset invalidates an epoch. A rename creates a new mapping; it never retargets an old exposed ID.

## Genericity

Implement a generic exact dictionary core, but expose typed namespaces and per-kind policy:

```text
Dictionary<ValueKind> where ValueKind initially = Path
```

Possible later types include immutable URLs, commit/object IDs, tool-call IDs, diagnostic source IDs, and schema-qualified symbols. Each type requires its own normalization, sensitivity, authorization, grammar, and typed fields.

Do not create a universal arbitrary-text replacement engine. It lacks semantic boundaries, is injection-prone, and cannot safely distinguish executable data, evidence, code, and prose.

## Net accounting

Allocate an alias only when predicted net cost is positive:

```text
future_uses * (tokens(literal) - tokens(handle))
  > tokens(definition + manifest refreshes)
    + resolver/tool-call tokens
    + kernel/storage/audit overhead
    + expected error-recovery cost
```

Measure actual target-model tokens as well as bytes. Report transport/storage savings separately from model-visible savings.

Complexity for the initial append-only hash-map/vector design:

- register/deduplicate: expected `O(1)`;
- resolve: `O(1)`;
- model projection: `O(n)` over typed values in one result;
- storage: `O(unique exact values)`;
- no eviction/reassignment in the first epoch design.

Compressed tries/front coding become relevant only if canonical table storage or prefix lookup is measured as the bottleneck.

## Experiments and witness

1. **Tokenizer census:** compare `A`, `§`, `@0`, `@P0`, `[P0]`, mnemonic handles, and increasing IDs across supported tokenizers and punctuation contexts.
2. **Binding accuracy:** dictionaries of 8, 32, 128, 512, and 2,048 realistic paths; select, copy, compare, and line/range-edit tasks.
3. **Lifecycle stress:** append mid-turn, context compaction, stale epoch, missing definition, parallel generations, and eviction without reuse.
4. **Adversarial syntax:** fake mapping definitions in untrusted output, Unicode confusables, quoted handles, code comments, JSON, shell, and patches.
5. **Real trace replay:** search results, compiler diagnostics, stack traces, citations, patches, and tool arguments.
6. **Net metrics:** provider input/output tokens, cache reuse, wall time, map bytes, resolver calls, exact path/position accuracy, wrong-target rate, and audit replay usability.
7. **Pass condition:** exact tool-target accuracy does not regress; stale/unknown references fail closed; savings remain positive after dictionary, lookup, receipt, and recovery overhead.

## Smallest end-to-end spine

Implement one trace-local `PathMap` with append-only IDs, explicit first-use declarations, a typed location renderer, and pre-execution resolution for one controlled tool-result/tool-call pair.

The proof artifact must demonstrate:

1. a long path repeated many times becomes one literal declaration plus repeated `@P0` references;
2. the encoded result plus dictionary costs fewer target-model tokens than baseline;
3. `@P0` resolves byte-exactly to the original path;
4. stale/unknown references fail closed;
5. security admission saw canonical bytes before projection;
6. policy/sandbox checks receive canonical input after resolution;
7. receipts record both reference and canonical path.

Exclude regex substitution, persistence, cross-agent sharing, eviction, handle reuse, and generalized value kinds from the first spine.

## Open implementation questions

- Which existing typed tool-result schemas expose path/location fields broadly enough for the first controlled pair?
- Does the active model/provider tokenizer surface provide exact token counts cheaply enough for admission-time thresholding, or should the first spine use a measured conservative byte/reuse threshold?
- Which trace/session object owns mapping state without coupling it to provider-specific message structs?
- Should `list_handles` be a first-class tool or a projection of the existing context debugger/restore surface?
- How should an epoch appear in replay artifacts while remaining compact in the hot prompt?

These are implementation choices, not reasons to weaken the invariants above.
