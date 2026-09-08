<!-- fak-qwen38-key: running-binary-provenance-observer -->
# feat(binstamp): observe running source and binary provenance

GitHub: #12211. Parent issues: #12096 and #12097.

```routing
lane: qwen38-running-binary-provenance
paths: ["internal/binstamp/binstamp.go", "internal/binstamp/binstamp_test.go"]
expected_steps: 6
```

## Parent context

Child prerequisite of #12096 and #12097. It does not close either parent and
earns no physical performance credit.

## Current state

The raw-decode executor now executes the real selected model/backend and the
fanout physical adapter maps only its observed output.
The adapter intentionally leaves source and binary identity empty, so physical
JSON and human modes fail before writing receipt bytes.

`internal/binstamp.Self` reads a VCS revision for stale-binary diagnostics, but
its fallback commit does not prove an observed `vcs.modified` setting and the
package does not hash the current executable. Command-local code in
`cmd/modelbench/native_profile.go` performs similar capture but is not
importable. There is therefore no strict runner-owned source/binary observer
that #12096 or #12097 can safely reuse.

## Working spine

The Go toolchain's `runtime/debug.ReadBuildInfo` is authoritative for the
revision and modified bit embedded in the process, and `os.Executable` locates
the running image whose opened bytes can be hashed. `internal/binstamp` already
owns running-build observation, so extending it keeps one provenance primitive
without moving command code or introducing a new package.

## Why this is next

The real raw-decode seam is now importable, but its fanout adapter cannot
honestly identify the process that executed it. Source and binary binding is
the smallest independent prerequisite and existing standard-library metadata
supports a fully device-free implementation.

## Core through-line

Add a strict importable observation in `internal/binstamp`:

`running Go build metadata + current executable bytes -> validated revision/tree state/binary digest observation`.

The observer reads both `vcs.revision` and `vcs.modified` from the running
binary's actual build metadata, opens the path returned for the current
executable, hashes the opened bytes, and returns only those observed values. It
rejects missing/malformed metadata, non-regular/empty executable files, and
unstable reads. It exposes no caller-provided revision, dirty flag, or digest.

This leaf deliberately does not attach the partial observation to the fanout
receipt. Source-archive identity, executed device/runtime/fallback identity,
model tokenizer/template/inventory identity, memory, dispatch, transfer, and
tensor-home counters remain unavailable. Therefore physical receipt output
continues to fail closed until the complete runner-owned envelope exists.

## Gold-plating boundary

- No appliance, SSH, service, or hardware access.
- No source archive reconstruction from a working tree.
- No caller-supplied provenance labels or environment-variable relabeling.
- No device, runtime, memory, dispatch, transfer, tensor-home, or hardware
  counter collection in this leaf.
- No physical receipt promotion and no performance claim.

## Done condition

- [ ] An importable observer returns a full source revision, observed clean or
  dirty state, current executable byte count, and SHA-256.
- [ ] Missing or malformed build settings fail closed.
- [ ] Empty, non-regular, unreadable, or changed executable observations fail
  closed without returning partial identity.
- [ ] The production entry point accepts no caller-supplied identity fields.
- [ ] Device-free fixtures prove missing and mismatched observations refuse and
  cannot be promoted by forged expected values.
- [ ] Focused tests, vet, leak audit, and exact-path validation pass.

## Done condition / witness

The source/binary observer is importable, derives every field from the running
process, and returns a zero observation plus error on ambiguity.

## Definition of done

All software criteria above pass and the production observer has no input by
which a caller can supply or relabel source revision, tree state, or binary
digest. The physical fanout adapter remains unpromoted pending the separately
routed complete observation envelope.

## Witness

```text
go test ./internal/binstamp -run 'Test.*ExecutableProvenance' -count=1
go vet ./internal/binstamp
fak validate --mine internal/binstamp/binstamp.go --mine internal/binstamp/binstamp_test.go
```

The witness is device-free and earns no `[HW-WITNESSED]` status.

## Acceptance gate

All focused tests and vet pass, the exact-path validator and public leak audit
are green, and review confirms that the exported production observer accepts no
identity arguments.

## Closure binding

The resolving implementation commit cites this child issue and carries the
`(fak binstamp)` leaf trailer. A documentation-only ticket commit does not close
the implementation issue.

## Likely files

- `internal/binstamp/binstamp.go:20-90` (`Stamp`, `Self`, `stampFrom`)
- `internal/binstamp/binstamp_test.go:1-130`

## Lane

`qwen38-running-binary-provenance`; one package, expected steps: 6.

## Verifiable witness details

- Repro: `go doc ./internal/binstamp.ExecutableProvenance` currently fails because no strict executable-provenance observation exists.
- Exact blocked seam: `internal/binstamp/binstamp.go:20-90` observes only a diagnostic stamp and cannot return current executable bytes or their digest.
- Blast radius: `internal/binstamp` only. Raw decode, compute, CLI, serving, and accelerator lanes remain unchanged.
- Fallback: retain the existing incomplete rawdecode adapter. It remains non-creditable and emits no physical receipt bytes.

## Classification

- Portfolio tier: 2 (serving), prerequisite for tier 1 fanout evidence.
- Centrality: Core
- P1 Context: advanced - closes the highest-value provenance gap with observed process evidence.
- P2 Net value: preserved - no physical or modeled performance number is emitted.
- P3 Adaptation: preserved - reusable existing `binstamp` ownership; no second framework.
- P4 Operations: advanced - all ambiguity fails closed.

## Work unit

leaf

## Expected steps

6

## Work estimate

Estimate: 2 points.

## Overall completion contribution

Contribution: 2/8 points toward the complete runner-owned observation envelope;
zero points toward the hardware witness required by #12096 and #12097.

## Completion standard

production

## Target operating envelope

- ambiguous source or executable observations admitted: = 0 percent

## Witnessed operating envelope

- ambiguous source or executable observations admitted: = 0 percent
