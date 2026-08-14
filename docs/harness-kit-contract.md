# Harness-kit public builder contract

Status: **v1alpha1**, issue #6786. Import `github.com/anthony-chaudhary/fak/pkg/harnesskit`; anything below `github.com/anthony-chaudhary/fak/internal/` is private and Go itself rejects that import from an external module.

## Value and centrality

- **For:** external builders differentiating an agent product while reusing the fak kernel.
- **Problem:** useful internal seams did not form a supported public contract.
- **Today:** builders could either copy private types or wait for SDK scaffolds, risking accidental ABI commitments.
- **Better because:** a small importable vocabulary and a versioned machine contract now precede those scaffolds.
- **Witness:** `internal/architest/harnesskit_external_test.go` creates a clean outside module, compiles a real product against `pkg/harnesskit`, and proves the same module cannot import `internal/harnesskit`.

This is **Core** work: it freezes the external/kernel boundary before public SDK work. P1-P4 all apply:

| Check | Contract answer |
|---|---|
| P1 managed context | Profiles and context/instruction planes declare portable setup without exposing context-MMU internals. |
| P2 net-true efficiency | One vocabulary is reused by products and tooling; no performance gain is claimed until a runtime witness measures it. |
| P3 bounded adaptation | Extensions are plane-scoped, provenance-pinned, compatibility-versioned, and capability-declaring. |
| P4 integrated operations | Lifecycle, cancellation, streams, backpressure, stable errors, ownership, and transport declarations are explicit. |

## Normative vocabulary

`Capability` names requested authority. `Extension` attaches a provenance-pinned implementation to one of `tools`, `models`, `context`, `instructions`, `transports`, or `events`. `Profile` groups extensions and requested capabilities. `ProductSpec` combines a profile and transport declarations; `Builder.Build` validates and freezes it. `Factory` is the host lifecycle seam, `Services` is its deliberately narrow capability-filtered reachability, and `Stream[T]` is the ordered streaming seam.

`harnesskit.PublicContract()` and `ContractJSON()` are normative machine-readable forms. The schema records all extension planes, lifecycle states, stable error codes, security reachability, ownership, and compatibility rules. Within `v1alpha1`, additions are allowed; removing a symbol, plane, state, code, or changing its meaning requires a new contract version.

## Operating semantics

- **Cancellation:** every blocking public operation accepts `context.Context`; cancellation returns an error that preserves the cause. `Drain` is bounded by its context.
- **Streaming and backpressure:** `Send`/`Recv` are ordered and block until accepted, completed, or canceled. `io.EOF` is clean completion. An adapter unable to wait may return `CodeBackpressure`.
- **Errors:** callers branch on `Error.Code` and use `errors.Is`/`errors.As` for causes. Error text is diagnostic, not an API.
- **Security reachability:** registration makes an extension reachable, never authoritative. The host supplies only `Services`; every `Invoke` remains subject to the effective session/tenant capability floor. This contract does not expose gateway, policy, engine, or adjudicator internals.
- **Resource ownership:** the host owns and closes a `Runtime` returned by `Factory.Start`; callers retain builder and service inputs. `Runtime.Close` is idempotent; `Drain` rejects new work and waits for accepted work.

## Existing issue routes (not duplicated)

- **#3265** owns executable custom-tool/MCP registration and the proof that calls cannot widen session/tenant authority. Harness-kit defines only the vocabulary and `Services.Invoke` security contract.
- **#6101** shipped pinned tool-plugin profiles and layered preferences in `fak.toml`; `Profile`/`Provenance` describe portable builder selection and do not reimplement configuration precedence.
- **#6672** owns canonical-instruction projection into host-specific envelopes; the `instructions` plane identifies the extension point without defining those adapters.
