# Versioned Codex and Claude host components — 2026-08-20

Issue: [#8212](https://github.com/anthony-chaudhary/fak/issues/8212)  
Parent: [#6777](https://github.com/anthony-chaudhary/fak/issues/6777)  
Companions: [#6805](https://github.com/anthony-chaudhary/fak/issues/6805), [#8200](https://github.com/anthony-chaudhary/fak/issues/8200), [#8226](https://github.com/anthony-chaudhary/fak/issues/8226), [#8227](https://github.com/anthony-chaudhary/fak/issues/8227), [#8228](https://github.com/anthony-chaudhary/fak/issues/8228), [#8229](https://github.com/anthony-chaudhary/fak/issues/8229)

## Verdict

FAK had all three necessary subsystems, but they stopped one seam short of dogfooding each other:
`internal/harnessprofile` described first-party Codex and Claude integrations,
`internal/harnessinit` generated an external product, and `internal/harnessresolve` produced a
versioned dependency lock. The shipped spine joins them. `fak harness init --host codex|claude`
now projects the same built-in descriptor used by guard into host, wire, and repoint components,
then drives the resolved lock through the generated product's real selfcheck.

The lifecycle rule is deliberately asymmetric: fak owns and versions its adapter contract;
upstream CLI releases are observed separately. An exact Codex or Claude binary pin would turn a
fast-moving compatible release into needless lock churn, while an unversioned adapter would hide
semantic drift. Each built-in therefore has a fak adapter version plus a digest of its runtime
descriptor. A snapshot test requires a deliberate version-and-snapshot decision when that digest
changes.

## Value frame and problem checks

- **Centrality:** Core. First-party hosts now exercise the same composition path external harness builders are expected to trust.
- **For:** maintainers and builders creating a fak product around Codex or Claude Code.
- **Problem:** host identity and repoint behavior were data-driven in guard but absent from generated product locks.
- **Today:** maintain the first-party profile and composition product independently, with no end-to-end evidence that they agree.
- **Better because:** one descriptor now generates a deterministic dependency graph and a checkable launch receipt.
- **Witness:** focused tests initialize both hosts in clean temporary modules, run `go run ./cmd/product --selfcheck`, and verify host, wire, and repoint identities in the receipt.

P1 managed context stays bounded to the selected host graph. P2 removes duplicate adapter
description but makes no unmeasured speed or token claim. P3 fails on unsupported hosts and on an
unacknowledged descriptor digest change. P4 connects initialization, dependency resolution,
lock verification, and launch observability.

## Dated source ledger

Observed at `2026-08-20T09:10:18-07:00` on Windows, with local `codex-cli 0.148.0` and Claude
Code `2.1.235`. The local version difference is evidence for separating observed upstream state
from fak's authored adapter version; it is not evidence that Claude `2.1.235` is incompatible.

| Source surface | State and source event | Immutable or versioned anchor | What changed in the decision | Refresh trigger |
|---|---|---|---|---|
| OpenAI Codex repository: implementation, config schema, tests, history | shipped `main`; commit event `2026-08-20T14:55:08Z` | [`9bf673718a4605b49e47d00762121d372af95439`](https://github.com/openai/codex/tree/9bf673718a4605b49e47d00762121d372af95439) | Confirmed a fast-moving host with structured config and hook layers; no source was copied. | Codex release or config/hook schema change |
| OpenAI Codex release | released `rust-v0.148.0`; published `2026-08-18T22:26:03Z` | [`rust-v0.148.0`](https://github.com/openai/codex/releases/tag/rust-v0.148.0) | Supplied the observed upstream version, separate from adapter `1.0.0`. | Next Codex release |
| OpenAI Codex configuration docs | shipped docs observed 2026-08-20 | [config reference](https://developers.openai.com/codex/config-reference) | User, trusted-project, and named profile layers support projecting a resolved fak product rather than treating one config file as the harness. | Documentation revision or profile precedence change |
| Codex issue/PR history | open and merged history observed 2026-08-20 | [profile override issue #33995](https://github.com/openai/codex/issues/33995), which links the file-profile PR lineage | A same-key profile override regression shows that declared layering can drift at runtime; a future witness must inspect effective behavior, not only file shape. | Issue resolution or profile implementation change |
| Claude Code repository and release feed | shipped `main`; commit event `2026-08-20T00:54:35Z` | [`770933ea1ad2fa7b858191e397a65e6644771c64`](https://github.com/anthropics/claude-code/tree/770933ea1ad2fa7b858191e397a65e6644771c64) | Confirmed published plugin examples/release history but not an auditable full CLI implementation. | Claude Code release or public-repo scope change |
| Claude Code release | released `v2.1.237`; published `2026-08-20T00:54:41Z` | [`v2.1.237`](https://github.com/anthropics/claude-code/releases/tag/v2.1.237) | Demonstrated upstream cadence and the two-release local lag without implying incompatibility. | Next Claude Code release |
| Claude Code settings, hooks, and plugins docs | shipped docs observed 2026-08-20 | [settings](https://code.claude.com/docs/en/settings), [hooks](https://code.claude.com/docs/en/hooks-guide), [plugins](https://code.claude.com/docs/en/plugins) | Scoped precedence, hot-reloaded settings, plugin versions, and merged hooks support versioning the projection contract while verifying effective composition separately. | Settings precedence, hook schema, or plugin manifest revision |
| Claude Code issue history | open reports observed 2026-08-20 | [hook loading #11544](https://github.com/anthropics/claude-code/issues/11544), [subdirectory scope #36793](https://github.com/anthropics/claude-code/issues/36793) | Reinforced that file presence is weaker than an effective-runtime witness. | Issue resolution or hook discovery change |

No relevant public RFC, discussion, or roadmap added a stronger version/dependency contract than
the released docs and issue/PR history. Claude Code's published repository does not expose the
full executable implementation, so internal implementation tests and blame are unavailable for
this comparison. Those are recorded omissions, not negative evidence.

## License and provenance disposition

- Codex at the pinned revision is Apache-2.0: **INSPIRE-ONLY** here because the useful input is its documented layering behavior and no upstream implementation is needed.
- GitHub exposed no detected license and no root license for Claude Code at the pinned revision: **INSPIRE-ONLY**. No expressive code, schema, test, comment, or asset was copied.
- The implementation is independently written against existing fak types. Its only external influence is the lifecycle principle: host configuration is layered and fast-moving, so keep adapter compatibility versioning distinct from observed executable versioning.

## Self-query witness and exact seam

Two capability queries returned `no matching capability`:

```text
fak capabilities "harness profile registry versioned component dependency init"
fak capabilities "Codex Claude host adapter compose harness init"
```

`fak-dev index` and raw `rg` found the adjacent parts separately. Verdict: **PARTIAL**, not
ABSENT. The pre-spine seams were `internal/harnessprofile/harnessprofile.go:143` (descriptor),
`internal/harnessinit/harnessinit.go:61` (generator), `internal/harnessresolve/harnessresolve.go:21`
(product manifest), and `pkg/harnesskit/lock.go:137` (external mixability check). The joining seam
is `internal/harnessinit/host.go:29`.

## Candidate and portfolio matrix

| Candidate | Fact or inference | Disposition | Fak seam and support budget | Disconfirming witness |
|---|---|---|---|---|
| Independent adapter version plus semantic digest | Inference from both hosts' release cadence and layered config | **DEFAULT**, shipped in #8212 | `internal/harnessprofile`; one version and snapshot per first-party adapter | Frequent compatible descriptor edits make version decisions noisier than the drift they expose |
| Project the selected host into host/wire/repoint dependencies | Inference combining host layering with fak's existing resolver | **DEFAULT**, shipped in #8212 | `internal/harnessinit` → `internal/harnessresolve`; no new resolver schema | A clean generated product cannot verify or explain the same components guard uses |
| Pin every generated product to one exact upstream CLI release | Neither host requires this for compatible releases | **EXCLUDE** | No seam; would add install ownership and needless churn | Upstream demonstrates that compatibility cannot be established without exact binary identity |
| Observe installed version and effective tool/config drift before launch | Issues show declared and effective layers can diverge | **WATCH** pending a focused runtime witness | Extend #8200 at the profile/launch boundary; read-only, no auto-update | Static descriptor plus existing tool reconciliation catches all witnessed drift classes |
| Allow third-party profile descriptors to satisfy the same init contract | Existing fak config already admits novel profiles; init currently names two built-ins | **DEFAULT** coverage frontier, contract-ready follow-on | Generalize `harnessprofile` conformance and `harnessinit` lookup; requires provenance and ownership rules | A third host needs semantics that cannot fit the descriptor/component vocabulary without unsafe flattening |
| Hot-reload a running product whenever host settings change | Claude docs describe most settings hot reload; cross-host behavior differs | **WATCH** | No spine change; revisit only with an atomic revalidation and session-boundary contract | Cross-host live mutation is less reliable than explicit restart/relock for supported workflows |
| Maintain separate hardcoded init logic per host | Existing duplicate path, not a source feature | **EXCLUDE** | Superseded by descriptor projection | A host requires irreducible behavior that cannot be expressed as a versioned component/adapter |

The **default frontier** is the shipped descriptor-to-lock path: a stable fak adapter version,
semantic digest, deterministic dependencies, and a runtime receipt. The **coverage frontier** is
third-party host admission, semantic upgrade diagnostics, upstream/effective-state drift, and a
cross-host dogfood matrix. Those belong in small leaves rather than making the spine install or
reverse-engineer host CLIs.

The filed portfolio binds those gaps without duplicating the two existing tickets:

- [#8226](https://github.com/anthony-chaudhary/fak/issues/8226) admits a provenance-checked third-party profile through the same init path;
- [#8227](https://github.com/anthony-chaudhary/fak/issues/8227) cross-dogfoods profile, guard plan, generated graph, lock, and receipt across three hosts;
- [#8229](https://github.com/anthony-chaudhary/fak/issues/8229) audits and retires remaining duplicate first-party semantics while registering legitimate exceptions;
- [#8228](https://github.com/anthony-chaudhary/fak/issues/8228) records authored adapter identity separately from observed executable identity;
- [#6805](https://github.com/anthony-chaudhary/fak/issues/6805) remains the owner for semantic compatibility/upgrade planning; and
- [#8200](https://github.com/anthony-chaudhary/fak/issues/8200) remains the owner for advertised tool-catalog drift and pre-launch remedy.

## Shipped proof and honest boundary

The source spine shipped at `5fc8a6f263aef401e888fbf069187d5bd7888b0c` as
`cmd/fak@r3112+g5fc8a6f`, `internal/harnessinit@r6+g5fc8a6f`, and
`internal/harnessprofile@r6+g5fc8a6f`.

The focused WSL witness is:

```text
go test ./internal/harnessprofile ./internal/harnessinit -count=1
go test ./cmd/fak -run 'TestHarnessInit' -count=1
```

The first command creates clean external Codex and Claude products, compiles each through
`go run`, verifies the resolved v1alpha2 lock through public `pkg/harnesskit`, and reads the
`harness.locked` receipt. A separate live WSL run through the real CLI emitted Codex lock
`sha256:3620468a4c497878a36b63e20791ec644a58d26c9e4aaac858ea672625d4988b` with
`host:codex`, `wire:openai-responses`, and both repoint mechanisms; the Claude run emitted lock
`sha256:1c866ff12579525911a58c7c2f2c7aa336864ae4fb7d4529e4cfc8482371fdc9` with
`host:claude`, `wire:anthropic`, and both repoint mechanisms. The temp products were removed
after capture.

The spine does not inspect installed CLI versions, write Codex/Claude
settings, install plugins, infer a host, or claim all future upstream releases are compatible.
Those remaining contracts are tracked as follow-on issues under #6777 rather than hidden in the
generator.
