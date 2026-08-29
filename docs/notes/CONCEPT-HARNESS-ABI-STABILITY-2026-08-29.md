---
title: "Harness ABI stability under rapid upstream churn"
description: "Pinned Codex, Claude Code, Gemini CLI, and OpenCode study: compatibility mechanisms, negative knowledge, FAK seams, and the default-versus-coverage frontier."
---

# Harness ABI stability under rapid upstream churn

**Verdict (observed 2026-08-29):** FAK already has a provider-neutral run
protocol, additive ABI/capability rules, immutable adapter descriptors, exact
Codex schema receipts, and binary-skew classification. It is still **partial**
as a harness compatibility system: the pieces do not yet produce one negotiated
per-seam envelope or a bidirectional released/current semantic-diff gate.
Product version alone is not that envelope; protocol/schema revision,
capabilities, build/channel, adapter digest, and persisted-state generation can
move independently.

This is the completion receipt for [#10020](https://github.com/anthony-chaudhary/fak/issues/10020),
under [#6805](https://github.com/anthony-chaudhary/fak/issues/6805). Durable study
receipt: `study_d2e87854f7d555384500becb1b8328cea3f3c8de0376fedee4526d6580069429`
(`fak study search --limit 20 'Harness ABI churn'` returned the stored record).

## Source ledger

`state` distinguishes code/release evidence from unresolved negative knowledge.
Every code anchor is a pinned `path:line@sha`; issue reports are evidence of an
observed failure report, not proof of the current implementation.

| Source | observed_at | source_event_at | state / platform | Pinned evidence | License / use | Refresh trigger |
|---|---|---|---|---|---|---|
| `openai/codex` | `2026-08-29T15:40:00Z` | `2026-08-29T06:05:28Z` | public `main` HEAD; Rust/TS, desktop + CLI + app-server; release line `0.151.0` | [`codex-rs/app-server/README.md:2757@6478a75`](https://github.com/openai/codex/blob/6478a751fde8884b2fdc76486fe23175a8e795d4/codex-rs/app-server/README.md#L2757), [`bazel/rules/testing/compat/exec_server_compat_test.rs:47@6478a75`](https://github.com/openai/codex/blob/6478a751fde8884b2fdc76486fe23175a8e795d4/bazel/rules/testing/compat/exec_server_compat_test.rs#L47), [`codex-rs/exec-server/BUILD.bazel:39@6478a75`](https://github.com/openai/codex/blob/6478a751fde8884b2fdc76486fe23175a8e795d4/codex-rs/exec-server/BUILD.bazel#L39) | [Apache-2.0 `LICENSE:1@6478a75`](https://github.com/openai/codex/blob/6478a751fde8884b2fdc76486fe23175a8e795d4/LICENSE#L1) + NOTICE; **ADAPT** concepts, independently implement | app-server schema/protocol/config removal, compatibility test pin, or stable/experimental boundary changes |
| `anthropics/claude-code` | `2026-08-29T15:40:00Z` | `2026-08-28T18:19:26Z` | public release/docs distribution at `main`; proprietary runtime; release `2.1.251` | [`CHANGELOG.md:1177@f1af9b1`](https://github.com/anthropics/claude-code/blob/f1af9b1f4b1fd4c776135381606edada82ef638e/CHANGELOG.md#L1177), [`CHANGELOG.md:1345@f1af9b1`](https://github.com/anthropics/claude-code/blob/f1af9b1f4b1fd4c776135381606edada82ef638e/CHANGELOG.md#L1345), [`CHANGELOG.md:1462@f1af9b1`](https://github.com/anthropics/claude-code/blob/f1af9b1f4b1fd4c776135381606edada82ef638e/CHANGELOG.md#L1462) | [`LICENSE.md:1@f1af9b1`](https://github.com/anthropics/claude-code/blob/f1af9b1f4b1fd4c776135381606edada82ef638e/LICENSE.md#L1): all rights reserved, use subject to Anthropic Commercial Terms; **INSPIRE-ONLY** | a public protocol/schema/test surface appears, or a release changes mixed-version/rename behavior |
| `google-gemini/gemini-cli` | `2026-08-29T15:40:00Z` | `2026-08-28T21:38:20Z` | public `main` HEAD; TypeScript CLI/extensions/hooks; nightly line `0.59.0-nightly.20260829.g0bd1d4397` | [`packages/core/src/hooks/hookTranslator.ts:19@0bd1d43`](https://github.com/google-gemini/gemini-cli/blob/0bd1d439751478771c45d3d0895a6a9760554bf4/packages/core/src/hooks/hookTranslator.ts#L19), [`packages/cli/src/config/extensions/update.ts:100@0bd1d43`](https://github.com/google-gemini/gemini-cli/blob/0bd1d439751478771c45d3d0895a6a9760554bf4/packages/cli/src/config/extensions/update.ts#L100), [`packages/cli/src/config/extensions/update.test.ts:278@0bd1d43`](https://github.com/google-gemini/gemini-cli/blob/0bd1d439751478771c45d3d0895a6a9760554bf4/packages/cli/src/config/extensions/update.test.ts#L278) | [Apache-2.0 `LICENSE:1@0bd1d43`](https://github.com/google-gemini/gemini-cli/blob/0bd1d439751478771c45d3d0895a6a9760554bf4/LICENSE#L1); **ADAPT** concepts, independently implement | hook DTO/event spelling, settings aliases, extension transaction, or stable release changes |
| `anomalyco/opencode` | `2026-08-29T15:40:00Z` | `2026-08-29T02:34:49Z` | public `dev` HEAD; TypeScript app/server/plugins; package line `1.18.25` | [`packages/app/src/utils/server-protocol.ts:24@dc4449d`](https://github.com/anomalyco/opencode/blob/dc4449df0d52199704ea4989a5a993ebbc605612/packages/app/src/utils/server-protocol.ts#L24), [`packages/app/src/utils/server-compat.ts:86@dc4449d`](https://github.com/anomalyco/opencode/blob/dc4449df0d52199704ea4989a5a993ebbc605612/packages/app/src/utils/server-compat.ts#L86), [`packages/opencode/src/plugin/shared.ts:194@dc4449d`](https://github.com/anomalyco/opencode/blob/dc4449df0d52199704ea4989a5a993ebbc605612/packages/opencode/src/plugin/shared.ts#L194), [`packages/opencode/src/plugin/loader.ts:123@dc4449d`](https://github.com/anomalyco/opencode/blob/dc4449df0d52199704ea4989a5a993ebbc605612/packages/opencode/src/plugin/loader.ts#L123) | [MIT `LICENSE:1@dc4449d`](https://github.com/anomalyco/opencode/blob/dc4449df0d52199704ea4989a5a993ebbc605612/LICENSE#L1); **ADAPT** concepts, independently implement | v1 support removal, protocol probe shape, plugin range/default pin, or migration semantics change |

Cursor was not promoted into the ledger. Its closed implementation did not add a
primary code/test anchor beyond the four repositories, and Codex issue
[#38359](https://github.com/openai/codex/issues/38359) is only an open report that
version-threshold feature detection fails under Cursor. Refresh from Cursor's
official release/docs only if it publishes a concrete compatibility contract.

### Coverage and completeness critic

The acquisition covered current pinned trees, implementation/tests, recent
release history, and targeted open/closed PR/issue searches for configuration,
protocol, update, extension, plugin, persistence, and mixed-version seams. The
Codex tree received the deepest pass because it is the primary target; the other
three were mechanism comparisons, not exhaustive inventories. Claude Code's
public repository does not contain the proprietary runtime or its tests, so only
dated release claims are asserted. Hosted rollouts, private telemetry, signed
artifacts, and live cross-version execution were not observed. README-only claims
were not used as sole support. The next disconfirming run is a generated schema
and behavioral matrix across two adjacent released binaries plus current HEAD.

## What FAK already proves

The self-query ran before candidate adjudication:

```text
fak capabilities 'harness ABI capability negotiation compatibility ranges schema digest'
  -> Enforce supporting capability floor; Attribute cache/token savings
fak capabilities 'extension update rollback plugin version migration'
  -> no matching capability
fak capabilities 'config migration deprecation semantic diff'
  -> no matching capability
fak-dev index docs|leaves|verbs|claims 'harness ABI compatibility version skew'
  -> harness protocol/contract docs; abi, appversion, harnesskit, harnessprofile,
     versionskew leaves; harness/version/capabilities/conformance verbs
```

The lexical/index fallback found the exact seams:

- [`internal/abi/types.go:32`](../../internal/abi/types.go#L32) defines additive
  major/minor capability negotiation; [`internal/market/market.go:161`](../../internal/market/market.go#L161)
  validates extension ABI ranges.
- [`pkg/harnesskit/protocol.go:9`](../../pkg/harnesskit/protocol.go#L9) owns the
  provider-neutral protocol and unknown-event behavior; [`pkg/harnesskit/contract.go:29`](../../pkg/harnesskit/contract.go#L29)
  makes its compatibility promise machine-readable.
- [`internal/harnessprofile/version.go:17`](../../internal/harnessprofile/version.go#L17)
  binds adapter semantic content to a version; its tests require a version bump
  when the digest changes.
- [`internal/codexsession/compatibility.go:23`](../../internal/codexsession/compatibility.go#L23)
  binds binary/protocol/schema/authority and fails closed on an untested digest,
  but it is Codex-specific and exact-match rather than a cross-release semantic
  compatibility classifier.
- [`internal/versionskew/versionskew.go:41`](../../internal/versionskew/versionskew.go#L41)
  classifies build ancestry and provenance. That is necessary but does not prove
  wire/config/state compatibility.
- [`docs/upgrade.md:33`](../upgrade.md#L33) deliberately avoids a blanket
  compatibility range; [`docs/generation-abi-compatibility-policy.md:147`](../generation-abi-compatibility-policy.md#L147)
  records that unknown JSON reader tolerance and breaking-change retirement are
  still assumptions outside the proven wire arm.

Thus the neighborhood is present, while the single harness-wide compatibility
decision requested by #6805 remains partial.

## Candidate matrix

`Fact` means observed at the pin. `Inference` is the FAK design consequence.
Each row has one decision axis so support cost and disconfirmation stay explicit.

| Axis | Evidence type and mechanism | FAK / worldview reason | Portfolio | Support cost | Disconfirming check | Routing |
|---|---|---|---|---|---|---|
| Startup compatibility identity | **Fact:** Codex separates stable/experimental initialize capabilities; FAK binds exact Codex binary/protocol/schema/authority. **Inference:** negotiate `{product build, protocol, schema digest, capabilities, channel, state generation}` per seam. | **PARTIAL**; a capability floor must be structural and fail closed, not inferred from a marketing version. | **DEFAULT** | M: shared envelope + receipts | two builds with the same version expose different authority/schema | [#6805](https://github.com/anthony-chaudhary/fak/issues/6805), [#8200](https://github.com/anthony-chaudhary/fak/issues/8200), [#8228](https://github.com/anthony-chaudhary/fak/issues/8228) |
| Released/current skew testing | **Fact:** Codex exercises app/current-executor/released in both directions and pins released artifacts. **Inference:** compare behavioral and schema meaning, not only same-revision fixture equality. | **PARTIAL**; FAK has exact fixtures, not the generic adjacent-release matrix. | **DEFAULT** | M: fixtures + CI artifact pins | an adjacent pair passes both directions and semantic diff with existing gates | [#8913](https://github.com/anthony-chaudhary/fak/issues/8913), [#9564](https://github.com/anthony-chaudhary/fak/issues/9564) |
| Stable DTO boundary | **Fact:** Gemini hook DTOs isolate SDK types; FAK protocol payloads avoid provider wire types. **Inference:** every provider adapter translates at one boundary. | **PRESENT** for the public protocol, **PARTIAL** across all adapters; kernel ownership stays provider-neutral. | **DEFAULT** | M per adapter | an adapter imports provider request/event types across its public FAK boundary | [#9564](https://github.com/anthony-chaudhary/fak/issues/9564) |
| Unknown additive events | **Fact:** FAK validates the envelope, advances the cursor, and withholds authority/rendering for unknown payloads. | **PRESENT**; conservative forward compatibility matches the additive-open-set worldview. | **DEFAULT** | L: keep conformance fixtures | a future event stops replay or gains authority | [#6456](https://github.com/anthony-chaudhary/fak/issues/6456) |
| Structural protocol probing | **Fact:** OpenCode probes v1/v2 health shapes and lazily selects an adapter. **Inference:** use a bounded probe only when negotiation metadata is absent. | **DIVERGENT** as a universal default: heuristic probing is weaker than an explicit contract, but useful at legacy edges. | **RECIPE** | M per legacy family | false-positive/ambiguous shapes cannot be bounded | [#6456](https://github.com/anthony-chaudhary/fak/issues/6456) |
| Extension update transaction | **Fact:** Gemini stages an update, retains prior config/version, and tests rollback on install failure. **Inference:** FAK should own rollback only for extensions it owns. | **ABSENT** as a general harness facility; external product replacement remains operator-owned by design. | **OPTIONAL-MODULE** | H: artifact retention, atomic swap, recovery tests | no FAK-owned extension can change on disk at runtime | [#7220](https://github.com/anthony-chaudhary/fak/issues/7220) |
| Plugin compatibility before execution | **Fact:** OpenCode reads `engines.opencode` and rejects an incompatible npm plugin before dynamic import. **Inference:** verify range, digest, trust, and capability profile before loading. | **PARTIAL**; market ABI ranges and harness locks exist, but all loaders need one pre-execution invariant. | **DEFAULT** | M: loader conformance | every executable extension path already proves checks precede load | [#8200](https://github.com/anthony-chaudhary/fak/issues/8200), [#8913](https://github.com/anthony-chaudhary/fak/issues/8913) |
| Runtime/build provenance | **Fact:** Codex exposes structured build info; Claude release notes use embedded build time for daemon handoff. **Inference:** record VCS/artifact/channel, never infer compatibility from recency. | **PARTIAL**; FAK classifies its binary but not a complete upstream harness envelope. | **DEFAULT** | L-M: extend receipt | `#8228` proves upstream harness build/artifact/channel at every admission | [#8228](https://github.com/anthony-chaudhary/fak/issues/8228), [#7220](https://github.com/anthony-chaudhary/fak/issues/7220) |
| Proprietary implementation borrowing | **Fact:** Claude public changelog reports mixed-writer and alias fixes; runtime/tests are unavailable. | **ABSENT by choice**; copying or asserting private mechanics would violate provenance and license discipline. | **EXCLUDE** implementation; **WATCH** behavior | L: release monitor | a permissively licensed public runtime/test contract appears | [#10020](https://github.com/anthony-chaudhary/fak/issues/10020) |

## Default versus coverage frontier

**Default spine:** explicit per-seam envelope; provider-neutral DTO translation;
unknown-additive-event safety; pre-load compatibility/trust checks; build/artifact
provenance; bidirectional adjacent-release schema and behavior tests. These are
growth-invariant and belong on every supported harness path.

**Coverage edge:** vendor config aliases, OpenCode v1 structural probes, Gemini
extension transactions, Codex experimental methods, and proprietary Claude/Cursor
release behavior. Keep these in adapters, optional modules, recipes, or watches;
do not widen the kernel contract for one product's churn.

## Negative knowledge: shipped versus open

The pinned anchors above are shipped tree/release evidence. The following remain
**open reports or proposals at observation time**, so they are refresh triggers,
not implementation claims: Codex [#41512](https://github.com/openai/codex/issues/41512)
(mixed-version thread resume), [#40865](https://github.com/openai/codex/issues/40865)
(non-atomic tool cutover), [#40411](https://github.com/openai/codex/issues/40411)
(plugin update rollback/trust), [#37403](https://github.com/openai/codex/issues/37403)
(writer handoff), and [#38359](https://github.com/openai/codex/issues/38359)
(version-threshold detection); Gemini [#29123](https://github.com/google-gemini/gemini-cli/issues/29123)
(hook event spelling) and PR [#29125](https://github.com/google-gemini/gemini-cli/pull/29125)
(timeout unit migration); OpenCode [#46161](https://github.com/anomalyco/opencode/issues/46161)
(initialization timeout) and [#46095](https://github.com/anomalyco/opencode/issues/46095)
(failed plugin import contamination). Re-check state and merged code before using
any of them as a test oracle.

No new child issue is justified by this study. Existing owners already partition
the work: runtime provenance [#8228](https://github.com/anthony-chaudhary/fak/issues/8228),
capability profiles [#8200](https://github.com/anthony-chaudhary/fak/issues/8200),
tool-schema drift [#8913](https://github.com/anthony-chaudhary/fak/issues/8913),
provider attribution/schema drift [#9564](https://github.com/anthony-chaudhary/fak/issues/9564),
lifecycle/checkpoint negotiation [#6456](https://github.com/anthony-chaudhary/fak/issues/6456),
and effective harness version/update posture [#7220](https://github.com/anthony-chaudhary/fak/issues/7220).
