# Self-update fast paths: check less, move less, rebuild only changed artifacts

- **Date:** 2026-08-29
- **Status:** design note; PRESENT is shipped, while PARTIAL may be implemented without its full operating-envelope witness
- **Study receipt:** `study_b4ce9b425907f4518a7c2e30fadb0aa7d1b71ee986a10631e63c583950f7a829`
- **First implementation spine:** [#10077](https://github.com/anthony-chaudhary/fak/issues/10077)

The fastest successful update is usually a proved no-op. FAK should make the common
launch path a local state read, make the common remote check a conditional metadata
request, and reserve patching, downloading, building, and activation for successively
smaller minorities of launches. "Delta" therefore means more than a binary patch: it
means a staged decision system that pays only for the first proof it needs.

This note records a design direction, not a new performance claim. In particular, the
existing 5x candidate-cache test uses a deterministic synthetic cost model; a live
same-host end-to-end witness remains part of #10077's acceptance envelope.

## Source ledger

`Observed` is when this study checked the source. `Event` is the upstream release or
source event used by the study; rolling pages are labeled rather than assigned a
fabricated release date.

| Source | Event | Observed | State at observation | License / use fence | Mechanism worth borrowing |
|---|---|---|---|---|---|
| [Omaha Protocol V3](https://github.com/google/omaha/blob/578a81bfb3cc77f1366b8d6d61e3dce10df2794d/doc/ServerProtocolV3.md) | pinned source commit, 2022-06-15 | 2026-08-29 | repository archived; historical protocol reference | Apache-2.0 | Explicit `noupdate`, client fingerprints, cohorts, and server backoff. Borrow protocol shape, not the archived implementation. |
| [Chromium Updater](https://chromium.googlesource.com/chromium/src/+/main/chrome/updater/) | rolling `main` | 2026-08-29 | active | Chromium BSD-style license | Ordered update pipeline and active successor context for Omaha-style selection. Re-pin before implementation. |
| [RFC 9110: HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110.html) | Standards Track, June 2022 | 2026-08-29 | active Internet Standard (STD 97) | IETF Trust Legal Provisions; standards text/use | `ETag` plus `If-None-Match` makes a retrieval conditional, allowing a matching representation to return `304 Not Modified` without transferring its body. |
| [Sparkle delta updates](https://sparkle-project.org/documentation/delta-updates/) | current release documentation; study observed a 2026-08-17 release | 2026-08-29 | active | MIT | Per-source signed deltas, preflight, patched-output verification, automatic full fallback, channels, and phased rollout. |
| [TUF specification](https://theupdateframework.github.io/specification/latest/) and [go-tuf](https://github.com/theupdateframework/go-tuf) | stable specification release recorded by the study on 2026-08-10 | 2026-08-29 | active | Community Specification License for the specification; Apache-2.0 for go-tuf | Signed root/targets/snapshot/timestamp metadata; expiry, rollback, freeze, and mix-and-match defenses; consistent snapshots. |
| [Go build and test caching](https://go.dev/cmd/go/#hdr-Build_and_test_caching) | rolling tool documentation | 2026-08-29 | active | BSD-3-Clause | Cache identity already includes source, compiler, environment, and build options; caches are safe for concurrent use. Keep this as the compile-delta layer. |
| [Zstandard `--patch-from`](https://github.com/facebook/zstd/blob/dev/programs/zstd.1.md) | feature introduced before the 2020-12-17 upstream patching note; current CLI docs are rolling | 2026-08-29 | active | upstream dual BSD/GPL license; select the compatible terms before embedding | Fast dictionary-based patch creation/application. The format supplies compression, not FAK's source/target identity or update authorization. |
| [Chromium Zucchini](https://chromium.googlesource.com/chromium/src/+/main/components/zucchini/) and [Courgette](https://chromium.googlesource.com/chromium/src/+/main/courgette/) | rolling `main` source trees | 2026-08-29 | Zucchini active; Courgette retained with older maintenance history | Chromium BSD-style license; research reference only | Executable-aware normalization can shrink patches beyond generic byte differencing, but its format complexity and platform-specific operating surface defer it until simpler signed deltas miss a measured FAK envelope. |
| [Windows MSIX differential updates](https://learn.microsoft.com/en-us/windows/msix/desktop/managing-your-msix-deployment-update) and [App Installer update settings](https://learn.microsoft.com/en-us/windows/msix/app-installer/update-settings) | rolling Microsoft Learn pages | 2026-08-29 | active platform facility | Microsoft documentation terms; Windows implementation is platform-provided | SHA-256 block maps over 64 KiB slices, reuse from any older package, launch/cadence configuration, repair, and downgrade controls. |

## Self-query: what FAK already has

Verdicts are about the tree observed on 2026-08-29, not the issue backlog.

### PRESENT

- `cmd/fak/selfupdate.go:selfUpdateFetchOrigin` uses Git's ordinary object transfer, so
  source acquisition is already delta/object based rather than a full repository download.
- `cmd/fak/selfupdate.go:selfUpdateShouldBuild` and its caller skip installation for an
  exact-current binary and preserve a provably ahead local build. `--check` deliberately
  does not fetch, so the observation-only mode is non-mutating.
- `internal/selfinstall.Install` implements an exact-commit candidate cache whose input
  digest binds the selected/source commit, host platform, Go toolchain/environment, build
  arguments, and vet arguments. It verifies artifact SHA-256 and reruns smoke/provenance
  before activation. Corrupt or mismatched entries fall through to build, vet, and smoke.
- `internal/selfinstall.RunLaunchTransaction` stages all copies, snapshots the prior
  deployed bytes, activates them as one transaction, and reports rollback or rollback
  failure rather than calling a partial install successful.
- `internal/launchshim` already owns configurable activation waiting/failure policy for
  launcher handoff; launcher configuration need not be reinvented inside an artifact
  patcher.

### PARTIAL

- `cmd/fak/selfupdate_install.go:selfUpdateAttemptOptions` now supplies
  `selfinstall.Options.CacheDir` from the repository's resolved Git common directory,
  making the existing verified cache a clone-shared live retry path. The real-Git test
  proves selected linked worktrees from the same clone share a cache, independent clones
  remain isolated, and unresolved Git state disables caching. This is implemented by
  #10077, but it is not yet an end-to-end speed claim.
- `go build` benefits from Go's build cache, but self-update has no explicit build-input
  receipt that can prove a source commit changed no runtime inputs. A new VCS stamp can
  therefore force a different binary even for a documentation-only commit.
- The current freshness decision avoids builds after comparison, but
  `internal/versionskew.AssessStamp` still performs ancestry-oriented Git work. An
  equal full SHA can be decided before those subprocesses.
- Hot-copy convergence is per target, while fak/fak-dev preparation still behaves as one
  broad update transaction. A stale companion can cause work that a component-specific
  receipt could avoid.

### ABSENT

- There is no cached remote update manifest with TTL plus `ETag`/`If-None-Match`, explicit
  `noupdate`, channel, cohort, or rollout admission.
- There is no signed artifact catalog separating metadata generation, application
  version, build-input digest, artifact digest, and source-commit provenance.
- There is no per-source binary delta catalog, patch selection policy, or verified
  delta-to-full fallback in FAK's installer.
- There is no proof-backed rule that lets a new source commit advance provenance without
  publishing or installing a new application artifact when build inputs and output bytes
  are unchanged.

## Candidate matrix

| Candidate | Expected win | Complexity / risk | Disposition |
|---|---|---|---|
| Wire the existing verified candidate cache into live retries | Removes repeat build+vet while preserving exact-commit smoke and full fallback | Small seam; live-host performance evidence still required | **SPINE IMPLEMENTED:** #10077; live speed witness pending |
| Equal-SHA local short-circuit | Avoids ancestry subprocesses on the dominant already-current launch | Must compare canonical full object IDs and retain dirty/unstamped handling | **NEXT:** small launcher/check optimization |
| Cached manifest TTL plus HTTP conditional request and explicit no-update | Most launches become one local read; remote checks commonly become 304/no body | Needs stale-policy, clock, force, and security-metadata rules | **NEXT:** metadata spine |
| Build-input receipt distinct from source commit | Lets docs/tests/research-only commits advance provenance without rebuilding runtime bytes | Unsafe if based only on path names; must derive the complete runtime input graph | **NEXT:** highest-leverage semantic skip |
| Per-source signed artifact deltas with full fallback | Moves only changed artifact bytes between known source and target digests | Catalog, patch generation, retention, and cross-platform QA | **LATER:** after signed full-artifact updates exist |
| zstd `--patch-from` codec | Practical first patch codec with fast application and modest operational surface | Patch format is not authorization; bind source/target hashes and cap CPU/RAM/ratio | **WATCH / optional codec** |
| TUF-style signed metadata | Supplies rollback/freeze/expiry/key-rotation defenses for remote artifacts | More ceremony than local source builds need | **ADAPT** when an artifact channel becomes a supported product path |
| MSIX/App Installer path | Native Windows block reuse, cadence, repair, and downgrade behavior | Platform-specific packaging and distribution contract | **OPTIONAL platform adapter**, not the portable core |

Complex binary-rewriting schemes such as Courgette/Zucchini are intentionally outside the
first implementation sequence. Reconsider them only after real FAK release artifacts show
that signed full downloads plus zstd deltas miss the declared size/latency envelope.

## Recommended staged architecture

Keep four stages separate so each can terminate the run:

1. **Check** reads local update state. Inside a configured TTL it returns a proved cached
   decision. Outside the TTL it conditionally fetches a small signed manifest and accepts
   `304 Not Modified` or explicit `noupdate` without touching artifact bytes.
2. **Select** applies channel, cohort, platform, architecture, minimum-supported-version,
   and rollback/freeze rules. It compares build-input and artifact digests before choosing
   any transfer.
3. **Acquire and materialize** chooses the cheapest valid route: exact verified local
   candidate, direct signed delta from the installed artifact, signed full artifact, or
   the existing pristine-source build. Every optimization falls back to a more complete
   route; none falls past verification.
4. **Verify and activate** binds the old artifact digest, patch/full payload digest, and
   expected new artifact digest; runs provenance and smoke checks; then uses the existing
   multi-copy transaction, atomic swap, and rollback receipt.

The check path and update path should be independently configurable. Useful policy knobs are
`channel`, stable `cohort_id`, `check_ttl`, allowed launch-time checking, maximum patch ratio,
maximum patch memory/time, retained source-artifact count, and `force_check`/`force_update`.
`force` may bypass latency skips; it must never bypass signature, digest, expiry, smoke, or
rollback protections.

## Local state and independent clocks

Persist one atomic, schema-versioned record beside the clone-shared cache:

- selected target/source commit and the last successfully activated source commit;
- application version and launcher/companion component versions;
- build-input digest and final artifact SHA-256;
- channel, stable cohort ID, platform/architecture, and rollout decision;
- manifest generation/version, signing-root version, expiry, and highest trusted version;
- last-check time, cache TTL, `ETag`, and server retry/backoff deadline;
- installed-slot, previous-good-slot, activation time, and last smoke/rollback result;
- available delta edges keyed by source artifact digest to target artifact digest.

These values are not aliases. `source_commit` answers where the inputs came from.
`build_input_digest` answers whether runtime-producing inputs changed. `artifact_digest`
answers whether deployed bytes changed. `app_version` names a user-visible artifact release.
`metadata_generation` orders catalog state even when no application artifact changes.

## Strict skip and version rules

Skip network, transfer, build, activation, or version publication only with a positive proof:

1. **Skip the remote check** while unexpired local TTL/backoff state applies, unless an
   operator forces a check or security policy requires metadata refresh.
2. **Skip metadata bytes** on a valid HTTP 304. Refresh the check timestamp without
   pretending that an application update occurred.
3. **Skip selection/download** for explicit signed `noupdate`, a cohort/channel hold, or a
   selected target artifact digest equal to the installed digest.
4. **Skip the build** when a complete, schema-versioned runtime input manifest produces the
   same `build_input_digest`. File-extension or directory heuristics alone are insufficient.
5. **Skip publication and the application-version bump** when a reproducible rebuild yields
   the same `artifact_digest`. Record the newer source provenance separately if policy needs it.
6. **Skip activation per component** when fak, fak-dev, or launcher receipt already matches
   that component's selected digest. Do not rebuild an unchanged primary merely because a
   companion is stale.

Never apply these skips when signed metadata is expired or regresses a trusted version, when
source/target identity is incomplete, when platform/toolchain/build options or the declared
runtime input graph changed, or when the installed artifact cannot attest clean provenance.
A failed delta, digest, signature, smoke, or activation check selects the signed full artifact
or pristine source-build path; it does not turn an update into a success.

## Implementation sequence after #10077

#10077 is deliberately narrow: set the live attempt's cache directory from the repository's
Git common directory and capture a real retry witness. Once that spine is green, later work can
be decomposed without coupling all mechanisms into one risky updater rewrite:

1. full-SHA short-circuit and phase receipts for the launch check;
2. per-component selection so a stale companion does not rebuild a current primary;
3. content-derived build-input receipts and a no-runtime-change decision;
4. cached/conditional signed metadata with channel, cohort, cadence, and `noupdate`;
5. signed full-artifact delivery and atomic versioned slots;
6. direct-delta catalog and zstd evaluation under size, CPU, memory, and fallback envelopes;
7. optional Windows MSIX/App Installer packaging adapter.

No follow-on ticket is implied to be shipped by this note. Fan-out belongs after the #10077
spine lands and its real operating-envelope witness identifies the actual next bottleneck.

## Live repository dogfood — August 30, 2026 (#10212)

The shipped spine was exercised twice against this repository's committed
`76e84ff50a20ba4d1b66fb939c0b15a3fcf1d89c` `HEAD`, using the standalone
`cmd/fak-selfupdate` entry point, the live clone-shared Git common directory, and disposable
`/tmp` updater/target copies. The updater intentionally built from committed `HEAD`, not the
peer-dirty working tree.

```text
run 1: status=divergent reused=false total_ms=37297
       build_ms=23893 vet_ms=5539 smoke_ms=1469
run 2: status=divergent reused=false total_ms=38513
       build_ms=24571 vet_ms=5630 smoke_ms=1649
both:  binary transaction completed but identity persistence failed:
       build-input digest is not SHA-256
```

This is a real dogfood failure, not a synthetic cache benchmark. The build-input identity
producer emits `sha256:<64 hex>`, while installed-identity persistence accepts only bare
64-hex SHA-256. Because persistence fails after installation, the verified candidate is not
committed for reuse and the second run performs build, vet, and smoke again. The required fix
is tracked by #10342. Two additional findings are filed rather than expanded into this
follow-on: typed cache-disposition observability (#10346) and stale source-path references
after the standalone-command migration (#10347). Current implementation and tests live under
`internal/selfupdate/cmd/`; commit `53d8ec8afd7195af22e519427e91da75f966077f`
moved them from `cmd/fak/`.

Promotion evidence: the live path reached the repository build, vet, smoke, and binary
transaction stages twice. Demotion/retirement evidence: no cache-hit or speed claim is promoted;
the measured retry rebuilt and the readout records the failure. Invalidating assumption: the
spine's cache can only serve real retries if build-input digest generation and identity
persistence use one canonical representation.

## Candidate-cache outcome readout — August 30, 2026 (#10214)

Every cache-enabled source-build attempt now joins the existing self-update progress stream with
one cumulative, machine-greppable readout:

```text
self-update: candidate-cache outcomes success=1 refusal=1 error=1
```

`success` counts an exact-input verified candidate restored and installed from the clone-shared
cache. `refusal` counts a cache-enabled attempt that safely fell back to a newly gated build.
`error` counts an attempt that did not produce an installed candidate. The captured contract test
feeds one real `selfinstall.Result` from each class through the production formatter and asserts
the exact cumulative readout; no dashboard, separate metrics file, receipt schema change, or
cache-acceptance change was added.

Promotion evidence: repeated dogfood runs retain zero `error` counts and show `success` increasing
on exact-input retries. Demotion or retirement evidence: unexplained `refusal` growth, any `error`
growth, or an operator surface that no longer captures the progress stream. Invalidating
assumption: process-local cumulative counts are sufficient for the existing invocation readout;
long-lived cross-process history would require a separately justified ledger fold.
