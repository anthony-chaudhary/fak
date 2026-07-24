# Air-gapped deployment kit + SBOM (#3279) — generation triage, 2026-07-23

Triage record for [#3279](https://github.com/anthony-chaudhary/fak/issues/3279)
("air-gapped single-binary deployment kit + SBOM — local-first but governed",
part of epic [#3256](https://github.com/anthony-chaudhary/fak/issues/3256),
workstream C / corporate adoption). This run was dispatched under a
**triage-only** generation frame (`stream=unclassified`, proof bar = classify
the horizon from issue evidence *before* implementation). This note is that
classification plus a build-verified correction the implementer needs before
writing the posture page as the issue currently specs it.

## Classification: `gen/next`

Evidence, per [`docs/generation.md`](../generation.md):

- The underlying **property already ships** — one static `CGO_ENABLED=0` Go
  binary that runs fully offline with a local model (`--gguf`) or the mock
  planner, with a required-bearer-token door (`--require-key-env`). Both flags
  are live in [`cmd/fak/serve.go`](../../cmd/fak/serve.go) today.
- What #3279 asks for that is **not yet shipped** is the *packaged kit* + a
  witnessed **air-gapped bring-up that runs a governed session end-to-end**, and
  a generated **SBOM**. That is a near-term foundation still needing a
  **dogfood/gate run**, which is the textbook `gen/next` definition — not
  `gen/now` (no current-build witness for the end-to-end acceptance) and not
  `gen/future` (the deliverable is concrete near-term docs+config, not a market
  memo).
- The ecosystem already fences it this way:
  [`docs/enterprise-positioning.md`](../enterprise-positioning.md) row 7 records
  the air-gapped kit + SBOM as **Partial — `[TICKETED #3279]`**: "the
  single-binary/offline *property* ships; the packaged *kit* + SBOM is ticketed."

## Invalidating assumption (build-verified, blocks a verbatim implementation)

The issue's Direction says to ship "a 'supply-chain posture' page (**zero
external Go deps, no `go.sum`**, reproducible build)". **That premise is now
stale.** Verified from the build on 2026-07-23:

- `go.mod` requires `golang.org/x/term v0.44.0` (direct) and
  `golang.org/x/sys v0.46.0` (indirect).
- A `go.sum` **exists** with 4 lines (the two modules' zip + go.mod hashes).
- `go list -m all` reports exactly: `golang.org/x/sys v0.46.0`,
  `golang.org/x/term v0.44.0`.

So the honest, verifiable posture is **"two `golang.org/x` extended-stdlib
modules and a 4-line `go.sum`"**, *not* "zero external deps, no `go.sum`". The
SBOM and posture page must state that — writing the issue's phrasing verbatim
would ship a false claim and red the honesty grader. (The stale one-liner also
survives as a comment in `go.mod` lines 5–7 and 8; correcting that comment is a
separate `cmd`/module-lane change, out of the docs lane — filed as follow-up
below.)

## Existing spines the implementer must reuse (do not duplicate)

- [`docs/supply-chain-reproducibility.md`](../supply-chain-reproducibility.md)
  (#3711) — the reproducible-build half is done: `CGO_ENABLED=0` + `-trimpath`
  + `go.mod`/`go.sum` determinism, witnessed by a double-build CI job. The #3279
  posture page should **link** this, not restate it.
- [`docs/enterprise-positioning.md`](../enterprise-positioning.md) row 7 + note 7
  — the honest status fence for #3279; keep it in sync when the kit lands.

## What full acceptance still needs (the `gen/next` gate)

1. A **packaged air-gapped bring-up doc**: a hardened `fak serve` recipe using
   the real flags (`--gguf` for the offline model, `--require-key-env` for the
   required token) plus a **bind-safety default** (never expose auth-less on a
   routable interface) — the bind-safety default appears to need building, not
   just documenting; verify before claiming it.
2. A **captured governed session end-to-end with no network** after the binary +
   model land — the witness for acceptance bullet #1. Needs a model + a running
   host (a fleet compute node, not this Windows dev box where `go test` is
   OS-blocked and no local model is resident).
3. A **generated SBOM** (SPDX/CycloneDX) built from the 2-module dependency set
   above, each entry checkable against `go.sum` — plus the corrected posture
   page. This is the achievable, build-verifiable half; it is `gen/next` only
   because it should land with the bring-up witness, not ahead of it.

## Generation bookkeeping (frame requirement)

- **Promotion evidence** (would move #3279 toward `now`/closed): the captured
  air-gapped governed-session run (item 2) + a merged SBOM/posture page whose
  every claim resolves from `go.mod`/`go.sum`/the build.
- **Demotion / retirement evidence**: if the corporate-adoption epic #3256 is
  re-scoped away from air-gap, or if a supported distro/image path (#3256
  product-images) subsumes the single-binary kit, #3279 demotes or retires in
  favor of that path.
- **Milestone/label intake tension**: #3279 already occupies the product
  milestone "All-in-one agent runtime (MLP) — corporate-ready". GitHub allows
  one milestone, so the `gen/next` label is added **without** binding the
  `Generation G1` milestone (which would clobber the product milestone). This is
  known intake drift for a discrete-deliverable epic child; do not "fix" it by
  overwriting the product milestone.

## Follow-up filed

- Correct the stale "zero external deps / no `go.sum`" comment in `go.mod`
  (lines 5–8) — a `cmd`/module-lane one-liner, out of scope for this docs-lane
  triage.
