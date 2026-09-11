# Quarantine: orphaned keep-awake / deploymanifest Power test WIP

Date: 2026-09-11
Repo: `fak` (public)
Status: WIP cannot land as-is — the implementation API referenced by the modified
tests **does not exist anywhere in git history**. The two tracked test files were
reverted to `HEAD` so the tree compiles; their uncommitted contents are preserved
here as `.orphan` reference copies.

## What was observed

Two tracked test files were modified in the working tree with new tests that
reference symbols which do not exist in any commit:

### 1. `cmd/fak/serve_config_test.go`

Added `TestServeManifestKeepAwakeDefaultsAndOverride` (working-tree lines 251-285).
References:

- `DefaultKeepAwakeServe` — undefined at `serve_config_test.go:258` and `:259`.
  Asserted behavior: with no `[power]` section, the new always-on built-in
  keep-awake default is what `newServeFlagSet()` initializes `sf.keepAwake` to.
- `applyServeManifestDefaults(sf, m)` pre-parse wiring and `KeepAwakeAlways` —
  asserted behavior: a manifest `[power] keep_awake = "off"` overrides the
  built-in default before flag parse; an explicit `--keep-awake=always` flag
  still wins over the manifest.

### 2. `internal/deploymanifest/deploymanifest_test.go`

Added `TestFakTomlPowerKeepAwake` (working-tree lines 301-358). References:

- `d.Power` / `d.Power.KeepAwake` — `Manifest has no field or method Power`
  at `deploymanifest_test.go:308` (`d := Defaults()`).
- `m.Power.KeepAwake` at `:321`, `:322` (per-manifest override).
  Asserted behavior:
  - `Defaults().Power.KeepAwake == "always"` (always-on wake-lock posture).
  - `Defaults().Present("power", "keep_awake")` is false (default not marked
    present).
  - `Parse` accepts `"off"`, `"while-active"`, `"always"` and records
    `Present("power","keep_awake") == true`.
  - `Parse` refuses `"snooze"` fail-closed with `*LoadError` where
    `Reason == ReasonBadValue` and locus `Section=="power"`, `Key=="keep_awake"`.
  - `Value(Key{Section:"power",Name:"keep_awake"})` projects the string value.
  - `Descriptors()` includes `power.keep_awake` with a non-empty description.

## No implementation exists in any git history

```
$ git log --all --oneline -S "DefaultKeepAwakeServe"
(empty)
$ git log --all --oneline -S "KeepAwake" -- internal/deploymanifest/
(empty)
$ git grep -n "DefaultKeepAwakeServe" -- .
cmd/fak/serve_config_test.go:258:	if *sfPlain.keepAwake != DefaultKeepAwakeServe {
cmd/fak/serve_config_test.go:259:		t.Fatalf("keep-awake default = %q, want %q", *sfPlain.keepAwake, DefaultKeepAwakeServe)
$ git grep -n "Power" -- internal/deploymanifest/   # only the test file matched
```

The matching implementation (the `[power]` section type, `Power` field on
`Manifest`, the `power.keep_awake` descriptor + validation, the serve-side
`DefaultKeepAwakeServe` / `KeepAwakeAlways` constants and
`applyServeManifestDefaults`) was never written, or was lost before the
available history.

## Compile-error evidence

```
$ go vet ./internal/deploymanifest ./cmd/fak
# github.com/anthony-chaudhary/fak/internal/deploymanifest
vet.exe: internal\deploymanifest\deploymanifest_test.go:308:7: d.Power undefined (type Manifest has no field or method Power)
# github.com/anthony-chaudhary/fak/cmd/fak
vet.exe: cmd\fak\serve_config_test.go:258:27: undefined: DefaultKeepAwakeServe
```

## Preservation

- `serve_config_test.go.orphan` — full working-tree content of
  `cmd/fak/serve_config_test.go` at quarantine time.
- `deploymanifest_test.go.orphan` — full working-tree content of
  `internal/deploymanifest/deploymanifest_test.go` at quarantine time.

## How to revive

1. Implement the missing API:
   - `internal/deploymanifest`: add a `Power` section type with a `KeepAwake`
     field, set `Power.KeepAwake = "always"` in `Defaults()`, accept
     `off|while-active|always`, refuse other values with `ReasonBadValue` and
     locus `power`/`keep_awake`, wire the typed projection (`Value`) and a
     `Descriptors()` entry for `power.keep_awake` with a description.
   - `cmd/fak`: define `DefaultKeepAwakeServe` (== the always-on built-in
     default) and `KeepAwakeAlways`, and implement
     `applyServeManifestDefaults(sf, m)` so a manifest `[power] keep_awake`
     pre-sets `sf.keepAwake` and an explicit flag still wins.
2. Copy each `.orphan` back over its tracked test file:
   ```
   Copy-Item docs/tickets/quarantine/keepawake-power-orphan-2026-09-11/serve_config_test.go.orphan cmd/fak/serve_config_test.go
   Copy-Item docs/tickets/quarantine/keepawake-power-orphan-2026-09-11/deploymanifest_test.go.orphan internal/deploymanifest/deploymanifest_test.go
   ```
3. Verify: `go test ./cmd/fak/... ./internal/deploymanifest/... -count=1`.

## Blocker-issue draft (do NOT file)

Title: `implement DefaultKeepAwakeServe and deploymanifest Power section (orphaned test WIP)`
Label suggestion: `blocker`

Body:

> **Witness / observation**
> - `cmd/fak/serve_config_test.go` (modified WIP) references
>   `DefaultKeepAwakeServe` (`:258`, `:259`) and `KeepAwakeAlways` +
>   `applyServeManifestDefaults` — none exist in tree or history.
> - `internal/deploymanifest/deploymanifest_test.go` (modified WIP) references
>   `d.Power.KeepAwake` / `m.Power` (lines 301-358) — `Manifest has no field or
>   method Power`.
> - `git log --all -S "DefaultKeepAwakeServe"` and `-S "KeepAwake" --
>   internal/deploymanifest/` both return empty: the implementation was never
>   written (or was lost pre-history).
> - `go vet ./internal/deploymanifest ./cmd/fak` fails to compile:
>   `d.Power undefined` and `undefined: DefaultKeepAwakeServe`.
> - WIP preserved at
>   `docs/tickets/quarantine/keepawake-power-orphan-2026-09-11/` (`.orphan`
>   copies + this README); tracked tests reverted to HEAD to restore a
>   compiling tree.
>
> **Evidence commands**
> - `git log --all --oneline -S "DefaultKeepAwakeServe"`
> - `git grep -n "Power" -- internal/deploymanifest/`
> - `go vet ./internal/deploymanifest ./cmd/fak`
>
> **Acceptance criteria**
> - `cmd/fak` defines `DefaultKeepAwakeServe` (always-on default) and
>   `KeepAwakeAlways`; `applyServeManifestDefaults` applies manifest
>   `[power] keep_awake` pre-parse, explicit flag wins.
> - `internal/deploymanifest` `Defaults().Power.KeepAwake == "always"`;
>   `Parse` accepts `off|while-active|always`, records `Present(power,
>   keep_awake)`, refuses unknown values with `ReasonBadValue` locus power/
>   keep_awake; `Value` and `Descriptors()` project `power.keep_awake`.
> - `.orphan` files restored over tracked tests.
> - `go test ./cmd/fak/... ./internal/deploymanifest/... -count=1` compiles and
>   passes.