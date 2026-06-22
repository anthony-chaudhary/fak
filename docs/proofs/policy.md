---
title: "fak proof: policy manifest capability floor"
description: "Soundness proof for fak's policy loader: the effective deny set is a superset of the declared floor, and dump-edit-load round-trips deterministically."
---

# D2 · policy

The `policy` package is the deployable form of fak's "permissions as the floor"
thesis: it loads the adjudicator's capability floor from a declarative, version-tagged
JSON manifest (`internal/policy/policy.go`) instead of a compiled-in Go literal, so an
operator configures *which* tools an agent may call by editing a reviewable file rather
than forking the kernel. The manifest maps 1:1 onto `adjudicator.Policy`: an `allow`/
`allow_prefix` carves affirmative exceptions, a `deny` map names an explicit refusal by
its stable closed-vocabulary reason, `arg_rules` add per-argument value constraints, and
everything unnamed resolves to the fail-closed `DEFAULT_DENY`. "Correct" for this module
is a **regime-D decision-procedure-soundness** property in two parts: (1) the load is
*sound* — it can only translate or tighten the declared floor, never silently loosen it
below what the manifest declares; and (2) the load is *deterministic and round-trips* —
the same bytes always yield the same policy, and a policy dumped to a manifest and read
back reconstructs identically (the `--dump` → edit → `--load` cycle an adopter relies on
must not silently drop a baked-in protection).

All witnesses below ran natively on this macOS node (go1.26 darwin/arm64); on the Windows
host run them through WSL via `.\fak\test.ps1`.

---

## Theorem 1 — the loaded capability floor is SOUND (effective deny ⊇ declared floor)

**THEOREM.** The capability floor loaded from a manifest is sound: the effective deny set
of the resolved `adjudicator.Policy` is a **superset** of the floor the manifest declares.
A declared deny is preserved as an effective `DENY` carrying its cited closed-vocabulary
reason; the default-deny posture is preserved so any tool not affirmatively allowed
resolves to `DEFAULT_DENY`; and a manifest that would silently *drop* a declared deny
(unknown reason name, unknown/misspelled field) **fails to load** rather than loosening the
floor. Loading can only tighten (add denies / arg-restrictions), never loosen below the
declared floor.

**REGIME.** D — decision-procedure soundness (the verdict never admits what the floor
forbids; the loader fails closed against silent widening).

**PROOF.** The loader maps the manifest 1:1 onto `adjudicator.Policy` and structurally
refuses to weaken it:

- Each manifest deny entry is copied into `p.Deny` with its reason resolved via
  `abi.ReasonByName` (`internal/policy/policy.go:204`,`:213`); the adjudicator then
  surfaces that as the effective verdict — witnessed below.
- The default-deny floor is **structural, not configured**: `PostureFailClosed = iota` is
  the zero value (`internal/adjudicator/decide.go:69`), so any tool absent from `Allow`/
  `AllowPrefix` resolves to `DEFAULT_DENY`. The floor *is* "deny"; explicit allows carve
  exceptions, never the reverse.
- The loader is **fail-loud against silent loosening**: an unknown deny reason aborts the
  load, listing the offending entries and the valid vocabulary
  (`internal/policy/policy.go:215`), and `DisallowUnknownFields`
  (`internal/policy/policy.go:164`) turns a misspelled `"allows"` for `"allow"` into a hard
  error rather than a dropped-and-ignored key that would have widened the floor.
- Arg rules only **restrict**: `ArgAllowGlob` is a positive containment requirement that
  fails closed on a missing or escaping arg (`internal/policy/policy.go:88`,
  `internal/adjudicator/decide.go:83`).

The witnesses pin these for the empty floor, an explicit load-bearing floor, the shipped
dogfood floor, and the malformed-manifest cases. `TestLoadedPolicyIsLoadBearing` asserts
declared deny `exfiltrate` → `DENY/SECRET_EXFIL` (reason preserved) and unlisted
`delete_account` → `DENY/DEFAULT_DENY` (floor preserved). `TestEmptyManifestIsFailClosed`
asserts an empty manifest denies *everything*. `TestUnknownDenyReasonRejected` /
`TestUnknownFieldRejected` assert the loosening-via-typo paths are rejected at load.
`TestArgRulesAreLoadBearing` asserts `./out/../secret.txt` → `DENY/POLICY_BLOCK` and
`git push --force` → `DENY`. `TestDogfoodManifestVerdictMatrix` (in the adjudicator package,
loading `examples/dogfood-claude-policy.json` through `policy.Parse`) locks the full deny
matrix (rm -rf, sudo, fork bomb, dd, git push → DENY; `.git/` and `internal/kernel/` edits →
`SELF_MODIFY`; unknown tool → `DEFAULT_DENY`) so a manifest edit that silently widens the
floor fails the suite.

**Honest residual.** The witnesses prove soundness for *representative* manifests. The
strong **universally-quantified** form — "∀ manifest M, ∀ t ∈ M.Deny, Parse(M) denies t" —
is **not** generator-witnessed; there is no `testing/quick` property over arbitrary deny
maps + allow sets. That stronger form is OPEN and would be closed by a property test that
generates random `(allow, deny, arg_rules)` triples and asserts the deny/default-deny
verdicts survive load.

**WITNESS.**
```
go test ./internal/policy/ -count=1 -timeout 120s \
  -run 'TestLoadedPolicyIsLoadBearing|TestEmptyManifestIsFailClosed|TestUnknownDenyReasonRejected|TestUnknownFieldRejected|TestArgRulesAreLoadBearing|TestArgRuleValidation' -v
go test ./internal/adjudicator/ -count=1 -run TestDogfoodManifestVerdictMatrix -v
```

**VERDICT.** PROVEN (representative-instance soundness) — 2026-06-20, all listed tests PASS
on go1.26 darwin/arm64. The ∀-manifest superset property is noted OPEN above.

**DOS.** bound at ship.

---

## Theorem 2 — policy load is deterministic and round-trips

**THEOREM.** For any `Policy` p built from a manifest, `FromPolicy(p).ToPolicy() == p`, and
the full JSON byte path `Parse(FromPolicy(p).JSON()) == p`, both under `reflect.DeepEqual`.
Version gating and posture parsing are total, deterministic functions of the input, and the
deny-reason name↔code map is a bijection (`ReasonByName` is the exact inverse of
`ReasonName` over the closed vocabulary).

**REGIME.** D — decision-procedure soundness (the `--dump`→edit→`--load` cycle is a
loss-free, deterministic involution; no protection is silently dropped or reordered).

**PROOF.** Round-trip is structural: `FromPolicy` (`internal/policy/policy.go:274`) and
`ToRuntime` (`internal/policy/policy.go:184`) are field-wise inverse maps — slices cloned,
the `Allow` set marshalled to a **sorted** slice (`internal/policy/policy.go:289`) so Go
map-iteration nondeterminism cannot perturb the dump, and deny codes rendered to names by
`abi.ReasonName` then resolved back by `abi.ReasonByName` (`internal/abi/reasons.go:78`),
which `TestReasonByNameInverse` pins as an exact bijection over `abi.ReasonNames()`.
Determinism of the gates: `validateVersion` (`internal/policy/policy.go:247`) and
`parsePosture` (`internal/policy/policy.go:258`) are pure total switches with no
clock/RNG/network; `ParseManifest` decodes with `DisallowUnknownFields`, so the same bytes
always yield the same `Manifest` or the same error.

`TestRoundTrip` asserts `FromPolicy(DefaultPolicy()).ToPolicy()` is `reflect.DeepEqual` to
the original (a dropped or reordered field would fail — it is a whole-struct compare, not a
spot-check). `TestParseFromDumpBytes` asserts the same through the marshalled JSON bytes.
`TestAdmitAndLogPostureLoadsAndRoundTrips` confirms the non-default posture survives the
round-trip and still drives the adjudicator. `TestVersionGating` (5 subtests) and
`TestUnknownPostureRejected` confirm the gates are total and reject the out-of-domain
inputs. The dump's only nondeterminism risk — map iteration over the 10-entry `Allow` map —
is eliminated by the explicit sort and exercised by the `DefaultPolicy` round-trip.

**WITNESS.**
```
go test ./internal/policy/ -count=1 -timeout 120s \
  -run 'TestRoundTrip|TestParseFromDumpBytes|TestVersionGating|TestAdmitAndLogPostureLoadsAndRoundTrips|TestReasonByNameInverse|TestUnknownPostureRejected' -v
```

**VERDICT.** PROVEN — 2026-06-20, all listed tests PASS on go1.26 darwin/arm64.

**DOS.** bound at ship.

---

## Reproduce

```
go test ./internal/policy/ -count=1 -timeout 120s          # whole package, green
go test ./internal/adjudicator/ -count=1 -run TestDogfoodManifestVerdictMatrix
```
