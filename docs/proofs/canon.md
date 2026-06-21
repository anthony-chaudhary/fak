# D5 · canon

> **Update — witness pass (2026-06-20, commit `3cb8ff9`).** 1 OPEN obligation(s) below were CLOSED to ✅ PROVEN by new deterministic tests added in `internal/canon/proofs_witness_test.go`. The body keeps the original analysis (the gap **and** the 'to close' plan that was then executed); the **current verdict is in the [master ledger](README.md)** and the executed closures are listed in *Closures* at the foot of this file.

`internal/canon` is the de-obfuscating canonicalizer and lexical threat detector
factored out of `normgate`, shared by the write-time gate (`normgate`) and the
read-time re-screen of a reloaded core image (`recall`). It canonicalizes a *copy*
of a body into several views — `Normalize` (drop invisibles, fold fullwidth and
ideographic space, map homoglyphs to ASCII), `Decoded` (base64/hex of long tokens),
`reverseRunes` (bidi), `squeezeAlnum` (de-separation) — and runs the secret/injection
detectors over those views, so an obfuscation that defeats a raw substring matcher is
undone before the match. It is a pure stdlib leaf, holds no state, and makes no policy
decision; it answers only the factual question "do the canonical views reveal a secret
or an injection marker?" (regime **D — decision-procedure soundness**: the canonical
projection and the fold over views must be well-defined and fail toward *catching*, not
missing, a hidden marker). "Correct" here has two parts: (1) the canonicalization is a
**normal form** — re-canonicalizing a canonical body is a no-op (idempotent), so a body
re-screened on reload yields a stable view; and (2) the canonical form **actually folds
the obfuscation classes it advertises**, so each named evasion family is caught.

---

## THEOREM 1 — canonicalization is idempotent (a normal form)

**THEOREM.** For all input strings `x`, `Normalize(Normalize(x)) == Normalize(x)`.
`Normalize` is a projection onto a canonical form; re-canonicalizing a canonical body
is a no-op. (Equivalently for the decision: `Scan` over a `Normalize`d body reveals the
same injection/secret findings as over the canonical input — the canonical view is a
fixed point under the canonicalizer.)

**REGIME.** D — decision-procedure soundness (normal-form / idempotence invariant).

**PROOF.** `Normalize` (`internal/canon/canon.go:98-115`) rewrites a body rune-by-rune:
invisibles are dropped (`isInvisible`, `canon.go:77-93`), fullwidth ASCII
`0xFF01..0xFF5E` is mapped down by `0xFEE0` (`canon.go:105-106`), ideographic space
`0x3000 → ' '` (`canon.go:107-108`), and homoglyphs are mapped to ASCII via the
`homoglyphs` table (`canon.go:67-73`, `canon.go:109-110`). Every output rune for a
transformed class is plain ASCII, and plain ASCII is **not** a key of `homoglyphs`, is
**not** in the fullwidth range, and is **not** invisible — so the canonicalized body is
a fixed point, hence the map is idempotent. This is the *argument*; it is not yet a
committed witness, so the theorem stays OPEN (see below).

**WITNESS.** `(go test ./internal/canon/ -count=1 -timeout 120s -run
'TestNormalizeIdempotent')` — **no such committed test exists yet.** A throwaway probe
(written, run, then deleted; not committed) confirmed the property empirically: a
fixed-point sweep over every `homoglyphs` key, the full `0xFF01..0xFF5E` fullwidth range,
and `0x3000/0x200B/0x202E/0xFEFF`, plus a `testing/quick` property `Normalize(Normalize(s))
== Normalize(s)` at `MaxCount=20000`, all PASS (0.51s). A `grep` for
`idempot|Normalize(Normalize|Scan(.*Scan` over the package returned nothing.

**VERDICT.** **OPEN** — 2026-06-20. The property is true-looking and empirically
corroborated, but no shipped test in the package asserts it; it is not promoted to
PROVEN on argument (or on a throwaway probe) alone. *Closing witness:* commit a
`TestNormalizeIdempotent` that asserts `Normalize(Normalize(s)) == Normalize(s)` via
`testing/quick` over random strings **and** sweeps the fixed point over every transformed
codepoint class (homoglyph keys, the fullwidth range, ideographic space, the invisible
set).

**DOS.** bound at ship.

---

## THEOREM 2 — the canonical form folds the obfuscation classes it claims

**THEOREM.** The canonical view folds each obfuscation class canon advertises —
homoglyph (Cyrillic/Greek look-alikes), fullwidth ASCII, zero-width / invisible splice,
bidi reversal, and character-spacing — so an injection marker hidden by *any* of these
classes is revealed (caught) on the canonical / decoded / reversed / squeezed views.

**REGIME.** D — decision-procedure soundness (the fold over views must catch, not miss).

**PROOF.** `Scan` (`internal/canon/canon.go:177-223`) builds `norm = Normalize(raw)`,
`dec`, and `rev = reverseRunes(norm)`, then matches `InjectionMarkers` over the
lower-cased views and `DistinctiveSqueezed` over the `squeezeAlnum` views. Each claimed
class is undone by a specific mechanism before the match: **homoglyph** by the
`homoglyphs` map (`canon.go:67-73`, applied in `Normalize`); **fullwidth** by the
`0xFF01..0xFF5E -= 0xFEE0` fold (`canon.go:105-106`); **zero-width/invisible** by the
`isInvisible` drop (`canon.go:77-93`, applied at `canon.go:102`); **bidi** by
`reverseRunes` (`canon.go:117-123`) feeding the reversed view (`canon.go:197`,
`canon.go:210`); **character-spacing** by `squeezeAlnum` (`canon.go:127-135`) plus the
de-separation squeeze pass (`canon.go:208-221`).

**WITNESS.** `(go test ./internal/canon/ -count=1 -timeout 120s -run
'TestObfuscatedInjectionCaught|TestNormalizeUndoesObfuscation' -v)`.
`TestObfuscatedInjectionCaught` (`internal/canon/canon_test.go:13-29`) enumerates one
case per claimed class — `homoglyph`, `fullwidth`, `zero-width`, `char-spacing`,
`squeeze-bidi` (plus `base64`, `exfil-marker`) — and asserts `Scan(body).Injection ==
true` for each. `TestNormalizeUndoesObfuscation` (`canon_test.go:72-77`) asserts directly
that `strings.ToLower(Normalize(homoglyph))` contains the clean ASCII marker
`ignore previous instructions`.

**VERDICT.** **PROVEN** — 2026-06-20. Ran green on the macOS native go1.26 toolchain:
`--- PASS: TestObfuscatedInjectionCaught (0.00s)` and
`--- PASS: TestNormalizeUndoesObfuscation (0.00s)`; `ok
github.com/anthony-chaudhary/fak/internal/canon 0.234s`. Each named obfuscation class has a
case that asserts the marker is caught, and the unit test confirms the homoglyph fold
yields the clean ASCII marker.

**DOS.** bound at ship.

---

## Closures (witness pass 2026-06-20, commit `3cb8ff9`)

Each obligation marked OPEN above was discharged by a new zero-dependency (stdlib `testing`/`testing/quick`) metamorphic/round-trip/invariant test that ASSERTS the property against an independently recomputed reference. Verified by `go test -count=1 ./internal/...` (45 packages green, 0 failures).

- **canonicalization-idempotent** → ✅ PROVEN by `TestNormalizeIdempotent_Deterministic`. Normalize(Normalize(x))==Normalize(x) holds. Found mechanism in canon.go: Normalize (exported, func Normalize(s string) string) is a per-rune projection -- it DROPS invisibles (isInvisible), folds fullwidth-ASCII [0xFF01..0xFF5E] by 0xFEE0 into plain ASCII [0x21..0x7E], maps ideographic space 0x3000 -> ' ', maps homoglyph keys (all non-ASCII) to plain ASCII targets, else passes through. Idempotence = every OUTPUT rune is a fixed point: dropped invisibles can't re-drop; the ASCII fold-image is disjoint from the fullwidth block, from 0x3000, and from the non-ASCII homoglyph keys; homoglyph targets are ASCII letters that are not themselves homoglyph keys (no chaining). Three independent deterministic witnesses, all PASS: (1) TestNormalizeIdempotent_Deterministic -- 8 hand-picked adversarial inputs exercising each branch (homoglyph/fullwidth/zero-width+BOM/ideographic-space/bidi/passthrough) plus a 5000-string randomized battery over an alphabet that is exactly the special runes (every homoglyph key, fullwidth block, the drop-set, special spaces), fixed seed 0xCA0, exact byte equality; (2) TestNormalizeIdempotent_Quick -- testing/quick, MaxCount=20000, fixed seed 20260620, over arbitrary Go strings; (3) TestNormalizeOutputIsFixedPoint -- directly witnesses image==fixed-point set (Normalize(out)==out for every produced out), seed 7. No counterexample found across ~28000 cases.
