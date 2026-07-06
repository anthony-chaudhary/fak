package hooks

import (
	"regexp"
	"sort"
	"strings"
)

// gate_swallowederror.go — issue #2899 (hermes-inspiration epic #2871): the whole-tree gate
// that catches a NEW silently-swallowed error before it reaches the shared trunk. Hermes has
// ~3,656 bare `except Exception:` / `except: pass` sites across its Python — each a place an
// error can vanish, including near credential refresh, subprocess, and file writes. Silent
// failure is the opposite of a witnessing kernel: fak's whole ethos is "report incomplete as
// `not yet` with evidence, never silently." This gate is the Go-tree floor for that ethos.
//
// The Go analogue of `except: pass` is the explicit error discard — assigning a call's result
// (which carries an `error`) to the blank identifier so it is thrown away un-witnessed:
//
//	_ = f.Close()          // the Close error vanishes
//	_ = os.Remove(tmp)     // the Remove error vanishes
//
// Unlike an UNCHECKED error (`f.Close()` with no assignment — a compile-time-visible naked call
// that `errcheck`/`go vet` families already grade), the `_ =` discard is a DELIBERATE, in-source
// silencing: the author typed `_` to make the compiler stop complaining. That is exactly the
// witnessing gap this gate names — a caught error must be logged/recorded with context or
// re-raised, never assigned to `_` with no witness. This gate grades that single, high-signal,
// low-false-positive idiom; the broader unchecked-return sweep stays the domain of the standard
// vet/errcheck pass the issue also names (run in `make ci`'s `go vet` step).
//
// Author opt-out: a `//nolint:errdiscard <reason>` directive on the SAME line as the discard (or
// the line immediately above it) marks a discard that is intentional and reviewed — a Close on a
// read-only handle, a best-effort cleanup whose failure is genuinely irrelevant. Keyed on RAW
// source (a directive, not prose), mirroring gate_deadcode.go's `//slop:keep` precedent.
//
// It ships DefaultOff, exactly like DEAD_CODE: the tree still carries pre-existing `_ =` error
// discards (~hundreds) that predate the floor, so wiring it always-on would red `make ci` for
// the whole fleet against known, not-yet-witnessed debt. It is the always-available audit sweep
// (`fak hygiene --gates SWALLOWED_ERROR`) that proves the retirement, and flips DefaultOff:false
// — the enforcement gate that HOLDS the line at zero un-witnessed discards — once the tree is
// clean.

// swallowedErrorGate is the SWALLOWED_ERROR gate name (also the Finding.Gate value).
const swallowedErrorGate = "SWALLOWED_ERROR"

// swallowedCapPerFile caps how many discards are reported per file so one pathological file
// cannot flood the finding list, mirroring gate_deadcode.go's deadCapPerFile.
const swallowedCapPerFile = 20

// swallowedDiscardRE matches a code-only line whose whole statement is `_ = <call>(...)`: the
// blank identifier, a single `=` (not `==`, `:=`, `<=`, `>=`, `!=`, or `+=` &c.), then an
// expression that ends in a call — the token immediately preceding the FINAL `(` is a selector
// or identifier, i.e. a function/method invocation whose (possibly error-typed) result is thrown
// away. Matched against the code-only line (comments/strings already blanked by codeOnlyLines),
// so a `_ =` inside a string literal or comment never matches. The trailing `.*\(` is greedy so
// a chained expression `_ = a.B().C()` still ends in a call.
//
// Restricting to the `_ = <call>()` form (a call result discarded) is what keeps the gate
// low-false-positive: `_ = someVar` (discarding a plain value, no error to lose) and multi-assign
// forms like `x, _ := f()` (the error position is a NAMED throwaway the errcheck family already
// covers, and the value is used) do NOT match.
var swallowedDiscardRE = regexp.MustCompile(`^_\s+=\s+.*\w[\w.]*\(`)

// swallowedNolintRE matches the `//nolint:errdiscard` author opt-out on the discard's own line or
// the line immediately above it. Keyed on RAW source (the directive text), so a discard that is
// intentional and reviewed is exempted with an explicit, non-gameable marker + reason.
var swallowedNolintRE = regexp.MustCompile(`//\s*nolint:errdiscard\b`)

// swallowedExcludeDirs mirrors gate_deadcode.go's deadCodeExcludeDirs: path segments that mark a
// non-first-party or scratch/copy subtree the floor never grades (a testdata fixture, a vendored
// dep). A discard there is not fak's kernel code.
var swallowedExcludeDirs = map[string]bool{
	".git": true, ".claude": true, ".fak": true, ".dos": true, ".tmp": true,
	"node_modules": true, "testdata": true, "vendor": true, "__pycache__": true,
}

// swallowedExcluded reports whether a repo-relative path lies under an excluded dir.
func swallowedExcluded(rel string) bool {
	for _, seg := range strings.Split(rel, "/") {
		if swallowedExcludeDirs[seg] {
			return true
		}
	}
	return false
}

// gateSwallowedErrorTree emits a SWALLOWED_ERROR finding for every `_ = <call>(...)` explicit
// error discard in a tracked non-test .go file, honoring the `//nolint:errdiscard` opt-out and
// the per-file cap. Test files (_test.go) are NOT graded: a discarded error in a test is a
// setup/teardown convenience, not a kernel witnessing gap (matching how gate_deadcode.go treats
// _test.go as reference-only, non-graded). The gate reads the code-only projection of each file
// so a `_ =` inside a string or comment is never a phantom match.
func gateSwallowedErrorTree(t *TrackedTree) ([]Finding, error) {
	var shipped []string
	for _, p := range t.Paths {
		if swallowedExcluded(p) {
			continue
		}
		if strings.HasSuffix(p, "_test.go") {
			continue // tests contribute no graded discard (setup/teardown convenience)
		}
		if strings.HasSuffix(p, ".go") {
			shipped = append(shipped, p)
		}
	}
	sort.Strings(shipped)

	var findings []Finding
	perFile := map[string]int{}
	for _, rel := range shipped {
		body, ok := t.FileBytes(rel)
		if !ok {
			continue
		}
		code := codeOnlyLines(string(body))
		raw := strings.Split(string(body), "\n")
		for idx, line := range code {
			trimmed := strings.TrimLeft(line, " \t")
			if !swallowedDiscardRE.MatchString(trimmed) {
				continue
			}
			// Author opt-out: `//nolint:errdiscard` on the discard's own raw line or the line
			// immediately above it (a comment-only line explaining the intentional discard).
			if idx < len(raw) && swallowedNolintRE.MatchString(raw[idx]) {
				continue
			}
			if idx >= 1 && idx-1 < len(raw) && swallowedNolintRE.MatchString(raw[idx-1]) {
				continue
			}
			if perFile[rel] >= swallowedCapPerFile {
				continue
			}
			perFile[rel]++
			findings = append(findings, Finding{
				Gate: swallowedErrorGate,
				File: rel,
				Line: idx + 1, // 1-based line number
				Detail: rel + ":" + itoa(int64(idx+1)) + " :: `_ = <call>()` discards a call result (any error) " +
					"un-witnessed — the Go analogue of `except: pass`. Witness the error (log/record it with " +
					"context, or return it), or if the discard is intentional and reviewed put " +
					"`//nolint:errdiscard <reason>` on its line or the line above. Issue #2899 (" + swallowedErrorGate + ").",
			})
		}
	}
	sort.Slice(findings, func(i, j int) bool {
		if findings[i].File != findings[j].File {
			return findings[i].File < findings[j].File
		}
		return findings[i].Line < findings[j].Line
	})
	return findings, nil
}
