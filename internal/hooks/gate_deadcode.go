package hooks

import (
	"regexp"
	"sort"
	"strings"
)

// gate_deadcode.go — the whole-tree gate that catches a NEW dead unexported symbol before it
// reaches the shared trunk. tools/code_slop_scorecard.py's kpi_dead_code already scores an
// unexported top-level symbol that is defined but referenced nowhere else, but that scorecard
// only runs post-hoc (the `demo-scorecards` CI step / a manual /slop-score pass) — so on a hot,
// many-session trunk a dead helper lands, and only minutes later does the scorecard notice, by
// which point the release-gating fast subset can already be red on CI_BASE_RED and a human has
// to chase it. This gate surfaces the same dead-code defect one boundary earlier, in
// `fak hygiene` (the pre-push --audit-tree backstop), so a contributor sees it BEFORE the
// trunk accretes the dead weight (the slop-PREVENTION twin of gate_pythongate.go).
//
// It is a HARD, deterministic, low-false-positive slop axis — unlike duplication, which the
// scorecard itself declines to gate ("scores but never gates" — a clone detector's group
// verdicts are too false-positive-prone for a commit boundary). This gate ports ONLY the
// dead_code KPI, byte-faithfully to the Python oracle, and does NOT import the tier-2
// scorecard package (an upward import a tier-1 hooks package may not make) — it re-derives the
// verdict from the tracked tree the same way, so it can never become a rival authority to the
// scorecard it fronts. A `parity_test.go` case asserts the two agree on canned trees.

// deadCodeGate is the DEAD_CODE gate name (also the Finding.Gate value).
const deadCodeGate = "DEAD_CODE"

// deadCapPerFile mirrors code_slop_scorecard.py DEAD_CAP_PER_FILE — at most this many dead
// symbols are reported per file, so one pathological file cannot flood the finding list (and
// the count matches the scorecard's capped debt exactly).
const deadCapPerFile = 5

// deadCodeIdentRE matches a Go identifier — the token the frequency scan counts. Same class as
// the Python _IDENT_RE ([A-Za-z_]\w*).
var deadCodeIdentRE = regexp.MustCompile(`[A-Za-z_]\w*`)

// deadCodeAsmSymRE matches the middle-dot assembly reference to a Go symbol (`·name`), so a
// symbol whose only caller is a hand-written .s file (a SIMD kernel body, an asm-read gate
// flag) is counted LIVE, never mis-flagged as dead. Mirrors the Python _ASM_SYM_RE.
var deadCodeAsmSymRE = regexp.MustCompile(`·(\w+)`)

// deadCodeFuncDeclRE / deadCodeTypeDeclRE / deadCodeVarConstDeclRE match a top-level
// declaration and capture its name — the symbol the gate grades. Byte-faithful to the Python
// _FUNC_DECL_RE / _TYPE_DECL_RE / _VARCONST_DECL_RE (matched against the code-only, left-trimmed
// line). The func pattern skips an optional receiver group so a method's name is captured.
var (
	deadCodeFuncDeclRE     = regexp.MustCompile(`^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*[\(\[]`)
	deadCodeTypeDeclRE     = regexp.MustCompile(`^type\s+([A-Za-z_]\w*)\b`)
	deadCodeVarConstDeclRE = regexp.MustCompile(`^(?:var|const)\s+([A-Za-z_]\w*)\b`)
)

// deadCodeExcludeDirs mirrors code_slop_scorecard.py GO_EXCLUDE_DIRS: path segments that mark a
// non-first-party or scratch/copy subtree the scorecard never grades. A tracked .go under any of
// these (a testdata fixture, a vendored dep) declares no kernel symbol and contributes no real
// reference, so excluding it keeps the gate's verdict identical to the scorecard's. (git ls-files
// already drops the untracked scratch checkouts; this drops the TRACKED excluded dirs too.)
var deadCodeExcludeDirs = map[string]bool{
	".git": true, ".claude": true, ".fak": true, ".dos": true, ".tmp": true,
	"node_modules": true, "testdata": true, "vendor": true, "__pycache__": true,
}

// deadCodeExcluded reports whether a repo-relative path lies under an excluded dir — the gate's
// twin of the scorecard's _excluded_go (minus the corrupt-path rune check, which git ls-files
// has already filtered out of the tracked set TrackedTree carries).
func deadCodeExcluded(rel string) bool {
	for _, seg := range strings.Split(rel, "/") {
		if deadCodeExcludeDirs[seg] {
			return true
		}
	}
	return false
}

// deadCodeKeepRE matches the `//slop:keep` author opt-out directive on the line immediately
// above a declaration — an explicit, non-gameable statement that an unexported symbol is
// intentionally unreferenced (a symbol-table provenance marker, a contract const checked only
// by an out-of-tree witness). Keyed on RAW source (a directive, not prose). Mirrors the Python
// _SLOP_KEEP_RE.
var deadCodeKeepRE = regexp.MustCompile(`^\s*//\s*slop:keep\b`)

// codeOnlyLines reduces each source line to its code-only form — string/rune literals and
// comments blanked, cross-line raw-string (backtick) and block-comment (/* */) spans tracked —
// so an identifier that appears only inside a string or comment is NOT counted as a reference.
// Index-aligned to the input lines. A byte-faithful port of code_slop_scorecard.py
// code_lines_of / _code_only.
func codeOnlyLines(text string) []string {
	lines := strings.Split(text, "\n")
	out := make([]string, 0, len(lines))
	inRaw, inBlock := false, false
	for _, raw := range lines {
		var code string
		code, inRaw, inBlock = codeOnly(raw, inRaw, inBlock)
		out = append(out, code)
	}
	return out
}

// codeOnly blanks the literals/comments of one line and advances the cross-line raw/block
// spans. Returns (codeOnlyText, inRaw, inBlock). Faithful to the Python _code_only: on entering
// a raw string or block comment the opening delimiter is NOT emitted, and the span content is
// dropped, so only real code characters reach the identifier scan.
func codeOnly(line string, inRaw, inBlock bool) (string, bool, bool) {
	var b strings.Builder
	rs := []rune(line)
	n := len(rs)
	for i := 0; i < n; {
		c := rs[i]
		switch {
		case inBlock:
			if c == '*' && i+1 < n && rs[i+1] == '/' {
				inBlock = false
				i += 2
				continue
			}
			i++
			continue
		case inRaw:
			if c == '`' {
				inRaw = false
			}
			i++
			continue
		}
		if c == '/' && i+1 < n && rs[i+1] == '/' {
			break // rest of the line is a // comment
		}
		if c == '/' && i+1 < n && rs[i+1] == '*' {
			inBlock = true
			i += 2
			continue
		}
		if c == '`' {
			inRaw = true
			i++
			continue
		}
		if c == '"' || c == '\'' {
			quote := c
			i++
			for i < n {
				if rs[i] == '\\' {
					i += 2
					continue
				}
				if rs[i] == quote {
					i++
					break
				}
				i++
			}
			continue
		}
		b.WriteRune(c)
		i++
	}
	return b.String(), inRaw, inBlock
}

// declName returns the captured symbol name if the (already code-only, left-trimmed) line is a
// top-level func / type / var / const declaration, or "" otherwise. Tries the three declaration
// patterns in the same order as the Python.
func declName(s string) string {
	for _, rx := range []*regexp.Regexp{deadCodeFuncDeclRE, deadCodeTypeDeclRE, deadCodeVarConstDeclRE} {
		if m := rx.FindStringSubmatch(s); m != nil {
			return m[1]
		}
	}
	return ""
}

// gateDeadCodeTree emits a DEAD_CODE finding for every UNEXPORTED top-level symbol in a tracked
// non-test .go file whose identifier appears exactly once across the whole first-party module
// (its own definition — referenced nowhere else), honoring the `//slop:keep` opt-out and the
// per-file cap. Test files (_test.go) and assembly (.s) contribute references but declare no
// graded symbols, exactly like the scorecard. Exported names, `_`, `init`, and `main` are
// never graded (external callers / runtime entry points a static scan cannot see).
//
// This gate does not fail open on an unreadable source the way the parse-a-baseline gates do:
// its "source of truth" is the tracked .go tree itself, which ReadTrackedTree already delivered
// — so there is nothing separate that can go missing. A tree with no .go files simply yields
// zero findings.
func gateDeadCodeTree(t *TrackedTree) ([]Finding, error) {
	// Partition the tracked tree into the three corpora the scorecard uses: shipped .go
	// (graded + referencing), _test.go (referencing only), and .s assembly (referencing only).
	var shipped, tests []string
	var asm []string
	for _, p := range t.Paths {
		if deadCodeExcluded(p) {
			continue
		}
		switch {
		case strings.HasSuffix(p, "_test.go"):
			tests = append(tests, p)
		case strings.HasSuffix(p, ".go"):
			shipped = append(shipped, p)
		case strings.HasSuffix(p, ".s"):
			asm = append(asm, p)
		}
	}

	// Token frequency across the WHOLE module (shipped + tests), code-only so a token inside a
	// string or comment is not a phantom reference. Assembly unions in both the bare-ident and
	// the ·name(SB) forms (over-counting there can only mark a symbol LIVE, never wrongly dead).
	freq := map[string]int{}
	countRefs := func(rel string) {
		body, ok := t.FileBytes(rel)
		if !ok {
			return
		}
		for _, c := range codeOnlyLines(string(body)) {
			for _, tok := range deadCodeIdentRE.FindAllString(c, -1) {
				freq[tok]++
			}
		}
	}
	for _, p := range shipped {
		countRefs(p)
	}
	for _, p := range tests {
		countRefs(p)
	}
	for _, p := range asm {
		body, ok := t.FileBytes(p)
		if !ok {
			continue
		}
		text := string(body)
		for _, tok := range deadCodeIdentRE.FindAllString(text, -1) {
			freq[tok]++
		}
		for _, m := range deadCodeAsmSymRE.FindAllStringSubmatch(text, -1) {
			freq[m[1]]++
		}
	}

	// Only non-test shipped files declare the symbols we grade. Iterate in sorted order so the
	// per-file cap and the finding order are deterministic (Paths is already sorted, but be
	// explicit).
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
			name := declName(strings.TrimLeft(line, " \t"))
			if name == "" {
				continue
			}
			if name == "_" || isExportedName(name) {
				continue // exported (API) or blank — a static scan can't see external callers
			}
			if name == "init" || name == "main" {
				continue // runtime entry points, never "referenced"
			}
			if freq[name] > 1 {
				continue // referenced somewhere beyond its own definition — live
			}
			// The author opt-out: the line immediately above the declaration is `//slop:keep`.
			if idx >= 1 && idx-1 < len(raw) && deadCodeKeepRE.MatchString(raw[idx-1]) {
				continue
			}
			if perFile[rel] >= deadCapPerFile {
				continue
			}
			perFile[rel]++
			findings = append(findings, Finding{
				Gate: deadCodeGate,
				File: rel,
				Detail: rel + " :: " + name + " is a dead unexported symbol (defined, never referenced) — " +
					"delete it, or wire it to its intended caller. If it is intentionally unreferenced " +
					"(a symbol-table provenance marker, a const checked only by an out-of-tree witness), " +
					"put `//slop:keep <reason>` on the line immediately above its declaration. Same verdict " +
					"tools/code_slop_scorecard.py kpi_dead_code computes, one boundary earlier (" + deadCodeGate + ").",
			})
		}
	}
	sort.Slice(findings, func(i, j int) bool {
		if findings[i].File != findings[j].File {
			return findings[i].File < findings[j].File
		}
		return findings[i].Detail < findings[j].Detail
	})
	return findings, nil
}

// isExportedName reports whether a Go identifier is exported (first rune is an uppercase
// letter). Matches the Python `name[0].isupper()` test for the ASCII identifier class the
// declaration regexes admit.
func isExportedName(name string) bool {
	if name == "" {
		return false
	}
	c := name[0]
	return c >= 'A' && c <= 'Z'
}
