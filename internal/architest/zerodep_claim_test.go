package architest

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// Stale "zero external dependencies / no go.sum" claim gate.
//
// THE DEFECT THIS EXISTS FOR. This module's dependency set went from zero modules to two
// (`golang.org/x/term` direct, `golang.org/x/sys` indirect) and grew a tracked 4-line
// `go.sum`. The old "zero external dependencies — there is no go.sum" line did not go with
// it. It survived in go.mod's own header comment, in INSTALL.md, in the Dockerfile, across a
// dozen docs pages, and in most of the unpublished launch copy — see the header of
// sbom_drift_test.go, which records that a build-verified triage caught the drift and named
// it, and docs/air-gapped-deployment-kit.md, which prints the corrected statement and says
// outright that the older phrasing is stale and must not be used. Correcting the prose again
// without a gate just resets the clock: the claim already rotted once, in the open, after
// someone had written down that it was rotten.
//
// A false supply-chain claim is worse than a missing one. A reviewer who reads "no go.sum"
// stops looking for a go.sum, and a stale slogan is exactly what an air-gapped or regulated
// adopter is invited to trust.
//
// WHY THIS LIVES IN architest AND NOT A NEW LEAF. Same reasoning as sbom_drift_test.go, and
// it still holds: this is a repo-hygiene contract over checked-in artifacts, stdlib-only,
// off the request path, never registered into the kernel. No new package leaf means no new
// architest tier row and the push gate's UNTIERED_LEAF check is untouched.
//
// SCOPE — WHAT IS AND IS NOT THE CORPUS. This gate reads the READER-FACING claim surface:
// the repo-root markdown, go.mod, the Dockerfile(s), and docs/**. Those are the pages that
// tell a reader what the artifact IS. Two corpora are deliberately outside it:
//
//   - Dated histories (docs/notes/**, docs/archive/**). A June note that described the tree
//     as it was in June is a record, not a false claim. Rewriting a record to make a checker
//     green is the falsification this repo forbids.
//   - Go source comments and examples/** prose. There, "zero-dependency" and
//     "dependency-free" are used constantly and TRUTHFULLY about one package, one test, or
//     one self-contained demo ("a dependency-free adapter that does not import crewai").
//     A gate that argues with true statements gets switched off, and a switched-off gate is
//     worth less than no gate. The stale claims that do live in Go comments are real and are
//     tracked as follow-on work; they are not laundered by being out of scope here.
//
// WHY THE RULE SET IS NARROW. Every rule below is a phrasing whose every occurrence in this
// corpus is a claim about the whole module. The adjectival forms — bare `zero-dependency`,
// `stdlib-only`, `dependency-free <adapter|example|proof>` — are NOT rules, because they are
// homographs: the same words, truthfully said about a smaller thing. Precision here is what
// makes the gate survivable; the cost is that a determined writer could smuggle the claim in
// through a form no rule names. That is the residual risk, and it is why the go.sum rule is
// unconditional: nearly every real occurrence of the stale claim pairs the dependency count
// with the go.sum line, and no context can make "there is no go.sum" true.

// claimKind is the closed vocabulary of stale claim this gate can report. The kind names the
// SHAPE of the false statement, because the cure differs: a go.sum-absence claim is repaired
// with a number (4 lines), a dependency-count claim with a module list.
type claimKind string

const (
	// claimGoSumAbsent asserts the repository has no go.sum. It is tracked, 4 lines, and
	// pins two modules. There is no context in which this claim is true, which is why this
	// rule carries no qualifier.
	claimGoSumAbsent claimKind = "GO_SUM_CLAIMED_ABSENT"
	// claimZeroExternal / claimNoExternal are the whole-module dependency-count claim in
	// its two polarities ("zero external dependencies" / "no external deps").
	claimZeroExternal claimKind = "ZERO_EXTERNAL_DEPENDENCIES"
	claimNoExternal   claimKind = "NO_EXTERNAL_DEPENDENCIES"
	// claimZeroDeps / claimNoDeps are the compressed marketing forms ("zero deps",
	// "no dependencies") that the launch copy reaches for.
	claimZeroDeps claimKind = "ZERO_DEPS"
	claimNoDeps   claimKind = "NO_DEPS"
	// claimDepFreeArtifact is "dependency-free" bound to the ARTIFACT noun (the binary, the
	// module, the repo, or fak itself). Unbound "dependency-free" is not matched — see the
	// homograph note in the header.
	claimDepFreeArtifact claimKind = "DEPENDENCY_FREE_ARTIFACT"
)

// staleClaimRule is one false phrasing. Why is the deliverable: a failure has to explain why
// the sentence is wrong, or the next fixer just reaches for a synonym.
type staleClaimRule struct {
	Kind claimKind
	Re   *regexp.Regexp
	Why  string
}

// theHonestStatement is the wording this repo already settled on, in
// docs/air-gapped-deployment-kit.md. Failures quote it so a fixer does not invent a
// thirteenth phrasing of the same fact.
const theHonestStatement = "fak is one static Go binary whose entire external dependency set is two " +
	"`golang.org/x` extended-standard-library modules (`x/term`, and `x/sys` indirectly through it), " +
	"pinned by a 4-line `go.sum`. Everything next to that claim is still true and must be kept: " +
	"one static pure-Go artifact, CGO_ENABLED=0, no Python, no PyTorch, no CUDA toolchain. " +
	"See docs/air-gapped-deployment-kit.md#supply-chain-posture"

var staleClaimRules = []staleClaimRule{
	{
		Kind: claimGoSumAbsent,
		Re:   regexp.MustCompile("(?i)(?:\\bno|\\bwithout(?:\\s+a)?|\\bzero)[\\s-]+`?go\\.sum"),
		Why:  "go.sum is tracked (`git ls-files go.sum`), 4 lines long, and pins golang.org/x/term v0.44.0 and golang.org/x/sys v0.46.0",
	},
	{
		Kind: claimZeroExternal,
		Re:   regexp.MustCompile(`(?i)zero[\s-]+external[\s-]+(?:go[\s-]+)?dep(?:s|endencies)\b`),
		Why:  "go.mod requires two external modules",
	},
	{
		Kind: claimNoExternal,
		Re:   regexp.MustCompile(`(?i)\bno[\s-]+external[\s-]+(?:go[\s-]+)?dep(?:s|endencies)\b`),
		Why:  "go.mod requires two external modules",
	},
	{
		Kind: claimZeroDeps,
		Re:   regexp.MustCompile(`(?i)\bzero[\s-]+dep(?:s|endencies)\b`),
		Why:  "the compressed form of the same false count; go.mod requires two external modules",
	},
	{
		Kind: claimNoDeps,
		Re:   regexp.MustCompile(`(?i)\bno[\s-]+dep(?:s|endencies)\b`),
		Why:  "the compressed form of the same false count; go.mod requires two external modules",
	},
	{
		Kind: claimDepFreeArtifact,
		Re: regexp.MustCompile(`(?i)\bfak\s+is\s+dependency-free\b` +
			`|\bdependency-free\s+(?:go\s+)?binary\b` +
			`|\bdependency-free\s+(?:module|repo|repository)\b`),
		Why: "\"dependency-free\" bound to the artifact is the same false count; it is only true of an individual example or package",
	},
}

// retractionCue is the ONE exemption that is not a path. A line may quote a stale phrasing
// if, on that same line, it says the phrasing is wrong — which is what
// docs/air-gapped-deployment-kit.md's "the older ... phrasing is stale — do not use it" does,
// and what any future page that wants to warn about the claim will have to do.
//
// WHY THIS IS NOT A LOOPHOLE. It is LINE-LOCAL: it cannot exempt a file, a section, or a
// paragraph, only the one line that carries the retraction, so it can never swallow the
// check. The cue vocabulary is closed and every member is an explicit retraction, so
// claiming the exemption means publishing, on the same line a reader sees, the sentence
// "this is stale" — you cannot quietly assert the claim and quietly take the exemption in
// one breath, because the two statements contradict each other in front of the reader.
// TestZeroDepClaimRetractionsArePinned then pins the exact per-file inventory of exempted
// lines, so a new exemption cannot appear without a reviewer seeing the number move.
var retractionCue = regexp.MustCompile(`(?i)\bstale\b|\bno longer true\b|\bnot true at this commit\b|\bwas true once\b|\bdo not use it\b`)

// zeroDepClaimDebt is the inventory of corpus files that STILL carry the stale claim and are
// not repaired here, each with the reason it could not be. It is a ratchet, not an excuse
// list: TestZeroDepClaimDebtIsAccurate reds if an entry no longer has a site (the entry must
// then be deleted, so the list can only shrink) and pins the size, so growing it is a
// review-visible act. Every entry below is a file this lane is fenced out of writing.
var zeroDepClaimDebt = map[string]string{
	"AGENTS.md": "line ~32 (\"Zero external deps, so no `go.sum`\"). The workspace agent contract; " +
		"out of this lane's fence.",
	"INDEX.md": "line ~658 (\"the zero-`go.sum` constraint\"), inside a one-line abstract of a dated " +
		"docs/notes survey. Out of this lane's fence; the abstract should be re-cut when the note is.",
	"docs/product-scorecard/README.md": "line ~74 (\"A single dependency-free Go binary (no Python, no CUDA, no go.sum)\"). " +
		"GENERATED by tools/product_scorecard.py from tools/product_scorecard.data/rows-platform.json; " +
		"hand-editing the .md is undone by the next regen, and the .json is under tools/, out of this " +
		"lane's fence. Fix the row, then regenerate.",
}

// zeroDepClaimDebtSize pins the debt. It may only ever go DOWN. Raising it means a new
// reader-facing page was allowed to ship the false claim.
const zeroDepClaimDebtSize = 3

// zeroDepRetractionSites pins which files may quote the stale phrasing in order to retract
// it, and how many lines each may spend doing so. See retractionCue for why this is pinned.
var zeroDepRetractionSites = map[string]int{
	"docs/air-gapped-deployment-kit.md": 2,
	"docs/enterprise-positioning.md":    1,
}

// zeroDepAnchors are corpus members whose ABSENCE means the walker stopped covering the
// claim surface. Without this, narrowing the corpus (a changed suffix test, a moved docs
// root, an over-eager skip) silently turns the gate into a no-op that still reports "ok".
var zeroDepAnchors = []string{
	"go.mod",
	"INSTALL.md",
	"CLAIMS.md",
	"Dockerfile",
	"docs/FAQ.md",
	"docs/cli-reference.md",
	"docs/air-gapped-deployment-kit.md",
}

// ---------------------------------------------------------------------------
// The scan
// ---------------------------------------------------------------------------

// claimSite is one line that makes (or retracts) a stale claim.
type claimSite struct {
	Path  string
	Line  int
	Kinds []claimKind
	Text  string
}

func (s claimSite) kindList() string {
	parts := make([]string, 0, len(s.Kinds))
	for _, k := range s.Kinds {
		parts = append(parts, string(k))
	}
	return strings.Join(parts, "+")
}

// scanStaleClaims is deliberately pure — path plus bytes in, sites out, no filesystem — so
// the mutation table below can prove every rule RED against synthetic text. A contract test
// that only runs over a corpus someone already cleaned proves nothing about whether it can
// still fail.
func scanStaleClaims(path string, data []byte) (claims, retracted []claimSite) {
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), 4<<20)
	n := 0
	for sc.Scan() {
		n++
		line := sc.Text()
		var kinds []claimKind
		for _, r := range staleClaimRules {
			if r.Re.MatchString(line) {
				kinds = append(kinds, r.Kind)
			}
		}
		if len(kinds) == 0 {
			continue
		}
		site := claimSite{Path: path, Line: n, Kinds: kinds, Text: excerpt(line)}
		if retractionCue.MatchString(line) {
			retracted = append(retracted, site)
			continue
		}
		claims = append(claims, site)
	}
	return claims, retracted
}

func excerpt(line string) string {
	line = strings.TrimSpace(line)
	const max = 160
	if len(line) <= max {
		return line
	}
	return line[:max] + " …"
}

// zeroDepCorpus returns the reader-facing claim surface, repo-relative and slash-separated:
// repo-root markdown plus go.mod plus the Dockerfile(s), and every .md under docs/ except the
// dated histories. Dot-prefixed root files are skipped — those are per-session scratch
// (.st_*) and tool config, not published claims.
func zeroDepCorpus(root string) ([]string, error) {
	var files []string

	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("read repo root %s: %w", root, err)
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasPrefix(name, ".") {
			continue
		}
		if strings.HasSuffix(name, ".md") || name == "go.mod" || strings.HasPrefix(name, "Dockerfile") {
			files = append(files, name)
		}
	}

	docs := filepath.Join(root, "docs")
	err = filepath.WalkDir(docs, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, rerr := filepath.Rel(root, p)
		if rerr != nil {
			return rerr
		}
		rel = filepath.ToSlash(rel)
		if d.IsDir() {
			// Dated histories: a record of what was true then is not a claim about now.
			if rel == "docs/notes" || rel == "docs/archive" {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(rel, ".md") {
			files = append(files, rel)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk docs/: %w", err)
	}

	sort.Strings(files)
	return files, nil
}

// ---------------------------------------------------------------------------
// The gates
// ---------------------------------------------------------------------------

func zeroDepRepoRoot(t *testing.T) string {
	t.Helper()
	return filepath.Dir(internalDir(t)) // repo root = parent of internal/
}

// scanZeroDepCorpus walks the corpus and returns every claim site, every retracted site, and
// the number of files it actually read. The file count is the fail-closed signal.
func scanZeroDepCorpus(t *testing.T, root string) (claims, retracted []claimSite, scanned int) {
	t.Helper()
	files, err := zeroDepCorpus(root)
	if err != nil {
		t.Fatalf("build the claim corpus: %v", err)
	}

	present := make(map[string]bool, len(files))
	for _, rel := range files {
		present[rel] = true
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatalf("read %s: %v", rel, err)
		}
		scanned++
		c, r := scanStaleClaims(rel, data)
		claims = append(claims, c...)
		retracted = append(retracted, r...)
	}

	// Fail closed, twice over. An empty corpus is the silently-inert failure this whole
	// class of gate rots into; a corpus that no longer contains the files the claim actually
	// lives in is the same failure wearing a large file count.
	if scanned == 0 {
		t.Fatal("the claim corpus is empty — this gate scanned nothing and would pass by looking at " +
			"nothing, which is exactly the silently-inert failure it exists to prevent; the repo-root " +
			"or docs/ layout changed under zeroDepCorpus")
	}
	for _, anchor := range zeroDepAnchors {
		if !present[anchor] {
			t.Fatalf("%s is not in the claim corpus (%d files scanned) — this file is a load-bearing "+
				"claim surface, so its absence means zeroDepCorpus stopped covering the thing this gate "+
				"is about. Failing closed rather than reporting a clean scan of the wrong tree.",
				anchor, scanned)
		}
	}
	return claims, retracted, scanned
}

// TestNoStaleZeroDependencyClaim is the gate itself. Any reader-facing page that tells a
// reader this module has no external dependencies, or that there is no go.sum, reds here.
func TestNoStaleZeroDependencyClaim(t *testing.T) {
	root := zeroDepRepoRoot(t)
	claims, retracted, scanned := scanZeroDepCorpus(t, root)

	byRule := map[claimKind]string{}
	for _, r := range staleClaimRules {
		byRule[r.Kind] = r.Why
	}

	live := 0
	for _, c := range claims {
		if _, known := zeroDepClaimDebt[c.Path]; known {
			continue
		}
		live++
		why := byRule[c.Kinds[0]]
		t.Errorf("[%s] %s:%d makes a claim about this module that is false — %s.\n"+
			"    line: %s\n"+
			"    cure: %s",
			c.kindList(), c.Path, c.Line, why, c.Text, theHonestStatement)
	}

	t.Logf("claim corpus: %d files scanned, %d live stale sites, %d retraction lines, %d files carrying known debt",
		scanned, live, len(retracted), len(zeroDepClaimDebt))
}

// TestZeroDepClaimDebtIsAccurate keeps the debt list from becoming a graveyard. An entry that
// no longer has a site must be DELETED — otherwise the list silently grants a permanent
// exemption to a file that has been clean for months, and the next stale claim added to it
// goes unreported.
func TestZeroDepClaimDebtIsAccurate(t *testing.T) {
	root := zeroDepRepoRoot(t)
	claims, _, _ := scanZeroDepCorpus(t, root)

	hit := map[string]int{}
	for _, c := range claims {
		hit[c.Path]++
	}

	corpus, err := zeroDepCorpus(root)
	if err != nil {
		t.Fatalf("build the claim corpus: %v", err)
	}
	inCorpus := make(map[string]bool, len(corpus))
	for _, rel := range corpus {
		inCorpus[rel] = true
	}

	for path, reason := range zeroDepClaimDebt {
		if !inCorpus[path] {
			t.Errorf("zeroDepClaimDebt names %q, which is not in the claim corpus at all — the file "+
				"moved or the corpus narrowed. Delete the entry or fix the corpus; a debt entry for an "+
				"unscanned file exempts nothing and hides everything. (reason on file: %s)", path, reason)
			continue
		}
		if hit[path] == 0 {
			t.Errorf("zeroDepClaimDebt still lists %q but the file no longer makes a stale claim. "+
				"DELETE the entry and drop zeroDepClaimDebtSize to %d — a debt list that outlives its "+
				"debt turns into a permanent blind spot for that file.", path, zeroDepClaimDebtSize-1)
		}
	}

	if len(zeroDepClaimDebt) != zeroDepClaimDebtSize {
		t.Errorf("zeroDepClaimDebt has %d entries, pinned at %d. This number may only go DOWN. "+
			"Raising it means a reader-facing page was allowed to keep shipping the false "+
			"\"zero external dependencies / no go.sum\" claim; fix the page instead.",
			len(zeroDepClaimDebt), zeroDepClaimDebtSize)
	}
}

// TestZeroDepClaimRetractionsArePinned pins the line-local retraction exemption to an exact
// per-file inventory. Without this the exemption is the one way the gate could be hollowed
// out quietly — a sentence containing the word "stale" anywhere would buy silence for the
// rest of the line. Pinned, a new exemption cannot appear without the count moving in review.
func TestZeroDepClaimRetractionsArePinned(t *testing.T) {
	root := zeroDepRepoRoot(t)
	_, retracted, _ := scanZeroDepCorpus(t, root)

	got := map[string]int{}
	for _, r := range retracted {
		got[r.Path]++
	}

	for path, n := range got {
		want, ok := zeroDepRetractionSites[path]
		if !ok {
			t.Errorf("%s takes the retraction exemption on %d line(s) but is not in "+
				"zeroDepRetractionSites. Quoting the stale phrasing to say it is stale is legitimate — "+
				"but it is pinned, so add the file and the count here in the same change, where a "+
				"reviewer sees it.", path, n)
			continue
		}
		if n != want {
			t.Errorf("%s takes the retraction exemption on %d line(s), pinned at %d. If the page "+
				"genuinely needs another retraction line, move the number here; if a line picked up a "+
				"retraction cue by accident, it is now silently exempt and must be reworded.", path, n, want)
		}
	}
	for path, want := range zeroDepRetractionSites {
		if got[path] == 0 {
			t.Errorf("zeroDepRetractionSites reserves %d retraction line(s) for %s, but the file takes "+
				"the exemption on none. Delete the reservation — an unused exemption is a standing "+
				"loophole for whatever that file says next.", want, path)
		}
	}
}

// TestZeroDepClaimGateCatchesMutations is the witness that the gate can FAIL. Every rule gets
// a case, in both the "must fire" and the "must NOT fire" direction, because a claim checker
// that over-fires on the true statements it lives next to is a checker someone deletes.
func TestZeroDepClaimGateCatchesMutations(t *testing.T) {
	cases := []struct {
		name string
		text string
		want []claimKind
	}{
		// --- must fire -------------------------------------------------------------
		{"go.mod's own stale header", "// Zero external dependencies (standard library only) — there is no go.sum.",
			[]claimKind{claimGoSumAbsent, claimZeroExternal}},
		{"install page", "fak is one static Go binary with **zero external dependencies** (there is no `go.sum`).",
			[]claimKind{claimGoSumAbsent, claimZeroExternal}},
		{"backticked go.sum", "One ~13 MB static Go binary (Apache-2.0, no `go.sum`) sits in front of an agent.",
			[]claimKind{claimGoSumAbsent}},
		{"hyphenated", "A zero-external-deps module.", []claimKind{claimZeroExternal}},
		{"no external dependencies", "One static Go binary, no external dependencies:", []claimKind{claimNoExternal}},
		{"no external Go deps", "**One static Go binary, Apache-2.0**, no external Go deps, drop-in.", []claimKind{claimNoExternal}},
		{"launch copy short form", "What you actually deploy: one ~13MB static Go binary. Zero deps.", []claimKind{claimZeroDeps}},
		{"spelled out", "one static Go binary, about thirteen megabytes, zero dependencies, Apache-2.0", []claimKind{claimZeroDeps}},
		{"no deps", "I built fak: one ~13MB static Go binary (Apache-2.0, no deps).", []claimKind{claimNoDeps}},
		{"no dependencies", "one static Go binary, no runtime, no dependencies:", []claimKind{claimNoDeps}},
		{"dependency-free binary", "fak is one ~13 MB dependency-free Go binary that drops in front of any agent.",
			[]claimKind{claimDepFreeArtifact}},
		{"fak is dependency-free", "events over ZMQ/msgpack. fak is dependency-free, so the adapter is out-of-tree.",
			[]claimKind{claimDepFreeArtifact}},
		{"without a go.sum", "It builds without a go.sum.", []claimKind{claimGoSumAbsent}},

		// --- must NOT fire ---------------------------------------------------------
		// These are the true neighbours. Damaging them would be its own honesty failure,
		// and over-firing on them is how this gate would get switched off.
		{"the corrected statement itself",
			"fak is one static Go binary whose entire external dependency set is two `golang.org/x` " +
				"extended-standard-library modules, pinned by a 4-line `go.sum`.", nil},
		{"the true neighbours", "A single static Go binary — no Python, no PyTorch, no CUDA toolchain; CGO_ENABLED=0.", nil},
		{"a stdlib-only package", "internal/sessionobs is the stdlib-only RSI scorecard, tier 1, off the hot path.", nil},
		{"a zero-dependency test", "discharged by a new zero-dependency (stdlib `testing`/`testing/quick`) property test", nil},
		{"a dependency-free example", "The proof is **dependency-free** — it does not import `crewai`, and does not call a model.", nil},
		{"a dependency-free adapter", "the coordination-overhead model with a dependency-free adapter", nil},
		{"go.sum named, not denied", "Either remove `golang.org/x/term`/`x/sys` and `go.sum`, or update the invariant.", nil},
		{"go.sum asserted present", "- go.sum exists (the zero-dep invariant broke)", nil},
		{"the module count stated truthfully", "cat go.sum   # -> exactly 4 lines; the h1: digests in the SBOM's sourceInfo", nil},
		{"no Python, then go.sum", "no Python, no CUDA toolchain, and a 4-line `go.sum` pinning two modules.", nil},
	}

	fired := map[claimKind]bool{}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			claims, retracted := scanStaleClaims("fixture.md", []byte(tc.text))
			if len(retracted) != 0 {
				t.Fatalf("fixture unexpectedly took the retraction exemption: %v", retracted)
			}
			var got []claimKind
			if len(claims) > 0 {
				got = claims[0].Kinds
			}
			if len(claims) > 1 {
				t.Fatalf("fixture is one line but produced %d sites", len(claims))
			}
			if !sameClaimKinds(got, tc.want) {
				t.Fatalf("kinds = %v, want %v\n  text: %s", got, tc.want, tc.text)
			}
			for _, k := range got {
				fired[k] = true
			}
		})
	}

	// Every rule must be exercised by at least one must-fire case. A rule no case can trip
	// is a rule whose regex may already be broken.
	for _, r := range staleClaimRules {
		if !fired[r.Kind] {
			t.Errorf("no mutation case fires rule %s — an unexercised rule is an unproven one; "+
				"add a case or delete the rule", r.Kind)
		}
	}
}

// TestZeroDepClaimRetractionExemptionIsLineLocal pins the exemption's shape: it must exempt
// exactly the line that carries the retraction and nothing else. A file-level or
// section-level exemption would be the loophole that swallows the check.
func TestZeroDepClaimRetractionExemptionIsLineLocal(t *testing.T) {
	const doc = "" +
		"> **The older \"zero external dependencies, no `go.sum`\" phrasing is stale — do not use it.**\n" +
		"\n" +
		"fak is one static Go binary with zero external dependencies and no `go.sum`.\n"

	claims, retracted := scanStaleClaims("mixed.md", []byte(doc))
	if len(retracted) != 1 || retracted[0].Line != 1 {
		t.Fatalf("retracted = %v, want exactly line 1 — the retraction line must be exempt", retracted)
	}
	if len(claims) != 1 || claims[0].Line != 3 {
		t.Fatalf("claims = %v, want exactly line 3 — a retraction on line 1 must NOT exempt the "+
			"assertion three lines later, or one warning sentence would license a whole page of the "+
			"stale claim", claims)
	}
}

// TestZeroDepClaimCorpusExcludesDatedHistories pins the path exemption in both directions.
// Dated notes must be skipped (rewriting a record to green a checker is falsification), and
// the live pages next to them must NOT be.
func TestZeroDepClaimCorpusExcludesDatedHistories(t *testing.T) {
	root := zeroDepRepoRoot(t)
	files, err := zeroDepCorpus(root)
	if err != nil {
		t.Fatalf("build the claim corpus: %v", err)
	}

	sawExplainer, sawNotes, sawArchive := false, false, false
	for _, rel := range files {
		switch {
		case strings.HasPrefix(rel, "docs/notes/"):
			sawNotes = true
		case strings.HasPrefix(rel, "docs/archive/"):
			sawArchive = true
		case strings.HasPrefix(rel, "docs/explainers/"):
			sawExplainer = true
		}
	}
	if sawNotes {
		t.Error("docs/notes/** is in the corpus — those are dated records, and a checker that reds " +
			"on them pressures someone to rewrite history to go green")
	}
	if sawArchive {
		t.Error("docs/archive/** is in the corpus — same reason as docs/notes/**")
	}
	if !sawExplainer {
		t.Error("docs/explainers/** is NOT in the corpus — the history exemption has widened into " +
			"the live pages, which is how a path skip turns a gate into a no-op")
	}
}

// TestZeroDepClaimGateFailsClosed pins the other half of "the gate can fail": a corpus it
// cannot honestly scan must be an ERROR, never a clean bill of health.
func TestZeroDepClaimGateFailsClosed(t *testing.T) {
	dir := t.TempDir()
	files, err := zeroDepCorpus(dir)
	if err == nil && len(files) == 0 {
		// A corpus builder that returns nothing is legal; the caller is what must refuse.
		// scanZeroDepCorpus t.Fatal's on scanned == 0 and on a missing anchor, which cannot
		// be asserted from inside the same *testing.T. Assert the precondition instead: an
		// empty directory yields an empty corpus, and every anchor is therefore absent.
		for _, anchor := range zeroDepAnchors {
			for _, f := range files {
				if f == anchor {
					t.Fatalf("anchor %q found in an empty tree", anchor)
				}
			}
		}
		return
	}
	if err != nil {
		// Missing docs/ is itself a fail-closed signal, which is also acceptable.
		if !strings.Contains(err.Error(), "walk docs/") {
			t.Fatalf("unexpected error building a corpus over an empty tree: %v", err)
		}
		return
	}
	t.Fatalf("an empty tree produced a %d-file corpus", len(files))
}

// ---------------------------------------------------------------------------
// The localized mirrors (docs/i18n/**)
// ---------------------------------------------------------------------------
//
// docs/i18n/** is inside the corpus above, but the rules above are English-phrase rules, so
// the translated pages sail through them while saying the same false thing in fifteen other
// languages — "cero dependencias externas", "ноль внешних зависимостей", "零外部依赖". That is
// the worst version of this defect, not the mildest: the reader least able to cross-check
// against go.mod is the one reading the translation.
//
// This layer is a LEDGER, not an exemption. Every site below is false today. It is pinned
// rather than repaired because repairing it means writing supply-chain wording in Bengali,
// Tamil, Telugu, Marathi, Arabic, Russian, Korean, Vietnamese and eight more, and a
// machine-guessed translation of a claim whose whole point is precision is not a repair — it
// is the same defect with better grammar. What the pin buys is that the debt is now counted,
// cannot rot, and cannot grow: a NEW localized page carrying the claim reds this gate, and a
// site that gets a native-language pass must be deleted from the map and the constant lowered.
//
// The detector is deliberately shaped differently from the English one. It looks for a
// "zero"-word within a short window of a "dependency"-word, in either order and ACROSS line
// breaks, because these pages hard-wrap mid-phrase (docs/i18n/de/install.md splits "ohne
// externe Abhängigkeiten" over two lines) and because several languages put the numeral after
// the noun (Korean "외부 의존성 제로", Tamil "வெளிச் சார்புகள் பூஜ்ஜியம்").
//
// The window is the honest limit: this is a FLOOR on the localized debt, not a ceiling. A
// translation that expresses the claim with no numeral at all is not detectable this way and
// is not claimed to be.

// localizedZeroWord / localizedDepWord are the two halves of the localized claim, in the
// languages docs/i18n actually ships. Each alternative was read in place in the file it comes
// from — none is a guess about a language not present in the corpus.
var (
	localizedZeroWord = regexp.MustCompile(`(?i)\bzero\b|\bzéro\b|\bcero\b|\bnull\b|\bsıfır\b|\btanpa\b|\bnol\b|` +
		`\bsin\b|\bsem\b|\bohne\b|\bsans\b|\bkhông\b|` +
		`ноль|нулев|ゼロ|제로|零|صفر|بلا|` +
		`শূন্য|शून्य|पूज्य|பூஜ்ஜிய|సున్నా`)
	localizedDepWord = regexp.MustCompile(`(?i)dependenc|dependên|dependênc|dépendance|abhängigkeit|` +
		`bağımlılık|dependensi|নির্ভর|依存|의존|зависим|依赖|依賴|phụ thuộc|சார்ப|சார்பு|اعتمادي|go\.sum`)
)

// localizedClaimWindow is how far apart the two words may be, in RUNES, and it is small on
// purpose: wide enough to cross one hard wrap, narrow enough that an unrelated "sin" three
// sentences away cannot manufacture a hit.
const localizedClaimWindow = 40

// claimLocalized is the kind reported for a translated stale claim.
const claimLocalized claimKind = "LOCALIZED_ZERO_DEPENDENCIES"

// scanLocalizedStaleClaims is pure for the same reason scanStaleClaims is: the mutation table
// has to be able to prove it fires. At most one site per line, so a line that pairs the two
// words twice still counts once.
func scanLocalizedStaleClaims(path string, data []byte) []claimSite {
	// Flatten line breaks to spaces WITHOUT changing any byte offset, so the window may span a
	// hard wrap while line numbers stay exact.
	flat := strings.NewReplacer("\n", " ", "\r", " ").Replace(string(data))
	runes := []rune(flat)

	// runeIndex[byteOffset] -> rune offset, built once.
	runeOf := make(map[int]int, len(runes))
	{
		r := 0
		for b := range flat {
			runeOf[b] = r
			r++
		}
	}

	lines := strings.Split(string(data), "\n")

	var sites []claimSite
	seen := map[int]bool{}
	for _, m := range localizedDepWord.FindAllStringIndex(flat, -1) {
		start, ok := runeOf[m[0]]
		if !ok {
			continue
		}
		end := start + len([]rune(flat[m[0]:m[1]]))

		lo := start - localizedClaimWindow
		if lo < 0 {
			lo = 0
		}
		hi := end + localizedClaimWindow
		if hi > len(runes) {
			hi = len(runes)
		}
		if !localizedZeroWord.MatchString(string(runes[lo:hi])) {
			continue
		}

		line := 1 + strings.Count(string(data[:m[0]]), "\n")
		if seen[line] {
			continue
		}
		seen[line] = true
		text := ""
		if line-1 < len(lines) {
			text = excerpt(lines[line-1])
		}
		sites = append(sites, claimSite{Path: path, Line: line, Kinds: []claimKind{claimLocalized}, Text: text})
	}
	sort.Slice(sites, func(i, j int) bool { return sites[i].Line < sites[j].Line })
	return sites
}

// localizedZeroDepDebt is the pinned per-file count of localized pages that still say the
// module has no external dependencies. Path -> number of stale lines. It may only shrink.
var localizedZeroDepDebt = map[string]int{
	"docs/i18n/ar/README.md":  1,
	"docs/i18n/ar/install.md": 1,
	"docs/i18n/bn/README.md":  1,
	"docs/i18n/bn/install.md": 1,
	"docs/i18n/de/README.md":  1,
	"docs/i18n/de/install.md": 1,
	"docs/i18n/es/README.md":  1,
	"docs/i18n/es/install.md": 1,
	"docs/i18n/fr/README.md":  1,
	"docs/i18n/fr/install.md": 1,
	"docs/i18n/hi/README.md":  1,
	"docs/i18n/hi/install.md": 1,
	"docs/i18n/id/README.md":  1,
	"docs/i18n/id/install.md": 1,
	"docs/i18n/ko/README.md":  1,
	"docs/i18n/mr/README.md":  1,
	"docs/i18n/mr/install.md": 1,
	"docs/i18n/pt/README.md":  1,
	"docs/i18n/pt/install.md": 1,
	"docs/i18n/ru/README.md":  1,
	"docs/i18n/ta/README.md":  1,
	"docs/i18n/ta/install.md": 1,
	"docs/i18n/te/README.md":  1,
	"docs/i18n/te/install.md": 1,
	"docs/i18n/tr/README.md":  1,
	"docs/i18n/vi/README.md":  1,
	"docs/i18n/vi/install.md": 1,
	"docs/i18n/zh/README.md":  1,
	"docs/i18n/zh/install.md": 1,
}

// localizedZeroDepDebtSites is the total pinned above. It may only ever go DOWN, one native
// -language pass at a time. Raising it means a translation shipped the false claim after this
// gate was in place.
const localizedZeroDepDebtSites = 29

// localizedCorpusAnchor is the hub page. Its absence means the i18n tree moved and the
// localized sweep is scanning nothing.
const localizedCorpusAnchor = "docs/i18n/README.md"

// TestLocalizedZeroDepClaimDebtIsPinned freezes the translated debt exactly: no new localized
// page may carry the claim, and a page that gets a native-language pass must be struck from
// the map. Fail-closed on an empty i18n corpus, same as the English gate.
func TestLocalizedZeroDepClaimDebtIsPinned(t *testing.T) {
	root := zeroDepRepoRoot(t)
	corpus, err := zeroDepCorpus(root)
	if err != nil {
		t.Fatalf("build the claim corpus: %v", err)
	}

	got := map[string]int{}
	scanned, sawAnchor := 0, false
	for _, rel := range corpus {
		if !strings.HasPrefix(rel, "docs/i18n/") {
			continue
		}
		scanned++
		if rel == localizedCorpusAnchor {
			sawAnchor = true
		}
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatalf("read %s: %v", rel, err)
		}
		for _, s := range scanLocalizedStaleClaims(rel, data) {
			got[s.Path]++
		}
	}

	if scanned == 0 {
		t.Fatal("the localized corpus is empty — the docs/i18n tree moved or zeroDepCorpus stopped " +
			"covering it, so this gate is scanning nothing while reporting a clean bill of health")
	}
	if !sawAnchor {
		t.Fatalf("%s is not in the localized corpus (%d files scanned) — failing closed rather than "+
			"pinning a debt list against the wrong tree", localizedCorpusAnchor, scanned)
	}

	total := 0
	for _, n := range got {
		total += n
	}
	for _, msg := range localizedDebtViolations(got, localizedZeroDepDebt) {
		t.Error(msg)
	}
	if total != localizedZeroDepDebtSites {
		t.Errorf("localized stale sites = %d, pinned at %d. This number may only go DOWN.",
			total, localizedZeroDepDebtSites)
	}

	t.Logf("localized corpus: %d files scanned, %d stale sites across %d pages, all pinned as debt "+
		"awaiting a native-language pass", scanned, total, len(got))
}

// localizedDebtViolations compares an observed per-page count against the pinned ledger. It is
// pure so the ledger's three failure modes — a NEW page carrying the claim, a page whose count
// moved, and an entry whose debt was paid — are provable without mutating the working tree.
func localizedDebtViolations(got, pinned map[string]int) []string {
	var msgs []string
	paths := make([]string, 0, len(got))
	for p := range got {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	for _, path := range paths {
		n := got[path]
		want, ok := pinned[path]
		if !ok {
			msgs = append(msgs, fmt.Sprintf("[%s] %s ships the stale \"no external dependencies\" claim on "+
				"%d line(s) and is NOT in localizedZeroDepDebt — a NEW translation of a claim this repo "+
				"already retracted in English.\n    cure: %s", claimLocalized, path, n, theHonestStatement))
			continue
		}
		if n != want {
			msgs = append(msgs, fmt.Sprintf("[%s] %s has %d stale line(s), pinned at %d. Down is the only "+
				"legal direction: lower the entry (or delete it and lower localizedZeroDepDebtSites) when a "+
				"native-language pass lands.", claimLocalized, path, n, want))
		}
	}
	retired := make([]string, 0, len(pinned))
	for p := range pinned {
		retired = append(retired, p)
	}
	sort.Strings(retired)
	for _, path := range retired {
		if got[path] == 0 {
			msgs = append(msgs, fmt.Sprintf("localizedZeroDepDebt still reserves %d stale line(s) for %s but "+
				"the file no longer makes the claim. DELETE the entry and lower localizedZeroDepDebtSites — a "+
				"ledger entry that outlives its debt is a permanent blind spot for that page.",
				pinned[path], path))
		}
	}
	return msgs
}

// TestLocalizedZeroDepDebtLedgerHasTeeth proves the ledger refuses growth, drift, and rot.
// Without this the pin is only ever exercised in its passing direction, which is how a ratchet
// quietly becomes a comment.
func TestLocalizedZeroDepDebtLedgerHasTeeth(t *testing.T) {
	pinned := map[string]int{"docs/i18n/es/README.md": 1}

	cases := []struct {
		name string
		got  map[string]int
		want int // number of violations
		must string
	}{
		{"clean", map[string]int{"docs/i18n/es/README.md": 1}, 0, ""},
		{"a new locale ships the claim",
			map[string]int{"docs/i18n/es/README.md": 1, "docs/i18n/pl/README.md": 1}, 1, "NOT in localizedZeroDepDebt"},
		{"an existing page grows a second stale line",
			map[string]int{"docs/i18n/es/README.md": 2}, 1, "pinned at 1"},
		{"the debt was paid but the entry stayed",
			map[string]int{}, 1, "no longer makes the claim"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			msgs := localizedDebtViolations(tc.got, pinned)
			if len(msgs) != tc.want {
				t.Fatalf("violations = %d, want %d: %v", len(msgs), tc.want, msgs)
			}
			if tc.must != "" && !strings.Contains(msgs[0], tc.must) {
				t.Fatalf("violation does not name the failure mode %q: %s", tc.must, msgs[0])
			}
		})
	}
}

// TestLocalizedZeroDepScannerCatchesMutations proves the localized detector fires — including
// across a hard wrap and with the numeral trailing the noun — and that it does not fire on a
// translated page that states the count truthfully.
func TestLocalizedZeroDepScannerCatchesMutations(t *testing.T) {
	cases := []struct {
		name  string
		text  string
		lines []int
	}{
		{"spanish", "- **Un binario estático, cero dependencias externas.** Ops simples", []int{1}},
		{"german wrapped over two lines", "fak ist eine einzige statische Go-Binary ohne\nexterne Abhängigkeiten — Gateway, Policy-Gate,", []int{2}},
		{"korean numeral trails the noun", "- **하나의 static 바이너리, 외부 의존성 제로.** 작은 팀에게", []int{1}},
		{"russian", "- **Один статический бинарник, ноль внешних зависимостей.**", []int{1}},
		{"chinese", "- **一个静态二进制，零外部依赖。** 小团队运维简单", []int{1}},
		{"tamil numeral trails the noun", "fak **ஒரே Go binary** — வெளிச் சார்புகள் பூஜ்ஜியம். clone-இலிருந்து", []int{1}},
		{"portuguese", "- **Um binário estático, zero dependência externa.** Ops simples", []int{1}},
		{"turkish", "- **Tek static binary, sıfır harici bağımlılık.** Küçük ekipler", []int{1}},
		{"vietnamese", "- **Một binary tĩnh, không phụ thuộc bên ngoài.** Vận hành gọn", []int{1}},
		{"arabic", "- **ثنائي واحد ثابت، بلا اعتماديات خارجية.** تشغيل بسيط", []int{1}},
		{"english go.sum denial inside a translation", "`fak` es un binario Go sin `go.sum`.", []int{1}},

		// Must NOT fire.
		{"the truthful translated count", "El conjunto de dependencias externas son dos módulos `golang.org/x`.", nil},
		{"far apart, unrelated", "sin duda es el mismo artefacto. " + strings.Repeat("y ", 40) + "las dependencias externas son dos.", nil},
		{"no numeral at all", "fak ist eine statische Go-Binary; die Abhängigkeiten stehen in go.mod.", nil},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sites := scanLocalizedStaleClaims("fixture.md", []byte(tc.text))
			var lines []int
			for _, s := range sites {
				lines = append(lines, s.Line)
			}
			if len(lines) != len(tc.lines) {
				t.Fatalf("lines = %v, want %v\n  text: %s", lines, tc.lines, tc.text)
			}
			for i := range lines {
				if lines[i] != tc.lines[i] {
					t.Fatalf("lines = %v, want %v", lines, tc.lines)
				}
			}
		})
	}
}

func sameClaimKinds(got, want []claimKind) bool {
	if len(got) != len(want) {
		return false
	}
	g := make([]string, 0, len(got))
	w := make([]string, 0, len(want))
	for _, k := range got {
		g = append(g, string(k))
	}
	for _, k := range want {
		w = append(w, string(k))
	}
	sort.Strings(g)
	sort.Strings(w)
	for i := range g {
		if g[i] != w[i] {
			return false
		}
	}
	return true
}
