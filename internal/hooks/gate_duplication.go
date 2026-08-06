package hooks

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/clonescan"
	"github.com/anthony-chaudhary/fak/internal/tokencache"
)

// gate_duplication.go — the DUPLICATION advisory gate. It brings fak's authoring-time clone
// engine (internal/clonescan, the same normalized-token-window definition the code-slop
// scorecard grades the whole tree with a cycle LATER) to the commit boundary, so a copy-pasted
// block is caught as it is authored instead of counted as debt afterward. See
// docs/notes/DEDUP-EARLIER-AND-MORE-OFTEN-2026-07-03.md.
//
// The existing `fak dup guard --staged` runs the SAME query, but two things keep it from being
// the durable commit-boundary answer: it is an opt-in CLI verb a worker must remember to run,
// and it tokenizes the WHOLE tracked .go tree (5.7k files, ~2-5s) — far too heavy to run on
// every commit. This gate is the in-process, always-on twin, and it makes ONE scoping choice to
// stay cheap: it compares a file's ADDED block only against the OTHER tracked .go files in the
// SAME directory (its Go package). That is both the cheapest neighborhood (you pay only for the
// directories a commit actually touched) and the most ACTIONABLE one — an intra-package clone is
// a "call the sibling helper" fix with no new import, exactly the slop the gate wants to stop at
// its source. Cross-package clones remain the whole-tree scorecard's job; the two compose.
//
// ADVISORY BY DEFAULT (DefaultMode "warn"), exactly like E2E_OVER_MOCKS / BARE_COMMIT_SWEEP: out
// of the box it never reds a shared trunk — it names the sibling site a block clones and the
// fix. Set FLEET_DUP_GUARD=block to hard-enforce it, or ALLOW_DUP=1 to skip it once. Precision is
// kept high (clonescan keeps identifiers, so distinct names never false-match; a window must
// carry real logic over ~6 lines), so an advisory warning on a shared trunk means a real
// near-duplicate, not an idiom.

const (
	// dupNeighborMatches caps how many sibling sites a single finding names — enough to point at
	// the duplicate without a wall of text when a block was copied across several files.
	dupNeighborMatches = 3
	// dupMaxNeighborhood is a safety valve: a directory holding more than this many tracked
	// non-test .go files is skipped (the block is left to the whole-tree scorecard) so the gate's
	// per-commit cost stays bounded even if a package grows pathologically large. Set well above
	// today's biggest package (cmd/fak ~1.1k) so every real directory is still covered; it exists
	// only to cap unbounded future growth.
	dupMaxNeighborhood = 2500
	// dupMinGradedWindows is the ABSOLUTE floor below which a finding is left UNGRADED
	// (Severity 0) however high its coverage fraction runs. Coverage alone would read a tiny
	// block whose every window is an idiom as a 100% copy, and an operator who opted into
	// near-total-copy escalation must not have a five-line `for … if err != nil` hard-block
	// their commit. 40 distinct qualifying windows is roughly a 10-15 line block of real logic
	// — the size at which "all of it already exists" stops being idiom and starts being a
	// paste. Below it the gate still WARNS exactly as before; it just refuses to grade.
	dupMinGradedWindows = 40
)

// gateDuplication fires ONE DUPLICATION finding per staged non-test .go file whose ADDED block
// token-clones another tracked .go file in the same directory. It reads the tracked .go listing
// once, groups it by directory, and reads sibling contents ONLY for the directories the commit
// touched — so the cost is bounded to the neighborhoods actually in play. Fail-open: if the
// tracked listing cannot be read it returns ErrCouldNotRun (the runner skips the gate), never a
// false DUPLICATION.
func gateDuplication(d *StagedDiff) ([]Finding, error) {
	return duplicationFindings(d, dupMaxNeighborhood, dupNeighborMatches, dupMinGradedWindows)
}

// duplicationFindings is gateDuplication's core with the neighborhood cap, per-finding match cap,
// and grading floor as parameters, so a unit test can drive the size-cap and ungraded branches
// without thousands of files or a hand-counted window budget.
func duplicationFindings(d *StagedDiff, maxNeighborhood, maxMatches, minGradedWindows int) ([]Finding, error) {
	// Candidate blocks: the concatenated added lines of each staged non-test .go file, tokenized to
	// their qualifying clone-window keys ONCE here. A trivial addition (below clonescan's ~34-token
	// window) yields an empty want-set — Query would no-op on it — so it is pruned up front and no
	// directory it lives in is marked `touched`. When EVERY candidate is trivial (the common
	// small-commit case: a one-line fix, a tiny helper, a bare `package x`) `touched` stays empty,
	// so trackedGoByDir's `git ls-files` subprocess + sibling disk reads are skipped entirely.
	type candidate struct {
		rel  string
		dir  string
		line int
		want map[string]bool
	}
	var candidates []candidate
	touched := map[string]bool{}
	for _, rel := range d.sortedFiles() {
		if !isGoNonTest(rel) {
			continue
		}
		lines := d.AddedByFile[rel]
		if len(lines) == 0 {
			continue
		}
		var sb strings.Builder
		for _, al := range lines {
			sb.WriteString(al.Text)
			sb.WriteByte('\n')
		}
		want := clonescan.CandidateKeys(sb.String())
		if len(want) == 0 {
			continue // no qualifying window: this block cannot clone anything, so it needs no neighborhood
		}
		relSlash := strings.ReplaceAll(rel, "\\", "/")
		dir := path.Dir(relSlash)
		candidates = append(candidates, candidate{
			rel:  relSlash,
			dir:  dir,
			line: lines[0].New,
			want: want,
		})
		touched[dir] = true
	}
	if len(candidates) == 0 {
		return nil, nil
	}

	byDir, err := trackedGoByDir(d, touched, maxNeighborhood)
	if err != nil {
		return nil, err
	}

	// Tokenize each neighborhood ONCE. Several staged candidates can share a
	// directory (its Go package), so building one TreeIndex per neighborhood and
	// reusing it across candidates avoids re-tokenizing the same siblings per
	// candidate (the quadratic N*M this ticket removes).
	//
	// A persisted, fleet-shared WindowCache (#4330) memoizes each sibling's
	// tokenization across commits, so an unchanged sibling is a content-addressed
	// lookup instead of a re-lex. Open once and reuse across neighborhoods; a nil
	// cache (disabled, or common dir unresolvable) degrades to the exact uncached path.
	cache := tokencache.Open(d.Root)
	indexByDir := make(map[string]*clonescan.TreeIndex, len(byDir))
	for dir, neighborhood := range byDir {
		indexByDir[dir] = clonescan.BuildTreeIndex(neighborhood, cache)
	}

	var findings []Finding
	for _, c := range candidates {
		index := indexByDir[c.dir]
		if index == nil {
			continue // no siblings, or the directory was over the cap and left to the scorecard
		}
		matches := index.Query(c.want, c.rel, maxMatches)
		if len(matches) == 0 {
			continue
		}
		findings = append(findings, Finding{
			Gate:     "DUPLICATION",
			File:     c.rel,
			Line:     c.line,
			Detail:   dupDetail(c.rel, matches),
			Severity: dupSeverity(index, c.want, c.rel, minGradedWindows),
		})
	}
	return findings, nil
}

// dupSeverity grades one already-matched candidate by COVERAGE — what fraction of the block just
// added already exists in the neighborhood — as a 0-100 percent. It runs only after Query found a
// match, so the clean path pays nothing for it.
//
// Two deliberate asymmetries. It grades on the candidate's own window count, never on
// Match.Windows: that number counts sibling-side hits and rises with sheer size, so a big honest
// addition would grade as a near-total copy. And it returns 0 — UNGRADED, per Finding.Severity —
// for any candidate under minGradedWindows, so a small all-idiom block can never reach a
// severity-driven refusal. The percent floors (integer division), so a reported severity is never
// higher than the true coverage; a consumer thresholding on it under-escalates rather than over.
func dupSeverity(index *clonescan.TreeIndex, want map[string]bool, selfPath string, minGradedWindows int) int {
	matched, total := index.Coverage(want, selfPath)
	if total < minGradedWindows || total <= 0 {
		return 0
	}
	return matched * 100 / total
}

// trackedGoByDir lists the tracked .go tree once and returns, for each WANTED directory, its
// sibling non-test .go files as rel-path -> source. A directory holding more than maxNeighborhood
// tracked files is returned absent (its blocks fall through to the whole-tree scorecard) so the
// per-commit read cost stays bounded. Only wanted, under-cap directories are read from disk.
func trackedGoByDir(d *StagedDiff, want map[string]bool, maxNeighborhood int) (map[string]map[string]string, error) {
	if d.run == nil {
		return nil, ErrCouldNotRun
	}
	out, code, err := d.run(d.ctx, d.Root, "ls-files", "*.go")
	if err != nil || code != 0 {
		return nil, ErrCouldNotRun
	}

	// Pass 1: bucket tracked non-test .go paths by directory, for wanted directories only.
	paths := map[string][]string{}
	for _, rel := range strings.Split(out, "\n") {
		rel = strings.TrimSpace(rel)
		if rel == "" || !isGoNonTest(rel) {
			continue
		}
		rel = strings.ReplaceAll(rel, "\\", "/")
		dir := path.Dir(rel)
		if !want[dir] {
			continue
		}
		paths[dir] = append(paths[dir], rel)
	}

	// Pass 2: read siblings only for wanted directories that are under the size cap.
	byDir := make(map[string]map[string]string, len(paths))
	for dir, rels := range paths {
		if len(rels) > maxNeighborhood {
			continue
		}
		tree := make(map[string]string, len(rels))
		for _, rel := range rels {
			if b, ok := d.neighborBytes(rel); ok {
				tree[rel] = b
			}
		}
		if len(tree) > 0 {
			byDir[dir] = tree
		}
	}
	return byDir, nil
}

// neighborBytes returns the source of a DUPLICATION neighborhood sibling WITHOUT the per-file
// `git show :<rel>` subprocess that StagedDiff.FileBytes spawns. That fan-out is the gate's real
// cost: one `git show` process per tracked non-test .go file in a touched directory — 720 on a
// cmd/fak commit — which under fleet git contention (a saturated fsmonitor daemon, many peer
// `git` processes) takes tens of seconds and WEDGES the pre-commit hook (the commit-lane hang).
// The neighborhood read only needs the sibling's current tracked source to tokenize it, so a
// single os.ReadFile of the working-tree copy is both far cheaper (no process spawn) and more
// faithful for a clone advisory: a block that was MOVED (deleted from the sibling on disk, added
// to the new file) is correctly NOT flagged, where the staged-blob read would still see the old
// copy and cry duplicate. Cache-aware: a warmed entry (or a unit test's pre-seeded fileCache) is
// reused verbatim, so no disk is touched there. A disk miss is a skip (fail-open, never a false
// DUPLICATION), matching FileBytes' own missing-file contract.
func (d *StagedDiff) neighborBytes(rel string) (string, bool) {
	if e, ok := d.cachedFile(rel); ok {
		return string(e.data), e.exists
	}
	b, err := os.ReadFile(filepath.Join(d.Root, filepath.FromSlash(rel)))
	exists := err == nil
	d.storeFile(rel, fileEntry{data: b, exists: exists})
	return string(b), exists
}

// isGoNonTest reports whether rel is a Go source file that is not a test file. Test files are
// out of scope for the first slice: table-driven tests are legitimately token-similar by shape,
// so scanning them would trade the gate's precision for noise on a shared trunk.
func isGoNonTest(rel string) bool {
	rel = strings.ReplaceAll(rel, "\\", "/")
	return strings.HasSuffix(rel, ".go") && !strings.HasSuffix(rel, "_test.go")
}

// dupDetail renders the one-line advisory: which added file clones a sibling, the sibling
// site(s) with their line span and overlap count, the fix, and the escape hatches.
func dupDetail(rel string, matches []clonescan.Match) string {
	sites := make([]string, 0, len(matches))
	for _, m := range matches {
		sites = append(sites, fmt.Sprintf("%s:%d-%d (%d windows)", m.File, m.StartLine, m.EndLine, m.Windows))
	}
	return fmt.Sprintf(
		"the added Go block in %s token-clones an existing site in the same package: %s — a shared "+
			"helper may already exist; call it instead of re-adding the block (`fak dup query --file %s` "+
			"to inspect). (advisory; FLEET_DUP_GUARD=block enforces, ALLOW_DUP=1 skips once)",
		rel, strings.Join(sites, ", "), rel,
	)
}
