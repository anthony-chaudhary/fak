package architest

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// This file is the regression gate for the CONCURRENCY CEILING that a missing lane
// declaration silently imposes on the whole fleet.
//
// dos.toml's own header states the invariant: "the honest partition is ONE LANE PER LEAF
// ... that is what lets the live fleet edit N leaves in parallel." The reason a violation
// is expensive rather than cosmetic is in internal/laneadmit/laneadmit.go: Decide falls
// back to the LANE's declared tree when a request carries none, and its geometric rung
// treats an EMPTY tree as conservatively overlapping EVERYTHING. So an internal/<leaf>
// with no [lanes] declaration and no [lanes.trees] entry resolves to an empty tree and
// collides with EVERY live lease -- one worker on an undeclared leaf serializes the whole
// fleet, and a `(fak <leaf>)` ship-stamp on it binds to a phantom unit. The gap is
// invisible at the surface where it is created (nothing fails when you add a leaf), which
// is exactly why it reopened at scale once before; this gate makes it fail loudly at the
// commit boundary that adds the leaf.

// laneRoster is the slice of dos.toml this gate needs: every lane name declared concurrent
// or exclusive, and the [lanes.trees] glob list per lane. The repo carries zero external
// deps (no TOML library), and lane names / tree globs are plain quoted tokens, so a line
// scan suffices -- the same deliberately tiny reader internal/hooks/commitstamp.go uses.
type laneRoster struct {
	concurrent []string            // in file order, WITH duplicates, so the dup gate can see them
	exclusive  []string            // in file order, WITH duplicates
	trees      map[string][]string // lane -> globs
}

func (r laneRoster) declared(lane string) bool {
	for _, l := range r.concurrent {
		if l == lane {
			return true
		}
	}
	for _, l := range r.exclusive {
		if l == lane {
			return true
		}
	}
	return false
}

// stripTOMLComment drops a trailing `#` comment, ignoring a `#` inside a quoted string.
// dos.toml comments routinely carry issue refs ("# #5059: ..."), so the scan has to be
// quote-aware rather than cutting at the first byte.
func stripTOMLComment(line string) string {
	var b strings.Builder
	inStr := false
	for i := 0; i < len(line); i++ {
		switch c := line[i]; c {
		case '"':
			inStr = !inStr
			b.WriteByte(c)
		case '#':
			if !inStr {
				return b.String()
			}
			b.WriteByte(c)
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}

// quotedTokens returns every "..." token on a line.
func quotedTokens(line string) []string {
	var out []string
	rest := line
	for {
		i := strings.IndexByte(rest, '"')
		if i < 0 {
			return out
		}
		rest = rest[i+1:]
		j := strings.IndexByte(rest, '"')
		if j < 0 {
			return out
		}
		out = append(out, rest[:j])
		rest = rest[j+1:]
	}
}

// readLaneRoster parses the `[lanes]` arrays and the `[lanes.trees]` table out of dos.toml.
func readLaneRoster(t *testing.T) laneRoster {
	t.Helper()
	path := filepath.Join(filepath.Dir(internalDir(t)), "dos.toml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v — dos.toml IS the lane taxonomy; without it the arbiter has no trees to prove disjoint.", path, err)
	}
	r := laneRoster{trees: map[string][]string{}}
	section, array := "", ""
	for _, rawLine := range strings.Split(string(raw), "\n") {
		line := stripTOMLComment(rawLine)
		trimmed := strings.TrimSpace(line)
		if array == "" && strings.HasPrefix(trimmed, "[") && !strings.Contains(trimmed, "=") {
			section, array = trimmed, ""
			continue
		}
		switch section {
		case "[lanes]":
			if array == "" {
				key, val, ok := strings.Cut(trimmed, "=")
				if !ok || !strings.Contains(val, "[") {
					continue
				}
				array = strings.TrimSpace(key)
				trimmed = val
			}
			switch array {
			case "concurrent":
				r.concurrent = append(r.concurrent, quotedTokens(trimmed)...)
			case "exclusive":
				r.exclusive = append(r.exclusive, quotedTokens(trimmed)...)
			}
			if strings.Contains(trimmed, "]") {
				array = ""
			}
		case "[lanes.trees]":
			key, val, ok := strings.Cut(trimmed, "=")
			if !ok {
				continue
			}
			if lane := strings.TrimSpace(key); lane != "" {
				r.trees[lane] = quotedTokens(val)
			}
		}
	}
	if len(r.concurrent) == 0 || len(r.trees) == 0 {
		t.Fatalf("parsed dos.toml but found %d concurrent lanes and %d trees — the reader is broken, not the config; fix it before trusting this gate.",
			len(r.concurrent), len(r.trees))
	}
	return r
}

// TestEveryLeafDeclaresLane is the coverage half: every internal/<leaf> package on disk has
// BOTH a [lanes] declaration (concurrent or exclusive) and a [lanes.trees] entry rooted at
// its own subtree. An undeclared leaf is not a missing label — it is an empty tree, and an
// empty tree conservatively overlaps every live lease (laneadmit.Decide), so it drops the
// fleet's parallel width to one for as long as anyone works on it.
//
// FLEET-SAFETY SCOPING (#2088, the same contract TestEveryPackageDeclaresTier uses):
// architest runs on EVERY push to the shared trunk, so a naive "any undeclared leaf fails"
// would let one agent's forgotten lane wedge every peer's unrelated push. The failure is
// therefore scoped to the leaves THIS push's commits actually deliver: an undeclared leaf
// this push touched is a hard failure its author must fix; a peer's leaf, or pre-existing
// trunk debt, is an advisory (t.Logf). With no trunk to diff against (a `git archive`
// checkout) it falls back to strict all-undeclared-fail, safe there because no concurrent
// peer push exists to wedge.
func TestEveryLeafDeclaresLane(t *testing.T) {
	internal := internalDir(t)
	roster := readLaneRoster(t)
	touched, scoped := leavesTouchedByPush(internal)

	type targetLeaf struct {
		base string
		leaf string
	}
	var targets []targetLeaf
	for _, leaf := range goPackageDirs(t, internal) {
		targets = append(targets, targetLeaf{base: "internal", leaf: leaf})
	}
	pkgDir := filepath.Join(filepath.Dir(internal), "pkg")
	if info, err := os.Stat(pkgDir); err == nil && info.IsDir() {
		for _, leaf := range goPackageDirs(t, pkgDir) {
			targets = append(targets, targetLeaf{base: "pkg", leaf: leaf})
		}
	}

	var missing, missingLeaves []string
	for _, tgt := range targets {
		var why []string
		if !roster.declared(tgt.leaf) {
			why = append(why, "no [lanes] declaration")
		}
		if !treeCoversLeaf(roster.trees[tgt.leaf], tgt.leaf) {
			why = append(why, "no [lanes.trees] entry rooted at "+tgt.base+"/"+tgt.leaf+"/")
		}
		if len(why) == 0 {
			continue
		}
		detail := tgt.base + "/" + tgt.leaf + ": " + strings.Join(why, " and ")
		if undeclaredLeafVerdict(scoped, touched[tgt.leaf]) {
			t.Logf("advisory: %s, but this push's commits do not touch it (a peer's leaf, or "+
				"pre-existing trunk debt). Its owner should declare it with:\n  %s\n"+
				"This push is not blocked for it.", detail, laneDeclarationFix(tgt.leaf))
			continue
		}
		missing = append(missing, detail)
		missingLeaves = append(missingLeaves, tgt.leaf)
	}
	if len(missing) == 0 {
		return
	}
	sort.Strings(missing)
	sort.Strings(missingLeaves)
	t.Errorf("%d leaf/leaves have no lane of their own:\n  %s\n"+
		"An undeclared leaf is NOT a cosmetic gap. laneadmit.Decide falls back to the lane's "+
		"declared tree when a request carries none, and an empty tree conservatively overlaps "+
		"EVERYTHING — so a worker on an undeclared leaf collides with every live lease and the "+
		"fleet's concurrency drops to one until it finishes. dos.toml's header states the "+
		"invariant: ONE LANE PER LEAF.\nFix — add both halves to dos.toml, e.g. for %q:\n%s",
		len(missing), strings.Join(missing, "\n  "), missingLeaves[0], laneDeclarationFix(missingLeaves[0]))
}

// treeCoversLeaf reports whether a lane's declared globs actually root at the leaf's own
// subtree. A lane declared with someone else's glob is the same empty-tree hazard wearing
// a name, so the entry has to point at internal/<leaf>/ or pkg/<leaf>/.
func treeCoversLeaf(globs []string, leaf string) bool {
	wantInternal := "internal/" + leaf + "/"
	wantPkg := "pkg/" + leaf + "/"
	for _, g := range globs {
		trimmed := strings.TrimSpace(g)
		if strings.HasPrefix(trimmed, wantInternal) || strings.HasPrefix(trimmed, wantPkg) {
			return true
		}
	}
	return false
}

// treePrefix reduces a lane's glob to the path prefix it actually claims:
// "internal/foo/**" -> "internal/foo/". A glob with no wildcard tail is an exact path and
// keeps its own text.
func treePrefix(glob string) string {
	return strings.TrimSuffix(strings.TrimSpace(glob), "**")
}

// pathContains reports whether prefix outer claims everything under inner. Containment is
// SEGMENT-WISE, not string-prefix: that is the trap treeCoversLeaf already documents
// ("internal/agentdojo/**" must not claim leaf "agent"), and it applies just as much when
// two lanes are compared against each other.
func pathContains(outer, inner string) bool {
	if outer == "" {
		// The empty tree is the hazard this whole file exists for: laneadmit treats it as
		// overlapping EVERYTHING, so it can never be disjoint from a peer.
		return true
	}
	if !strings.HasPrefix(inner, outer) {
		return false
	}
	return strings.HasSuffix(outer, "/") || strings.HasPrefix(inner[len(outer):], "/")
}

// treesOverlap reports whether two lane globs can ever claim the same file.
func treesOverlap(a, b string) bool {
	pa, pb := treePrefix(a), treePrefix(b)
	return pa == pb || pathContains(pa, pb) || pathContains(pb, pa)
}

// TestLeafTreesArePairwiseDisjoint is the third half of the same invariant: dos.toml's
// header says "the honest partition is ONE LANE PER LEAF", and a partition is only a
// partition if the parts do not overlap. Coverage (TestEveryLeafDeclaresLane) proves every
// leaf HAS a tree and uniqueness (TestLaneRosterHasNoDuplicates) proves no lane is declared
// twice; neither notices two DIFFERENT lanes both claiming internal/<leaf>/. That case is
// not cosmetic either — laneadmit's geometric rung would then refuse two workers on lanes
// that were meant to be independent, so the fleet's parallel width silently drops, which is
// the exact cost the coverage gate was written to prevent.
//
// Scope: globs rooted at internal/, i.e. the LEAF trees the header invariant is about. The
// non-leaf trees (cmd/**, docs/**, the release file list) are deliberately nesting umbrella
// scopes and are not part of the leaf partition.
//
// Unscoped by push, for the same reason TestLaneRosterHasNoDuplicates is: dos.toml is
// itself an EXCLUSIVE lane, so only one worker at a time may edit it and a new overlap is
// always attributable to the commit that wrote it — there is no peer to wedge.
func TestLeafTreesArePairwiseDisjoint(t *testing.T) {
	roster := readLaneRoster(t)
	type claim struct{ lane, glob string }
	var claims []claim
	for lane, globs := range roster.trees {
		for _, g := range globs {
			if g = strings.TrimSpace(g); strings.HasPrefix(g, "internal/") {
				claims = append(claims, claim{lane, g})
			}
		}
	}
	if len(claims) < 100 {
		t.Fatalf("only %d internal/-rooted lane trees parsed out of dos.toml — the reader is "+
			"broken, not the config; a gate that inspects nothing cannot fire.", len(claims))
	}
	sort.Slice(claims, func(i, j int) bool {
		if claims[i].glob != claims[j].glob {
			return claims[i].glob < claims[j].glob
		}
		return claims[i].lane < claims[j].lane
	})
	var bad []string
	for i := range claims {
		for j := i + 1; j < len(claims); j++ {
			if claims[i].lane == claims[j].lane || !treesOverlap(claims[i].glob, claims[j].glob) {
				continue
			}
			bad = append(bad, "lane "+claims[i].lane+" ("+claims[i].glob+") overlaps lane "+
				claims[j].lane+" ("+claims[j].glob+")")
		}
	}
	if len(bad) > 0 {
		t.Errorf("dos.toml [lanes.trees] declares %d overlapping leaf tree pair(s):\n  %s\n"+
			"Leaf trees must be PAIRWISE DISJOINT — one lane per leaf, each rooted at its own "+
			"internal/<leaf>/ prefix and nothing else. Move the shared path onto exactly one "+
			"lane (a cmd/fak shim belongs to the `cmd` lane, not to its leaf's lane).",
			len(bad), strings.Join(bad, "\n  "))
	}
}

// duplicateLanes returns each lane that appears more than once, once per extra occurrence.
// Split out as a pure helper so the uniqueness rule is testable without the real dos.toml.
func duplicateLanes(lanes []string) []string {
	seen := map[string]bool{}
	var dups []string
	for _, lane := range lanes {
		if seen[lane] {
			dups = append(dups, lane)
			continue
		}
		seen[lane] = true
	}
	return dups
}

func laneDeclarationFix(leaf string) string {
	return "  [lanes].concurrent  += \"" + leaf + "\"\n" +
		"  [lanes.trees]        " + leaf + " = [\"internal/" + leaf + "/**\"]\n" +
		"  (or let `fak new-leaf " + leaf + "` write both at the `# new-leaf:` markers). " +
		"Keep the tree at the leaf's own prefix so leaf trees stay pairwise disjoint, and " +
		"declare the lane EXACTLY ONCE."
}

// TestLaneCoverageRulesRejectTheRegression proves the two gates above actually FIRE — a
// green coverage test over a healthy dos.toml is otherwise indistinguishable from a gate
// that can never fail. These are the pure rules, exercised on synthetic input.
func TestLaneCoverageRulesRejectTheRegression(t *testing.T) {
	for _, tc := range []struct {
		name  string
		globs []string
		leaf  string
		want  bool
	}{
		{"own subtree", []string{"internal/agent/**"}, "agent", true},
		{"pkg subtree", []string{"pkg/fakclient/**"}, "fakclient", true},
		{"one of several", []string{"experiments/**", "internal/experiments/**"}, "experiments", true},
		{"no entry at all", nil, "agent", false},
		{"someone else's tree", []string{"internal/agenttopo/**"}, "agent", false},
		{"someone else's pkg tree", []string{"pkg/fakclientextra/**"}, "fakclient", false},
		{"prefix is not a path boundary", []string{"internal/agentdojo/**"}, "agent", false},
		{"repo-wide glob does not claim a leaf", []string{"**/*"}, "agent", false},
	} {
		if got := treeCoversLeaf(tc.globs, tc.leaf); got != tc.want {
			t.Errorf("treeCoversLeaf(%q, %q) = %v, want %v (%s)", tc.globs, tc.leaf, got, tc.want, tc.name)
		}
	}
	for _, tc := range []struct {
		name string
		a, b string
		want bool
	}{
		{"two distinct leaves", "internal/mixedprecision/**", "internal/quantpolicy/**", false},
		{"the string-prefix trap", "internal/agent/**", "internal/agenttopo/**", false},
		{"same tree twice", "internal/agent/**", "internal/agent/**", true},
		{"a leaf swallowing a nested leaf", "internal/resume/**", "internal/resume/scan/**", true},
		{"an empty tree overlaps everything", "internal/agent/**", "", true},
	} {
		if got := treesOverlap(tc.a, tc.b); got != tc.want {
			t.Errorf("treesOverlap(%q, %q) = %v, want %v (%s)", tc.a, tc.b, got, tc.want, tc.name)
		}
	}
	if got := duplicateLanes([]string{"a", "b", "a", "c", "b"}); len(got) != 2 {
		t.Errorf("duplicateLanes found %v, want the two repeats — the dup gate cannot fire.", got)
	}
	if got := duplicateLanes([]string{"a", "b", "c"}); len(got) != 0 {
		t.Errorf("duplicateLanes reported %v on a clean roster — false positive.", got)
	}
}

func TestSyntheticUndeclaredPackageFiresLaneGate(t *testing.T) {
	roster := laneRoster{
		concurrent: []string{"gateway", "fakclient"},
		trees: map[string][]string{
			"gateway":   {"internal/gateway/**"},
			"fakclient": {"pkg/fakclient/**"},
		},
	}

	targets := []struct {
		base string
		leaf string
	}{
		{"internal", "gateway"},         // declared & covered -> ok
		{"pkg", "fakclient"},            // declared & covered in pkg -> ok
		{"internal", "mockinternalgap"}, // undeclared in internal -> must fail
		{"pkg", "mockpkggap"},           // undeclared in pkg -> must fail
		{"pkg", "notree"},               // declared in concurrent but missing tree -> must fail
	}
	roster.concurrent = append(roster.concurrent, "notree")

	var missing []string
	for _, tgt := range targets {
		var why []string
		if !roster.declared(tgt.leaf) {
			why = append(why, "no [lanes] declaration")
		}
		if !treeCoversLeaf(roster.trees[tgt.leaf], tgt.leaf) {
			why = append(why, "no [lanes.trees] entry rooted at "+tgt.base+"/"+tgt.leaf+"/")
		}
		if len(why) > 0 {
			missing = append(missing, tgt.base+"/"+tgt.leaf+": "+strings.Join(why, " and "))
		}
	}

	if len(missing) != 3 {
		t.Fatalf("expected 3 failures (mockinternalgap, mockpkggap, notree), got %d: %v", len(missing), missing)
	}

	wantMissing := []string{
		"internal/mockinternalgap: no [lanes] declaration and no [lanes.trees] entry rooted at internal/mockinternalgap/",
		"pkg/mockpkggap: no [lanes] declaration and no [lanes.trees] entry rooted at pkg/mockpkggap/",
		"pkg/notree: no [lanes.trees] entry rooted at pkg/notree/",
	}
	for i, want := range wantMissing {
		if missing[i] != want {
			t.Errorf("missing[%d] = %q, want %q", i, missing[i], want)
		}
	}
}

// TestLaneRosterHasNoDuplicates is the uniqueness half. dos.toml carries a documented
// history of this exact regression (the 2026-06-28 / 2026-06-29 CONCURRENT_LANES_OVERLAP
// notes): a lane listed twice in the concurrent roster is a self-overlap that makes the
// arbiter's roster SPAWN-ORDER-SENSITIVE — the same request can be admitted or refused
// depending on which occurrence the walk reaches first. A lane declared both concurrent
// and exclusive is worse: the two rules contradict (run-alone vs run-with-disjoint-peers).
//
// Unlike the coverage gate this is unscoped. dos.toml is itself an EXCLUSIVE lane, so only
// one worker at a time may edit it and a duplicate is always attributable to the commit
// that wrote it — there is no peer to wedge.
func TestLaneRosterHasNoDuplicates(t *testing.T) {
	roster := readLaneRoster(t)
	for _, arr := range []struct {
		name  string
		lanes []string
	}{{"concurrent", roster.concurrent}, {"exclusive", roster.exclusive}} {
		if dups := duplicateLanes(arr.lanes); len(dups) > 0 {
			sort.Strings(dups)
			t.Errorf("dos.toml [lanes].%s declares %d lane(s) more than once: %s\n"+
				"Remove the duplicate second occurrence — a self-overlapping roster makes the "+
				"arbiter spawn-order-sensitive for those lanes (CONCURRENT_LANES_OVERLAP). "+
				"Each lane must be declared EXACTLY ONCE.", arr.name, len(dups), strings.Join(dups, ", "))
		}
	}
	excl := map[string]bool{}
	for _, lane := range roster.exclusive {
		excl[lane] = true
	}
	var both []string
	for _, lane := range roster.concurrent {
		if excl[lane] {
			both = append(both, lane)
		}
	}
	if len(both) > 0 {
		sort.Strings(both)
		t.Errorf("dos.toml declares %s in BOTH [lanes].concurrent and [lanes].exclusive.\n"+
			"The two rules contradict: an exclusive lane runs alone, a concurrent one runs "+
			"alongside any disjoint peer. Pick one.", strings.Join(both, ", "))
	}
}
