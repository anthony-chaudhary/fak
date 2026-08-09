package hooks

import (
	"sort"
	"strings"
)

// gate_untieredleaf.go — the STAGED (commit-boundary) sibling of gate_tierdeclared.go's
// whole-tree TIER_DECLARED gate.
//
// gateTierDeclaredTree runs only in `fak hygiene` / `make ci`, over the WHOLE tree — so it
// catches an internal/<leaf> package that is missing its architest tier row only AFTER the
// leaf is committed and pushed: architest's TestEveryPackageDeclaresTier then reds the entire
// module, and `fak sync push` refuses on that drift, so one worker's hand-made untiered leaf
// STALLS every peer's push (#3614). The opt-in `fak new-leaf` path declares the row for you,
// but a worker who hand-creates internal/foo/ and commits it never goes through new-leaf.
//
// This gate moves the same check one boundary EARLIER, onto the staged diff of the single
// commit that introduces the leaf: it fires when a commit adds a non-test .go under
// internal/<leaf>/ whose leaf is absent from the tier table, and names the exact one-line edit
// (or the `fak new-leaf` command). The blast radius shrinks from fleet-wide-at-push-time to the
// one author, refused at their own commit. It reuses the tier-table parser gate_tierdeclared.go
// owns (parseDeclaredTierKeys) so it can never become a rival authority on the taxonomy.
//
// ADVISORY by default (DefaultMode "warn" in PreCommitGates): it never reds a shared trunk out
// of the box — FLEET_TIER_GUARD=block hard-enforces it once a soak validates the signal, the
// same ratchet BARE_COMMIT_SWEEP / E2E_OVER_MOCKS / PRIOR_ART use. UNTIERED_LEAF is declared in
// dos.toml [reasons] so `dos_check_reason UNTIERED_LEAF` classifies it as a known OPERATOR_GATE.

// tierInsertMarker mirrors internal/newleaf.TierMarker ("// new-leaf:tier") — the anchor that
// `fak new-leaf` inserts a tier row before. internal/hooks is tier 1 and imports nothing
// internal (it is off the hot path), and newleaf is also tier 1, so hooks cannot import newleaf
// without a forbidden same-tier edge; the marker is a stable literal named only in a finding's
// fix hint, so a local copy carries no rival logic.
const tierInsertMarker = "// new-leaf:tier"

// gateUntieredLeaf emits an UNTIERED_LEAF finding for every internal/<leaf> the staged diff
// NEWLY adds a non-test .go file under (AddedPaths / AddedRenamedPaths — the same "newly
// present" scope FILE_ADMISSION uses) whose leaf is absent from the architest tier table. It
// fails open (ErrCouldNotRun) when the tier table cannot be read or parsed, so the whole-tree
// TIER_DECLARED gate and architest's own test stay the backstop — the pre-commit runner skips a
// could-not-run gate rather than blocking on it.
func gateUntieredLeaf(d *StagedDiff) ([]Finding, error) {
	body, exists := d.FileBytes(tierTableFile)
	if !exists {
		return nil, ErrCouldNotRun
	}
	declared, ok := parseDeclaredTierKeys(body)
	if !ok {
		return nil, ErrCouldNotRun
	}

	// Leaves this commit introduces a non-test .go for. AddedPaths (--diff-filter=A) plus
	// AddedRenamedPaths (=AR) is the "newly present in the tree" set; a leaf already carrying a
	// tier row is skipped, so re-touching a declared leaf never fires.
	newLeaves := map[string]bool{}
	for _, p := range d.AddedPaths {
		if leaf, ok := internalLeafOf(p); ok {
			newLeaves[leaf] = true
		}
	}
	for _, p := range d.AddedRenamedPaths {
		if leaf, ok := internalLeafOf(p); ok {
			newLeaves[leaf] = true
		}
	}

	var findings []Finding
	for leaf := range newLeaves {
		if leaf == "architest" { // architest excludes itself from its own tier scan
			continue
		}
		if declared[leaf] {
			continue
		}
		findings = append(findings, Finding{
			Gate: "UNTIERED_LEAF",
			File: "internal/" + leaf + "/",
			Detail: "new leaf internal/" + leaf + " is being committed without a tier row — architest's " +
				"TestEveryPackageDeclaresTier will red the whole module at push time and stall every peer's " +
				"`fak sync push`. Add `\"" + leaf + "\": <tier>,` before `" + tierInsertMarker + "` in " +
				tierTableFile + " (the LOWEST tier whose role it fits), or run `fak new-leaf " + leaf +
				" --tier <tier>`, and stage the tier table in THIS commit so the leaf never lands untiered.",
		})
	}
	sort.Slice(findings, func(i, j int) bool { return findings[i].File < findings[j].File })
	// The denominator is the set of NEW leaves this commit introduces — almost always zero, and
	// that zero is the point: it separates "no new leaf to check" from "every new leaf declared
	// its tier", which this gate reported identically before (#5602).
	d.NoteCandidates("UNTIERED_LEAF", len(newLeaves), "new internal/<leaf>/ package(s)")
	return findings, nil
}

// internalLeafOf returns the leaf package name of a repo-relative path shaped
// internal/<leaf>/<file>.go, and false otherwise. It mirrors gate_tierdeclared.go's on-disk
// scan exactly: only a non-test .go file DIRECTLY under internal/<leaf>/ counts (a deeper path
// leaves seg[2] a directory name, which architest's own scan also skips), so the staged gate and
// the whole-tree gate agree on which package a file makes "declarable".
func internalLeafOf(p string) (string, bool) {
	seg := strings.Split(p, "/")
	if len(seg) < 3 || seg[0] != "internal" {
		return "", false
	}
	if !strings.HasSuffix(seg[2], ".go") || strings.HasSuffix(seg[2], "_test.go") {
		return "", false
	}
	return strings.ToLower(seg[1]), true
}
