package hooks

import (
	"path"
	"sort"
	"strings"
)

// tree_gates.go — the six remaining `--audit-tree` twins, collected in one file the way
// gate_brandconsistency.go holds the BRAND_CONSISTENCY twin. Each function here is the
// whole-tree counterpart of the staged gate in its gate_*.go sibling: where the staged gate
// scans `git diff --cached`, the tree twin scans the whole `git ls-files` set (t.Paths) and
// reads each in-scope file in full, exactly as the matching tools/check_*.py `--audit-tree`
// branch does. The per-item decision (path predicate, per-line regex, per-file link resolve)
// is IDENTICAL to the staged gate — only the input set and the dropped staged-only sub-rules
// differ — so each twin reuses its sibling's package-level helpers/regexes verbatim.
//
// Parity anchors (Python --audit-tree branch is ground truth):
//   gateDocPlacementTree    ← tools/check_doc_placement.py     (Rule-1 only; unindexed-notes is staged-only)
//   gateBrokenLinkTree      ← tools/check_links.py             (front-door files that EXIST)
//   gateFileAdmissionTree   ← tools/check_committed_files.py   (_classify over the whole tree)
//   gateSecretShapeTree     ← tools/check_secret_shapes.py     (_tracked_text: whole-file scan)
//   gateProvenanceLabelTree ← tools/check_provenance_labels.py (scan_file: readlines, 1-based)
//   gateIndexSyncTree       ← tools/check_index_sync.py        (dangling over existing indexes, orphans over the tree)

// gateDocPlacementTree is the --audit-tree DOC_PLACEMENT gate. The Python tree branch runs ONLY
// _violations over _tracked_paths — the new-note INDEX completeness rule is `args.audit_staged`
// gated (check_doc_placement.py L144: `_unindexed_new_notes(root) if args.audit_staged else []`),
// so the tree twin drops it, exactly like tree.go's header note about DOC_PLACEMENT's staged-only
// sub-rule. The Rule-1 finding wording matches the staged gateDocPlacement.
func gateDocPlacementTree(t *TrackedTree) ([]Finding, error) {
	var findings []Finding
	for _, n := range t.Paths {
		if strings.HasSuffix(n, ".md") && !strings.Contains(n, "/") && !allowedRootMD[n] {
			findings = append(findings, Finding{
				Gate: "DOC_PLACEMENT", File: n,
				Detail: "dated/research doc at the repo root — belongs under docs/notes/ (reached via INDEX.md): " + n + "  ->  docs/notes/" + n,
			})
		}
	}
	return findings, nil
}

// gateBrokenLinkTree is the --audit-tree BROKEN_LINK gate. The Python tree branch checks every
// front-door file that EXISTS (`[f for f in FRONT_DOOR if os.path.exists(...)]`) and runs all
// three sub-checks (dead markdown links, dead inline `code.md` refs, scrub-private refs) over the
// whole file — the same per-file logic as the staged gate, only the file set differs. The resolve
// rule (normpath(join(dir,ref)) / ref / fak-stripped) is replicated against the tree reader.
func gateBrokenLinkTree(t *TrackedTree) ([]Finding, error) {
	var findings []Finding
	for _, f := range frontDoor {
		body, ok := t.FileBytes(f)
		if !ok {
			continue // not present on disk — os.path.exists() false, nothing to check
		}
		content := string(body)
		dir := path.Dir(f)
		findings = append(findings, findDeadLinks(t, f, dir, content)...)
		findings = append(findings, findDeadInlineRefs(t, f, dir, content)...)
		// scrubPrivateRefs is a pure-text check (no disk read) — reuse the staged helper verbatim.
		findings = append(findings, scrubPrivateRefs(f, dir, content)...)
	}
	return findings, nil
}

// gateFileAdmissionTree is the --audit-tree FILE_ADMISSION gate. The Python tree branch runs
// _classify over `sorted(set(_tracked(root)))` — the same classifier as staged, just over the
// whole tree. t.Paths is already sorted+unique (git ls-files), so it matches sorted(set(...)).
// The classifier's only disk dependency is the size cap, taken here from t.Size.
func gateFileAdmissionTree(t *TrackedTree) ([]Finding, error) {
	seen := map[string]bool{}
	var findings []Finding
	for _, p := range t.Paths {
		if seen[p] {
			continue
		}
		seen[p] = true
		if why := classifyFileWith(t, p); why != "" {
			findings = append(findings, Finding{Gate: "FILE_ADMISSION", File: p, Detail: why})
		}
	}
	return findings, nil
}

// gateSecretShapeTree is the --audit-tree SECRET_SHAPE gate. The Python tree branch (_tracked_text)
// reads every tracked file whose path ends in a TEXT_EXT and runs _scan_text over the WHOLE file
// at once (no line numbers in tree mode), deduping findings on (file, hit). This twin matches that:
// scan the full body, emit Line 0 (the Python tree path carries no line number — it readraw().read()
// rather than the staged readlines-with-numbers). selfRefShape + textExt are the shared package sets.
func gateSecretShapeTree(t *TrackedTree) ([]Finding, error) {
	seen := map[string]bool{} // dedupe on (file, hit) like the Python report
	var findings []Finding
	for _, f := range t.Paths {
		norm := strings.ReplaceAll(f, "\\", "/")
		if selfRefShape[norm] {
			continue
		}
		if !textExt[lowerExt(norm)] {
			continue
		}
		body, ok := t.FileBytes(f)
		if !ok {
			continue // OSError / UnicodeDecodeError in Python => skipped
		}
		for _, hit := range scanShapes(string(body)) {
			key := f + "\x00" + hit.text
			if seen[key] {
				continue
			}
			seen[key] = true
			findings = append(findings, Finding{
				Gate: "SECRET_SHAPE", File: f, Line: 0,
				Detail: "[" + hit.shape + "] " + hit.text,
			})
		}
	}
	return findings, nil
}

// gateProvenanceLabelTree is the --audit-tree PROVENANCE_LABEL gate. The Python tree branch
// (scan_file) reads each tracked SCAN_GLOBS file with readlines() and runs _line_violates on each
// line, 1-based, after the SKIP_PREFIXES / SKIP_BASENAMES filter. SCAN_GLOBS (*.md *.html *.txt
// *.json) is exactly provenanceScanExts. lineViolates + the skip sets are the shared helpers; the
// finding wording matches the staged gateProvenanceLabel (trim160(text) + " — fix: " + fix).
func gateProvenanceLabelTree(t *TrackedTree) ([]Finding, error) {
	var findings []Finding
	for _, f := range t.Paths {
		if startsWithAny(f, provenanceSkipPrefixes) || provenanceSkipBasenames[baseName(f)] {
			continue
		}
		if !provenanceScanExts[lowerExt(f)] {
			continue
		}
		body, ok := t.FileBytes(f)
		if !ok {
			continue // OSError in Python scan_file => []
		}
		for i, line := range strings.Split(string(body), "\n") {
			if fix, bad := lineViolates(line); bad {
				findings = append(findings, Finding{
					Gate: "PROVENANCE_LABEL", File: f, Line: i + 1,
					Detail: trim160(line) + " — fix: " + fix,
				})
			}
		}
	}
	return findings, nil
}

// gateIndexSyncTree is the --audit-tree INDEX_SYNC gate. The Python tree branch runs both directions
// unconditionally: DANGLING over each index file that READS (dangling() returns [] on OSError, so a
// missing index is simply skipped), and ORPHAN over `orphans(root)` = every tracked dated docs/notes
// note (git ls-files docs/notes filtered by _is_dated_note), sorted by path, whose basename is absent
// from INDEX.md. indexLinks / isDatedNote / joinClean / dirOf are the shared staged helpers.
func gateIndexSyncTree(t *TrackedTree) ([]Finding, error) {
	var findings []Finding

	// DANGLING: for each index file present in the tree, every relative .md link must resolve.
	for _, idx := range indexFiles {
		body, ok := t.FileBytes(idx)
		if !ok {
			continue // dangling() -> [] on OSError
		}
		idxDir := dirOf(idx)
		for _, link := range indexLinks(string(body)) {
			if !t.Exists(joinClean(idxDir, link)) {
				findings = append(findings, Finding{
					Gate: "INDEX_SYNC", File: idx,
					Detail: "](" + link + ")  ->  missing file",
				})
			}
		}
	}

	// ORPHAN: every tracked dated docs/notes note not listed in INDEX.md, sorted by path
	// (matching orphans(root)'s `sorted(...)`). A missing INDEX.md => "" => orphans() returns []
	// in Python (OSError), so guard on the index being readable.
	index, idxOK := t.IndexMD()
	if idxOK {
		var orphans []string
		for _, p := range t.Paths {
			if !strings.HasPrefix(p, "docs/notes/") {
				continue
			}
			if !isDatedNote(p) {
				continue
			}
			if !strings.Contains(index, baseName(p)) {
				orphans = append(orphans, p)
			}
		}
		sort.Strings(orphans)
		for _, p := range orphans {
			findings = append(findings, Finding{
				Gate: "INDEX_SYNC", File: p,
				Detail: "dated note not listed in INDEX.md: " + p + "  —  add a one-line entry to INDEX.md",
			})
		}
	}
	return findings, nil
}
