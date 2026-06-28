package hooks

import (
	"path"
	"strings"
)

// filesource.go — the shared seam between the staged gates (over *StagedDiff) and their
// whole-tree twins (over *TrackedTree). Both change-set views read the same working tree and
// expose the same path-probe methods, so the per-item decisions the gates duplicated
// (link resolve, dead-link/dead-ref scan, file classification) are collapsed here into one
// helper each, parameterised over a small fileProbe interface. *StagedDiff and *TrackedTree
// both satisfy fileProbe, so the staged gate and its tree twin now call the SAME body — the
// only difference between the two modes stays in the caller (which input set it iterates).

// fileProbe is the path-probe surface the resolve/classify helpers need. Both *StagedDiff and
// *TrackedTree implement it (Exists + Size over the working tree).
type fileProbe interface {
	Exists(rel string) bool
	Size(rel string) (int64, bool)
}

// fileReader adds the whole-file read both change-set views expose, on top of fileProbe. Used by
// the tree-twin loop helpers that read each in-scope file in full (the --audit-tree branches).
type fileReader interface {
	fileProbe
	FileBytes(rel string) ([]byte, bool)
}

// rootMDPlacementFindings ports the DOC_PLACEMENT Rule-1 scan shared by the staged gate and its
// tree twin: every path that is a root-level *.md outside the front-door allowlist is misplaced.
// The only difference between the two modes was the input set (d.StagedPaths vs t.Paths), so the
// per-item decision + finding wording live here once.
func rootMDPlacementFindings(paths []string) []Finding {
	var findings []Finding
	for _, n := range paths {
		if strings.HasSuffix(n, ".md") && !strings.Contains(n, "/") && !allowedRootMD[n] {
			findings = append(findings, Finding{
				Gate: "DOC_PLACEMENT", File: n,
				Detail: "dated/research doc at the repo root — belongs under docs/notes/ (reached via INDEX.md): " + n + "  ->  docs/notes/" + n,
			})
		}
	}
	return findings
}

// classifyPathsFindings ports the FILE_ADMISSION scan shared by the staged gate and its tree twin:
// run _classify over a de-duplicated path set, emitting a FILE_ADMISSION finding for each refusal.
// Staged mode feeds d.AddedRenamedPaths (--diff-filter=AR); tree mode feeds the whole t.Paths set.
func classifyPathsFindings(fp fileProbe, paths []string) []Finding {
	seen := map[string]bool{}
	var findings []Finding
	for _, p := range paths {
		if seen[p] {
			continue
		}
		seen[p] = true
		if why := classifyFileWith(fp, p); why != "" {
			findings = append(findings, Finding{Gate: "FILE_ADMISSION", File: p, Detail: why})
		}
	}
	return findings
}

// danglingIndexLinkFindings ports the INDEX_SYNC DANGLING scan shared by the staged gate and its
// tree twin: for one index file's body, every relative .md link that does not resolve under the
// index's own directory is a dangling INDEX_SYNC finding. The only difference between the two modes
// was which index files the caller iterates (staged-only vs every present index).
func danglingIndexLinkFindings(fp fileProbe, idx, body string) []Finding {
	var findings []Finding
	idxDir := dirOf(idx)
	for _, link := range indexLinks(body) {
		if !fp.Exists(joinClean(idxDir, link)) {
			findings = append(findings, Finding{
				Gate: "INDEX_SYNC", File: idx,
				Detail: "](" + link + ")  ->  missing file",
			})
		}
	}
	return findings
}

// orphanNoteFindings ports the INDEX_SYNC ORPHAN scan shared by the staged gate and its tree twin:
// every dated docs/notes/ note in the input set whose basename is absent from INDEX.md is an
// orphan finding. Staged mode feeds the newly-added paths (in diff order); tree mode feeds the
// whole tracked set (the caller sorts) — the per-path predicate + wording are identical and live here.
func orphanNoteFindings(paths []string, index string) []Finding {
	var findings []Finding
	for _, p := range paths {
		if !strings.HasPrefix(p, "docs/notes/") || !isDatedNote(p) {
			continue
		}
		if !strings.Contains(index, baseName(p)) {
			findings = append(findings, Finding{
				Gate: "INDEX_SYNC", File: p,
				Detail: "dated note not listed in INDEX.md: " + p + "  —  add a one-line entry to INDEX.md",
			})
		}
	}
	return findings
}

// scanTreeFileLines ports the per-file-then-per-line tree scan shared by the BRAND_CONSISTENCY and
// PROVENANCE_LABEL tree twins (and any future line-oriented --audit-tree gate): for each in-scope
// tracked file (inScope), read it in full and run a per-line predicate, emitting a Finding{Gate,
// File, Line:i+1, Detail} for each 1-based line the predicate marks. The gate name and the
// inScope/perLine decisions stay with the caller; the read+split+enumerate scaffold lives here once.
func scanTreeFileLines(fr fileReader, paths []string, gate string, inScope func(rel string) bool, perLine func(line string) (detail string, hit bool)) []Finding {
	var findings []Finding
	for _, f := range paths {
		if !inScope(f) {
			continue
		}
		body, ok := fr.FileBytes(f)
		if !ok {
			continue
		}
		for i, line := range strings.Split(string(body), "\n") {
			if detail, hit := perLine(line); hit {
				findings = append(findings, Finding{Gate: gate, File: f, Line: i + 1, Detail: detail})
			}
		}
	}
	return findings
}

// resolveRef ports _resolves (check_links.py L66-74): try normpath(join(dir,ref)), ref, and the
// fak/-stripped form; resolve if any exists. Shared by the staged BROKEN_LINK gate and its tree
// twin (formerly StagedDiff.resolves / treeResolves).
func resolveRef(fp fileProbe, dir, ref string) bool {
	cands := []string{path.Clean(path.Join(dir, ref)), ref}
	if strings.HasPrefix(ref, "fak/") {
		cands = append(cands, ref[len("fak/"):])
	}
	for _, c := range cands {
		if fp.Exists(c) {
			return true
		}
	}
	return false
}

// findDeadLinks mirrors the dead-markdown-link scan shared by deadLinks / treeDeadLinks: every
// non-skipped relative ]( link ) whose stripped target does not resolve is a BROKEN_LINK.
func findDeadLinks(fp fileProbe, f, dir, content string) []Finding {
	var out []Finding
	for _, m := range linkRE.FindAllStringSubmatch(content, -1) {
		link := m[1]
		if skipLinkTarget(link) {
			continue
		}
		p := stripFragment(link)
		if p == "" || resolveRef(fp, dir, p) {
			continue
		}
		out = append(out, Finding{Gate: "BROKEN_LINK", File: f, Detail: "](" + link + ")  ->  missing " + p})
	}
	return out
}

// findDeadInlineRefs mirrors the dead inline `code.md` ref scan shared by deadInlineRefs /
// treeDeadInlineRefs: every distinct `…` token that looks like a .md path and does not resolve.
func findDeadInlineRefs(fp fileProbe, f, dir, content string) []Finding {
	seen := map[string]bool{}
	var out []Finding
	for _, m := range inlineRE.FindAllStringSubmatch(content, -1) {
		span := m[1]
		ref := stripFragment(firstField(span))
		if !mdTokenRE.MatchString(ref) || seen[ref] {
			continue
		}
		seen[ref] = true
		if !resolveRef(fp, dir, ref) {
			out = append(out, Finding{Gate: "BROKEN_LINK", File: f, Detail: "`" + span + "`  ->  missing " + ref})
		}
	}
	return out
}

// classifyFileWith reproduces _classify's exact precedence (check_committed_files.py L127-156),
// shared by the staged FILE_ADMISSION gate and its tree twin (formerly classifyFile /
// classifyFileTree). The only disk dependency is the size cap, read through fp.Size.
func classifyFileWith(fp fileProbe, p string) string {
	for _, sf := range secretFiles {
		if sf.re.MatchString(p) {
			return sf.why
		}
	}
	for _, po := range privateOnly {
		if po.re.MatchString(p) {
			return po.why
		}
	}
	if keepExceptions[p] {
		if sz, ok := fp.Size(p); ok && sz > fileAdmissionMaxBytes {
			return largeFileMsg(sz)
		}
		return ""
	}
	for _, hj := range hardJunk {
		if hj.MatchString(p) {
			return "build artifact / cache / compiled output"
		}
	}
	if !startsWithAny(p, exemptDataDirs) {
		for _, sj := range softJunk {
			if sj.MatchString(p) {
				return "log / temp / demo-output (regenerable)"
			}
		}
	}
	if sz, ok := fp.Size(p); ok && sz > fileAdmissionMaxBytes {
		return oversizedBlobMsg(sz)
	}
	return ""
}
