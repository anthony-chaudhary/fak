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
