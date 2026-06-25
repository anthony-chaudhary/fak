package hooks

import (
	"path"
	"regexp"
	"strings"
)

// gate_brokenlink.go — the BROKEN_LINK gate, a port of tools/check_links.py. For each staged
// front-door doc it checks three things: dead markdown links, dead inline `code.md` refs, and
// references to scrub-private docs. HTTP/mailto/anchor/absolute/data targets are skipped — it is
// an offline relative-path resolver only.

var frontDoor = []string{
	"README.md", "START-HERE.md", "INSTALL.md", "INDEX.md", "AGENTS.md",
	"CLAUDE.md", "CONTRIBUTING.md", "SECURITY.md", "CLA.md",
	"LEARNING-PATH.md", "docs/index.md", "docs/FAQ.md",
}

var (
	linkRE    = regexp.MustCompile(`\]\(([^)]+)\)`)
	inlineRE  = regexp.MustCompile("`([^`]+)`")
	mdTokenRE = regexp.MustCompile(`\A[\w./-]+\.md\z`)
)

var scrubPrivateMD = map[string]bool{"CLAUDE.md": true, "PUBLIC-SCRUB-POLICY.md": true}

func gateBrokenLink(d *StagedDiff) ([]Finding, error) {
	staged := map[string]bool{}
	for _, p := range d.StagedPaths {
		staged[p] = true
	}
	var findings []Finding
	for _, f := range frontDoor {
		if !staged[f] {
			continue
		}
		b, ok := d.FileBytes(f)
		if !ok {
			continue
		}
		content := string(b)
		dir := path.Dir(f)
		findings = append(findings, deadLinks(d, f, dir, content)...)
		findings = append(findings, deadInlineRefs(d, f, dir, content)...)
		findings = append(findings, scrubPrivateRefs(f, dir, content)...)
	}
	return findings, nil
}

// resolves ports _resolves (check_links.py L66-74): try normpath(join(dir,ref)), ref, and the
// fak/-stripped form; resolve if any exists.
func (d *StagedDiff) resolves(dir, ref string) bool {
	cands := []string{path.Clean(path.Join(dir, ref)), ref}
	if strings.HasPrefix(ref, "fak/") {
		cands = append(cands, ref[len("fak/"):])
	}
	for _, c := range cands {
		if d.Exists(c) {
			return true
		}
	}
	return false
}

func skipLinkTarget(link string) bool {
	return strings.HasPrefix(link, "http://") || strings.HasPrefix(link, "https://") ||
		strings.HasPrefix(link, "mailto:") || strings.HasPrefix(link, "#") ||
		strings.HasPrefix(link, "/") || strings.HasPrefix(link, "data:")
}

func deadLinks(d *StagedDiff, f, dir, content string) []Finding {
	var out []Finding
	for _, m := range linkRE.FindAllStringSubmatch(content, -1) {
		link := m[1]
		if skipLinkTarget(link) {
			continue
		}
		p := stripFragment(link)
		if p == "" || d.resolves(dir, p) {
			continue
		}
		out = append(out, Finding{Gate: "BROKEN_LINK", File: f, Detail: "](" + link + ")  ->  missing " + p})
	}
	return out
}

func deadInlineRefs(d *StagedDiff, f, dir, content string) []Finding {
	seen := map[string]bool{}
	var out []Finding
	for _, m := range inlineRE.FindAllStringSubmatch(content, -1) {
		span := m[1]
		ref := stripFragment(firstField(span))
		if !mdTokenRE.MatchString(ref) || seen[ref] {
			continue
		}
		seen[ref] = true
		if !d.resolves(dir, ref) {
			out = append(out, Finding{Gate: "BROKEN_LINK", File: f, Detail: "`" + span + "`  ->  missing " + ref})
		}
	}
	return out
}

func scrubPrivateRefs(f, dir, content string) []Finding {
	if scrubPrivateMD[baseName(f)] {
		return nil
	}
	var out []Finding
	targets := func(ref string) bool {
		p := stripFragment(ref)
		cands := []string{path.Clean(path.Join(dir, p)), p}
		if strings.HasPrefix(p, "fak/") {
			cands = append(cands, p[len("fak/"):])
		}
		for _, c := range cands {
			if scrubPrivateMD[baseName(c)] {
				return true
			}
		}
		return false
	}
	for _, m := range linkRE.FindAllStringSubmatch(content, -1) {
		if targets(m[1]) {
			out = append(out, Finding{Gate: "BROKEN_LINK", File: f, Detail: "](" + m[1] + ")  ->  references a scrub-private doc"})
		}
	}
	for _, m := range inlineRE.FindAllStringSubmatch(content, -1) {
		span := firstField(m[1])
		if span != "" && targets(span) {
			out = append(out, Finding{Gate: "BROKEN_LINK", File: f, Detail: "`" + m[1] + "`  ->  references a scrub-private doc"})
		}
	}
	return out
}

// stripFragment drops a #anchor and ?query, like link.split("#")[0].split("?")[0].
func stripFragment(s string) string {
	if i := strings.IndexByte(s, '#'); i >= 0 {
		s = s[:i]
	}
	if i := strings.IndexByte(s, '?'); i >= 0 {
		s = s[:i]
	}
	return s
}

// firstField returns the first whitespace-delimited token (parts[0] in the Python).
func firstField(s string) string {
	if f := strings.Fields(s); len(f) > 0 {
		return f[0]
	}
	return ""
}
