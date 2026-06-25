package hooks

import "strings"

// gate_indexsync.go — the INDEX_SYNC gate, a port of tools/check_index_sync.py. It triggers only
// when an index file (INDEX.md / llms.txt) or a docs/notes/ file is staged. Two checks: dangling
// .md links in a staged index file, and dated docs/notes/ notes (newly added) not listed in
// INDEX.md.

var indexFiles = []string{"INDEX.md", "llms.txt"}

func gateIndexSync(d *StagedDiff) ([]Finding, error) {
	staged := map[string]bool{}
	for _, s := range d.StagedPaths {
		staged[s] = true
	}
	// also consider --name-only (the Python uses `git diff --cached --name-only` for the trigger)
	relevant := false
	for _, s := range d.StagedPaths {
		if isIndexFile(s) || strings.HasPrefix(s, "docs/notes/") {
			relevant = true
			break
		}
	}
	if !relevant {
		return nil, nil
	}

	var findings []Finding

	// DANGLING: for each STAGED index file, every relative .md link must resolve.
	for _, idx := range indexFiles {
		if !staged[idx] {
			continue
		}
		body, ok := d.FileBytes(idx)
		if !ok {
			continue
		}
		idxDir := dirOf(idx)
		for _, link := range indexLinks(string(body)) {
			if !d.Exists(joinClean(idxDir, link)) {
				findings = append(findings, Finding{
					Gate: "INDEX_SYNC", File: idx,
					Detail: "](" + link + ")  ->  missing file",
				})
			}
		}
	}

	// ORPHAN: newly-added dated docs/notes/ notes not listed in INDEX.md.
	index, _ := d.IndexMD()
	for _, p := range d.AddedPaths {
		if !strings.HasPrefix(p, "docs/notes/") {
			continue
		}
		if !isDatedNote(p) {
			continue
		}
		base := baseName(p)
		if !strings.Contains(index, base) {
			findings = append(findings, Finding{
				Gate: "INDEX_SYNC", File: p,
				Detail: "dated note not listed in INDEX.md: " + p + "  —  add a one-line entry to INDEX.md",
			})
		}
	}
	return findings, nil
}

func isIndexFile(s string) bool {
	for _, f := range indexFiles {
		if s == f {
			return true
		}
	}
	return false
}

// indexLinks extracts .md link targets, skipping http/https/mailto/#/absolute (NOTE: check_index_sync
// skips the same set as check_links MINUS data:), de-duped in order. (check_index_sync.py _links.)
func indexLinks(body string) []string {
	var out []string
	for _, m := range linkRE.FindAllStringSubmatch(body, -1) {
		link := m[1]
		if strings.HasPrefix(link, "http://") || strings.HasPrefix(link, "https://") ||
			strings.HasPrefix(link, "mailto:") || strings.HasPrefix(link, "#") ||
			strings.HasPrefix(link, "/") {
			continue
		}
		p := stripFragment(link)
		if p != "" && strings.HasSuffix(p, ".md") {
			out = append(out, p)
		}
	}
	return dedupePreserveOrder(out)
}

// isDatedNote ports _is_dated_note: basename != README.md, ends .md, and (has a date or starts PLAN-).
func isDatedNote(rel string) bool {
	base := baseName(rel)
	if base == "README.md" || !strings.HasSuffix(rel, ".md") {
		return false
	}
	return datedRE.MatchString(base) || strings.HasPrefix(base, "PLAN-")
}

func dirOf(p string) string {
	if i := strings.LastIndex(p, "/"); i >= 0 {
		return p[:i]
	}
	return ""
}

// joinClean joins idxDir and a relative ref and normalizes — os.path.normpath(os.path.join(...)).
func joinClean(dir, ref string) string {
	if dir == "" {
		return cleanSlash(ref)
	}
	return cleanSlash(dir + "/" + ref)
}
