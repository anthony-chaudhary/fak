package devindex

// devindex builds a QUERYABLE VIEW over fak's own committed dev facts so an agent
// can ASK instead of re-survey prose every session ("query, don't survey", epic
// #1287 / C1 #1288). It is a VIEW, never a new source of truth: every fact is read
// live from the file that already owns it —
//
//   - the leaf catalog (lane name -> tree glob + the inline `# …` description) and
//     the path->lane resolver come from dos.toml `[lanes.trees]`, the SAME taxonomy
//     the commit-stamp lint and the DOS arbiter bind to;
//   - the doc map (title -> path + blurb) comes from the curated INDEX.md;
//   - the maturity rollup (how many of a leaf's claims are SHIPPED / SIMULATED /
//     STUB) comes from CLAIMS.md, the lint-enforced honesty ledger (C2 #1289). Each
//     ledger line names the package paths it touches; we resolve those to lanes with
//     the SAME LaneForPath the stamp lint uses, so the join cannot drift off-taxonomy.
//
// Because it reads the sources rather than caching them, it cannot drift into a
// parallel reality — a freshness gate (C6 #1293) reds the build if it ever does.
// Pure and stdlib-only (tier foundation): no network, no hot-path coupling.

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Leaf is one entry of fak's lane taxonomy: a lane/leaf name, the tree glob(s) it
// owns, the package directory that tree resolves to, whether that directory exists
// on disk, and the one-line description maintained as the inline dos.toml comment.
type Leaf struct {
	Name   string `json:"name"`
	Tree   string `json:"tree"`
	Dir    string `json:"dir,omitempty"`
	Exists bool   `json:"exists"`
	Desc   string `json:"desc,omitempty"`
	// Status is the CLAIMS.md maturity rollup for this leaf (C2 #1289): how many of
	// the ledger claims that name a path under this leaf are SHIPPED / SIMULATED /
	// STUB. The zero value means the honesty ledger names no capability here.
	Status Status `json:"status"`
}

// Status is a per-leaf rollup of the CLAIMS.md maturity tags that bind to a leaf.
// It answers the recurring "what's shipped vs simulated vs stub for X" without
// reading the ledger prose.
type Status struct {
	Shipped   int `json:"shipped"`
	Simulated int `json:"simulated"`
	Stub      int `json:"stub"`
}

// Total is the number of ledger claims bound to the leaf (across all three tags).
func (s Status) Total() int { return s.Shipped + s.Simulated + s.Stub }

// Claim is one line of the CLAIMS.md honesty ledger: its maturity tag, the `##`
// section it sits under, the lanes its in-line package-path references resolve to
// (via LaneForPath — the SAME taxonomy the commit-stamp lint binds to), and the
// claim prose (the `- [TAG] ` prefix stripped). fak index reads it so an agent
// asks the ledger instead of grepping it.
type Claim struct {
	Tag     string   `json:"tag"`               // SHIPPED | SIMULATED | STUB
	Section string   `json:"section,omitempty"` // the nearest `##`/`###` header above it
	Lanes   []string `json:"lanes,omitempty"`   // leaves the claim's path refs bind to
	Text    string   `json:"text"`              // the claim prose, tag prefix removed
}

// Doc is one entry of the curated doc map (INDEX.md): a human title, the path or
// URL it points at, and the one-line blurb that follows it.
type Doc struct {
	Title string `json:"title"`
	Path  string `json:"path"`
	Blurb string `json:"blurb,omitempty"`
}

// Catalog is the loaded self-index: the leaf taxonomy and the doc map, plus the
// path-prefix maps the lane resolver needs. Build it with Load.
type Catalog struct {
	Root   string  `json:"root"`
	Leaves []Leaf  `json:"leaves"`
	Docs   []Doc   `json:"docs"`
	Claims []Claim `json:"claims,omitempty"`

	// prefixes maps a tree prefix ("internal/gateway/") to its lane ("gateway");
	// exact maps a bare file entry ("version") to its lane. Both lowercased.
	prefixes map[string]string
	exact    map[string]string
}

// FindRoot walks up from start looking for the dos.toml that marks the repo root,
// so `fak index` works from any subdirectory. It returns the first ancestor that
// contains dos.toml, or start unchanged if none is found.
func FindRoot(start string) string {
	dir := start
	if dir == "" {
		dir = "."
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return dir
	}
	for {
		if _, err := os.Stat(filepath.Join(abs, "dos.toml")); err == nil {
			return abs
		}
		parent := filepath.Dir(abs)
		if parent == abs {
			return dir // hit the filesystem root without finding dos.toml
		}
		abs = parent
	}
}

// Load reads the catalog from root (the repo root holding dos.toml and INDEX.md).
// A missing INDEX.md degrades to an empty doc map rather than an error — the leaf
// taxonomy is the load-bearing half. Load only errors when dos.toml is unreadable,
// because without it there is no taxonomy to serve.
func Load(root string) (*Catalog, error) {
	c := &Catalog{Root: root, prefixes: map[string]string{}, exact: map[string]string{}}
	b, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		return nil, err
	}
	c.parseLanes(string(b))

	if idx, err := os.ReadFile(filepath.Join(root, "INDEX.md")); err == nil {
		c.parseDocs(string(idx))
	}
	// CLAIMS.md is parsed AFTER the lanes so the claim->lane join can use the
	// resolver. A missing ledger degrades to an empty rollup, not an error.
	if cl, err := os.ReadFile(filepath.Join(root, "CLAIMS.md")); err == nil {
		c.parseClaims(string(cl))
	}
	return c, nil
}

// laneLineRE captures the comment that trails a `[lanes.trees]` entry. The globs
// never contain '#', so the first '#' after the closing ']' starts the comment.
var laneTokenRE = regexp.MustCompile(`"([^"]+)"`)

// parseLanes scans the `[lanes.trees]` table out of dos.toml. It is a deliberately
// tiny line scanner (the repo carries no TOML dependency): a lane entry is
// `name = ["glob", ...]  # description`, and the comment after the array is the
// per-leaf description this view surfaces (the commit-stamp reader strips it; we
// keep it).
func (c *Catalog) parseLanes(text string) {
	section := ""
	for _, raw := range strings.Split(text, "\n") {
		t := strings.TrimSpace(raw)
		if t == "" || strings.HasPrefix(t, "#") {
			continue // blank or comment-only line (e.g. the new-leaf:tree marker)
		}
		if strings.HasPrefix(t, "[") {
			section = strings.Trim(t, "[]")
			continue
		}
		if section != "lanes.trees" {
			continue
		}
		eq := strings.IndexByte(t, '=')
		if eq < 0 {
			continue
		}
		name := strings.ToLower(strings.TrimSpace(t[:eq]))
		if name == "" {
			continue
		}
		rhs := t[eq+1:]
		arrayPart, desc := rhs, ""
		if h := strings.IndexByte(rhs, '#'); h >= 0 {
			arrayPart, desc = rhs[:h], strings.TrimSpace(rhs[h+1:])
		}

		var globs []string
		for _, m := range laneTokenRE.FindAllStringSubmatch(arrayPart, -1) {
			glob := m[1]
			globs = append(globs, glob)
			if strings.HasSuffix(glob, "**") {
				p := strings.TrimSuffix(strings.TrimSuffix(glob, "**"), "/")
				if p != "" && !strings.Contains(p, "*") {
					c.prefixes[strings.ToLower(p)+"/"] = name
				}
			} else if !strings.Contains(glob, "*") {
				c.exact[strings.ToLower(glob)] = name
			}
		}
		if len(globs) == 0 {
			continue
		}
		leaf := Leaf{Name: name, Tree: strings.Join(globs, ", "), Desc: desc}
		// The leaf's package directory is the first subtree glob's prefix.
		for _, g := range globs {
			if strings.HasSuffix(g, "**") {
				dir := strings.TrimSuffix(strings.TrimSuffix(g, "**"), "/")
				if dir != "" && !strings.Contains(dir, "*") {
					leaf.Dir = dir
					if fi, err := os.Stat(filepath.Join(c.Root, filepath.FromSlash(dir))); err == nil && fi.IsDir() {
						leaf.Exists = true
					}
					break
				}
			}
		}
		c.Leaves = append(c.Leaves, leaf)
	}
	sort.Slice(c.Leaves, func(i, j int) bool { return c.Leaves[i].Name < c.Leaves[j].Name })
}

// docLineRE matches a curated doc-map bullet: `- [Title](path) — blurb`. The blurb
// is optional and may follow an em dash or a hyphen.
var docLineRE = regexp.MustCompile(`^\s*[-*]\s*\[(.+?)\]\(([^)]+)\)\s*(?:[—–-]\s*(.*))?$`)

// parseDocs extracts the curated doc map out of INDEX.md. Only markdown link
// bullets are taken; prose lines are skipped. Titles keep their text minus
// surrounding backticks.
func (c *Catalog) parseDocs(text string) {
	seen := map[string]bool{}
	for _, raw := range strings.Split(text, "\n") {
		m := docLineRE.FindStringSubmatch(raw)
		if m == nil {
			continue
		}
		// Titles carry markdown backticks (often around one word, e.g. "`fleet`
		// console"); strip them all so the title is clean for display AND search.
		title := strings.TrimSpace(strings.ReplaceAll(m[1], "`", ""))
		path := strings.TrimSpace(m[2])
		if path == "" || seen[title+"\x00"+path] {
			continue
		}
		seen[title+"\x00"+path] = true
		c.Docs = append(c.Docs, Doc{Title: title, Path: path, Blurb: strings.TrimSpace(m[3])})
	}
}

// claimTagRE matches a real ledger claim line: `- [TAG] prose`. The legend lines
// at the top of CLAIMS.md write the tag in backticks (“ - `[SHIPPED]` — … “), so
// the literal `[` right after the bullet excludes them — only the lint-enforced
// `- [TAG]` capability lines (unit 96) are taken.
var claimTagRE = regexp.MustCompile(`^-\s*\[(SHIPPED|SIMULATED|STUB)\]\s*(.*)$`)

// pkgRefRE finds the in-line package/path references a claim names (a lane dir like
// `internal/gitgate`, `cmd/ctxbench`, `tools/...`), in or out of backticks. It
// captures only the top dir + FIRST path segment — the part that determines the
// lane — so a package-qualified Go SYMBOL (`internal/engine.RunCapacityPressure…`)
// resolves to its package lane (`engine`), not a bogus dotted pseudo-lane. Each
// match is resolved by LaneForPath so the join binds to the SAME taxonomy the
// commit-stamp lint uses.
var pkgRefRE = regexp.MustCompile(`(?:internal|cmd|tools|docs|examples|visuals|experiments)/[A-Za-z0-9_-]+`)

// parseClaims scans the CLAIMS.md honesty ledger into c.Claims and folds each
// claim's maturity tag onto every leaf its path references resolve to. It is a
// VIEW: it never rewrites the ledger, only reads the bytes the lint already
// guards. Section headers (`##` / `###`) are tracked so a claim carries the
// subsystem it sits under.
func (c *Catalog) parseClaims(text string) {
	section := ""
	for _, raw := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(raw)
		if strings.HasPrefix(trimmed, "#") {
			section = strings.TrimSpace(strings.TrimLeft(trimmed, "# "))
			continue
		}
		m := claimTagRE.FindStringSubmatch(trimmed)
		if m == nil {
			continue
		}
		c.Claims = append(c.Claims, Claim{
			Tag:     m[1],
			Section: section,
			Lanes:   c.lanesInText(m[2]),
			Text:    strings.TrimSpace(m[2]),
		})
	}

	idx := make(map[string]int, len(c.Leaves))
	for i := range c.Leaves {
		idx[c.Leaves[i].Name] = i
	}
	for _, cl := range c.Claims {
		for _, lane := range cl.Lanes {
			i, ok := idx[lane]
			if !ok {
				continue // a claim may name a path with no declared leaf; still searchable
			}
			switch cl.Tag {
			case "SHIPPED":
				c.Leaves[i].Status.Shipped++
			case "SIMULATED":
				c.Leaves[i].Status.Simulated++
			case "STUB":
				c.Leaves[i].Status.Stub++
			}
		}
	}
}

// lanesInText resolves every package-path reference in a claim line to its lane,
// de-duplicated and sorted, so a claim that names a path three times counts once.
func (c *Catalog) lanesInText(text string) []string {
	seen := map[string]bool{}
	var out []string
	for _, ref := range pkgRefRE.FindAllString(text, -1) {
		lane := c.LaneForPath(ref)
		if lane == "" || seen[lane] {
			continue
		}
		seen[lane] = true
		out = append(out, lane)
	}
	sort.Strings(out)
	return out
}

// ClaimsForLeaf returns the ledger claims that bind to the named (case-insensitive)
// leaf, in ledger order — the detail behind a leaf's Status rollup.
func (c *Catalog) ClaimsForLeaf(name string) []Claim {
	n := strings.ToLower(strings.TrimSpace(name))
	var out []Claim
	for _, cl := range c.Claims {
		for _, l := range cl.Lanes {
			if l == n {
				out = append(out, cl)
				break
			}
		}
	}
	return out
}

// SearchClaims returns the ledger claims matching the query, lexically scored (a
// lane match weighs most, then the section, then the prose) and ranked best-first.
// An empty query returns nothing — a ledger search with no terms is a usage error
// the caller surfaces. This is the "what's shipped vs simulated vs stub for X" ask.
func (c *Catalog) SearchClaims(query string) []Claim {
	toks := tokens(query)
	if len(toks) == 0 {
		return nil
	}
	type scored struct {
		cl Claim
		s  int
	}
	var hits []scored
	for _, cl := range c.Claims {
		lanes := strings.ToLower(strings.Join(cl.Lanes, " "))
		section, text := strings.ToLower(cl.Section), strings.ToLower(cl.Text)
		score := 0
		for _, tk := range toks {
			if strings.Contains(lanes, tk) {
				score += 3
			}
			if strings.Contains(section, tk) {
				score += 2
			}
			if strings.Contains(text, tk) {
				score++
			}
		}
		if score > 0 {
			hits = append(hits, scored{cl, score})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].s != hits[j].s {
			return hits[i].s > hits[j].s
		}
		return hits[i].cl.Text < hits[j].cl.Text
	})
	out := make([]Claim, len(hits))
	for i, h := range hits {
		out[i] = h.cl
	}
	return out
}

// LaneForPath maps one repo-relative path to its lane: the exact-file map first,
// then the longest matching [lanes.trees] subtree prefix (authoritative), then the
// directory convention (internal/<X> -> X, cmd/** -> cmd, a top-level lane dir ->
// itself). It mirrors internal/hooks.laneForPath so `fak index lane` and the
// commit-stamp lint reach the SAME answer. "" when no lane can be inferred.
func (c *Catalog) LaneForPath(path string) string {
	p := normPath(path)
	lp := strings.ToLower(p)
	if lane, ok := c.exact[lp]; ok {
		return lane
	}
	best, bestLane := "", ""
	for prefix, lane := range c.prefixes {
		if strings.HasPrefix(lp, prefix) && len(prefix) > len(best) {
			best, bestLane = prefix, lane
		}
	}
	if bestLane != "" {
		return bestLane
	}
	seg := strings.Split(p, "/")
	if len(seg) >= 2 {
		switch seg[0] {
		case "internal":
			return strings.ToLower(seg[1])
		case "cmd":
			return "cmd"
		case "docs", "tools", "examples", "visuals", "experiments":
			return seg[0]
		}
	}
	return ""
}

// SuggestStamp renders the `(fak <leaf>)` ship-stamp trailer the path implies, or
// "" when no lane can be inferred — the answer an agent otherwise greps dos.toml
// for before every commit.
func (c *Catalog) SuggestStamp(path string) string {
	if lane := c.LaneForPath(path); lane != "" {
		return "(fak " + lane + ")"
	}
	return ""
}

// LeafByName returns the leaf with the given (case-insensitive) name, or false.
func (c *Catalog) LeafByName(name string) (Leaf, bool) {
	n := strings.ToLower(strings.TrimSpace(name))
	for _, l := range c.Leaves {
		if l.Name == n {
			return l, true
		}
	}
	return Leaf{}, false
}

// SearchLeaves returns the leaves whose name, tree, or description matches every
// whitespace-separated query token (case-insensitive), ranked by where the match
// landed (a name hit outranks a description hit). An empty query returns every
// leaf in name order.
func (c *Catalog) SearchLeaves(query string) []Leaf {
	toks := tokens(query)
	if len(toks) == 0 {
		out := make([]Leaf, len(c.Leaves))
		copy(out, c.Leaves)
		return out
	}
	type scored struct {
		l Leaf
		s int
	}
	var hits []scored
	for _, l := range c.Leaves {
		name, tree, desc := strings.ToLower(l.Name), strings.ToLower(l.Tree), strings.ToLower(l.Desc)
		score, all := 0, true
		for _, tk := range toks {
			switch {
			case strings.Contains(name, tk):
				score += 3
			case strings.Contains(tree, tk):
				score += 2
			case strings.Contains(desc, tk):
				score++
			default:
				all = false
			}
		}
		if all {
			hits = append(hits, scored{l, score})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].s != hits[j].s {
			return hits[i].s > hits[j].s
		}
		return hits[i].l.Name < hits[j].l.Name
	})
	out := make([]Leaf, len(hits))
	for i, h := range hits {
		out[i] = h.l
	}
	return out
}

// SearchDocs returns the doc-map entries matching the query, lexically scored
// (a title hit weighs most, then the path, then the blurb) and ranked best-first.
// A doc must match at least one query token. An empty query returns nothing — a
// doc search with no terms is a usage error the caller surfaces.
func (c *Catalog) SearchDocs(query string) []Doc {
	toks := tokens(query)
	if len(toks) == 0 {
		return nil
	}
	type scored struct {
		d Doc
		s int
	}
	var hits []scored
	for _, d := range c.Docs {
		title, path, blurb := strings.ToLower(d.Title), strings.ToLower(d.Path), strings.ToLower(d.Blurb)
		score := 0
		for _, tk := range toks {
			if strings.Contains(title, tk) {
				score += 3
			}
			if strings.Contains(path, tk) {
				score += 2
			}
			if strings.Contains(blurb, tk) {
				score++
			}
		}
		if score > 0 {
			hits = append(hits, scored{d, score})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].s != hits[j].s {
			return hits[i].s > hits[j].s
		}
		return hits[i].d.Title < hits[j].d.Title
	})
	out := make([]Doc, len(hits))
	for i, h := range hits {
		out[i] = h.d
	}
	return out
}

// tokens lowercases and splits a query on whitespace, dropping empties.
func tokens(q string) []string {
	var out []string
	for _, f := range strings.Fields(strings.ToLower(q)) {
		if f != "" {
			out = append(out, f)
		}
	}
	return out
}

// normPath canonicalizes a path to forward slashes with no leading "./" so the
// lane resolver compares against the dos.toml globs uniformly.
func normPath(path string) string {
	p := strings.ReplaceAll(path, "\\", "/")
	return strings.TrimPrefix(p, "./")
}
