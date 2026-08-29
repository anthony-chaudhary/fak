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
// Foundation tier, off the hot path: no network, no hot-path coupling. It composes a
// single sibling foundation primitive — internal/trigram — for the fuzzy fallback
// that kills Search* false-ABSENTs (#3925), so it is a foundation COMPOSITE rather
// than a pureRoot primitive (see internal/architest pureRoot).

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/docsearch"
	"github.com/anthony-chaudhary/fak/internal/trigram"
)

// Leaf is one entry of fak's lane taxonomy: a lane/leaf name, the tree glob(s) it
// owns, the package directory that tree resolves to, whether that directory exists
// on disk, the one-line description maintained as the inline dos.toml comment, and
// the module's current derived version (the version-everything spine, #2465).
type Leaf struct {
	Name   string `json:"name"`
	Tree   string `json:"tree"`
	Dir    string `json:"dir,omitempty"`
	Exists bool   `json:"exists"`
	Desc   string `json:"desc,omitempty"`
	// Version is the leaf's current module version — "r<rev>+g<sha>", the SAME string
	// `fak version modules` prints — read from the last fak-module-versions/1 ledger
	// row whose module equals this leaf's Dir (#2465). It lets an agent reason about a
	// leaf's staleness from the index payload without shelling out. Empty when the
	// ledger names no version for this Dir, or the ledger is absent: a staleness hint,
	// never load-bearing, so it degrades quietly.
	Version string `json:"version,omitempty"`
	// Status is the CLAIMS.md maturity rollup for this leaf (C2 #1289): how many of
	// the ledger claims that name a path under this leaf are SHIPPED / SIMULATED /
	// STUB. The zero value means the honesty ledger names no capability here.
	Status Status `json:"status"`
	// Approx marks a leaf returned by the trigram fuzzy fallback (#3925) rather than an
	// exact substring hit: the query only NEAR-matched this leaf's name/tree/desc, so a
	// caller can flag it as approximate. False (and omitted) on every exact hit — the
	// fallback engages only when exact scoring found nothing, so exact and approximate
	// results never mix in one response.
	Approx bool `json:"approx,omitempty"`
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
	// Approx marks a claim returned by the trigram fuzzy fallback (#3925) — a near-miss
	// on the lanes/section/text rather than an exact substring hit. Omitted on exact hits.
	Approx bool `json:"approx,omitempty"`
}

// Doc remains the development index's public documentation row while the shared
// loader/search authority lives in docsearch, outside the runtime-forbidden package.
type Doc = docsearch.Doc

// Catalog is the loaded self-index: the leaf taxonomy and the doc map, plus the
// path-prefix maps the lane resolver needs. Build it with Load.
type Catalog struct {
	Root        string       `json:"root"`
	Leaves      []Leaf       `json:"leaves"`
	Docs        []Doc        `json:"docs"`
	Claims      []Claim      `json:"claims,omitempty"`
	Generations []Generation `json:"generations,omitempty"`

	// prefixes maps a tree prefix ("internal/gateway/") to its lane ("gateway");
	// exact maps a bare file entry ("version") to its lane. Both lowercased.
	prefixes map[string]string
	exact    map[string]string
	tiers    map[string]int
	// declared is the full set of lane names dos.toml declares, lowercased — the
	// names in the flat [lanes] concurrency-class arrays AND the [lanes.trees] keys.
	// It mirrors internal/hooks.readLaneTaxonomy's `declared` exactly, so the
	// undeclared-leaf freshness detector agrees with the authoritative lane-audit gate.
	// Counting only [lanes.trees] keys (as this once did) falsely flags a leaf that is
	// declared in [lanes] but carries no explicit tree glob of its own.
	declared map[string]bool
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
	c := &Catalog{Root: root, prefixes: map[string]string{}, exact: map[string]string{}, tiers: map[string]int{}, declared: map[string]bool{}}
	b, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		return nil, err
	}
	c.parseLanes(string(b))
	if tiers, err := os.ReadFile(filepath.Join(root, "internal", "architest", "architest_test.go")); err == nil {
		c.parseTiers(string(tiers))
	}

	c.Docs = docsearch.LoadDocs(root).Docs
	if gen, err := os.ReadFile(filepath.Join(root, "docs", "generation.md")); err == nil {
		c.Generations = parseGenerations(string(gen))
	} else {
		c.Generations = defaultGenerations()
	}
	// CLAIMS.md is parsed AFTER the lanes so the claim->lane join can use the
	// resolver. A missing ledger degrades to an empty rollup, not an error.
	if cl, err := os.ReadFile(filepath.Join(root, "CLAIMS.md")); err == nil {
		c.parseClaims(string(cl))
	}
	// The module-versions ledger is joined AFTER the lanes so each leaf's Dir is
	// known. A missing ledger degrades to empty versions, not an error — the version
	// is a staleness hint (#2465), never load-bearing.
	if mv, err := os.ReadFile(filepath.Join(root, "docs", "nightrun", "module-versions.jsonl")); err == nil {
		c.joinModuleVersions(mv)
	}
	return c, nil
}

// modVersionRow is the minimal shape this view reads from a fak-module-versions/1
// ledger line: the module path and its "r<rev>+g<sha>" version string. The ledger
// schema is additive-only (a breaking row change is a /2 with its own contract), so
// binding to just these two fields is stable — every other column is ignored.
type modVersionRow struct {
	Module  string `json:"module"`
	Version string `json:"version"`
}

// joinModuleVersions folds the module-versions ledger onto the leaves: it takes the
// LAST row per module (the ledger is append-only, so the final row is the current
// version) and sets Leaf.Version for every leaf whose Dir names that module. It is a
// VIEW — it never rewrites the ledger, only reads the bytes `fak version modules
// --stamp` already produced. A malformed line is skipped, never fatal: a bad ledger
// row must not break the leaf map.
func (c *Catalog) joinModuleVersions(ledger []byte) {
	latest := map[string]string{}
	for _, raw := range strings.Split(string(ledger), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		var row modVersionRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if row.Module == "" || row.Version == "" {
			continue
		}
		latest[row.Module] = row.Version // append-only ⇒ last row wins
	}
	for i := range c.Leaves {
		if v, ok := latest[c.Leaves[i].Dir]; ok {
			c.Leaves[i].Version = v
		}
	}
}

// laneLineRE captures the comment that trails a `[lanes.trees]` entry. The globs
// never contain '#', so the first '#' after the closing ']' starts the comment.
var laneTokenRE = regexp.MustCompile(`"([^"]+)"`)

// splitTOMLComment cuts a line at the first '#' that is outside a quoted string,
// so a glob containing '#' cannot truncate the entry. Globs do not contain '#'
// today; the quote-awareness is here so the array joiner below cannot be the
// thing that breaks when one does.
func splitTOMLComment(line string) (body, comment string) {
	inQuote := false
	for i := 0; i < len(line); i++ {
		switch line[i] {
		case '"':
			inQuote = !inQuote
		case '#':
			if !inQuote {
				return line[:i], strings.TrimSpace(line[i+1:])
			}
		}
	}
	return line, ""
}

// arrayDepthDelta counts unquoted '[' minus unquoted ']' in a line's body. A
// section header (`[lanes.trees]`) and an inline entry both net to zero; only an
// array left open at end-of-line is positive.
func arrayDepthDelta(body string) int {
	depth, inQuote := 0, false
	for i := 0; i < len(body); i++ {
		switch body[i] {
		case '"':
			inQuote = !inQuote
		case '[':
			if !inQuote {
				depth++
			}
		case ']':
			if !inQuote {
				depth--
			}
		}
	}
	return depth
}

// joinLaneArrays collapses a TOML array spanning several lines into the single
// logical line parseLanes's scanner expects.
//
// Why this exists: parseLanes is a line scanner, so the multi-line spelling puts
// a lane's NAME and its GLOBS on different lines. The name alone reaches
// `c.declared[name] = true`, which is what every "is this lane real?" check
// consults — so the lane validates while contributing no prefixes and no exact
// entries, and LaneForPath falls through to `unknown` for every path it owns.
// Nothing errors and no gate goes red; resolution just quietly stops. fak's own
// dos.toml writes every tree inline so the tree is blind to it; it was reported
// by a downstream adopter whose dos.toml wraps its wider doc trees across lines.
//
// Only the final line's comment survives, because the per-leaf description is by
// convention the comment trailing the closing bracket.
func joinLaneArrays(text string) string {
	lines := strings.Split(text, "\n")
	out := make([]string, 0, len(lines))
	var buf strings.Builder
	depth := 0
	for _, raw := range lines {
		body, comment := splitTOMLComment(strings.TrimSpace(raw))
		if depth == 0 {
			if arrayDepthDelta(body) <= 0 {
				out = append(out, raw) // untouched: headers, inline entries, comments
				continue
			}
			buf.Reset()
			buf.WriteString(strings.TrimSpace(body))
			depth = arrayDepthDelta(body)
			continue
		}
		buf.WriteByte(' ')
		buf.WriteString(strings.TrimSpace(body))
		if depth += arrayDepthDelta(body); depth <= 0 {
			if comment != "" {
				buf.WriteString(" # " + comment)
			}
			out = append(out, buf.String())
			depth = 0
		}
	}
	if depth > 0 { // unterminated array: emit what we have rather than drop the lane
		out = append(out, buf.String())
	}
	return strings.Join(out, "\n")
}

// parseLanes scans the `[lanes.trees]` table out of dos.toml. It is a deliberately
// tiny line scanner (the repo carries no TOML dependency): a lane entry is
// `name = ["glob", ...]  # description`, and the comment after the array is the
// per-leaf description this view surfaces (the commit-stamp reader strips it; we
// keep it). Multi-line arrays are folded to that one-line shape first, by
// joinLaneArrays, so the scanner never sees a name without its globs.
func (c *Catalog) parseLanes(text string) {
	section := ""
	for _, raw := range strings.Split(joinLaneArrays(text), "\n") {
		t := strings.TrimSpace(raw)
		if t == "" || strings.HasPrefix(t, "#") {
			continue // blank or comment-only line (e.g. the new-leaf:tree marker)
		}
		if strings.HasPrefix(t, "[") {
			section = strings.Trim(t, "[]")
			continue
		}
		if section == "lanes" {
			// The [lanes] table is a set of concurrency-class arrays whose VALUES are
			// lane names (`concurrent = ["agent", "gateway", ...]`), often spanning many
			// lines. Every quoted token on a line is a declared lane name — the SAME rule
			// internal/hooks.readLaneTaxonomy applies — so a leaf declared here but with
			// no explicit [lanes.trees] glob still counts as declared. Strip an inline
			// comment first so a quoted word in a trailing note is not read as a lane.
			body := t
			if h := strings.IndexByte(body, '#'); h >= 0 {
				body = body[:h]
			}
			for _, m := range laneTokenRE.FindAllStringSubmatch(body, -1) {
				c.declared[strings.ToLower(m[1])] = true
			}
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
		c.declared[name] = true // the [lanes.trees] key is itself a declared lane
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

// markdownInlinePaths is retained for the devindex package's historical parser
// tests; docsearch owns the implementation used by both artifacts.
func markdownInlinePaths(line string) []string {
	return docsearch.InlinePaths(line)
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
		claimText := m[2]
		laneText := claimText + "\n" + c.linkedClaimText(claimText)
		c.Claims = append(c.Claims, Claim{
			Tag:     m[1],
			Section: section,
			Lanes:   c.lanesInText(laneText),
			Text:    strings.TrimSpace(claimText),
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

// linkedClaimText follows only the generated claim-page links in the compact
// CLAIMS.md document set. The index line preserves the maturity tag while the
// page owns the package-bound witness prose; joining both keeps lane rollups
// authoritative after the ledger was split into addressable pages.
func (c *Catalog) linkedClaimText(claimText string) string {
	var joined strings.Builder
	for _, raw := range markdownInlinePaths(claimText) {
		clean := strings.TrimSpace(raw)
		if i := strings.IndexAny(clean, "#?"); i >= 0 {
			clean = clean[:i]
		}
		clean = filepath.ToSlash(filepath.Clean(clean))
		if !strings.HasPrefix(clean, "docs/claims/") || !strings.HasSuffix(clean, ".md") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(c.Root, filepath.FromSlash(clean)))
		if err != nil {
			continue
		}
		joined.Write(b)
		joined.WriteByte('\n')
	}
	return joined.String()
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
// When exact scoring yields NO hit, it falls back to a trigram fuzzy pass over the
// same fields (#3925), returning the best near-misses flagged Approx.
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
	if len(hits) == 0 {
		return c.fuzzyClaims(toks)
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

// fuzzyClaims is SearchClaims' near-miss fallback: trigram similarity over each
// claim's lanes/section/text (same weighting as the exact scorer), best-first,
// flagged Approx. Empty when nothing clears fuzzyThreshold.
func (c *Catalog) fuzzyClaims(toks []string) []Claim {
	type fscored struct {
		cl Claim
		s  float64
	}
	var hits []fscored
	for _, cl := range c.Claims {
		lanes := strings.ToLower(strings.Join(cl.Lanes, " "))
		section, text := strings.ToLower(cl.Section), strings.ToLower(cl.Text)
		if s := fuzzyScore(toks, wfield{lanes, 3}, wfield{section, 2}, wfield{text, 1}); s > 0 {
			cl.Approx = true
			hits = append(hits, fscored{cl, s})
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

// fuzzyThreshold is the minimum trigram similarity a query token must reach against
// a field word for the fuzzy fallback to treat it as a near-miss match. Below this
// the two words share too few 3-rune shingles to be a plausible typo or synonym, so
// admitting them would return noise instead of the intended false-ABSENT fix (#3925).
const fuzzyThreshold = 0.34

// wfield pairs a lowercased field's text with the rank weight an exact hit there
// earns, so the fuzzy fallback ranks a near-miss on a name/title/lane above one on a
// description — mirroring the exact scorer's field weighting.
type wfield struct {
	text   string
	weight int
}

// fieldWords splits a field into the words the fuzzy scorer shingles against, breaking
// on any run of non-alphanumeric runes so a path segment ("internal/gateway"), an
// underscore- or dash-joined identifier, or ordinary prose all yield matchable words.
func fieldWords(s string) []string {
	return strings.FieldsFunc(s, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

// fuzzyScore is the fallback scorer the Search* verbs use ONLY when exact substring
// scoring found nothing. It returns the best weighted trigram similarity between any
// query token and any word of the given fields, or 0 when nothing clears
// fuzzyThreshold. Reusing internal/trigram's shingle ratio — rather than standing up a
// second fuzzy engine — is the whole point of #3925: a synonym or near-miss spelling
// of a real capability still scores, so fak_feature_query stops returning a
// false-ABSENT on a capability that exists.
func fuzzyScore(toks []string, fields ...wfield) float64 {
	best := 0.0
	for _, f := range fields {
		for _, word := range fieldWords(f.text) {
			for _, tk := range toks {
				sim := trigram.Similarity(tk, word)
				if sim < fuzzyThreshold {
					continue
				}
				if s := sim * float64(f.weight); s > best {
					best = s
				}
			}
		}
	}
	return best
}

// SearchLeaves returns the leaves whose name, tree, or description matches every
// whitespace-separated query token (case-insensitive), ranked by where the match
// landed (a name hit outranks a description hit). An empty query returns every leaf
// in name order. When exact substring scoring yields NO hit, it falls back to a
// trigram fuzzy pass over the same fields (#3925), returning the best near-misses
// flagged Approx rather than an empty result.
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
	if len(hits) == 0 {
		return c.fuzzyLeaves(toks)
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

// fuzzyLeaves is SearchLeaves' near-miss fallback: it scores every leaf by trigram
// similarity over its name/tree/desc (same weighting as the exact scorer) and returns
// those clearing fuzzyThreshold, flagged Approx, best-first. Empty when nothing is
// close enough — a genuinely absent capability still returns nothing.
func (c *Catalog) fuzzyLeaves(toks []string) []Leaf {
	type fscored struct {
		l Leaf
		s float64
	}
	var hits []fscored
	for _, l := range c.Leaves {
		name, tree, desc := strings.ToLower(l.Name), strings.ToLower(l.Tree), strings.ToLower(l.Desc)
		if s := fuzzyScore(toks, wfield{name, 3}, wfield{tree, 2}, wfield{desc, 1}); s > 0 {
			l.Approx = true
			hits = append(hits, fscored{l, s})
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
// Multi-term searches rank token coverage first and prefer reference docs over
// historical notes at equal coverage. A doc must match at least one query token.
// An empty query returns nothing — a doc search with no terms is a usage error the
// caller surfaces. When exact scoring yields NO hit, it falls back to a trigram
// fuzzy pass over the same fields (#3925), returning the best near-misses flagged
// Approx rather than an empty result.
func (c *Catalog) SearchDocs(query string) []Doc {
	return (&docsearch.Catalog{Root: c.Root, Docs: c.Docs}).SearchDocs(query)
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

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
