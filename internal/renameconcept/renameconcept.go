// Package renameconcept plans — and mechanically applies — a CONCEPT rename
// across the whole tree: the "dgxbridge -> slackbridge" class of change, where
// one name is baked into directory names, Go identifiers, docs prose, config
// lanes, ignore rules, and historical data records, each wanting a different
// treatment.
//
// The problem it solves is not string replacement — `sed` does that — it is the
// INVENTORY and the TRIAGE. A concept rename in a shared, guarded tree fails in
// three characteristic ways: (1) a site is missed because it uses a different
// case form (DgxBridge / DGX_BRIDGE / dgx-bridge); (2) history is rewritten
// that should have been left alone (dated notes, append-only JSONL ledgers);
// (3) a generated artifact is patched byte-wise instead of regenerated. So the
// core verb here is BuildPlan: expand the concept into its case-form variants,
// scan the tree, and split every touched site into MECHANICAL (the tool may
// rewrite it), HOLDOUT (history/binary — rename forward, never rewrite), and
// IRREGULAR (a casing no variant covers — a human decision). Apply then does
// exactly the mechanical share, deepest-path-first, and re-scans so the
// residual is reported from evidence, not assumed zero.
//
// Pure stdlib, imports nothing internal, off the hot path. The filesystem is
// the only dependency, so the whole plan/apply cycle is testable on a temp
// tree.
package renameconcept

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
)

const Schema = "fak-rename-concept/1"

// maxScanBytes caps how much of one file the scanner will read. Anything larger
// is held out as oversized rather than silently skipped — the plan must never
// under-report a site.
const maxScanBytes = 16 << 20

// skipDirs are trees the scanner never descends into: vendored/derived caches.
// Dot-directories (tool state: .git, .dos, .fak scratch mirrors, .goal-runs
// session inputs, ...) are skipped wholesale by skipDir — with the one
// exception of .github, whose workflows genuinely carry concept names.
// Everything else is scanned and, if untouchable, surfaces as a HOLDOUT —
// visible, not invisible.
var skipDirs = map[string]bool{
	"node_modules": true, "__pycache__": true, "vendor": true,
}

func skipDir(base string) bool {
	if skipDirs[base] {
		return true
	}
	return strings.HasPrefix(base, ".") && base != ".github"
}

// holdoutPrefixes are in-repo trees that record history: rewriting a concept
// name inside them would forge the past. Sites here are planned as holdouts
// unless Options.IncludeHistorical opts in.
var holdoutPrefixes = []string{
	"docs/notes/", "docs/nightrun/", ".dispatch-runs/", "dos.runs/", "coverage/",
}

// binaryExts short-circuit the content scan: these are generated artifacts. A
// name hit on one plans as "regenerate", never as a byte patch.
var binaryExts = map[string]bool{
	".exe": true, ".dll": true, ".so": true, ".dylib": true, ".a": true, ".o": true,
	".png": true, ".jpg": true, ".jpeg": true, ".gif": true, ".ico": true, ".pdf": true,
	".gguf": true, ".bin": true, ".pt": true, ".safetensors": true, ".onnx": true,
	".zip": true, ".gz": true, ".tar": true, ".woff": true, ".woff2": true, ".ttf": true,
}

// Options configures one plan/apply cycle.
type Options struct {
	// Root is the workspace root to scan (required).
	Root string
	// From/To are the concept's current and replacement spellings. Word
	// boundaries may be marked with '-', '_', a space, or camel humps —
	// "dgx-bridge" expands to more case forms than "dgxbridge" can.
	From, To string
	// IncludeHistorical also rewrites the history-shaped holdouts (dated notes,
	// JSONL/CSV ledgers). Default false: rename forward, keep history intact.
	IncludeHistorical bool
}

// Variant is one case-form spelling pair the scan matches and Apply rewrites.
type Variant struct {
	Old string `json:"old"`
	New string `json:"new"`
}

// Site is one file's involvement in the rename.
type Site struct {
	Path string `json:"path"` // repo-relative, forward slashes
	Kind string `json:"kind"` // go|docs|config|script|data|binary|other
	// NameHit — the file/dir base name itself carries a variant (a PathRename
	// covers it).
	NameHit bool `json:"name_hit,omitempty"`
	// Matches is the exact-variant content match count (the mechanical share).
	Matches int `json:"matches"`
	// Irregular counts case-insensitive hits NO variant covers (e.g. DGXBridge
	// when the concept was given as one word). Never rewritten mechanically.
	Irregular  int            `json:"irregular,omitempty"`
	PerVariant map[string]int `json:"per_variant,omitempty"`
	// Hold names why Apply must not touch this site ("" = mechanical).
	Hold string `json:"hold,omitempty"`
}

// PathRename is one file/dir rename, planned deepest-first so children move
// before their parents.
type PathRename struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// Plan is the dry-run product: the full inventory, triaged.
type Plan struct {
	Schema     string       `json:"schema"`
	Workspace  string       `json:"workspace"`
	From       string       `json:"from"`
	To         string       `json:"to"`
	Variants   []Variant    `json:"variants"`
	Renames    []PathRename `json:"path_renames"`
	Mechanical []Site       `json:"mechanical_sites"`
	Holdouts   []Site       `json:"holdout_sites"`
	// FilesScanned / TotalMatches / IrregularMatches summarize scan coverage.
	FilesScanned     int `json:"files_scanned"`
	TotalMatches     int `json:"total_matches"`
	IrregularMatches int `json:"irregular_matches"`
	// CommitPaths is every path an explicit-pathspec commit of this rename
	// must name: rewritten files plus both sides of each rename.
	CommitPaths []string `json:"commit_paths"`
	OK          bool     `json:"ok"`
	Finding     string   `json:"finding"`
	NextAction  string   `json:"next_action"`
}

// ApplyResult reports what Apply actually did, with the residual re-derived
// from a fresh scan — never assumed.
type ApplyResult struct {
	Schema         string       `json:"schema"`
	Workspace      string       `json:"workspace"`
	RewrittenFiles []string     `json:"rewritten_files"`
	RenamedPaths   []PathRename `json:"renamed_paths"`
	SkippedHolds   []Site       `json:"skipped_holdouts"`
	Errors         []string     `json:"errors,omitempty"`
	// Residual is the post-apply rescan: what a follow-up (human or opted-in
	// historical pass) still has to handle.
	Residual   *Plan  `json:"residual,omitempty"`
	OK         bool   `json:"ok"`
	NextAction string `json:"next_action"`
}

// ---- variant expansion --------------------------------------------------------------

// splitWords breaks a concept spelling into words: explicit separators
// ('-', '_', space) first, then camel humps within each token (dgxBridge ->
// dgx, bridge; HTTPServer -> http, server). All words come back lowercase.
func splitWords(s string) []string {
	var out []string
	for _, tok := range strings.FieldsFunc(s, func(r rune) bool {
		return r == '-' || r == '_' || unicode.IsSpace(r)
	}) {
		out = append(out, splitHumps(tok)...)
	}
	return out
}

func splitHumps(tok string) []string {
	runes := []rune(tok)
	var words []string
	start := 0
	for i := 1; i < len(runes); i++ {
		prev, cur := runes[i-1], runes[i]
		boundary := false
		if unicode.IsUpper(cur) && (unicode.IsLower(prev) || unicode.IsDigit(prev)) {
			boundary = true // aB / 1B
		}
		if unicode.IsUpper(prev) && unicode.IsUpper(cur) && i+1 < len(runes) && unicode.IsLower(runes[i+1]) {
			boundary = true // ABc -> A | Bc (acronym then word)
		}
		if boundary {
			words = append(words, strings.ToLower(string(runes[start:i])))
			start = i
		}
	}
	words = append(words, strings.ToLower(string(runes[start:])))
	return words
}

func title(w string) string {
	if w == "" {
		return w
	}
	r := []rune(w)
	return string(unicode.ToUpper(r[0])) + string(r[1:])
}

// spellings renders one word list in every case form the scan matches. Order
// matters only for stable output; matching is by exact string.
func spellings(ws []string) []string {
	if len(ws) == 0 {
		return nil
	}
	joined := strings.Join(ws, "")
	titled := make([]string, len(ws))
	for i, w := range ws {
		titled[i] = title(w)
	}
	camel := ws[0] + strings.Join(titled[1:], "")
	return []string{
		strings.Join(ws, "-"),                  // kebab: dgx-bridge
		strings.Join(ws, "_"),                  // snake: dgx_bridge
		strings.ToUpper(strings.Join(ws, "_")), // SCREAMING_SNAKE
		strings.ToUpper(strings.Join(ws, "-")), // SCREAMING-KEBAB
		joined,                                 // joined lower: dgxbridge
		strings.ToUpper(joined),                // joined upper: DGXBRIDGE
		strings.Join(titled, ""),               // Pascal: DgxBridge
		camel,                                  // camel: dgxBridge
		title(joined),                          // capitalized joined: Dgxbridge
	}
}

// Variants expands (from, to) into matched case-form pairs, deduped by the old
// spelling, longest-old-first so replacement can never bite a substring of a
// longer form.
func Variants(from, to string) []Variant {
	oldS := spellings(splitWords(from))
	newS := spellings(splitWords(to))
	if len(oldS) == 0 || len(newS) == 0 || len(oldS) != len(newS) {
		return nil
	}
	seen := map[string]bool{}
	var out []Variant
	for i, o := range oldS {
		if o == newS[i] || seen[o] {
			continue
		}
		seen[o] = true
		out = append(out, Variant{Old: o, New: newS[i]})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if len(out[i].Old) != len(out[j].Old) {
			return len(out[i].Old) > len(out[j].Old)
		}
		return out[i].Old < out[j].Old
	})
	return out
}

// baseNeedles are the lowercase forms the IRREGULAR detector searches
// case-insensitively: any hit whose exact casing is not a known variant is a
// spelling the plan cannot rewrite mechanically.
func baseNeedles(from string) []string {
	ws := splitWords(from)
	if len(ws) == 0 {
		return nil
	}
	set := map[string]bool{}
	var out []string
	for _, n := range []string{strings.Join(ws, ""), strings.Join(ws, "-"), strings.Join(ws, "_")} {
		if !set[n] {
			set[n] = true
			out = append(out, n)
		}
	}
	return out
}

// ---- scanning -----------------------------------------------------------------------

func kindOf(rel string) string {
	base := filepath.Base(rel)
	ext := strings.ToLower(filepath.Ext(base))
	switch {
	case binaryExts[ext]:
		return "binary"
	case ext == ".go":
		return "go"
	case ext == ".md" || ext == ".rst" || ext == ".txt" || ext == ".html":
		return "docs"
	case ext == ".toml" || ext == ".json" || ext == ".yaml" || ext == ".yml" ||
		base == ".gitignore" || base == ".gitattributes" || base == "Makefile":
		return "config"
	case ext == ".py" || ext == ".ps1" || ext == ".sh" || ext == ".bat":
		return "script"
	case ext == ".jsonl" || ext == ".csv":
		return "data"
	default:
		return "other"
	}
}

// underHoldoutPrefix reports whether rel sits in a history-shaped tree.
func underHoldoutPrefix(rel string) bool {
	for _, p := range holdoutPrefixes {
		if strings.HasPrefix(rel, p) || rel+"/" == p {
			return true
		}
	}
	return false
}

// holdReason triages a site: "" means Apply may rewrite it.
func holdReason(rel, kind string, opts Options) string {
	if kind == "binary" {
		return "binary artifact: regenerate from source after the rename, do not patch bytes"
	}
	if opts.IncludeHistorical {
		return ""
	}
	if kind == "data" {
		return "append-only record: rename forward, do not rewrite history (--include-historical overrides)"
	}
	for _, p := range holdoutPrefixes {
		if strings.HasPrefix(rel, p) {
			return "historical record under " + p + ": leave the old name in history (--include-historical overrides)"
		}
	}
	return ""
}

// countMatches counts exact-variant hits (per variant) and irregular
// case-insensitive hits no variant covers.
func countMatches(content string, variants []Variant, needles []string) (perVariant map[string]int, irregular int) {
	perVariant = map[string]int{}
	covered := map[int]int{} // offset -> match length, for irregular overlap checks
	for _, v := range variants {
		n := 0
		for at := 0; ; {
			i := strings.Index(content[at:], v.Old)
			if i < 0 {
				break
			}
			covered[at+i] = len(v.Old)
			at += i + len(v.Old)
			n++
		}
		if n > 0 {
			perVariant[v.Old] = n
		}
	}
	lower := strings.ToLower(content)
	for _, needle := range needles {
		for at := 0; ; {
			i := strings.Index(lower[at:], needle)
			if i < 0 {
				break
			}
			pos := at + i
			at = pos + len(needle)
			if inCovered(covered, pos, len(needle)) {
				continue
			}
			irregular++
		}
	}
	return perVariant, irregular
}

// inCovered reports whether [pos, pos+n) overlaps any exact-variant hit.
func inCovered(covered map[int]int, pos, n int) bool {
	for c, l := range covered {
		if pos < c+l && c < pos+n {
			return true
		}
	}
	return false
}

func renameBase(base string, variants []Variant) (string, bool) {
	out := base
	for _, v := range variants {
		out = strings.ReplaceAll(out, v.Old, v.New)
	}
	return out, out != base
}

func isBinaryContent(b []byte) bool {
	head := b
	if len(head) > 8000 {
		head = head[:8000]
	}
	for _, c := range head {
		if c == 0 {
			return true
		}
	}
	return false
}

// BuildPlan scans opts.Root and triages every touched site. It never mutates
// the tree.
func BuildPlan(opts Options) (Plan, error) {
	p := Plan{Schema: Schema, From: opts.From, To: opts.To}
	root, err := filepath.Abs(opts.Root)
	if err != nil {
		return p, err
	}
	p.Workspace = root
	if strings.TrimSpace(opts.From) == "" || strings.TrimSpace(opts.To) == "" {
		return p, errors.New("both a from- and a to-spelling are required")
	}
	variants := Variants(opts.From, opts.To)
	if len(variants) == 0 {
		return p, errors.New("from and to expand to identical spellings — nothing to rename")
	}
	p.Variants = variants
	needles := baseNeedles(opts.From)

	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil // unreadable entry: skip, don't abort the whole plan
		}
		rel, rerr := filepath.Rel(root, path)
		if rerr != nil || rel == "." {
			return nil
		}
		rel = filepath.ToSlash(rel)
		base := d.Name()
		if d.IsDir() {
			if skipDir(base) {
				return filepath.SkipDir
			}
			if newBase, hit := renameBase(base, variants); hit && !underHoldoutPrefix(rel) {
				p.Renames = append(p.Renames, PathRename{From: rel, To: filepath.ToSlash(filepath.Join(filepath.Dir(rel), newBase))})
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		p.FilesScanned++

		site := Site{Path: rel, Kind: kindOf(rel)}
		newBase, nameHit := renameBase(base, variants)
		site.NameHit = nameHit

		if site.Kind != "binary" {
			info, ierr := d.Info()
			switch {
			case ierr != nil:
				return nil
			case info.Size() > maxScanBytes:
				site.Hold = "oversized: scan and rewrite manually"
			default:
				b, rerr := os.ReadFile(path)
				if rerr != nil {
					return nil
				}
				if isBinaryContent(b) {
					site.Kind = "binary"
				} else {
					per, irregular := countMatches(string(b), variants, needles)
					site.PerVariant = per
					site.Irregular = irregular
					for _, n := range per {
						site.Matches += n
					}
				}
			}
		}

		if site.Matches == 0 && site.Irregular == 0 && !site.NameHit {
			return nil
		}
		if site.Hold == "" {
			site.Hold = holdReason(rel, site.Kind, opts)
		}
		if site.Hold == "" && site.Matches == 0 && site.Irregular > 0 && !site.NameHit {
			site.Hold = "irregular casing only: no variant covers these hits, rewrite manually"
		}
		p.TotalMatches += site.Matches
		p.IrregularMatches += site.Irregular
		if site.Hold != "" {
			// A holdout keeps its old NAME too: renaming a historical note breaks
			// links into history, and a binary artifact regenerates under the new
			// name rather than being moved.
			p.Holdouts = append(p.Holdouts, site)
		} else {
			if site.NameHit {
				p.Renames = append(p.Renames, PathRename{From: rel, To: filepath.ToSlash(filepath.Join(filepath.Dir(rel), newBase))})
			}
			p.Mechanical = append(p.Mechanical, site)
		}
		return nil
	})
	if err != nil {
		return p, err
	}

	sortSites(p.Mechanical)
	sortSites(p.Holdouts)
	sortRenames(p.Renames)
	p.CommitPaths = commitPaths(p)
	p.OK = true
	switch {
	case p.TotalMatches == 0 && p.IrregularMatches == 0 && len(p.Renames) == 0:
		p.Finding = "concept_not_found"
		p.NextAction = "no occurrence of " + opts.From + " under " + root + "; check the spelling or the workspace"
	case len(p.Holdouts) == 0 && p.IrregularMatches == 0:
		p.Finding = "fully_mechanical"
		p.NextAction = "re-run with --apply, then rebuild and commit the listed paths explicitly"
	default:
		p.Finding = "mechanical_with_holdouts"
		p.NextAction = "re-run with --apply for the mechanical share; the holdout list is the manual follow-up (history stays, binaries regenerate)"
	}
	return p, nil
}

// sortRenames orders deepest-first (children before parents), then by path.
func sortRenames(rs []PathRename) {
	sort.SliceStable(rs, func(i, j int) bool {
		di, dj := strings.Count(rs[i].From, "/"), strings.Count(rs[j].From, "/")
		if di != dj {
			return di > dj
		}
		return rs[i].From > rs[j].From
	})
}

func sortSites(ss []Site) {
	sort.SliceStable(ss, func(i, j int) bool { return ss[i].Path < ss[j].Path })
}

// commitPaths folds every path an explicit-pathspec commit must name: each
// mechanical site plus both sides of each rename, deduped and sorted.
func commitPaths(p Plan) []string {
	set := map[string]bool{}
	for _, s := range p.Mechanical {
		set[s.Path] = true
	}
	for _, r := range p.Renames {
		// A rename inside an already-renamed parent dir commits under the NEW
		// parent path; naming both sides of every rename covers that superset.
		set[r.From] = true
		set[r.To] = true
	}
	out := make([]string, 0, len(set))
	for s := range set {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

// ---- apply --------------------------------------------------------------------------

// Apply executes the plan's mechanical share: content rewrites first (at the
// original paths), then path renames deepest-first. It finishes with a fresh
// BuildPlan so the residual is evidence, not assumption.
func Apply(opts Options) (ApplyResult, error) {
	res := ApplyResult{Schema: Schema}
	plan, err := BuildPlan(opts)
	if err != nil {
		return res, err
	}
	res.Workspace = plan.Workspace
	res.SkippedHolds = plan.Holdouts

	for _, s := range plan.Mechanical {
		if s.Matches == 0 {
			continue // name-only site: the rename pass covers it
		}
		abs := filepath.Join(plan.Workspace, filepath.FromSlash(s.Path))
		b, rerr := os.ReadFile(abs)
		if rerr != nil {
			res.Errors = append(res.Errors, s.Path+": "+rerr.Error())
			continue
		}
		content := string(b)
		for _, v := range plan.Variants {
			content = strings.ReplaceAll(content, v.Old, v.New)
		}
		info, _ := os.Stat(abs)
		mode := fs.FileMode(0o644)
		if info != nil {
			mode = info.Mode().Perm()
		}
		if werr := os.WriteFile(abs, []byte(content), mode); werr != nil {
			res.Errors = append(res.Errors, s.Path+": "+werr.Error())
			continue
		}
		res.RewrittenFiles = append(res.RewrittenFiles, s.Path)
	}

	for _, r := range plan.Renames { // already deepest-first
		from := filepath.Join(plan.Workspace, filepath.FromSlash(r.From))
		to := filepath.Join(plan.Workspace, filepath.FromSlash(r.To))
		if _, serr := os.Stat(to); serr == nil {
			res.Errors = append(res.Errors, r.From+": rename target "+r.To+" already exists")
			continue
		}
		if rerr := os.Rename(from, to); rerr != nil {
			res.Errors = append(res.Errors, r.From+": "+rerr.Error())
			continue
		}
		res.RenamedPaths = append(res.RenamedPaths, r)
	}

	residual, rerr := BuildPlan(opts)
	if rerr == nil {
		res.Residual = &residual
	}
	res.OK = len(res.Errors) == 0 && (res.Residual == nil || len(res.Residual.Mechanical) == 0)
	switch {
	case !res.OK:
		res.NextAction = "resolve the listed errors/residual mechanical sites, then re-run"
	case res.Residual != nil && (len(res.Residual.Holdouts) > 0 || res.Residual.IrregularMatches > 0):
		res.NextAction = "mechanical share applied; work the holdout list manually (history stays, binaries regenerate), then rebuild and commit the plan's commit_paths explicitly"
	default:
		res.NextAction = "applied clean; rebuild, run the affected tests, and commit the plan's commit_paths explicitly"
	}
	return res, nil
}
