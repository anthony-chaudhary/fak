package memoryindex

import (
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/frontmatter"
)

// IndexName is the file a cold session reads first, and the one whose absence
// means "no index at all".
const IndexName = "MEMORY.md"

// IndexGlob matches every index TIER, not just the primary.
//
// The index outgrows a single file: a reader has a byte ceiling on what it will
// load, so a store past that ceiling spills the remainder into a second tier
// (fak's own committed mirror already names MEMORY_archive.md that way, and
// internal/memoryread hardcodes it) and points the header at it. A checker that
// went on quantifying over MEMORY.md alone would report every file indexed only
// in the spill tier as unreachable — a wall of false positives, which is worse
// than no check at all, because a report an operator learns to skip is how the
// next TRUE orphan goes unread too.
//
// A glob rather than a list of two names, so adding a third tier cannot silently
// reintroduce that error.
const IndexGlob = "MEMORY*.md"

// IsIndexTier reports whether a basename is an index tier rather than a memory.
// One predicate, used by the census and the row parser alike, so the two halves
// cannot drift about what counts as an index.
func IsIndexTier(base string) bool {
	ok, _ := path.Match(IndexGlob, base)
	return ok
}

// frontmatterScanLimit caps how far the parser looks for the closing `---`. Deep
// enough for a real description block, bounded so a file with no closing fence
// at all is a finding rather than a full-file scan.
const frontmatterScanLimit = 64

// Slugify is the auto-memory directory-name transform: EVERY non-alphanumeric
// rune becomes '-', not just the path separators. That distinction is the whole
// rule — a path holding a dot or a colon (a Windows drive letter, a version in a
// hostname, a username with a period) resolves to a directory that does not
// exist under a separators-only transform, the existence guard then passes, and
// the entire checker becomes a SILENT no-op that reports clean forever. Runes,
// not bytes, so a non-ASCII path yields one '-' per character.
func Slugify(p string) string {
	var b strings.Builder
	for _, r := range p {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	return b.String()
}

// AutoMemoryDir is the Claude Code auto-memory location for a session whose
// working directory is cwd: ~/.claude/projects/<slug>/memory.
//
// The directory is DERIVED from a working directory, so a caller that changed
// directory must hand the session's real one back. Getting that wrong is a trap
// rather than a bug: the derived path simply does not exist, and "no store" is
// this checker's quietest answer — which is why the CLI carries a --require-index
// flag for gate callers who need "never looked" to be distinguishable from
// "looked and clean".
func AutoMemoryDir(home, cwd string) string {
	return filepath.Join(home, ".claude", "projects", Slugify(cwd), "memory")
}

// rowRe pulls markdown link targets naming a .md file out of one index line,
// tolerating a `#fragment` after the extension. `- [Title](a.md#the-fact)` still
// points at a.md, and a checker that missed that would call an indexed file an
// orphan.
var rowRe = regexp.MustCompile(`\[([^\]]*)\]\(([^)\s]+?\.md)((?:#[^)]*)?)\)`)

// wikilinkRe matches the [[name]] form memory bodies use to link each other,
// including the [[name|alias]] and [[name#section]] variants. The inner text is
// reduced to a slug by NormalizeLink.
var wikilinkRe = regexp.MustCompile(`\[\[([^\[\]\n]+)\]\]`)

// ParseRows extracts the pointer rows from one index tier's text. Line numbers
// are 1-based and refer to that tier.
//
// A web link is not a claim about a local file: `](https://host/spec.md)` must
// never be reported as a MISSING file, and a `mailto:` even less so.
func ParseRows(tier, text string) []Row {
	var out []Row
	for i, line := range strings.Split(text, "\n") {
		for _, m := range rowRe.FindAllStringSubmatch(line, -1) {
			target := m[2]
			if target == "" || strings.Contains(target, "://") || strings.HasPrefix(target, "mailto:") {
				continue
			}
			out = append(out, Row{
				Tier:   tier,
				Line:   i + 1,
				Title:  strings.TrimSpace(m[1]),
				Target: target,
			})
		}
	}
	return out
}

// ParseFrontmatter reads the leading YAML block by line scan. Values use the
// shared dependency-free scalar decoder so every frontmatter consumer agrees
// on quoted escapes and preserves malformed quoted input verbatim. Only
// `name`, `description` and the nested `metadata.type` are extracted, because
// those are the three the grammar requires.
func ParseFrontmatter(body string) Frontmatter {
	var fm Frontmatter
	lines := strings.Split(body, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return fm
	}
	fm.Present = true
	inMetadata := false
	for i := 1; i < len(lines) && i <= frontmatterScanLimit; i++ {
		line := strings.TrimRight(lines[i], "\r")
		if strings.TrimSpace(line) == "---" {
			fm.Terminated = true
			break
		}
		indented := strings.HasPrefix(line, " ") || strings.HasPrefix(line, "\t")
		key, val, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val, _ = frontmatter.DecodeScalar(val)
		if indented {
			if inMetadata && key == "type" {
				fm.Type = val
			}
			continue
		}
		inMetadata = key == "metadata"
		switch key {
		case "name":
			fm.Name = val
		case "description":
			fm.Description = val
		}
	}
	return fm
}

// ParseWikilinks returns the raw inner text of every [[...]] in a body, in
// source order.
func ParseWikilinks(body string) []string {
	var out []string
	for _, m := range wikilinkRe.FindAllStringSubmatch(body, -1) {
		out = append(out, m[1])
	}
	return out
}

// Load reads a memory directory into a Store.
//
// ok is false for the "nothing to check" case — no such directory, or a
// directory with no MEMORY.md. A brand-new project has no memories, and a spill
// tier without a head is not an index because nothing would point a cold session
// at it. Callers decide whether that absence is benign (a hook) or a refusal (a
// gate); the answer is deliberately not baked in here.
func Load(dir string) (Store, bool) {
	st, err := os.Stat(dir)
	if err != nil || !st.IsDir() {
		return Store{Dir: dir}, false
	}
	if _, err := os.Stat(filepath.Join(dir, IndexName)); err != nil {
		return Store{Dir: dir}, false
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		return Store{Dir: dir}, false
	}

	s := Store{Dir: dir}
	var memories []string
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		switch {
		case IsIndexTier(name):
			s.Tiers = append(s.Tiers, name)
			if name != IndexName {
				memories = append(memories, name)
			}
		case strings.HasSuffix(name, ".md") && name != "README.md":
			memories = append(memories, name)
		default:
			// Not a memory and not an index: it still EXISTS, so a row aimed at
			// it resolves, but it is never checked for frontmatter or coverage.
			s.Present = append(s.Present, name)
		}
	}
	// ReadDir is already sorted; sorting explicitly makes the report stable
	// regardless of that guarantee.
	sort.Strings(s.Tiers)
	sort.Strings(s.Present)
	sort.Strings(memories)

	for _, tier := range s.Tiers {
		s.Rows = append(s.Rows, ParseRows(tier, readFile(filepath.Join(dir, tier)))...)
	}
	for _, name := range memories {
		body := readFile(filepath.Join(dir, name))
		s.Files = append(s.Files, File{
			Name:      name,
			Front:     ParseFrontmatter(body),
			Wikilinks: ParseWikilinks(body),
		})
	}
	return s, true
}

// Check is Load plus Reconcile: the whole read-only pass over a directory.
func Check(dir string, opt Options) (Report, bool) {
	s, ok := Load(dir)
	if !ok {
		return Report{Schema: Schema, Dir: dir, Counts: zeroCounts()}, false
	}
	return Reconcile(s, opt), true
}

func zeroCounts() map[string]int {
	m := map[string]int{}
	for _, k := range Kinds() {
		m[k] = 0
	}
	return m
}

// readFile returns "" for anything unreadable: an unreadable memory has no
// frontmatter and no links, which is already a finding, and a checker that dies
// on one bad file reports nothing about the other four hundred.
func readFile(p string) string {
	b, err := os.ReadFile(p)
	if err != nil {
		return ""
	}
	return string(b)
}
