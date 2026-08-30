package capindex

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	frontmatteryaml "github.com/anthony-chaudhary/fak/internal/frontmatter"
)

// SkillResolver is the `skill` kind: a Resolver over a directory of
// .claude/skills/*/SKILL.md files. Index() parses each SKILL.md's YAML
// frontmatter (name / description / tags) into a cheap CapCard — the at-rest
// cost. Fault() reads the full SKILL.md body lazily, only when asked. A skill is
// thus a paged capability: the card is held for free, the body faults on demand.
type SkillResolver struct {
	// Root is the skills directory (e.g. ".claude/skills"). Each immediate
	// subdirectory holding a SKILL.md is one skill.
	Root string
}

// NewSkillResolver builds a skill resolver rooted at the given skills directory.
func NewSkillResolver(root string) *SkillResolver {
	return &SkillResolver{Root: root}
}

// SkillIntentMaxBytes caps one skill's RESIDENT intent line (#5560, epic #3229).
//
// The cap is what makes the at-rest floor bounded PER SKILL rather than "however
// much prose the author's leading sentence happened to run to" — which is the exact
// mechanism by which the resident floor grew 30% in twenty days unopposed (#5444).
// At 320 B a skill's resident line costs ~80 estimated tokens at the house divisor,
// and 58 skills fit in ~4.6 kB even in the worst case.
//
// It is a BINDING cap, not a generous one: measured over fak's own corpus, most
// leading sentences fit and a handful elide (the longest runs 664 B). Eliding is the
// designed nudge, not a defect — a skill whose leading sentence will not fit is
// telling its author to write an explicit frontmatter `intent:`, which then wins
// outright (score-2x carries the worked example). TestResidentIntentInventory names
// the ones still eliding, so that migration is a list rather than a hunt.
const SkillIntentMaxBytes = 320

// skillFrontmatter is the subset of SKILL.md YAML frontmatter the card needs.
type skillFrontmatter struct {
	name        string
	version     string
	description string // the full prose: the in-process ranking key
	intent      string // OPTIONAL `intent:` override for the resident one-liner
	tags        []string
}

// Index returns one cheap CapCard per skill — frontmatter only, no body paged.
// The Digest is SHA-256 over the full SKILL.md bytes so a change to the body
// (not just the frontmatter) yields a new digest and a re-index of exactly that
// one entry. Cards are sorted by name for determinism.
// skillDir is one subdirectory under Root that carries a readable SKILL.md, with
// its frontmatter and resolved name already in hand.
type skillDir struct {
	path string
	body []byte
	fm   skillFrontmatter
	name string // frontmatter name, or the directory name as a fallback
}

// scanSkillDirs walks Root and returns one skillDir per subdirectory that has a
// readable SKILL.md, resolving each skill's name (frontmatter name, else the
// directory name). It is the shared directory walk behind Index (build every
// card) and locate (find one ref) — both previously inlined this same scan.
func (r *SkillResolver) scanSkillDirs() []skillDir {
	dirs, err := os.ReadDir(r.Root)
	if err != nil {
		return nil
	}
	out := make([]skillDir, 0, len(dirs))
	for _, d := range dirs {
		if !d.IsDir() {
			continue
		}
		path := filepath.Join(r.Root, d.Name(), "SKILL.md")
		body, err := os.ReadFile(path)
		if err != nil {
			continue // not a skill dir (no SKILL.md)
		}
		fm := parseFrontmatter(body)
		name := fm.name
		if name == "" {
			name = d.Name() // fall back to the directory name
		}
		out = append(out, skillDir{path: path, body: body, fm: fm, name: name})
	}
	return out
}

func (r *SkillResolver) Index() []CapCard {
	entries := r.scanSkillDirs()
	cards := make([]CapCard, 0, len(entries))
	for _, e := range entries {
		// The at-rest card carries only what is needed to DECIDE whether to load
		// this skill: name, version, tags, and one intent line. The full
		// description stays out of the serialized card (#5560) — it is held as the
		// in-process ranking key below, and the whole SKILL.md faults in on demand.
		intent := intentLine(e.fm.description, e.fm.intent)
		cardBytes, _ := json.Marshal(map[string]any{
			"name":    e.name,
			"version": e.fm.version,
			"intent":  intent,
			"tags":    e.fm.tags,
		})

		tags := append([]string{"skill"}, e.fm.tags...)
		cards = append(cards, CapCard{
			Ref: CapRef{
				Kind:    CapKindSkill,
				Name:    e.name,
				Version: e.fm.version,
			},
			Digest:    Digest(e.body), // hash the WHOLE SKILL.md, not just the card
			Intent:    intent,
			Trigger:   e.fm.description,
			Tags:      tags,
			CardBytes: cardBytes,
		})
	}

	sort.Slice(cards, func(i, j int) bool {
		if cards[i].Ref.Name != cards[j].Ref.Name {
			return cards[i].Ref.Name < cards[j].Ref.Name
		}
		return cards[i].Ref.Version < cards[j].Ref.Version
	})
	return cards
}

// Fault pages in the full SKILL.md body for one skill ref. The body is NOT read
// up front: it is wired into Capability.Resolve as a closure, so the file is
// only touched if and when something materializes the capability. The returned
// Digest matches the card's digest (same SHA-256 over the same bytes).
func (r *SkillResolver) Fault(ref CapRef) (Capability, error) {
	if ref.Kind != CapKindSkill {
		return Capability{}, ErrKindMismatch
	}

	path, body, ok := r.locate(ref)
	if !ok {
		return Capability{}, ErrNotFound
	}

	cardBytes, _ := json.Marshal(map[string]any{
		"name":    ref.Name,
		"version": ref.Version,
	})

	// Lazy fault: the body is read once, on demand, by Resolve — not here.
	return Capability{
		Ref:    ref,
		Digest: Digest(body),
		Card:   cardBytes,
		Scope:  abi.ScopeAgent, // a skill is private to one agent by default
		Resolve: func() []byte {
			full, err := os.ReadFile(path)
			if err != nil {
				return nil
			}
			return full
		},
	}, nil
}

// locate finds the SKILL.md for a ref. It matches on the frontmatter name first
// (and version when the ref pins one), falling back to the directory name. It
// returns the path, the bytes already read (for digesting), and whether a match
// was found.
func (r *SkillResolver) locate(ref CapRef) (string, []byte, bool) {
	for _, e := range r.scanSkillDirs() {
		if e.name != ref.Name {
			continue
		}
		if ref.Version != "" && e.fm.version != ref.Version {
			continue
		}
		return e.path, e.body, true
	}
	return "", nil, false
}

// parseFrontmatter extracts name/version/description/tags from a SKILL.md's
// leading YAML frontmatter block (delimited by lines of exactly "---"). It is a
// deliberately small, dependency-free parser: it reads only the flat scalar keys
// the card needs and the inline "[a, b]" tag list. Anything it does not
// recognize is ignored.
func parseFrontmatter(body []byte) skillFrontmatter {
	var fm skillFrontmatter
	sc := bufio.NewScanner(strings.NewReader(string(body)))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	inBlock := false
	started := false
	for sc.Scan() {
		line := sc.Text()
		trimmed := strings.TrimSpace(line)
		if trimmed == "---" {
			if !started {
				started = true
				inBlock = true
				continue
			}
			break // closing delimiter — frontmatter done
		}
		if !inBlock {
			continue
		}

		key, val, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		switch key {
		case "name":
			fm.name, _ = frontmatteryaml.DecodeScalar(val)
		case "version":
			fm.version, _ = frontmatteryaml.DecodeScalar(val)
		case "description":
			fm.description, _ = frontmatteryaml.DecodeScalar(val)
		case "intent":
			fm.intent, _ = frontmatteryaml.DecodeScalar(val)
		case "tags":
			fm.tags = parseInlineList(val)
		}
	}
	return fm
}

// intentLine picks the RESIDENT one-liner for a capability (#5560).
//
// An explicit frontmatter `intent:` wins — an author who knows their skill's first
// sentence is not self-contained says so once, in the file, instead of the index
// guessing forever. Otherwise the leading sentence of the description is used, which
// needs no migration of the 58 existing SKILL.md files. Either way the result is
// capped at SkillIntentMaxBytes, so no single skill can quietly become a resident
// paragraph again.
func intentLine(description, explicit string) string {
	line := strings.TrimSpace(explicit)
	if line == "" {
		line = FirstSentence(description)
	}
	return capWords(line, SkillIntentMaxBytes)
}

// sentenceAbbrevs are the tokens whose trailing period does NOT end a sentence. The
// single-letter case ("e.g.", "i.e.") is handled structurally, not listed.
var sentenceAbbrevs = map[string]bool{
	"vs": true, "cf": true, "al": true, "approx": true, "incl": true, "resp": true,
}

// FirstSentence returns the leading sentence of s — the text up to and including the
// first sentence terminator that is FOLLOWED BY WHITESPACE and not preceded by an
// abbreviation. Requiring the whitespace is what keeps a path (".claude/skills/x.md")
// or a version ("v0.42.0") from being read as a sentence end; the abbreviation check
// is what keeps "e.g." and "vs." from cutting mid-clause. Text with no terminator is
// returned whole.
func FirstSentence(s string) string {
	for i, r := range s {
		if r != '.' && r != '?' && r != '!' {
			continue
		}
		rest := s[i+1:]
		if rest != "" && !isSpaceByte(rest[0]) {
			continue // mid-token punctuation: a path, a decimal, an abbreviation's inner dot
		}
		if r == '.' && sentenceAbbrevs[lastWord(s[:i])] {
			continue
		}
		return strings.TrimSpace(s[:i+1])
	}
	return strings.TrimSpace(s)
}

func isSpaceByte(b byte) bool { return b == ' ' || b == '\t' || b == '\n' || b == '\r' }

// lastWord returns the trailing run of letters in s, lowercased — the token whose
// period is being judged. A single letter yields a one-character result, which the
// caller treats as an abbreviation ("e.g.", "i.e.").
func lastWord(s string) string {
	end := len(s)
	start := end
	for start > 0 {
		c := s[start-1]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') {
			start--
			continue
		}
		break
	}
	w := strings.ToLower(s[start:end])
	if len(w) == 1 {
		return "vs" // structural: a lone letter before a period is an abbreviation
	}
	return w
}

// capWords truncates s to at most max BYTES on a word boundary, marking the elision
// with a horizontal ellipsis so a reader can tell prose was left behind rather than
// the author having written a fragment. It never splits a multi-byte rune: the cut
// falls back to the last rune boundary when the line holds no space to break on.
func capWords(s string, max int) string {
	if len(s) <= max {
		return s
	}
	const mark = "…"
	budget := max - len(mark)
	if budget <= 0 {
		return ""
	}
	cut := strings.LastIndexByte(s[:budget+1], ' ')
	if cut <= 0 {
		cut = budget
		for cut > 0 && s[cut]&0xC0 == 0x80 { // back off to a rune boundary
			cut--
		}
	}
	return strings.TrimRight(s[:cut], " \t") + mark
}

// parseInlineList parses a YAML inline list "[a, b, c]" into a slice. A bare
// scalar (no brackets) is treated as a single-element list. Empty yields nil.
func parseInlineList(val string) []string {
	val = strings.TrimSpace(val)
	val = strings.TrimPrefix(val, "[")
	val = strings.TrimSuffix(val, "]")
	if val == "" {
		return nil
	}
	parts := strings.Split(val, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p, _ = frontmatteryaml.DecodeScalar(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
