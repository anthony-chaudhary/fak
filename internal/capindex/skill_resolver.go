package capindex

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
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

// skillFrontmatter is the subset of SKILL.md YAML frontmatter the card needs.
type skillFrontmatter struct {
	name        string
	version     string
	description string
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
		cardBytes, _ := json.Marshal(map[string]any{
			"name":        e.name,
			"version":     e.fm.version,
			"description": e.fm.description,
			"tags":        e.fm.tags,
		})

		tags := append([]string{"skill"}, e.fm.tags...)
		cards = append(cards, CapCard{
			Ref: CapRef{
				Kind:    CapKindSkill,
				Name:    e.name,
				Version: e.fm.version,
			},
			Digest:    Digest(e.body), // hash the WHOLE SKILL.md, not just the card
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
		val = strings.Trim(val, `"'`)
		switch key {
		case "name":
			fm.name = val
		case "version":
			fm.version = val
		case "description":
			fm.description = val
		case "tags":
			fm.tags = parseInlineList(val)
		}
	}
	return fm
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
		p = strings.TrimSpace(p)
		p = strings.Trim(p, `"'`)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
