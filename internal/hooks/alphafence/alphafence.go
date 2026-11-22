// Package alphafence protects an engineer's private workflow files (skills, prompts, runbooks,
// strategy notes) from leaving the machine without requiring them to be listed in plaintext.
//
// A fence holds only salted hashes of the distinctive lines of the declared private files. A
// leak gate asks the fence whether an outbound line reproduces one of them; the fence never
// stores, prints, or returns the private text itself.
package alphafence

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"unicode"
)

const (
	Schema = "fak.alpha-fence/1"
	// DefaultRel is where the gates look for a fence, relative to the repository root. The
	// directory is conventionally untracked, so the fence stays on the engineer's machine.
	DefaultRel        = ".fak/alpha-fence.json"
	DefaultMinLineLen = 24
	DefaultMinHits    = 1

	hashBytes = 12
	saltBytes = 16
)

// Fence is the on-disk record. It is membership-testable by anyone holding it (a guessed line
// can be checked against it), so it is kept local by default rather than committed.
type Fence struct {
	Schema     string  `json:"schema"`
	Salt       string  `json:"salt"`
	MinLineLen int     `json:"min_line_len"`
	MinHits    int     `json:"min_hits"`
	Entries    []Entry `json:"entries"`
}

// Entry is one fenced source. Label is opaque unless the builder was asked to keep paths.
type Entry struct {
	Label  string   `json:"label"`
	Lines  int      `json:"lines"`
	Hashes []string `json:"hashes"`
}

// Source is one private file handed to Build.
type Source struct {
	Label string
	Body  []byte
}

// Options tunes Build. Zero values select the defaults; an empty Salt draws a fresh random one.
type Options struct {
	Salt       []byte
	MinLineLen int
	MinHits    int
}

// Normalize folds a line to the form that is hashed: lower-cased, whitespace runs collapsed,
// invisible format characters removed, and leading or trailing markup stripped. Reflowing,
// re-indenting, quoting, bulleting or commenting out a private line therefore does not evade
// the fence.
func Normalize(line string) string {
	visible := strings.Map(func(r rune) rune {
		if unicode.Is(unicode.Cf, r) {
			return -1
		}
		return r
	}, line)
	folded := strings.Join(strings.Fields(strings.ToLower(visible)), " ")
	return strings.TrimFunc(folded, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

// eligible reports whether a normalized line is distinctive enough to fence: long enough, and
// mostly letters or digits rather than punctuation or rule lines.
func eligible(norm string, minLen int) bool {
	if len(norm) < minLen {
		return false
	}
	alnum := 0
	for _, r := range norm {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			alnum++
		}
	}
	return alnum*2 >= minLen
}

type digest [hashBytes]byte

// lineDigest is a salted, truncated SHA-256. The salt only has to make one fence's hashes
// useless against another's, so a keyed MAC construction would buy nothing here.
func lineDigest(salt []byte, norm string) digest {
	buf := make([]byte, 0, len(salt)+len(norm))
	sum := sha256.Sum256(append(append(buf, salt...), norm...))
	var d digest
	copy(d[:], sum[:])
	return d
}

func lineHash(salt []byte, norm string) string {
	d := lineDigest(salt, norm)
	return hex.EncodeToString(d[:])
}

// Build fences the distinctive lines of sources. Lines that also occur in public are dropped:
// text that is already published is not alpha, and keeping it would make shared boilerplate
// trip the gate.
func Build(opts Options, sources []Source, public [][]byte) (Fence, error) {
	salt := opts.Salt
	if len(salt) == 0 {
		salt = make([]byte, saltBytes)
		if _, err := rand.Read(salt); err != nil {
			return Fence{}, fmt.Errorf("alphafence: draw salt: %w", err)
		}
	}
	minLen := opts.MinLineLen
	if minLen <= 0 {
		minLen = DefaultMinLineLen
	}
	minHits := opts.MinHits
	if minHits <= 0 {
		minHits = DefaultMinHits
	}
	known := map[string]bool{}
	for _, body := range public {
		for line := range strings.SplitSeq(string(body), "\n") {
			if norm := Normalize(line); eligible(norm, minLen) {
				known[norm] = true
			}
		}
	}
	f := Fence{Schema: Schema, Salt: hex.EncodeToString(salt), MinLineLen: minLen, MinHits: minHits, Entries: []Entry{}}
	for _, src := range sources {
		seen := map[string]bool{}
		var hashes []string
		for line := range strings.SplitSeq(string(src.Body), "\n") {
			norm := Normalize(line)
			if !eligible(norm, minLen) || known[norm] || seen[norm] {
				continue
			}
			seen[norm] = true
			hashes = append(hashes, lineHash(salt, norm))
		}
		if len(hashes) == 0 {
			continue
		}
		sort.Strings(hashes)
		f.Entries = append(f.Entries, Entry{Label: src.Label, Lines: len(hashes), Hashes: hashes})
	}
	return f, nil
}

// Marshal renders the fence as stable, indented JSON.
func (f Fence) Marshal() ([]byte, error) {
	b, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

// Lines is the number of fenced lines across all entries.
func (f Fence) Lines() int {
	n := 0
	for _, e := range f.Entries {
		n += e.Lines
	}
	return n
}

// Parse decodes and validates a fence. Any defect is an error: a fence that silently matches
// nothing would report a leak-free tree it never inspected.
func Parse(raw []byte) (Fence, error) {
	var f Fence
	if err := json.Unmarshal(raw, &f); err != nil {
		return Fence{}, fmt.Errorf("alphafence: decode: %w", err)
	}
	if f.Schema != Schema {
		return Fence{}, fmt.Errorf("alphafence: schema %q, want %q", f.Schema, Schema)
	}
	if salt, err := hex.DecodeString(f.Salt); err != nil || len(salt) == 0 {
		return Fence{}, errors.New("alphafence: salt is not non-empty hex")
	}
	if f.MinLineLen <= 0 || f.MinHits <= 0 {
		return Fence{}, errors.New("alphafence: min_line_len and min_hits must be positive")
	}
	hashes := 0
	for _, e := range f.Entries {
		for _, h := range e.Hashes {
			if raw, err := hex.DecodeString(h); err != nil || len(raw) != hashBytes {
				return Fence{}, fmt.Errorf("alphafence: entry %q carries a malformed hash", e.Label)
			}
			hashes++
		}
	}
	if hashes == 0 {
		return Fence{}, errors.New("alphafence: fence holds no hashes and would protect nothing")
	}
	return f, nil
}

// Matcher answers membership queries against one parsed fence.
type Matcher struct {
	salt    []byte
	minLen  int
	minHits int
	labels  map[digest]string
}

// NewMatcher indexes a fence for scanning.
func NewMatcher(f Fence) (*Matcher, error) {
	salt, err := hex.DecodeString(f.Salt)
	if err != nil || len(salt) == 0 {
		return nil, errors.New("alphafence: salt is not non-empty hex")
	}
	m := &Matcher{salt: salt, minLen: f.MinLineLen, minHits: f.MinHits, labels: make(map[digest]string, f.Lines())}
	if m.minLen <= 0 {
		m.minLen = DefaultMinLineLen
	}
	if m.minHits <= 0 {
		m.minHits = DefaultMinHits
	}
	for _, e := range f.Entries {
		for _, h := range e.Hashes {
			raw, err := hex.DecodeString(h)
			if err != nil || len(raw) != hashBytes {
				return nil, fmt.Errorf("alphafence: entry %q carries a malformed hash", e.Label)
			}
			d := digest(raw)
			if _, dup := m.labels[d]; !dup {
				m.labels[d] = e.Label
			}
		}
	}
	return m, nil
}

// Match reports whether line reproduces a fenced line, and which entry fenced it.
func (m *Matcher) Match(line string) (label string, ok bool) {
	if m == nil || len(m.labels) == 0 {
		return "", false
	}
	norm := Normalize(line)
	if !eligible(norm, m.minLen) {
		return "", false
	}
	label, ok = m.labels[lineDigest(m.salt, norm)]
	return label, ok
}

// Hit is one outbound line that reproduces fenced text. It carries a position and the fencing
// entry, never the text.
type Hit struct {
	Line  int
	Label string
}

// Collector groups hits for one outbound unit (a file, a message, a payload) so the fence's
// min_hits threshold is applied per unit rather than per line.
type Collector struct {
	m        *Matcher
	hits     []Hit
	distinct map[string]bool
}

// NewCollector starts a unit. A nil matcher yields a collector that never reports.
func (m *Matcher) NewCollector() *Collector {
	return &Collector{m: m, distinct: map[string]bool{}}
}

// embedDelims bound text embedded in a longer line: string literals, code spans, table cells.
const embedDelims = "\"`|"

// escapedNewline is a line break as it appears inside a JSON or source string literal.
const escapedNewline = `\n`

// Observe tests one line of the unit: the line as a whole, then each piece of it that a
// string literal, code span, table cell or escaped newline sets apart. A private line wrapped
// in JSON or pasted into a table is therefore still seen. A line counts once.
func (c *Collector) Observe(lineNo int, line string) {
	if c.m == nil || len(c.m.labels) == 0 || len(line) < c.m.minLen {
		return
	}
	if c.observe(lineNo, line) {
		return
	}
	if !strings.ContainsAny(line, embedDelims) && !strings.Contains(line, escapedNewline) {
		return
	}
	pieces := strings.FieldsFunc(strings.ReplaceAll(line, escapedNewline, "|"), func(r rune) bool {
		return strings.ContainsRune(embedDelims, r)
	})
	for _, piece := range pieces {
		if c.observe(lineNo, piece) {
			return
		}
	}
}

func (c *Collector) observe(lineNo int, text string) bool {
	label, ok := c.m.Match(text)
	if !ok {
		return false
	}
	c.hits = append(c.hits, Hit{Line: lineNo, Label: label})
	c.distinct[Normalize(text)] = true
	return true
}

// Hits returns the unit's hits once enough distinct fenced lines were reproduced, else nil.
func (c *Collector) Hits() []Hit {
	if c.m == nil || len(c.distinct) < c.m.minHits {
		return nil
	}
	return c.hits
}
