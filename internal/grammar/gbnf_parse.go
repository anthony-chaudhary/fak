// GBNF character-rule parsing and repetition lowering are adapted from llama.cpp
// at cd8427d018c07eefab6c110f7f115c31ec4c6157 (MIT).
// See LLAMA-GBNF-LICENSE.txt and gbnf.README.md.
package grammar

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

const (
	gbnfMaxSourceBytes  = 64 * 1024
	gbnfMaxRules        = 2048
	gbnfMaxSymbols      = 32768
	gbnfMaxRanges       = 32768
	gbnfMaxAlternatives = 256
	gbnfMaxNesting      = 64
	gbnfMaxRepetition   = 256
	gbnfMaxWork         = 8 * 1024 * 1024
)

// ErrGBNF identifies a bounded parse/compile/match refusal. Diagnostics contain
// fixed reason codes and byte offsets, never the caller's grammar or output.
var ErrGBNF = errors.New("GBNF constraint refused")

type GBNFError struct {
	Reason string
	Offset int
}

func (e *GBNFError) Error() string {
	return fmt.Sprintf("%v: %s at byte %d", ErrGBNF, e.Reason, e.Offset)
}
func (e *GBNFError) Unwrap() error { return ErrGBNF }

type gbnfRange struct{ lo, hi rune }
type gbnfSymbol struct{ ref, class int }
type gbnfRule struct {
	alternatives [][]gbnfSymbol
	defined      bool
}
type gbnfProgram struct {
	rules   []gbnfRule
	classes [][]gbnfRange
	root    int
	symbols int
}
type gbnfParser struct {
	source  string
	at      int
	program gbnfProgram
	names   map[string]int
	ranges  int
	// pendingSymbols owns retained sequences not yet transferred into a rule,
	// including alternatives and enclosing groups still being parsed. The cap
	// is a logical symbol bound, not RSS: one bounded unexpanded atom and
	// transient slice-copy capacity are additional parser scratch.
	pendingSymbols int
}
type gbnfWork struct{ left int }

func (w *gbnfWork) spend(n int) error {
	if n < 0 || n > w.left {
		return &GBNFError{Reason: "work_limit", Offset: 0}
	}
	w.left -= n
	return nil
}
func (p *gbnfParser) fail(reason string) error { return &GBNFError{Reason: reason, Offset: p.at} }
func (p *gbnfParser) peek() byte {
	if p.at >= len(p.source) {
		return 0
	}
	return p.source[p.at]
}
func (p *gbnfParser) space(newlines bool) {
	for p.at < len(p.source) {
		c := p.source[p.at]
		if c == '#' {
			for p.at < len(p.source) && p.source[p.at] != '\n' && p.source[p.at] != '\r' {
				p.at++
			}
		} else if c == ' ' || c == '\t' || (newlines && (c == '\n' || c == '\r')) {
			p.at++
		} else {
			break
		}
	}
}
func gbnfNameByte(b byte) bool {
	return b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9' || b == '-'
}
func (p *gbnfParser) name() (string, error) {
	start := p.at
	for gbnfNameByte(p.peek()) {
		p.at++
	}
	if start == p.at {
		return "", p.fail("expected_rule_name")
	}
	return p.source[start:p.at], nil
}
func (p *gbnfParser) newRule() (int, error) {
	if len(p.program.rules) >= gbnfMaxRules {
		return 0, p.fail("rule_limit")
	}
	id := len(p.program.rules)
	p.program.rules = append(p.program.rules, gbnfRule{})
	return id, nil
}
func (p *gbnfParser) namedRule(name string) (int, error) {
	if id, ok := p.names[name]; ok {
		return id, nil
	}
	id, err := p.newRule()
	if err != nil {
		return 0, err
	}
	p.names[name] = id
	return id, nil
}
func (p *gbnfParser) reserveSymbols(n int) error {
	if n < 0 || n > gbnfMaxSymbols-p.program.symbols-p.pendingSymbols {
		return p.fail("symbol_limit")
	}
	p.pendingSymbols += n
	return nil
}
func (p *gbnfParser) define(id int, alternatives [][]gbnfSymbol) error {
	if p.program.rules[id].defined {
		return p.fail("duplicate_rule")
	}
	if len(alternatives) > gbnfMaxAlternatives {
		return p.fail("alternative_limit")
	}
	n := 0
	for _, seq := range alternatives {
		n += len(seq)
	}
	if n > p.pendingSymbols || n > gbnfMaxSymbols-p.program.symbols {
		return p.fail("symbol_limit")
	}
	p.pendingSymbols -= n
	p.program.symbols += n
	p.program.rules[id] = gbnfRule{alternatives: alternatives, defined: true}
	return nil
}
func gbnfScalar(r rune) bool { return r >= 0 && r <= utf8.MaxRune && (r < 0xD800 || r > 0xDFFF) }
func (p *gbnfParser) character() (rune, error) {
	if p.at >= len(p.source) {
		return 0, p.fail("unterminated_character")
	}
	if p.source[p.at] != '\\' {
		r, n := utf8.DecodeRuneInString(p.source[p.at:])
		if r == utf8.RuneError && n == 1 {
			return 0, p.fail("invalid_utf8")
		}
		p.at += n
		return r, nil
	}
	p.at++
	if p.at >= len(p.source) {
		return 0, p.fail("unterminated_escape")
	}
	c := p.source[p.at]
	p.at++
	switch c {
	case 't':
		return '\t', nil
	case 'r':
		return '\r', nil
	case 'n':
		return '\n', nil
	case '\\', '"', '[', ']', '-':
		return rune(c), nil
	case 'x', 'u', 'U':
		n := 2
		if c == 'u' {
			n = 4
		}
		if c == 'U' {
			n = 8
		}
		if n > len(p.source)-p.at {
			return 0, p.fail("short_hex_escape")
		}
		v, err := strconv.ParseUint(p.source[p.at:p.at+n], 16, 32)
		if err != nil || !gbnfScalar(rune(v)) {
			return 0, p.fail("invalid_scalar_escape")
		}
		p.at += n
		return rune(v), nil
	default:
		return 0, p.fail("unsupported_escape")
	}
}
func gbnfNormalizeRanges(in []gbnfRange, negate bool) []gbnfRange {
	sort.Slice(in, func(i, j int) bool {
		if in[i].lo == in[j].lo {
			return in[i].hi < in[j].hi
		}
		return in[i].lo < in[j].lo
	})
	var merged []gbnfRange
	for _, r := range in {
		if len(merged) > 0 && r.lo <= merged[len(merged)-1].hi+1 {
			if r.hi > merged[len(merged)-1].hi {
				merged[len(merged)-1].hi = r.hi
			}
		} else {
			merged = append(merged, r)
		}
	}
	if negate {
		var complement []gbnfRange
		at := rune(0)
		for _, r := range merged {
			if at < r.lo {
				complement = append(complement, gbnfRange{at, r.lo - 1})
			}
			at = r.hi + 1
		}
		if at <= utf8.MaxRune {
			complement = append(complement, gbnfRange{at, utf8.MaxRune})
		}
		merged = complement
	}
	// Unicode scalar values exclude surrogates, including within a wide range.
	var out []gbnfRange
	for _, r := range merged {
		for _, valid := range []gbnfRange{{0, 0xD7FF}, {0xE000, utf8.MaxRune}} {
			lo, hi := max(r.lo, valid.lo), min(r.hi, valid.hi)
			if lo <= hi {
				out = append(out, gbnfRange{lo, hi})
			}
		}
	}
	return out
}
func (p *gbnfParser) class(ranges []gbnfRange, negate bool) (gbnfSymbol, error) {
	if len(ranges) == 0 {
		return gbnfSymbol{}, p.fail("empty_character_class")
	}
	ranges = gbnfNormalizeRanges(ranges, negate)
	if len(ranges) == 0 {
		return gbnfSymbol{}, p.fail("empty_scalar_domain")
	}
	if len(p.program.classes) >= gbnfMaxSymbols || len(ranges) > gbnfMaxRanges-p.ranges {
		return gbnfSymbol{}, p.fail("class_limit")
	}
	id := len(p.program.classes)
	p.program.classes = append(p.program.classes, ranges)
	p.ranges += len(ranges)
	return gbnfSymbol{ref: -1, class: id}, nil
}
func (p *gbnfParser) atom(depth int) ([]gbnfSymbol, error) {
	switch p.peek() {
	case '"':
		p.at++
		var seq []gbnfSymbol
		for p.peek() != '"' {
			r, err := p.character()
			if err != nil {
				return nil, err
			}
			s, err := p.class([]gbnfRange{{r, r}}, false)
			if err != nil {
				return nil, err
			}
			seq = append(seq, s)
			if len(seq) > gbnfMaxSymbols {
				return nil, p.fail("symbol_limit")
			}
		}
		p.at++
		return seq, nil
	case '[':
		p.at++
		neg := false
		if p.peek() == '^' {
			neg = true
			p.at++
		}
		var ranges []gbnfRange
		for p.peek() != ']' {
			lo, err := p.character()
			if err != nil {
				return nil, err
			}
			hi := lo
			if p.peek() == '-' && p.at+1 < len(p.source) && p.source[p.at+1] != ']' {
				p.at++
				hi, err = p.character()
				if err != nil {
					return nil, err
				}
				if hi < lo {
					return nil, p.fail("reversed_range")
				}
			}
			ranges = append(ranges, gbnfRange{lo, hi})
			if len(ranges) > 256 {
				return nil, p.fail("class_range_limit")
			}
		}
		p.at++
		s, err := p.class(ranges, neg)
		return []gbnfSymbol{s}, err
	case '.':
		p.at++
		s, err := p.class([]gbnfRange{{0, utf8.MaxRune}}, false)
		return []gbnfSymbol{s}, err
	case '(':
		if depth >= gbnfMaxNesting {
			return nil, p.fail("nesting_limit")
		}
		p.at++
		p.space(true)
		id, err := p.newRule()
		if err != nil {
			return nil, err
		}
		alts, err := p.alternatives(true, depth+1)
		if err != nil {
			return nil, err
		}
		if p.peek() != ')' {
			return nil, p.fail("expected_group_close")
		}
		p.at++
		if err = p.define(id, alts); err != nil {
			return nil, err
		}
		return []gbnfSymbol{{ref: id, class: -1}}, nil
	default:
		if !gbnfNameByte(p.peek()) {
			return nil, p.fail("unsupported_or_missing_atom")
		}
		name, err := p.name()
		if err != nil {
			return nil, err
		}
		id, err := p.namedRule(name)
		return []gbnfSymbol{{ref: id, class: -1}}, err
	}
}
func (p *gbnfParser) integer() (int, error) {
	start := p.at
	for p.peek() >= '0' && p.peek() <= '9' {
		p.at++
	}
	if start == p.at {
		return 0, p.fail("expected_repeat_bound")
	}
	v, err := strconv.ParseUint(p.source[start:p.at], 10, 32)
	if err != nil || v > gbnfMaxRepetition {
		return 0, p.fail("repeat_limit")
	}
	return int(v), nil
}
func (p *gbnfParser) repeatBounds(nested bool) (int, int, error) {
	c := p.peek()
	p.at++
	switch c {
	case '*':
		return 0, -1, nil
	case '+':
		return 1, -1, nil
	case '?':
		return 0, 1, nil
	}
	p.space(nested)
	lo, err := p.integer()
	if err != nil {
		return 0, 0, err
	}
	p.space(nested)
	hi := lo
	if p.peek() == ',' {
		p.at++
		p.space(nested)
		hi = -1
		if p.peek() >= '0' && p.peek() <= '9' {
			hi, err = p.integer()
			if err != nil {
				return 0, 0, err
			}
			p.space(nested)
		}
	}
	if p.peek() != '}' {
		return 0, 0, p.fail("expected_repeat_close")
	}
	p.at++
	if hi >= 0 && hi < lo {
		return 0, 0, p.fail("reversed_repeat")
	}
	return lo, hi, nil
}
func (p *gbnfParser) repeated(body []gbnfSymbol, lo, hi int) ([]gbnfSymbol, error) {
	if len(body) == 0 {
		return nil, p.fail("empty_repetition")
	}
	remaining := gbnfMaxSymbols - p.program.symbols - p.pendingSymbols
	if lo > 0 && len(body) > remaining/lo {
		return nil, p.fail("symbol_limit")
	}
	optional := hi - lo
	if hi < 0 {
		optional = 1
	}
	n := len(body) * lo
	if optional > 0 {
		n++ // The output owns one reference to the optional-rule chain.
	}
	if err := p.reserveSymbols(n); err != nil {
		return nil, err
	}
	out := make([]gbnfSymbol, 0, n)
	for i := 0; i < lo; i++ {
		out = append(out, body...)
	}
	tail := -1
	for i := 0; i < optional; i++ {
		n := len(body)
		if hi < 0 || tail >= 0 {
			n++
		}
		if err := p.reserveSymbols(n); err != nil {
			return nil, err
		}
		id, err := p.newRule()
		if err != nil {
			return nil, err
		}
		seq := make([]gbnfSymbol, len(body), n)
		copy(seq, body)
		if hi < 0 {
			seq = append(seq, gbnfSymbol{ref: id, class: -1})
		} else if tail >= 0 {
			seq = append(seq, gbnfSymbol{ref: tail, class: -1})
		}
		if err = p.define(id, [][]gbnfSymbol{seq, nil}); err != nil {
			return nil, err
		}
		tail = id
	}
	if tail >= 0 {
		out = append(out, gbnfSymbol{ref: tail, class: -1})
	}
	return out, nil
}
func (p *gbnfParser) sequence(nested bool, depth int) ([]gbnfSymbol, error) {
	var out []gbnfSymbol
	for {
		p.space(nested)
		c := p.peek()
		if c == 0 || c == '|' || c == ')' || (!nested && (c == '\r' || c == '\n')) {
			return out, nil
		}
		body, err := p.atom(depth)
		if err != nil {
			return nil, err
		}
		p.space(nested)
		switch p.peek() {
		case '*', '+', '?', '{':
			lo, hi, err := p.repeatBounds(nested)
			if err != nil {
				return nil, err
			}
			body, err = p.repeated(body, lo, hi)
			if err != nil {
				return nil, err
			}
		default:
			if err = p.reserveSymbols(len(body)); err != nil {
				return nil, err
			}
		}
		if len(body) > gbnfMaxSymbols-len(out) {
			return nil, p.fail("symbol_limit")
		}
		out = append(out, body...)
	}
}
func (p *gbnfParser) alternatives(nested bool, depth int) ([][]gbnfSymbol, error) {
	var out [][]gbnfSymbol
	for {
		seq, err := p.sequence(nested, depth)
		if err != nil {
			return nil, err
		}
		out = append(out, seq)
		if len(out) > gbnfMaxAlternatives {
			return nil, p.fail("alternative_limit")
		}
		if p.peek() != '|' {
			return out, nil
		}
		p.at++
		p.space(true)
	}
}
func parseGBNF(source string) (*gbnfProgram, error) {
	if len(source) == 0 || len(source) > gbnfMaxSourceBytes || !utf8.ValidString(source) || strings.IndexByte(source, 0) >= 0 {
		return nil, &GBNFError{Reason: "source_size_or_encoding", Offset: 0}
	}
	p := &gbnfParser{source: source, names: make(map[string]int)}
	p.space(true)
	for p.at < len(source) {
		name, err := p.name()
		if err != nil {
			return nil, err
		}
		id, err := p.namedRule(name)
		if err != nil {
			return nil, err
		}
		p.space(false)
		if !strings.HasPrefix(source[p.at:], "::=") {
			return nil, p.fail("expected_definition")
		}
		p.at += 3
		p.space(true)
		alts, err := p.alternatives(false, 0)
		if err != nil {
			return nil, err
		}
		if err = p.define(id, alts); err != nil {
			return nil, err
		}
		if p.peek() != 0 && p.peek() != '\r' && p.peek() != '\n' {
			return nil, p.fail("expected_rule_newline")
		}
		p.space(true)
	}
	root, ok := p.names["root"]
	if !ok {
		return nil, p.fail("missing_root")
	}
	p.program.root = root
	if err := p.program.validate(); err != nil {
		return nil, err
	}
	return &p.program, nil
}
func (p *gbnfProgram) validate() error {
	fail := func(reason string) error { return &GBNFError{Reason: reason, Offset: 0} }
	for _, r := range p.rules {
		if !r.defined {
			return fail("undefined_rule")
		}
	}
	nullable := make([]bool, len(p.rules))
	productive := make([]bool, len(p.rules))
	work := &gbnfWork{left: gbnfMaxWork}
	for changed := true; changed; {
		changed = false
		for id, r := range p.rules {
			if err := work.spend(1); err != nil {
				return err
			}
			for _, seq := range r.alternatives {
				if err := work.spend(1); err != nil {
					return err
				}
				n, prod := true, true
				for _, s := range seq {
					if err := work.spend(1); err != nil {
						return err
					}
					if s.ref < 0 {
						n = false
					} else {
						n = n && nullable[s.ref]
						prod = prod && productive[s.ref]
					}
				}
				if n && !nullable[id] {
					nullable[id] = true
					changed = true
				}
				if prod && !productive[id] {
					productive[id] = true
					changed = true
				}
			}
		}
	}
	for _, v := range productive {
		if !v {
			return fail("unproductive_rule")
		}
	}
	edges := make([][]int, len(p.rules))
	for id, r := range p.rules {
		if err := work.spend(1); err != nil {
			return err
		}
		seen := make(map[int]bool)
		for _, seq := range r.alternatives {
			if err := work.spend(1); err != nil {
				return err
			}
			for _, s := range seq {
				if err := work.spend(1); err != nil {
					return err
				}
				if s.ref < 0 {
					break
				}
				if !seen[s.ref] {
					edges[id] = append(edges[id], s.ref)
					seen[s.ref] = true
				}
				if !nullable[s.ref] {
					break
				}
			}
		}
	}
	color := make([]uint8, len(p.rules))
	var visit func(int, int) error
	visit = func(id, depth int) error {
		if depth > gbnfMaxRules {
			return fail("reference_depth_limit")
		}
		if color[id] == 1 {
			return fail("left_recursion_or_epsilon_cycle")
		}
		if color[id] == 2 {
			return nil
		}
		color[id] = 1
		for _, next := range edges[id] {
			if err := work.spend(1); err != nil {
				return err
			}
			if err := visit(next, depth+1); err != nil {
				return err
			}
		}
		color[id] = 2
		return nil
	}
	for id := range p.rules {
		if err := visit(id, 0); err != nil {
			return err
		}
	}
	return nil
}
