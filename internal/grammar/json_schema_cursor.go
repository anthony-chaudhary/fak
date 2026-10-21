package grammar

import (
	"bytes"
	"encoding/binary"
	"sort"
)

// The JSON-schema mask steps one resumable cursor per canonical object shape,
// O(1) per byte, and caches the token-admission bitset per cursor state. Inside
// a free string every position shares one state, so long values reuse one mask.
// The per-state token-mask cache adapts the adaptive token-mask cache that
// DeepInfer's structured output inherits from XGrammar (Apache-2.0,
// deepseek-ai/DeepInfer deepinfer/structure_output at
// 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01); no upstream source is copied.

const (
	numStart = iota
	numAfterMinus
	numIntZero
	numIntDigits
	numAfterDot
	numFracDigits
)

// jsonFieldProgram is a field's value grammar in stepping form. Literal sets
// (string enums, booleans, integer enums) are sorted bytewise so the members
// sharing a consumed prefix form one contiguous range.
type jsonFieldProgram struct {
	kind  fieldKind
	cands [][]byte
}

type jsonShapeProgram struct {
	segs   []segment
	fields []*jsonFieldProgram // parallel to segs; nil for literal segments
}

// jsonCursor is one shape's parse position. seg == len(segs) means the object
// is complete; any further byte is dead.
type jsonCursor struct {
	seg  int32
	off  int32 // literal offset, or bytes consumed inside the current value
	lo   int32 // literal-set alive range [lo, hi)
	hi   int32
	num  int8
	dead bool
}

func newJSONShapeProgram(t compiledTool) jsonShapeProgram {
	p := jsonShapeProgram{segs: t.segs, fields: make([]*jsonFieldProgram, len(t.segs))}
	for i, seg := range t.segs {
		if seg.field == nil {
			continue
		}
		fp := &jsonFieldProgram{kind: seg.field.kind}
		switch seg.field.kind {
		case fieldStringEnum:
			fp.cands = sortedCandidates(seg.field.quotedEnum)
		case fieldBool:
			fp.cands = sortedCandidates(boolLits)
		case fieldIntegerEnum:
			fp.cands = sortedCandidates(seg.field.integerEnum)
		}
		p.fields[i] = fp
	}
	return p
}

func sortedCandidates(in [][]byte) [][]byte {
	out := make([][]byte, len(in))
	copy(out, in)
	sort.Slice(out, func(i, j int) bool { return bytes.Compare(out[i], out[j]) < 0 })
	return out
}

func (p *jsonShapeProgram) start() jsonCursor {
	var c jsonCursor
	p.enter(&c, 0)
	return c
}

func (p *jsonShapeProgram) enter(c *jsonCursor, seg int32) {
	c.seg, c.off, c.num = seg, 0, numStart
	c.lo, c.hi = 0, 0
	if int(seg) < len(p.fields) && p.fields[seg] != nil {
		c.hi = int32(len(p.fields[seg].cands))
	}
}

func (p *jsonShapeProgram) complete(c *jsonCursor) bool {
	return !c.dead && int(c.seg) == len(p.segs)
}

// step advances c by one byte and reports whether the shape is still alive.
// A number or integer member that can end hands an unextending byte to the
// literal that always follows a value segment.
func (p *jsonShapeProgram) step(c *jsonCursor, b byte) bool {
	if c.dead {
		return false
	}
	for {
		if int(c.seg) >= len(p.segs) {
			c.dead = true
			return false
		}
		fp := p.fields[c.seg]
		if fp == nil {
			lit := p.segs[c.seg].lit
			if lit[c.off] != b {
				c.dead = true
				return false
			}
			c.off++
			if int(c.off) == len(lit) {
				p.enter(c, c.seg+1)
			}
			return true
		}
		switch fp.kind {
		case fieldString:
			switch {
			case c.off == 0:
				if b != '"' {
					c.dead = true
					return false
				}
				c.off = 1
			case b == '"':
				p.enter(c, c.seg+1)
			case !stringContentOK(b):
				c.dead = true
				return false
			}
			return true
		case fieldStringEnum, fieldBool:
			if !narrowCandidates(fp.cands, c, b) {
				c.dead = true
				return false
			}
			// Quoted variants and true/false are prefix-free: a member completed
			// here cannot be extended by any surviving member.
			if len(fp.cands[c.lo]) == int(c.off) {
				p.enter(c, c.seg+1)
			}
			return true
		case fieldIntegerEnum:
			exact := c.hi > c.lo && len(fp.cands[c.lo]) == int(c.off)
			if narrowCandidates(fp.cands, c, b) {
				return true
			}
			if !exact {
				c.dead = true
				return false
			}
			p.enter(c, c.seg+1)
			continue
		default:
			if stepNumber(c, b, fp.kind == fieldInt) {
				return true
			}
			if c.dead {
				return false
			}
			p.enter(c, c.seg+1)
			continue
		}
	}
}

// narrowCandidates keeps the members whose next byte is b. It leaves c
// unchanged when none survive.
func narrowCandidates(cands [][]byte, c *jsonCursor, b byte) bool {
	k := int(c.off)
	lo, hi := int(c.lo), int(c.hi)
	first := -1
	last := -1
	for i := lo; i < hi; i++ {
		if len(cands[i]) > k && cands[i][k] == b {
			if first < 0 {
				first = i
			}
			last = i
		} else if first >= 0 {
			break
		}
	}
	if first < 0 {
		return false
	}
	c.lo, c.hi, c.off = int32(first), int32(last+1), c.off+1
	return true
}

// stepNumber consumes b when it extends the canonical number. When it does not,
// it reports false and marks c dead unless the number can already end.
func stepNumber(c *jsonCursor, b byte, integerOnly bool) bool {
	digit := b >= '0' && b <= '9'
	switch c.num {
	case numStart:
		switch {
		case b == '-':
			c.num = numAfterMinus
		case b == '0':
			c.num = numIntZero
		case digit:
			c.num = numIntDigits
		default:
			c.dead = true
			return false
		}
	case numAfterMinus:
		switch {
		case b == '0':
			c.num = numIntZero
		case digit:
			c.num = numIntDigits
		default:
			c.dead = true
			return false
		}
	case numIntZero:
		if integerOnly || b != '.' {
			return false
		}
		c.num = numAfterDot
	case numIntDigits:
		switch {
		case digit:
		case !integerOnly && b == '.':
			c.num = numAfterDot
		default:
			return false
		}
	case numAfterDot:
		if !digit {
			c.dead = true
			return false
		}
		c.num = numFracDigits
	case numFracDigits:
		if !digit {
			return false
		}
	}
	return true
}

// jsonMaskState is the alive cursor of every shape after a history.
type jsonMaskState struct {
	cursors []jsonCursor
}

func (s *jsonMaskState) alive() bool {
	for i := range s.cursors {
		if !s.cursors[i].dead {
			return true
		}
	}
	return false
}

func (s *jsonMaskState) key() string {
	buf := make([]byte, 0, len(s.cursors)*17)
	for _, c := range s.cursors {
		if c.dead {
			buf = append(buf, 0)
			continue
		}
		buf = append(buf, 1)
		buf = binary.LittleEndian.AppendUint32(buf, uint32(c.seg))
		buf = binary.LittleEndian.AppendUint32(buf, uint32(c.off))
		buf = binary.LittleEndian.AppendUint32(buf, uint32(c.lo))
		buf = binary.LittleEndian.AppendUint32(buf, uint32(c.hi))
		buf = append(buf, byte(c.num))
	}
	return string(buf)
}
