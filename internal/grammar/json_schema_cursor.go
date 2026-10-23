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

	item     *jsonFieldProgram // fieldArray item grammar
	minItems int32
	maxItems int32 // -1 when unbounded

	object *jsonShapeProgram // fieldObject item segments
}

type jsonShapeProgram struct {
	segs   []segment
	fields []*jsonFieldProgram // parallel to segs; nil for literal segments
}

// jsonCursor is one shape's parse position. seg == len(segs) means the object
// is complete; any further byte is dead. Inside an array, off/lo/hi/num describe
// the current item and count saturates at the largest bound that matters. An
// object item adds its segment index sub and, inside a scalar array member, the
// inner frame iarr/icount; all three are zero outside an object item.
type jsonCursor struct {
	seg    int32
	off    int32 // literal offset, or bytes consumed inside the current value
	lo     int32 // literal-set alive range [lo, hi)
	hi     int32
	num    int8
	arr    int8
	count  int32
	sub    int32
	iarr   int8
	icount int32
	dead   bool
}

const (
	arrOpen  = iota // expecting '['
	arrFirst        // after '[': first item or ']'
	arrItem         // inside an item
	arrAfter        // after an item: ',' or ']'
	arrNext         // after ',': next item
)

const (
	valueDead      = iota // b is illegal here
	valueConsumed         // b extended the value, which is still open
	valueCompleted        // b completed the value
	valueHandoff          // the value had already ended; b belongs to what follows
)

func newJSONShapeProgram(t compiledTool) jsonShapeProgram {
	return newJSONSegmentsProgram(t.segs)
}

func newJSONSegmentsProgram(segs []segment) jsonShapeProgram {
	p := jsonShapeProgram{segs: segs, fields: make([]*jsonFieldProgram, len(segs))}
	for i, seg := range segs {
		if seg.field != nil {
			p.fields[i] = newJSONFieldProgram(seg.field)
		}
	}
	return p
}

func newJSONFieldProgram(f *compiledField) *jsonFieldProgram {
	fp := &jsonFieldProgram{kind: f.kind}
	switch f.kind {
	case fieldStringEnum:
		fp.cands = sortedCandidates(f.quotedEnum)
	case fieldBool:
		fp.cands = sortedCandidates(boolLits)
	case fieldIntegerEnum:
		fp.cands = sortedCandidates(f.integerEnum)
	case fieldArray:
		fp.item = newJSONFieldProgram(f.item)
		fp.minItems, fp.maxItems = int32(f.minItems), int32(f.maxItems)
	case fieldObject:
		object := newJSONSegmentsProgram(f.segs)
		fp.object = &object
	case fieldString, fieldInt, fieldNumber:
		// Free-form scalars carry no precomputed candidates or sub-program.
	}
	return fp
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
	c.seg, c.arr, c.count = seg, arrOpen, 0
	var fp *jsonFieldProgram
	if int(seg) < len(p.fields) {
		fp = p.fields[seg]
	}
	startValue(c, fp)
}

// startValue resets the per-value state for fp (nil for a literal segment).
func startValue(c *jsonCursor, fp *jsonFieldProgram) {
	c.off, c.num, c.lo, c.hi = 0, numStart, 0, 0
	if fp != nil {
		c.hi = int32(len(fp.cands))
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
		if fp.kind == fieldArray {
			switch stepArray(c, &c.arr, &c.count, fp, b) {
			case valueConsumed:
				return true
			case valueCompleted:
				p.enter(c, c.seg+1)
				return true
			default:
				c.dead = true
				return false
			}
		}
		switch stepValue(c, fp, b) {
		case valueConsumed:
			return true
		case valueCompleted:
			p.enter(c, c.seg+1)
			return true
		case valueHandoff:
			p.enter(c, c.seg+1)
			continue
		default:
			c.dead = true
			return false
		}
	}
}

// stepValue advances one scalar value by b. It never marks c dead itself.
func stepValue(c *jsonCursor, fp *jsonFieldProgram, b byte) int {
	switch fp.kind {
	case fieldString:
		switch {
		case c.off == 0:
			if b != '"' {
				return valueDead
			}
			c.off = 1
		case b == '"':
			return valueCompleted
		case !stringContentOK(b):
			return valueDead
		}
		return valueConsumed
	case fieldStringEnum, fieldBool:
		if !narrowCandidates(fp.cands, c, b) {
			return valueDead
		}
		// Quoted variants and true/false are prefix-free: a member completed
		// here cannot be extended by any surviving member.
		if len(fp.cands[c.lo]) == int(c.off) {
			return valueCompleted
		}
		return valueConsumed
	case fieldIntegerEnum:
		exact := c.hi > c.lo && len(fp.cands[c.lo]) == int(c.off)
		if narrowCandidates(fp.cands, c, b) {
			return valueConsumed
		}
		if exact {
			return valueHandoff
		}
		return valueDead
	case fieldInt, fieldNumber:
		dead := false
		if stepNumber(c, b, fp.kind == fieldInt, &dead) {
			return valueConsumed
		}
		if dead {
			return valueDead
		}
		return valueHandoff
	case fieldObject:
		return fp.object.stepItem(c, b)
	default:
		return valueDead
	}
}

// stepItem advances an object array item through its segments. The last
// segment is the closing '}' literal, so an item completes and never hands a
// byte off; completing resets sub so every position after an item shares one
// cache key.
func (p *jsonShapeProgram) stepItem(c *jsonCursor, b byte) int {
	for {
		fp := p.fields[c.sub]
		var r int
		switch {
		case fp == nil:
			lit := p.segs[c.sub].lit
			if lit[c.off] != b {
				return valueDead
			}
			c.off++
			if int(c.off) < len(lit) {
				return valueConsumed
			}
			if int(c.sub)+1 == len(p.segs) {
				c.sub = 0
				return valueCompleted
			}
			r = valueCompleted
		case fp.kind == fieldArray:
			r = stepArray(c, &c.iarr, &c.icount, fp, b)
			if r == valueCompleted {
				c.iarr, c.icount = arrOpen, 0
			}
		default:
			r = stepValue(c, fp, b)
		}
		switch r {
		case valueConsumed:
			return valueConsumed
		case valueCompleted:
			p.enterItem(c, c.sub+1)
			return valueConsumed
		case valueHandoff:
			p.enterItem(c, c.sub+1)
		default:
			return valueDead
		}
	}
}

func (p *jsonShapeProgram) enterItem(c *jsonCursor, sub int32) {
	c.sub = sub
	startValue(c, p.fields[sub])
}

// stepArray advances an array value held in the frame (arr, count): a shape's
// own frame, or the inner frame of a scalar array inside an object item. It
// reports valueCompleted once the closing ']' is consumed.
func stepArray(c *jsonCursor, arr *int8, count *int32, fp *jsonFieldProgram, b byte) int {
	for {
		switch *arr {
		case arrOpen:
			if b != '[' {
				return valueDead
			}
			*arr = arrFirst
			return valueConsumed
		case arrFirst:
			if b == ']' {
				if fp.minItems > 0 {
					return valueDead
				}
				return valueCompleted
			}
			if fp.maxItems == 0 {
				return valueDead
			}
			startValue(c, fp.item)
			*arr = arrItem
		case arrNext:
			startValue(c, fp.item)
			*arr = arrItem
		case arrItem:
			// A finished item's scanner state is reset so every position after
			// an item shares one cache key.
			switch stepValue(c, fp.item, b) {
			case valueConsumed:
				return valueConsumed
			case valueCompleted:
				startValue(c, nil)
				*arr, *count = arrAfter, arrayCount(*count+1, fp)
				return valueConsumed
			case valueHandoff:
				startValue(c, nil)
				*arr, *count = arrAfter, arrayCount(*count+1, fp)
			default:
				return valueDead
			}
		default: // arrAfter
			switch {
			case b == ',' && (fp.maxItems < 0 || *count < fp.maxItems):
				*arr = arrNext
				return valueConsumed
			case b == ']' && *count >= fp.minItems:
				return valueCompleted
			default:
				return valueDead
			}
		}
	}
}

// arrayCount saturates the item count at the largest bound that still matters,
// keeping cursor states (and cache keys) finite for unbounded arrays.
func arrayCount(n int32, fp *jsonFieldProgram) int32 {
	limit := fp.minItems
	if fp.maxItems >= 0 {
		limit = fp.maxItems
	}
	return min(n, limit)
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
// it reports false and sets *dead unless the number can already end.
func stepNumber(c *jsonCursor, b byte, integerOnly bool, dead *bool) bool {
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
			*dead = true
			return false
		}
	case numAfterMinus:
		switch {
		case b == '0':
			c.num = numIntZero
		case digit:
			c.num = numIntDigits
		default:
			*dead = true
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
			*dead = true
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
	buf := make([]byte, 0, len(s.cursors)*32)
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
		buf = append(buf, byte(c.num), byte(c.arr))
		buf = binary.LittleEndian.AppendUint32(buf, uint32(c.count))
		buf = binary.LittleEndian.AppendUint32(buf, uint32(c.sub))
		buf = append(buf, byte(c.iarr))
		buf = binary.LittleEndian.AppendUint32(buf, uint32(c.icount))
	}
	return string(buf)
}
