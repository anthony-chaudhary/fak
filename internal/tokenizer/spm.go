// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 The ggml authors
// Adapted from llama.cpp cd8427d018c07eefab6c110f7f115c31ec4c6157,
// src/llama-vocab.cpp, llm_tokenizer_spm_session. See THIRD_PARTY_NOTICES.md.
package tokenizer

import (
	"container/heap"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode/utf8"
)

// SPMOptions are explicit llama SentencePiece-family semantics, not BPE hints.
// IDs are -1 when absent. Options are copied into the immutable tokenizer.
type SPMOptions struct {
	BOS, EOS                       int
	AddBOS, AddEOS, AddSpacePrefix bool
}
type spmVocabulary struct {
	scores  []float32
	types   []int32
	bytes   [256]int
	options SPMOptions
}

const (
	spmNormal      int32 = 1
	spmUnknown     int32 = 2
	spmControl     int32 = 3
	spmUserDefined int32 = 4
	spmUnused      int32 = 5
	spmByte        int32 = 6
)

// FromGGMLSPM admits the score-based llama SPM family. It does not infer a
// tokenizer from absent BPE merges, and does not implement Unigram/WordPiece.
// Every byte fallback must be explicitly typed, so any valid UTF-8 ordinary
// input can be encoded without manufacturing a control or unknown token.
func FromGGMLSPM(tokens []string, scores []float32, types []int32, opts SPMOptions) (*Tokenizer, error) {
	if len(tokens) == 0 || len(tokens) != len(scores) || len(tokens) != len(types) {
		return nil, fmt.Errorf("tokenizer: SPM requires equally sized nonempty tokens, scores and types")
	}
	v := &spmVocabulary{scores: append([]float32(nil), scores...), types: append([]int32(nil), types...), options: opts}
	for i := range v.bytes {
		v.bytes[i] = -1
	}
	t := &Tokenizer{idToToken: append([]string(nil), tokens...), tokenToID: make(map[string]int, len(tokens)), special: make(map[int]string), metaspace: true, spm: v}
	for id, piece := range tokens {
		if piece == "" || !utf8.ValidString(piece) || math.IsNaN(float64(scores[id])) || math.IsInf(float64(scores[id]), 0) {
			return nil, fmt.Errorf("tokenizer: invalid SPM vocabulary row %d", id)
		}
		if _, exists := t.tokenToID[piece]; exists {
			return nil, fmt.Errorf("tokenizer: duplicate SPM vocabulary piece at %d", id)
		}
		t.tokenToID[piece] = id
		switch types[id] {
		case spmNormal:
		case spmUnknown, spmControl, spmUserDefined:
			t.special[id] = piece
			t.addedByContent = append(t.addedByContent, specialToken{piece, id})
		case spmUnused:
		case spmByte:
			b, ok := byteFallback(piece)
			if !ok || v.bytes[b] >= 0 {
				return nil, fmt.Errorf("tokenizer: invalid SPM byte row %d", id)
			}
			v.bytes[b] = id
		default:
			return nil, fmt.Errorf("tokenizer: unsupported SPM token type %d", types[id])
		}
	}
	for _, id := range v.bytes {
		if id < 0 {
			return nil, fmt.Errorf("tokenizer: SPM requires all 256 typed byte fallbacks")
		}
	}
	for _, item := range []struct {
		id      int
		enabled bool
	}{{opts.BOS, opts.AddBOS}, {opts.EOS, opts.AddEOS}} {
		if item.id < -1 || item.id >= len(tokens) || (item.enabled && item.id < 0) {
			return nil, fmt.Errorf("tokenizer: invalid SPM BOS/EOS identity")
		}
		if item.id >= 0 && types[item.id] != spmControl {
			return nil, fmt.Errorf("tokenizer: SPM BOS/EOS must be control tokens")
		}
	}
	sort.Slice(t.addedByContent, func(i, j int) bool {
		a, b := t.addedByContent[i], t.addedByContent[j]
		if len(a.content) != len(b.content) {
			return len(a.content) > len(b.content)
		}
		return a.id < b.id
	})
	t.specialByContent = append([]specialToken(nil), t.addedByContent...)
	return t, nil
}

type spmSymbol struct{ start, size, prev, next int }
type spmBigram struct {
	left, right, size int
	score             float32
}
type spmQueue []spmBigram

func (q spmQueue) Len() int { return len(q) }
func (q spmQueue) Less(i, j int) bool {
	if q[i].score != q[j].score {
		return q[i].score > q[j].score
	}
	return q[i].left < q[j].left
}
func (q spmQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *spmQueue) Push(x any)   { *q = append(*q, x.(spmBigram)) }
func (q *spmQueue) Pop() any     { old := *q; n := len(old); x := old[n-1]; *q = old[:n-1]; return x }

// spmOrdinary uses byte offsets into an immutable input. No token returned here
// may be UNKNOWN/CONTROL/USER_DEFINED; trusted fragments are handled separately.
func (t *Tokenizer) spmOrdinary(text string) ([]int, error) {
	if text == "" {
		return nil, nil
	}
	var syms []spmSymbol
	for off := 0; off < len(text); {
		lengths := [16]int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4}
		n := lengths[text[off]>>4]
		if n > len(text)-off {
			n = len(text) - off
		}
		i := len(syms)
		syms = append(syms, spmSymbol{off, n, i - 1, i + 1})
		off += n
	}
	syms[len(syms)-1].next = -1
	q := spmQueue{}
	add := func(left, right int) {
		if left < 0 || right < 0 {
			return
		}
		a, b := syms[left], syms[right]
		piece := text[a.start : b.start+b.size]
		id, ok := t.tokenToID[piece]
		if !ok || t.spm.types[id] != spmNormal {
			return
		}
		heap.Push(&q, spmBigram{left, right, a.size + b.size, t.spm.scores[id]})
	}
	for i := 1; i < len(syms); i++ {
		add(i-1, i)
	}
	for q.Len() > 0 {
		pair := heap.Pop(&q).(spmBigram)
		left, right := &syms[pair.left], &syms[pair.right]
		if left.size == 0 || right.size == 0 || left.next != pair.right || left.size+right.size != pair.size {
			continue
		}
		left.size += right.size
		right.size = 0
		left.next = right.next
		if right.next >= 0 {
			syms[right.next].prev = pair.left
		}
		add(left.prev, pair.left)
		add(pair.left, left.next)
	}
	ids := make([]int, 0, len(syms))
	for i := 0; i >= 0; i = syms[i].next {
		s := syms[i]
		piece := text[s.start : s.start+s.size]
		if id, ok := t.tokenToID[piece]; ok && t.spm.types[id] == spmNormal {
			ids = append(ids, id)
			continue
		}
		// Every accepted merge is a normal vocabulary piece; only unmerged Unicode
		// symbols need iterative byte resegmentation. No recursion or unknown fallback.
		for j := 0; j < len(piece); j++ {
			ids = append(ids, t.spm.bytes[piece[j]])
		}
	}
	return ids, nil
}

func (t *Tokenizer) spmEncodeSegments(segs []Segment, addSpecial, parseControl bool) ([]int, error) {
	type interval struct{ start, end int }
	var text strings.Builder
	var trusted []interval
	for _, seg := range segs {
		start := text.Len()
		text.WriteString(seg.Text)
		if seg.Trusted && seg.Text != "" {
			n := len(trusted)
			if n > 0 && trusted[n-1].end == start {
				trusted[n-1].end = text.Len()
			} else {
				trusted = append(trusted, interval{start, text.Len()})
			}
		}
	}
	full := text.String()
	var ids []int
	if addSpecial && t.spm.options.AddBOS {
		ids = append(ids, t.spm.options.BOS)
	}
	prevSpecial := true
	ordinary := func(s string) error {
		if s == "" {
			return nil
		}
		if prevSpecial && t.spm.options.AddSpacePrefix {
			s = " " + s
		}
		part, err := t.spmOrdinary(metaspaceEncode(s))
		if err != nil {
			return err
		}
		ids = append(ids, part...)
		prevSpecial = false
		return nil
	}
	run := 0
	for _, span := range trusted {
		for pos := span.start; pos < span.end; {
			at, sp, ok := t.spmNextMatch(full[pos:span.end], parseControl)
			if !ok {
				break
			}
			if err := ordinary(full[run : pos+at]); err != nil {
				return nil, err
			}
			ids = append(ids, sp.id)
			prevSpecial = true
			pos += at + len(sp.content)
			run = pos
		}
	}
	if err := ordinary(full[run:]); err != nil {
		return nil, err
	}
	if addSpecial && t.spm.options.AddEOS {
		ids = append(ids, t.spm.options.EOS)
	}
	return ids, nil
}

// spmIdentity binds scores, types and automatic special/space policy in addition
// to the existing ordered vocabulary and added-token tables.
func (t *Tokenizer) spmIdentity(h interface{ Write([]byte) (int, error) }) {
	if t.spm == nil {
		return
	}
	writeIdentityPart(h, "llama-spm-v1")
	writeIdentityPart(h, fmt.Sprintf("%d/%d/%t/%t/%t", t.spm.options.BOS, t.spm.options.EOS, t.spm.options.AddBOS, t.spm.options.AddEOS, t.spm.options.AddSpacePrefix))
	for i, score := range t.spm.scores {
		writeIdentityPart(h, fmt.Sprintf("%08x/%d", math.Float32bits(score), t.spm.types[i]))
	}
}

func (t *Tokenizer) spmDecodeToken(id int) (decodedToken, error) {
	if id < 0 || id >= len(t.idToToken) {
		return decodedToken{}, fmt.Errorf("tokenizer: SPM id %d out of range", id)
	}
	piece := t.idToToken[id]
	switch t.spm.types[id] {
	case spmUnknown, spmControl, spmUserDefined:
		return decodedToken{special: piece, isSpecial: true}, nil
	case spmNormal:
		return decodedToken{bytes: []byte(strings.ReplaceAll(piece, string(metaspaceRune), " "))}, nil
	case spmByte:
		b, _ := byteFallback(piece)
		return decodedToken{bytes: []byte{b}}, nil
	default:
		return decodedToken{}, fmt.Errorf("tokenizer: SPM id %d is not emittable", id)
	}
}

// EncodeSPM exposes the donor's explicit special-token policy. Generic Encode
// and EncodeSegments never add BOS/EOS; the explicit native SPM raw-completion
// caller owns this policy. Unsupported SPM chat is refused before rendering.
func (t *Tokenizer) EncodeSPM(text string, addSpecial, parseSpecial bool) ([]int, error) {
	if t == nil || t.spm == nil {
		return nil, ErrEncodeUnsupported
	}
	return t.spmEncodeSegments([]Segment{{Text: text, Trusted: true}}, addSpecial, parseSpecial)
}

func (t *Tokenizer) spmNextMatch(text string, parseControl bool) (int, specialToken, bool) {
	at := -1
	var best specialToken
	for _, sp := range t.addedByContent {
		if !parseControl && t.spm.types[sp.id] != spmUserDefined {
			continue
		}
		i := strings.Index(text, sp.content)
		if i >= 0 && (at < 0 || i < at) {
			at, best = i, sp
		}
	}
	return at, best, at >= 0
}

// IsSPM identifies the explicitly admitted score-based tokenizer family.
func (t *Tokenizer) IsSPM() bool { return t != nil && t.spm != nil }

// SPMPolicy returns a value copy for explicit ordinary-caller consistency checks.
func (t *Tokenizer) SPMPolicy() (SPMOptions, bool) {
	if t == nil || t.spm == nil {
		return SPMOptions{}, false
	}
	return t.spm.options, true
}
