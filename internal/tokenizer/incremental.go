package tokenizer

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// IncrementalDecoder turns token ids into streaming text deltas while holding
// bytes that end in a not-yet-complete UTF-8 code point.
type IncrementalDecoder struct {
	tok     *Tokenizer
	pending []byte
}

type decodedToken struct {
	bytes     []byte
	special   string
	isSpecial bool
}

// NewIncrementalDecoder returns a stateful decoder backed by this tokenizer.
func (t *Tokenizer) NewIncrementalDecoder() *IncrementalDecoder {
	return &IncrementalDecoder{tok: t}
}

// NewIncrementalDecoder returns a stateful decoder backed by tok.
func NewIncrementalDecoder(tok *Tokenizer) *IncrementalDecoder {
	if tok == nil {
		return &IncrementalDecoder{}
	}
	return tok.NewIncrementalDecoder()
}

// Push decodes one token id and returns the text that became safe to emit.
func (d *IncrementalDecoder) Push(id int) (string, error) {
	if d == nil || d.tok == nil {
		return "", fmt.Errorf("tokenizer: nil incremental decoder")
	}
	decoded, err := d.tok.decodeToken(id)
	if err != nil {
		return "", err
	}
	if decoded.isSpecial {
		return d.flushAll() + decoded.special, nil
	}
	d.pending = append(d.pending, decoded.bytes...)
	return d.flushReady(), nil
}

// Finalize flushes any trailing bytes, including an incomplete final UTF-8
// sequence, matching Tokenizer.Decode's byte-preserving one-shot behavior.
func (d *IncrementalDecoder) Finalize() (string, error) {
	if d == nil || d.tok == nil {
		return "", fmt.Errorf("tokenizer: nil incremental decoder")
	}
	return d.flushAll(), nil
}

// Reset drops buffered partial bytes so the decoder can be reused for a new stream.
func (d *IncrementalDecoder) Reset() {
	if d == nil {
		return
	}
	d.pending = d.pending[:0]
}

func (d *IncrementalDecoder) flushReady() string {
	n := readyUTF8PrefixLen(d.pending)
	if n == 0 {
		return ""
	}
	delta := string(d.pending[:n])
	copy(d.pending, d.pending[n:])
	d.pending = d.pending[:len(d.pending)-n]
	return delta
}

func (d *IncrementalDecoder) flushAll() string {
	if len(d.pending) == 0 {
		return ""
	}
	delta := string(d.pending)
	d.pending = d.pending[:0]
	return delta
}

func readyUTF8PrefixLen(b []byte) int {
	n := 0
	for n < len(b) {
		r, size := utf8.DecodeRune(b[n:])
		if r == utf8.RuneError && size == 1 && !utf8.FullRune(b[n:]) {
			break
		}
		n += size
	}
	return n
}

func (t *Tokenizer) decodeToken(id int) (decodedToken, error) {
	if id < 0 || id >= len(t.idToToken) {
		return decodedToken{}, fmt.Errorf("tokenizer: id %d out of range", id)
	}
	piece := t.idToToken[id]
	if piece == "" {
		return decodedToken{}, fmt.Errorf("tokenizer: missing token for id %d", id)
	}
	if special, ok := t.special[id]; ok {
		return decodedToken{special: special, isSpecial: true}, nil
	}
	if t.metaspace {
		if b, ok := byteFallback(piece); ok {
			return decodedToken{bytes: []byte{b}}, nil
		}
		return decodedToken{bytes: []byte(strings.ReplaceAll(piece, string(metaspaceRune), " "))}, nil
	}
	out := make([]byte, 0, len(piece))
	for _, r := range piece {
		b, ok := byteLevelDecode[r]
		if !ok {
			return decodedToken{}, fmt.Errorf("tokenizer: token %q contains non-ByteLevel rune %U", piece, r)
		}
		out = append(out, b)
	}
	return decodedToken{bytes: out}, nil
}
