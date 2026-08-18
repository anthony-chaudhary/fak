package quality

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// uniReplacement is the UTF-8 encoding of U+FFFD, the replacement character a
// sloppy streaming detokenizer emits when it flushes a partial multibyte
// sequence instead of buffering it across a token boundary.
const uniReplacement = "�"

// IncrementalUnicode is the incremental Unicode / byte-fallback decoding oracle
// (#4533): token-by-token (streaming) detokenization must reconstruct exactly
// the same valid UTF-8 text as a full decode, even when a multibyte codepoint
// is split across two tokens and when byte-fallback tokens (raw bytes spelled
// "<0xE2>"-style) reassemble a codepoint. Tokens are modeled as byte fragments;
// the reference Text is the full/correct decoded string. The oracle asserts the
// engine's incrementally-assembled Text equals that reference decode and that
// no U+FFFD appears at a boundary a faithful decoder would have buffered. On
// mismatch, FirstDivergence.Index is the token index where decoding first
// diverged and Detail describes the bad boundary.
type IncrementalUnicode struct{}

func (IncrementalUnicode) Name() string { return "incremental-unicode" }
func (IncrementalUnicode) Kind() string { return "differential" }

func init() { Register(IncrementalUnicode{}) }

func (IncrementalUnicode) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "incremental-unicode", Kind: "differential", Pass: true}
	frags := eng.Tokens
	if len(frags) == 0 {
		frags = ref.Tokens
	}
	steps, full, splits := uniDecodeSteps(frags)
	want := ref.Text
	if want == "" {
		want = full
	}
	if eng.Text == want {
		v.Detail = fmt.Sprintf("incremental decode matched the full decode: %d token(s), %d rune(s), %d split boundary(ies) buffered correctly, no replacement character introduced",
			len(frags), utf8.RuneCountInString(eng.Text), splits)
		return v
	}

	// Replay a faithful incremental decoder over the same fragments and find the
	// first token at which the engine's assembled text departs from it.
	for i, st := range steps {
		if !strings.HasPrefix(eng.Text, st.prefix) {
			m := uniCommonPrefixLen(st.prefix, eng.Text)
			refWin, engWin := uniWindow(st.prefix, m), uniWindow(eng.Text, m)
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: refWin, Engine: engWin}
			v.Detail = fmt.Sprintf("incremental decode diverged at token %d, byte offset %d: a faithful decoder has emitted %q there, engine text has %q",
				i, m, refWin, engWin)
			return v
		}
		off := len(st.prefix)
		if len(st.pending) > 0 && off < len(eng.Text) &&
			strings.HasPrefix(eng.Text[off:], uniReplacement) && !strings.HasPrefix(full[off:], uniReplacement) {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: uniWindow(full, off), Engine: uniWindow(eng.Text, off)}
			v.Detail = fmt.Sprintf("token %d ends mid-codepoint (% X buffered); engine emitted U+FFFD at the boundary instead of buffering the partial sequence until it completes",
				i, st.pending)
			return v
		}
	}

	// Every per-token prefix held and no boundary was flushed early, yet the
	// final texts still differ: the divergence is at or past the last token
	// (trailing extra text, or a reference text that disagrees with its own
	// fragments).
	last := len(frags) - 1
	if last < 0 {
		last = 0
	}
	m := uniCommonPrefixLen(want, eng.Text)
	v.Pass = false
	v.FirstDivergence = &Divergence{Index: last, Reference: uniWindow(want, m), Engine: uniWindow(eng.Text, m)}
	v.Detail = fmt.Sprintf("assembled text diverged from the reference decode at byte %d (at/after token %d): reference has %q, engine has %q",
		m, last, uniWindow(want, m), uniWindow(eng.Text, m))
	return v
}

// uniStep is the state of a faithful incremental decoder after consuming one
// token: the text emitted so far (complete runes only) and the bytes of the
// partial multibyte sequence still buffered across the token boundary.
type uniStep struct {
	prefix  string
	pending []byte
}

// uniDecodeSteps replays a faithful incremental decoder over the token byte
// fragments: complete runes are emitted as soon as their last byte arrives, a
// trailing partial multibyte sequence is buffered across token boundaries, and
// a byte that can never begin or continue a valid sequence becomes one U+FFFD
// (encoding/utf8 width-1 error semantics). It returns the per-token states, the
// full decode (any dangling tail flushed as replacement characters at
// end-of-stream), and the count of token boundaries that required buffering.
func uniDecodeSteps(frags []string) (steps []uniStep, full string, splits int) {
	var out, buf []byte
	steps = make([]uniStep, 0, len(frags))
	for i, f := range frags {
		buf = append(buf, uniFragmentBytes(f)...)
		for len(buf) > 0 {
			if !utf8.FullRune(buf) {
				break // incomplete prefix of a valid sequence: buffer across the boundary
			}
			r, size := utf8.DecodeRune(buf)
			if r == utf8.RuneError && size == 1 {
				out = append(out, uniReplacement...)
				buf = buf[1:]
				continue
			}
			out = append(out, buf[:size]...)
			buf = buf[size:]
		}
		if len(buf) > 0 && i < len(frags)-1 {
			splits++
		}
		steps = append(steps, uniStep{prefix: string(out), pending: append([]byte(nil), buf...)})
	}
	for range buf {
		out = append(out, uniReplacement...)
	}
	return steps, string(out), splits
}

// uniFragmentBytes returns the raw bytes a token fragment contributes to the
// stream. A byte-fallback token spelled in the SentencePiece style — exactly
// "<0xNN>" with two hex digits — contributes that single raw byte; any other
// fragment contributes its bytes verbatim (split fragments carry raw bytes via
// \x escapes).
func uniFragmentBytes(tok string) []byte {
	if b, ok := uniFallbackByte(tok); ok {
		return []byte{b}
	}
	return []byte(tok)
}

// uniFallbackByte parses a "<0xNN>" byte-fallback token into its raw byte.
func uniFallbackByte(tok string) (byte, bool) {
	if len(tok) != 6 || !strings.HasPrefix(tok, "<0x") || tok[5] != '>' {
		return 0, false
	}
	hi, ok := mathx.HexNibble(tok[3])
	if !ok {
		return 0, false
	}
	lo, ok := mathx.HexNibble(tok[4])
	if !ok {
		return 0, false
	}
	return hi<<4 | lo, true
}

// uniCommonPrefixLen returns the byte length of the longest common prefix — the offset
// of the first divergent byte, which is exactly what a decode audit reports. It is
// strmatch.CommonPrefixLen; `fak benchmarks`' name matcher carried an identical copy.
func uniCommonPrefixLen(a, b string) int { return strmatch.CommonPrefixLen(a, b) }

// uniWindow returns a short slice of s starting at byte offset off — up to 16
// bytes, trimmed back to a rune boundary — or "<end>" past the end: enough
// context to show a divergence without dumping the whole text.
func uniWindow(s string, off int) string {
	if off >= len(s) {
		return "<end>"
	}
	end := off + 16
	if end >= len(s) {
		return s[off:]
	}
	for end > off && !utf8.RuneStart(s[end]) {
		end--
	}
	return s[off:end]
}
