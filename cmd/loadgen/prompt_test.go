package main

import (
	"errors"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

// dictTokenizer is a deterministic greedy longest-match "BPE-like" encoder
// over a fixed word table: at each position the longest word (max 3 bytes in
// these fixtures) wins, and unmatched runes fall back to one-rune tokens in
// the negative ID space. It stands in for a real BPE just where the prompt
// seam cares: merges that span a segment boundary change the token count.
type dictTokenizer struct {
	words map[string]int
	rev   map[int]string
}

func newDictTokenizer(words map[string]int) *dictTokenizer {
	rev := make(map[int]string, len(words))
	for w, id := range words {
		rev[id] = w
	}
	return &dictTokenizer{words: words, rev: rev}
}

func (d *dictTokenizer) Encode(s string) []int {
	var out []int
	for i := 0; i < len(s); {
		matched := false
		for l := 3; l >= 1; l-- {
			if i+l <= len(s) {
				if id, ok := d.words[s[i:i+l]]; ok {
					out = append(out, id)
					i += l
					matched = true
					break
				}
			}
		}
		if !matched {
			r, size := utf8.DecodeRuneInString(s[i:])
			out = append(out, -int(r)-1)
			i += size
		}
	}
	return out
}

func (d *dictTokenizer) Decode(tokens []int) string {
	var b strings.Builder
	for _, t := range tokens {
		if w, ok := d.rev[t]; ok {
			b.WriteString(w)
		}
	}
	return b.String()
}

// spaceNormTok wraps a dictTokenizer whose Decode inserts a space between
// consecutive tokens (as real detokenizers add spacing) while its Encode
// collapses space runs to a single space before matching. Re-encoding a
// decoded pair therefore runs long until the correction trims it back, which
// exercises the trim branch of CorrectToExactLength.
type spaceNormTok struct{ base *dictTokenizer }

func collapseSpaces(x string) string {
	var b strings.Builder
	lastSpace := false
	for i := 0; i < len(x); i++ {
		if x[i] == ' ' {
			if lastSpace {
				continue
			}
			lastSpace = true
		} else {
			lastSpace = false
		}
		b.WriteByte(x[i])
	}
	return b.String()
}

func (s *spaceNormTok) Encode(x string) []int {
	return s.base.Encode(collapseSpaces(x))
}

func (s *spaceNormTok) Decode(tokens []int) string {
	parts := make([]string, 0, len(tokens))
	for _, t := range tokens {
		if w, ok := s.base.rev[t]; ok {
			parts = append(parts, w)
		}
	}
	return strings.Join(parts, " ")
}

// probeCorpus builds a 300-token corpus whose [100:164] window decodes to 64
// copies of "a" and whose [200:264] window decodes to 64 copies of "cd"; the
// remaining slots are filler ("b").
func probeCorpus(aID, bID int) []int {
	corpus := make([]int, 300)
	for i := range corpus {
		corpus[i] = 9 // filler: "b"
	}
	for i := 100; i < 164; i++ {
		corpus[i] = aID
	}
	for i := 200; i < 264; i++ {
		corpus[i] = bID
	}
	return corpus
}

func TestPromptTokenLevelSeam(t *testing.T) {
	t.Run("token-level join keeps the exact token count", func(t *testing.T) {
		// "ab" and "cd" are single tokens; the string-concatenated form
		// "ab cd" charges the separator as its own token.
		tok := newDictTokenizer(map[string]int{
			"ab": 1, "cd": 10, "a": 4, "b": 9, "c": 3, "d": 16,
			" ": 5, "\n": 6, "\n\n": 7,
		})
		prefix, body := []int{1}, []int{10}
		text, err := AssemblePrompt(tok, prefix, body, nil)
		if err != nil {
			t.Fatalf("AssemblePrompt: %v", err)
		}
		if text != "abcd" {
			t.Fatalf("text=%q want %q", text, "abcd")
		}
		// Closed contract: the wire prompt re-encodes to exactly
		// len(prefix)+len(body) tokens — no seam charge.
		want := len(prefix) + len(body)
		if got := len(tok.Encode(text)); got != want {
			t.Fatalf("re-encoded length=%d want %d", got, want)
		}
		// An empty prefix must give the identical closed contract.
		if text2, err := AssemblePrompt(tok, nil, []int{1, 10}, nil); err != nil || text2 != "abcd" {
			t.Fatalf("empty-prefix assembly=%q err=%v want %q", text2, err, "abcd")
		}
		// Contrast: the string-concatenation seam costs one extra token
		// under this BPE (the AIP-1118 drift class in closed form).
		if got := len(tok.Encode("ab" + " " + "cd")); got != want+1 {
			t.Fatalf("string-join length=%d want %d", got, want+1)
		}
	})

	t.Run("empty prompt reports the sentinel", func(t *testing.T) {
		tok := newDictTokenizer(map[string]int{"a": 4})
		if _, err := AssemblePrompt(tok, nil, nil, nil); !errors.Is(err, ErrEmptyPrompt) {
			t.Fatalf("err=%v want ErrEmptyPrompt", err)
		}
		if _, err := AssemblePrompt(nil, []int{4}, nil, nil); !errors.Is(err, ErrEmptyPrompt) {
			t.Fatalf("nil tokenizer err=%v want ErrEmptyPrompt", err)
		}
	})

	t.Run("top-up correction reaches the exact length", func(t *testing.T) {
		// "a"+"bc" decodes to "abc", which re-encodes to the single merged
		// token 8: one token short of the target, so the loop tops up.
		tok := newDictTokenizer(map[string]int{"abc": 8, "a": 4, "bc": 2, "b": 9, "c": 3})
		draws := 0
		topup := func(n int) []int {
			draws++
			if n != 1 {
				t.Fatalf("topup draw n=%d want 1", n)
			}
			return []int{3} // "c"
		}
		text, err := AssemblePrompt(tok, []int{4}, []int{2}, topup)
		if err != nil {
			t.Fatalf("AssemblePrompt: %v", err)
		}
		if text != "abcc" {
			t.Fatalf("text=%q want %q", text, "abcc")
		}
		if draws != 1 {
			t.Fatalf("top-up draws=%d want 1", draws)
		}
		if got := len(tok.Encode(text)); got != 2 {
			t.Fatalf("re-encoded length=%d want 2", got)
		}
	})

	t.Run("trim correction reaches the exact length", func(t *testing.T) {
		// Decode adds a seam space, so the re-encoding of [4,9] is
		// [4,5,9]: one token long, forcing a trim; the trimmed prefix then
		// round-trips exactly.
		base := newDictTokenizer(map[string]int{"a": 4, "b": 9, " ": 5})
		tok := &spaceNormTok{base: base}
		fin, text, err := CorrectToExactLength(tok, []int{4, 9}, 2, nil)
		if err != nil {
			t.Fatalf("CorrectToExactLength: %v", err)
		}
		if !reflect.DeepEqual(fin, []int{4, 5}) {
			t.Fatalf("final tokens=%v want [4 5]", fin)
		}
		// text is the decode of the final sequence; its re-encoding (which
		// normalizes the doubled space) has exactly the target count.
		if text != "a  " {
			t.Fatalf("text=%q want %q", text, "a  ")
		}
		if got := len(tok.Encode(text)); got != 2 {
			t.Fatalf("re-encoded length=%d want 2", got)
		}
	})

	t.Run("uncorrectable length reports the sentinel", func(t *testing.T) {
		tok := newDictTokenizer(map[string]int{"abc": 8, "a": 4, "bc": 2, "b": 9, "c": 3})
		if _, err := AssemblePrompt(tok, []int{4}, []int{2}, nil); !errors.Is(err, ErrLengthMismatch) {
			t.Fatalf("err=%v want ErrLengthMismatch", err)
		}
		stalled := func(int) []int { return nil } // top-up that makes no progress
		if _, err := AssemblePrompt(tok, []int{4}, []int{2}, stalled); !errors.Is(err, ErrLengthMismatch) {
			t.Fatalf("stalled top-up err=%v want ErrLengthMismatch", err)
		}
	})

	t.Run("probe picks the first stable candidate", func(t *testing.T) {
		// No cross-seam merges: "\n\n" passes the byte-exact join probe and
		// is returned first by the ladder.
		tok := newDictTokenizer(map[string]int{
			"a": 4, "cd": 10, "b": 9, " ": 5, "\n": 6, "\n\n": 7,
		})
		if got := ProbeBPETerminator(tok, probeCorpus(4, 10)); !reflect.DeepEqual(got, []int{7}) {
			t.Fatalf("terminator=%v want [7]", got)
		}
	})

	t.Run("probe skips an unstable candidate", func(t *testing.T) {
		// "a\n\n" merges across the first seam, so "\n\n" loses one token in
		// the join probe and the ladder falls through to "\n".
		tok := newDictTokenizer(map[string]int{
			"a": 4, "cd": 10, "b": 9, " ": 5, "\n": 6, "\n\n": 7, "a\n\n": 14,
		})
		if got := ProbeBPETerminator(tok, probeCorpus(4, 10)); !reflect.DeepEqual(got, []int{6}) {
			t.Fatalf("terminator=%v want [6]", got)
		}
	})

	t.Run("probe reports nil when no candidate is stable", func(t *testing.T) {
		// Each seam merge defeats one candidate: "a\n\n" kills "\n\n",
		// "a\n" kills "\n", "a " kills " ".
		tok := newDictTokenizer(map[string]int{
			"a": 4, "cd": 10, "b": 9, " ": 5, "\n": 6, "\n\n": 7,
			"a\n\n": 14, "a\n": 13, "a ": 15,
		})
		if got := ProbeBPETerminator(tok, probeCorpus(4, 10)); got != nil {
			t.Fatalf("terminator=%v want nil", got)
		}
	})

	t.Run("probe needs a sufficient corpus", func(t *testing.T) {
		tok := newDictTokenizer(map[string]int{"a": 4, "cd": 10, "b": 9, " ": 5, "\n": 6, "\n\n": 7})
		if got := ProbeBPETerminator(tok, make([]int, 100)); got != nil {
			t.Fatalf("short corpus terminator=%v want nil", got)
		}
		if got := ProbeBPETerminator(nil, probeCorpus(4, 10)); got != nil {
			t.Fatalf("nil tokenizer terminator=%v want nil", got)
		}
	})
}
