package tokenizer

import (
	"strings"
	"testing"
)

func TestIncrementalDecoderEmitsOnlyCompleteUTF8(t *testing.T) {
	tok := byteVocabTokenizer()
	cases := []struct {
		name       string
		raw        []byte
		wantDelta  []string
		wantJoined string
	}{
		{
			name:       "split-e-acute",
			raw:        []byte{0xc3, 0xa9},
			wantDelta:  []string{"", "\u00e9"},
			wantJoined: "\u00e9",
		},
		{
			name:       "split-emoji",
			raw:        []byte("\U0001f600"),
			wantDelta:  []string{"", "", "", "\U0001f600"},
			wantJoined: "\U0001f600",
		},
		{
			name:       "split-cjk",
			raw:        []byte("\u4e2d"),
			wantDelta:  []string{"", "", "\u4e2d"},
			wantJoined: "\u4e2d",
		},
		{
			name:       "split-combining-mark",
			raw:        []byte("e\u0301"),
			wantDelta:  []string{"e", "", "\u0301"},
			wantJoined: "e\u0301",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dec := tok.NewIncrementalDecoder()
			var joined strings.Builder
			if len(tc.raw) != len(tc.wantDelta) {
				t.Fatalf("test bug: %d raw bytes for %d expected deltas", len(tc.raw), len(tc.wantDelta))
			}
			for i, b := range tc.raw {
				delta, err := dec.Push(int(b))
				if err != nil {
					t.Fatalf("Push(%d): %v", b, err)
				}
				if delta != tc.wantDelta[i] {
					t.Fatalf("Push byte %d delta = %q, want %q", i, delta, tc.wantDelta[i])
				}
				joined.WriteString(delta)
			}
			tail, err := dec.Finalize()
			if err != nil {
				t.Fatalf("Finalize: %v", err)
			}
			joined.WriteString(tail)
			if got := joined.String(); got != tc.wantJoined {
				t.Fatalf("joined deltas = %q, want %q", got, tc.wantJoined)
			}
		})
	}
}

func TestIncrementalDecoderMatchesOneShotDecode(t *testing.T) {
	tok := loadFixtureTokenizer(t)
	ids := []int{1, 504, 198, 7042, 30, 198, 2}
	got := decodeIncremental(t, tok, ids)
	want, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if got != want {
		t.Fatalf("incremental decode = %q, want one-shot %q", got, want)
	}
}

func TestIncrementalDecoderFinalizeFlushesDanglingUTF8Bytes(t *testing.T) {
	tok := byteVocabTokenizer()
	dec := tok.NewIncrementalDecoder()
	delta, err := dec.Push(0xe2)
	if err != nil {
		t.Fatalf("Push: %v", err)
	}
	if delta != "" {
		t.Fatalf("dangling start byte emitted early: %q", delta)
	}
	tail, err := dec.Finalize()
	if err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	want := string([]byte{0xe2})
	if tail != want {
		t.Fatalf("Finalize dangling bytes = %q, want raw one-shot bytes %q", tail, want)
	}
}

func TestOptionalSmolLM2IncrementalMatchesDecode(t *testing.T) {
	path := findSmolLM2TokenizerJSON(t)
	tok, err := LoadJSON(path)
	if err != nil {
		t.Fatalf("LoadJSON: %v", err)
	}
	for _, text := range []string{
		"The capital of France is",
		"emoji \U0001f600\U0001f603 plus \u00a9 and \u2122",
		"\u4e2d\u6587 mixed \u0627\u0644\u0639\u0631\u0628\u064a\u0629",
		"e\u0301 cafe\u0301 split combining marks",
	} {
		ids, err := tok.Encode(text)
		if err != nil {
			t.Fatalf("Encode(%q): %v", text, err)
		}
		got := decodeIncremental(t, tok, ids)
		want, err := tok.Decode(ids)
		if err != nil {
			t.Fatalf("Decode(%q ids): %v", text, err)
		}
		if got != want {
			t.Fatalf("incremental decode %q = %q, want %q", text, got, want)
		}
		if got != text {
			t.Fatalf("round trip %q = %q", text, got)
		}
	}
}

func FuzzIncrementalDecoderMatchesDecode(f *testing.F) {
	for _, seed := range [][]byte{
		[]byte("hello"),
		[]byte{0xc3, 0xa9},
		[]byte("\U0001f600"),
		[]byte{0xe2},
		[]byte{0xe2, 0x28},
	} {
		f.Add(seed)
	}
	tok := byteVocabTokenizer()
	f.Fuzz(func(t *testing.T, raw []byte) {
		if len(raw) > 1024 {
			raw = raw[:1024]
		}
		ids := make([]int, len(raw))
		for i, b := range raw {
			ids[i] = int(b)
		}
		got := decodeIncremental(t, tok, ids)
		want, err := tok.Decode(ids)
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
		if got != want {
			t.Fatalf("incremental decode mismatch for bytes % x: got %q want %q", raw, got, want)
		}
	})
}

func decodeIncremental(t *testing.T, tok *Tokenizer, ids []int) string {
	t.Helper()
	dec := tok.NewIncrementalDecoder()
	var out strings.Builder
	for _, id := range ids {
		delta, err := dec.Push(id)
		if err != nil {
			t.Fatalf("Push(%d): %v", id, err)
		}
		out.WriteString(delta)
	}
	tail, err := dec.Finalize()
	if err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	out.WriteString(tail)
	return out.String()
}

func byteVocabTokenizer() *Tokenizer {
	tokens := make([]string, 256)
	for i := range tokens {
		tokens[i] = byteLevelPiece(byte(i))
	}
	return &Tokenizer{
		idToToken: tokens,
		tokenToID: map[string]int{},
		special:   map[int]string{},
		split:     preTokenizeByteLevel,
	}
}

func byteLevelPiece(bs ...byte) string {
	var b strings.Builder
	for _, raw := range bs {
		b.WriteRune(byteLevelEncodeRune[raw])
	}
	return b.String()
}
