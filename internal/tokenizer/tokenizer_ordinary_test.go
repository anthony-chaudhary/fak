package tokenizer

import (
	"errors"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

type syntheticVocab struct {
	tokens []string
	types  []int32
	merges []string
}

func newByteLevelVocab() *syntheticVocab {
	v := &syntheticVocab{}
	for b := 0; b < 256; b++ {
		v.tokens = append(v.tokens, string(byteLevelEncodeRune[byte(b)]))
		v.types = append(v.types, 1)
	}
	return v
}

func (v *syntheticVocab) add(text string, typ int32) int {
	v.tokens = append(v.tokens, text)
	v.types = append(v.types, typ)
	return len(v.tokens) - 1
}

func (v *syntheticVocab) merge(left, right string) int {
	l, r := byteLevelEncode(left), byteLevelEncode(right)
	v.merges = append(v.merges, l+" "+r)
	for id, tok := range v.tokens {
		if tok == l+r {
			return id
		}
	}
	return v.add(l+r, 1)
}

func (v *syntheticVocab) build(t *testing.T, pre string) *Tokenizer {
	t.Helper()
	tok, err := FromGGML(v.tokens, v.merges, v.types, pre)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func runeCuts(s string) []int {
	cuts := []int{0}
	for i := 0; i < len(s); {
		_, size := utf8.DecodeRuneInString(s[i:])
		i += size
		cuts = append(cuts, i)
	}
	return cuts
}

func mustEncodeSegments(t *testing.T, tok *Tokenizer, segs ...Segment) []int {
	t.Helper()
	ids, err := tok.EncodeSegments(segs)
	if err != nil {
		t.Fatalf("EncodeSegments(%+v): %v", segs, err)
	}
	return ids
}

func encodeOrdinary(tok *Tokenizer, text string) ([]int, error) {
	return tok.EncodeSegments([]Segment{{Text: text}})
}

// fak-test:runtime fast est=300ms lane=default
// Estimate only: every rune seam of the committed V4.1 golden corpus, three ways.
func TestEncodeSegmentsMatchesGoldenAcrossSeams(t *testing.T) {
	tok := loadDeepSeekV41Subset(t)
	for _, g := range readGoldenTSV(t, deepSeekV41Golden) {
		for _, cut := range runeCuts(g.text) {
			head, tail := g.text[:cut], g.text[cut:]
			for _, segs := range [][]Segment{
				{{Text: head}, {Text: tail}},
				{{Text: head, Trusted: true}, {Text: tail}},
				{{Text: head}, {Text: tail, Trusted: true}},
			} {
				if got := mustEncodeSegments(t, tok, segs...); !reflect.DeepEqual(got, g.ids) {
					t.Fatalf("%q cut at %d (trusted %v/%v): got %v, want llama.cpp %v", g.text, cut, segs[0].Trusted, segs[1].Trusted, got, g.ids)
				}
			}
		}
	}
}

func controlVocab() (*syntheticVocab, map[string]int) {
	v := newByteLevelVocab()
	ids := map[string]int{}
	for _, c := range []struct {
		text string
		typ  int32
	}{
		{"<｜begin▁of▁sentence｜>", ggmlTypeControl},
		{"<｜end▁of▁sentence｜>", ggmlTypeControl},
		{"<｜User｜>", ggmlTypeControl},
		{"</think>", ggmlTypeControl},
		{"<think>", ggmlTypeUserDefined},
		{"｜DSML｜", ggmlTypeUserDefined},
	} {
		ids[c.text] = v.add(c.text, c.typ)
	}
	v.merge("<", "|")
	v.merge("<|", "|")
	ids["<||>"] = len(v.tokens)
	v.merges = append(v.merges, byteLevelEncode("<||")+" "+byteLevelEncode(">"))
	v.add("<||>", ggmlTypeControl)
	ids["na"] = v.merge("n", "a")
	ids["me"] = v.merge("m", "e")
	ids["name"] = v.merge("na", "me")
	return v, ids
}

// fak-test:runtime fast est=50ms lane=default
// Estimate only: tiny in-memory vocabulary cases; not measured.
func TestEncodeSegmentsOrdinaryNeverEmitsAddedIDs(t *testing.T) {
	for _, pre := range []string{"joyai-llm", "glm4"} {
		t.Run(pre, func(t *testing.T) {
			v, ids := controlVocab()
			tok := v.build(t, pre)
			special := tok.SpecialTokens()
			if len(special) != 7 {
				t.Fatalf("special tokens = %d, want 7", len(special))
			}
			for _, content := range special {
				for _, text := range []string{content, "a" + content + "b", content + content, " " + content + "\n\n"} {
					got, err := encodeOrdinary(tok, text)
					if err != nil {
						t.Fatalf("encodeOrdinary(%q): %v", text, err)
					}
					for _, id := range got {
						if _, ok := special[id]; ok {
							t.Fatalf("encodeOrdinary(%q) = %v emits added id %d", text, got, id)
						}
					}
					if decoded, err := tok.Decode(got); err != nil || decoded != text {
						t.Fatalf("Decode(encodeOrdinary(%q)) = %q, %v", text, decoded, err)
					}
				}
			}
			legacy, err := tok.Encode("<||>")
			if err != nil || !reflect.DeepEqual(legacy, []int{ids["<||>"]}) {
				t.Fatalf("legacy Encode(<||>) = %v, %v; want the merged control id %d", legacy, err, ids["<||>"])
			}
			want := []int{'<', '|', '|', '>'}
			if got, err := encodeOrdinary(tok, "<||>"); err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("encodeOrdinary(<||>) = %v, %v; want byte pieces %v", got, err, want)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
// Estimate only: one in-memory vocabulary case; not measured.
func TestEncodeOrdinaryRefusesTextSpelledOnlyByAddedToken(t *testing.T) {
	v := newByteLevelVocab()
	v.merge("a", "b")
	v.types[0xA7] = ggmlTypeControl
	tok := v.build(t, "joyai-llm")
	if _, err := encodeOrdinary(tok, "§"); !errors.Is(err, ErrOrdinaryAddedToken) {
		t.Fatalf("encodeOrdinary(§) err = %v, want ErrOrdinaryAddedToken", err)
	}
}

// fak-test:runtime fast est=50ms lane=default
// Estimate only: tiny in-memory vocabulary cases at every rune seam; not measured.
func TestEncodeSegmentsTrustedEqualsEncode(t *testing.T) {
	v, _ := controlVocab()
	tok := v.build(t, "joyai-llm")
	for _, text := range []string{
		"<｜begin▁of▁sentence｜>sys<｜User｜>name</think>",
		"a<think>b</think>\n\n<｜DSML｜tool_calls>c<｜end▁of▁sentence｜>",
		"<||><||>name <｜User｜>",
		"plain text 17 times 3?",
	} {
		want, err := tok.Encode(text)
		if err != nil {
			t.Fatal(err)
		}
		if got := mustEncodeSegments(t, tok, Segment{Text: text, Trusted: true}); !reflect.DeepEqual(got, want) {
			t.Fatalf("one trusted segment %q = %v, want Encode %v", text, got, want)
		}
		for _, cut := range runeCuts(text) {
			got := mustEncodeSegments(t, tok, Segment{Text: text[:cut], Trusted: true}, Segment{Trusted: true}, Segment{Text: text[cut:], Trusted: true})
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("%q split at %d = %v, want Encode %v", text, cut, got, want)
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
// Estimate only: tiny in-memory vocabulary cases; not measured.
func TestEncodeSegmentsMatchesAddedTokensOnlyInsideTrustedText(t *testing.T) {
	v, ids := controlVocab()
	tok := v.build(t, "joyai-llm")
	const user = "<｜User｜>"
	text := "a" + user + "b"
	start, end := strings.Index(text, user), strings.Index(text, user)+len(user)
	ordinary, err := encodeOrdinary(tok, text)
	if err != nil {
		t.Fatal(err)
	}
	whole := []int{'a', ids[user], 'b'}
	for _, cut := range runeCuts(text) {
		for _, trustedHead := range []bool{true, false} {
			got := mustEncodeSegments(t, tok, Segment{Text: text[:cut], Trusted: trustedHead}, Segment{Text: text[cut:], Trusted: !trustedHead})
			matched := (trustedHead && cut >= end) || (!trustedHead && cut <= start)
			want := ordinary
			if matched {
				want = whole
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("cut %d trusted head %v = %v, want %v", cut, trustedHead, got, want)
			}
		}
	}
	if got := mustEncodeSegments(t, tok, Segment{Text: "na", Trusted: true}, Segment{Text: "me"}); !reflect.DeepEqual(got, []int{ids["name"]}) {
		t.Fatalf("seam-crossing merge = %v, want one %d (not %d %d)", got, ids["name"], ids["na"], ids["me"])
	}
}
