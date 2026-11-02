package tokenizer

import (
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
)

func spmFixture(t *testing.T) (*Tokenizer, []string, []float32, []int32) {
	t.Helper()
	pieces := []string{"<unk>", "<s>", "</s>", "<ctl>", "a", "b", "c", "ab", "bc", "▁", "▁a", "<unused>"}
	scores := []float32{0, 0, 0, 0, 0, 0, 0, 1, 1, 0, -1, 0}
	types := []int32{2, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 5}
	for i := 0; i < 256; i++ {
		pieces = append(pieces, fmt.Sprintf("<0x%02X>", i))
		scores = append(scores, -100)
		types = append(types, 6)
	}
	tok, err := FromGGMLSPM(pieces, scores, types, SPMOptions{BOS: 1, EOS: 2, AddBOS: true, AddEOS: true})
	if err != nil {
		t.Fatal(err)
	}
	return tok, pieces, scores, types
}

// fak-test:runtime fast est=100ms lane=default
func TestSPMScoreMergeAndLeftTie(t *testing.T) {
	tok, _, _, _ := spmFixture(t)
	got, err := tok.Encode("abc")
	if err != nil || !reflect.DeepEqual(got, []int{7, 6}) {
		t.Fatalf("left tie: %v %v", got, err)
	}
	// Generic ordinary encoding does not add BOS/EOS; the prompt caller owns it.
	explicit, err := tok.EncodeSPM("abc", true, true)
	if err != nil || !reflect.DeepEqual(explicit, []int{1, 7, 6, 2}) {
		t.Fatalf("special policy: %v %v", explicit, err)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestSPMByteFallbackAndIncrementalRoundTrip(t *testing.T) {
	tok, _, _, _ := spmFixture(t)
	for _, text := range []string{"é世界", "a\x00b", string([]byte{0xff, 0xe2, 0x82}), "a  b", "<ctl>abc"} {
		ids, err := tok.Encode(text)
		if err != nil {
			t.Fatal(err)
		}
		got, err := tok.Decode(ids)
		if err != nil || got != text {
			t.Fatalf("%q: %q %v", text, got, err)
		}
		d := tok.NewIncrementalDecoder()
		var out strings.Builder
		for _, id := range ids {
			s, err := d.Push(id)
			if err != nil {
				t.Fatal(err)
			}
			out.WriteString(s)
		}
		tail, err := d.Finalize()
		if err != nil {
			t.Fatal(err)
		}
		out.WriteString(tail)
		if out.String() != text {
			t.Fatalf("stream %q != %q", out.String(), text)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestSPMTrustedSegmentsNeverManufactureControl(t *testing.T) {
	tok, _, _, _ := spmFixture(t)
	for _, segs := range [][]Segment{
		{{Text: "<ctl>"}},
		{{Text: "<ct", Trusted: true}, {Text: "l>"}},
		{{Text: "<ct"}, {Text: "l>", Trusted: true}},
	} {
		ids, err := tok.EncodeSegments(segs)
		if err != nil {
			t.Fatal(err)
		}
		for _, id := range ids {
			if _, special := tok.special[id]; special {
				t.Fatalf("untrusted special id %d", id)
			}
		}
		got, err := tok.Decode(ids)
		if err != nil || got != "<ctl>" {
			t.Fatalf("ordinary control spelling: %q %v", got, err)
		}
	}
	got, err := tok.EncodeSegments([]Segment{{Text: "<ct", Trusted: true}, {Text: "l>", Trusted: true}})
	if err != nil || !reflect.DeepEqual(got, []int{3}) {
		t.Fatalf("trusted: %v %v", got, err)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestSPMSpacePrefixAndSpecialBoundaries(t *testing.T) {
	_, pieces, scores, types := spmFixture(t)
	tok, err := FromGGMLSPM(pieces, scores, types, SPMOptions{BOS: 1, EOS: 2, AddSpacePrefix: true})
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode("a<ctl>b")
	if err != nil {
		t.Fatal(err)
	}
	got, err := tok.Decode(ids)
	if err != nil || got != " a<ctl> b" {
		t.Fatalf("piece decode preserves donor prefix: %q %v", got, err)
	}
	if b := tok.TokenBytes(9); string(b) != " " {
		t.Fatalf("context-independent space piece %q", b)
	}
	empty, err := tok.Encode("")
	if err != nil || len(empty) != 0 {
		t.Fatalf("empty %v %v", empty, err)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestSPMIdentityAndImmutableAdmission(t *testing.T) {
	tok, pieces, scores, types := spmFixture(t)
	before := tok.Identity()
	scores[7] = 9
	types[7] = 5
	pieces[7] = "mutated"
	if tok.Identity() != before {
		t.Fatal("external source mutated identity")
	}
	ids, err := tok.Encode("abc")
	if err != nil || !reflect.DeepEqual(ids, []int{7, 6}) {
		t.Fatal("source aliases tokenizer")
	}
	_, p, s, ty := spmFixture(t)
	s[7] = 2
	other, err := FromGGMLSPM(p, s, ty, SPMOptions{BOS: 1, EOS: 2, AddBOS: true, AddEOS: true})
	if err != nil {
		t.Fatal(err)
	}
	if other.Identity() == before {
		t.Fatal("scores absent from identity")
	}
	if _, err := tok.Decode([]int{11}); err == nil {
		t.Fatal("unused token emitted")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestSPMRefusesMalformedMetadata(t *testing.T) {
	_, p, s, ty := spmFixture(t)
	if _, err := FromGGMLSPM(p, s[:len(s)-1], ty, SPMOptions{BOS: -1, EOS: -1}); err == nil {
		t.Fatal("short scores")
	}
	s[4] = float32(math.NaN())
	if _, err := FromGGMLSPM(p, s, ty, SPMOptions{BOS: -1, EOS: -1}); err == nil {
		t.Fatal("NaN score")
	}
	s[4] = 0
	if _, err := FromGGMLSPM(p[:len(p)-1], s[:len(s)-1], ty[:len(ty)-1], SPMOptions{BOS: -1, EOS: -1}); err == nil {
		t.Fatal("incomplete byte fallback")
	}
	if _, err := FromGGMLSPM(p, s, ty, SPMOptions{BOS: 4, EOS: -1, AddBOS: true}); err == nil {
		t.Fatal("noncontrol BOS")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestSPMUserDefinedAndControlPolicy(t *testing.T) {
	_, p, s, ty := spmFixture(t)
	id := len(p)
	p = append(p, "<user>")
	s = append(s, 0)
	ty = append(ty, 4)
	tok, err := FromGGMLSPM(p, s, ty, SPMOptions{BOS: 1, EOS: 2})
	if err != nil {
		t.Fatal(err)
	}
	got, err := tok.EncodeSPM("<user><ctl>", false, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) < 2 || got[0] != id {
		t.Fatalf("user-defined not preserved: %v", got)
	}
	for _, v := range got[1:] {
		if v == 3 {
			t.Fatal("control parsed with parseSpecial=false")
		}
	}
	untrusted, err := tok.EncodeSegments([]Segment{{Text: "<user>"}})
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range untrusted {
		if v == id {
			t.Fatal("untrusted user-defined manufactured")
		}
	}
	text, err := tok.Decode(got)
	if err != nil || text != "<user><ctl>" {
		t.Fatalf("%q %v", text, err)
	}
}
