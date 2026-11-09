package tokenizer

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestMetaspaceTransformEncodeConsumers(t *testing.T) {
	tok := transformSelectionFixture(t, "gemma4")
	if !tok.metaspace {
		t.Fatal("fixture did not select the real metaspace BPE route")
	}
	// With no added tokens, each entry point handles one ordinary chunk. The
	// metaspace route retains that entire chunk as one piece rather than splitting.
	for _, tc := range []struct {
		name string
		run  func() ([]int, error)
	}{
		{"encode", func() ([]int, error) { return tok.Encode("ab ab") }},
		{"segments", func() ([]int, error) {
			return tok.EncodeSegments([]Segment{{Text: "ab "}, {Text: "ab"}})
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.run()
			if err != nil || !reflect.DeepEqual(got, []int{3, 2, 3}) {
				t.Fatalf("metaspace IDs=%v err=%v, want [3 2 3]", got, err)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestByteLevelTransformEncodeConsumers(t *testing.T) {
	tok := transformSelectionFixture(t, "gpt2")
	if tok.metaspace || tok.split == nil || !reflect.DeepEqual(tok.split("ab"), []string{"ab"}) {
		t.Fatal("fixture did not select one real ByteLevel piece")
	}
	for _, tc := range []struct {
		name string
		run  func() ([]int, error)
	}{
		{"encode", func() ([]int, error) { return tok.Encode("ab") }},
		{"segments", func() ([]int, error) {
			return tok.EncodeSegments([]Segment{{Text: "a"}, {Text: "b"}})
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.run()
			if err != nil || !reflect.DeepEqual(got, []int{3}) {
				t.Fatalf("ByteLevel IDs=%v err=%v, want [3]", got, err)
			}
		})
	}
}

func transformSelectionFixture(t *testing.T, pre string) *Tokenizer {
	t.Helper()
	tok, err := FromGGML([]string{"a", "b", "▁", "ab"}, []string{"a b"}, nil, pre)
	if err != nil {
		t.Fatal(err)
	}
	if tok.spm != nil || tok.ignoreMerges || len(tok.addedByContent) != 0 {
		t.Fatal("fixture must use ordinary BPE without added-token boundaries")
	}
	return tok
}
