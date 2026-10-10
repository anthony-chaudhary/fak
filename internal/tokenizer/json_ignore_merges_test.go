package tokenizer

import (
	"fmt"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
// Estimate only: three tiny in-memory JSON vocabularies; not measured.
func TestJSONIgnoreMerges(t *testing.T) {
	identities := make(map[string]string)
	for _, tc := range []struct {
		name  string
		field string
		want  []int
	}{
		{"enabled", `,"ignore_merges":true`, []int{4}},
		{"disabled", `,"ignore_merges":false`, []int{3, 2}},
		{"omitted", "", []int{3, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// "abc" exists in the vocabulary but cannot be produced by the
			// only merge. The JSON flag decides whether the whole piece wins.
			doc := fmt.Sprintf(`{"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"abc":4},"merges":["a b"]%s},"pre_tokenizer":{"type":"ByteLevel"},"decoder":{"type":"ByteLevel"}}`, tc.field)
			tok, err := ParseJSON([]byte(doc))
			if err != nil {
				t.Fatal(err)
			}
			got, err := tok.Encode("abc")
			if err != nil || !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("Encode(abc) = %v, %v; want %v", got, err, tc.want)
			}
			// A missing whole piece still follows ranked BPE in every mode.
			got, err = tok.Encode("abca")
			if err != nil || !reflect.DeepEqual(got, []int{3, 2, 0}) {
				t.Fatalf("Encode(abca) = %v, %v; want [3 2 0]", got, err)
			}
			identities[tc.name] = tok.Identity()
		})
	}
	if identities["enabled"] == identities["disabled"] {
		t.Fatal("different merge behavior has the same tokenizer identity")
	}
	if identities["disabled"] != identities["omitted"] {
		t.Fatal("explicit false changed the default tokenizer identity")
	}
}
