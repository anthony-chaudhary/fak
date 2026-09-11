package tokenizer

import (
	"regexp"
	"testing"
)

func TestTokenizerOperationalIdentity(t *testing.T) {
	base, err := FromGGML([]string{"a", "b", "ab", "<special>"}, []string{"a b"}, []int32{1, 1, 1, 4}, "default")
	if err != nil {
		t.Fatal(err)
	}
	id := base.Identity()
	if !regexp.MustCompile(`^sha256:[0-9a-f]{64}$`).MatchString(id) || base.Identity() != id {
		t.Fatalf("Identity() = %q; want stable closed sha256 identity", id)
	}
	if got := (*Tokenizer)(nil).Identity(); got != "" {
		t.Fatalf("nil Identity() = %q, want empty", got)
	}

	jsonA := []byte(`{"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2,"<special>":3},"merges":["a b"]},"pre_tokenizer":{"type":"ByteLevel"},"decoder":{"type":"ByteLevel"},"added_tokens":[{"id":3,"content":"<special>","special":true}]}`)
	jsonB := []byte(`{"added_tokens":[{"special":true,"content":"<special>","id":3}],"decoder":{"type":"ByteLevel"},"pre_tokenizer":{"type":"ByteLevel"},"model":{"merges":["a b"],"vocab":{"<special>":3,"ab":2,"b":1,"a":0},"type":"BPE"}}`)
	parsedA, err := ParseJSON(jsonA)
	if err != nil {
		t.Fatal(err)
	}
	parsedB, err := ParseJSON(jsonB)
	if err != nil {
		t.Fatal(err)
	}
	if parsedA.Identity() != parsedB.Identity() {
		t.Fatalf("serialization/map ordering changed identity: %q != %q", parsedA.Identity(), parsedB.Identity())
	}
	if parsedA.Identity() != base.Identity() {
		t.Fatalf("equivalent JSON and GGML operational states differ: %q != %q", parsedA.Identity(), base.Identity())
	}

	// Both fresh constructor inputs have the same decode-order vocabulary, merge ranks,
	// and pre-tokenizer. Reversing duplicate added tokens changes both last-write encode
	// lookup and first-match precedence, so their operational identities must differ.
	duplicateA, err := ParseJSON([]byte(`{"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]},"pre_tokenizer":{"type":"ByteLevel"},"decoder":{"type":"ByteLevel"},"added_tokens":[{"id":3,"content":"<dup>","special":true},{"id":4,"content":"<dup>","special":true}]}`))
	if err != nil {
		t.Fatal(err)
	}
	duplicateB, err := ParseJSON([]byte(`{"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]},"pre_tokenizer":{"type":"ByteLevel"},"decoder":{"type":"ByteLevel"},"added_tokens":[{"id":4,"content":"<dup>","special":true},{"id":3,"content":"<dup>","special":true}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if duplicateA.Identity() == duplicateB.Identity() {
		t.Fatal("different duplicate added-token lookup and match precedence retained identity")
	}

	variants := []struct {
		name   string
		tokens []string
		merges []string
		types  []int32
		pre    string
	}{
		{name: "token bytes", tokens: []string{"a", "c", "ac", "<special>"}, merges: []string{"a c"}, types: []int32{1, 1, 1, 4}, pre: "default"},
		{name: "token ids", tokens: []string{"b", "a", "ab", "<special>"}, merges: []string{"a b"}, types: []int32{1, 1, 1, 4}, pre: "default"},
		{name: "merge rank", tokens: []string{"a", "b", "ab", "ba", "<special>"}, merges: []string{"b a", "a b"}, types: []int32{1, 1, 1, 1, 4}, pre: "default"},
		{name: "special content", tokens: []string{"a", "b", "ab", "<other>"}, merges: []string{"a b"}, types: []int32{1, 1, 1, 4}, pre: "default"},
		{name: "special type", tokens: []string{"a", "b", "ab", "<special>"}, merges: []string{"a b"}, types: []int32{1, 1, 1, 1}, pre: "default"},
		{name: "qwen pretokenizer", tokens: []string{"a", "b", "ab", "<special>"}, merges: []string{"a b"}, types: []int32{1, 1, 1, 4}, pre: "qwen2"},
		{name: "glm4 pretokenizer", tokens: []string{"a", "b", "ab", "<special>"}, merges: []string{"a b"}, types: []int32{1, 1, 1, 4}, pre: "glm4"},
		{name: "metaspace", tokens: []string{"a", "b", "ab", "<special>"}, merges: []string{"a b"}, types: []int32{1, 1, 1, 4}, pre: "gemma4"},
	}
	for _, tt := range variants {
		t.Run(tt.name, func(t *testing.T) {
			changed, err := FromGGML(tt.tokens, tt.merges, tt.types, tt.pre)
			if err != nil {
				t.Fatal(err)
			}
			if changed.Identity() == id {
				t.Fatalf("encode-affecting %s mutation retained identity %q", tt.name, id)
			}
		})
	}
}
