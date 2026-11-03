package tokenizer

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func nativeSPMFixtureJSON(t *testing.T) []byte {
	t.Helper()
	_, p, s, ty := spmFixture(t)
	b, err := json.Marshal(map[string]any{
		"fak_tokenizer_profile": "llama-spm-v1", "normalization": "identity",
		"tokens": p, "scores": s, "types": ty,
		"options": map[string]any{"bos_id": 1, "eos_id": 2, "add_bos": true, "add_eos": true, "add_space_prefix": false},
	})
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// fak-test:runtime fast est=100ms lane=default
func TestNativeSPMJSONLoadAndPolicy(t *testing.T) {
	b := nativeSPMFixtureJSON(t)
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	tok, err := LoadJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	direct, _, _, _ := spmFixture(t)
	if !tok.IsSPM() || tok.Identity() != direct.Identity() {
		t.Fatal("profile lost immutable policy")
	}
	ids, err := tok.EncodeSPM("abc", true, false)
	if err != nil || !reflect.DeepEqual(ids, []int{1, 7, 6, 2}) {
		t.Fatalf("%v %v", ids, err)
	}
	plain, err := tok.Encode("abc")
	if err != nil || !reflect.DeepEqual(plain, []int{7, 6}) {
		t.Fatal("generic encoding added special tokens")
	}
	for _, text := range []string{"é世界", "a\x00b", string([]byte{255, 226, 130}), "<ctl>abc"} {
		ids, err := tok.Encode(text)
		if err != nil {
			t.Fatal(err)
		}
		got, err := tok.Decode(ids)
		if err != nil || got != text {
			t.Fatalf("roundtrip %q %q %v", text, got, err)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestNativeSPMJSONRejectsFormatAmbiguity(t *testing.T) {
	b := string(nativeSPMFixtureJSON(t))
	cases := []string{
		strings.Replace(b, "llama-spm-v1", "Unigram", 1),
		strings.Replace(b, "\"identity\"", "\"NFKC\"", 1),
		strings.Replace(b, "\"add_bos\":true", "\"add_bos\":null", 1),
		strings.Replace(b, "\"add_bos\":true", "\"add_bos\":true,\"add_bos\":false", 1),
		strings.Replace(b, "\"bos_id\":1", "\"bos_id\":1.5", 1),
		strings.Replace(b, "\"scores\":[0,", "\"scores\":[null,", 1),
		strings.Replace(b, "\"normalization\":\"identity\",", "", 1),
		strings.Replace(b, "{", "{\"unknown_transform\":true,", 1),
		strings.Replace(b, "{", "{\"fak_tokenizer_profile\":\"llama-spm-v1\",", 1),
		b + " {}",
	}
	for i, bad := range cases {
		tok, err := ParseJSON([]byte(bad))
		if err == nil || tok != nil {
			t.Fatalf("case %d admitted", i)
		}
	}
	if _, err := ParseJSON([]byte(strings.Replace(b, "\"identity\"", "\"NFKC\"", 1))); !errors.Is(err, ErrNativeSPMProfile) {
		t.Fatalf("typed error %v", err)
	}
	if _, err := ParseJSON([]byte(`{"model":{"type":"Unigram","vocab":[["a",0]]}}`)); err == nil {
		t.Fatal("generic Unigram silently admitted")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestNativeSPMJSONPreservesBPE(t *testing.T) {
	tok, err := ParseJSON([]byte(`{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"decoder":{"type":"ByteLevel"}}`))
	if err != nil || tok.IsSPM() {
		t.Fatalf("legacy BPE %v", err)
	}
	ids, err := tok.Encode("a")
	if err != nil || !reflect.DeepEqual(ids, []int{0}) {
		t.Fatalf("%v %v", ids, err)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestNativeSPMJSONUnicodeRefusal(t *testing.T) {
	b := string(nativeSPMFixtureJSON(t))
	for _, piece := range []string{string([]byte{255}), `\uD800`, `\uDC00`, `\uD800x`} {
		bad := strings.Replace(b, "\"a\"", "\""+piece+"\"", 1)
		if tok, err := ParseJSON([]byte(bad)); err == nil || tok != nil {
			t.Fatal("malformed Unicode admitted")
		}
	}
	for _, s := range []string{`"\uD83D\uDE00"`, `"\\uD800"`, `"é"`} {
		if !nativeSPMUnicode([]byte(s)) {
			t.Fatalf("valid Unicode refused: %s", s)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestNativeSPMJSONForbiddenLegacyShapeIsTyped(t *testing.T) {
	b := string(nativeSPMFixtureJSON(t))
	for _, extra := range []string{`"model":{"vocab":17},`, `"decoder":5,`} {
		bad := strings.Replace(b, "{", "{"+extra, 1)
		if tok, err := ParseJSON([]byte(bad)); tok != nil || !errors.Is(err, ErrNativeSPMProfile) {
			t.Fatalf("selected profile escaped typed refusal: %v", err)
		}
	}
}
