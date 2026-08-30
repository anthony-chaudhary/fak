package vdso

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"
)

func TestNonsemanticPathFieldsNormalizeBeforeCacheHashing(t *testing.T) {
	v := New(8)
	v.RegisterNonsemanticPathFields("compile", "install_path")

	first := []byte(`{"install_path":"/tmp/root-a/pkg","semantic_path":"src/main.mo","target":"host"}`)
	want := nonsemanticPathHashOracle(t, first, "install_path")
	if got := v.argHashFor("compile", first); got != want {
		t.Fatalf("declared nonsemantic path hash=%q, want oracle %q", got, want)
	}

	variants := []string{
		`{"install_path":"/tmp/root-b/pkg","semantic_path":"src/main.mo","target":"host"}`,
		`{"install_path":"../pkg","semantic_path":"src/main.mo","target":"host"}`,
		`{"install_path":"/tmp/symlink/pkg","semantic_path":"src/main.mo","target":"host"}`,
		`{"install_path":"C:\\TMP\\PKG","semantic_path":"src/main.mo","target":"host"}`,
		`{"install_path":"/TMP/PKG","semantic_path":"src/main.mo","target":"host"}`,
	}
	for _, args := range variants {
		if got := v.argHashFor("compile", []byte(args)); got != want {
			t.Errorf("declared path variant %s hash=%q, want %q", args, got, want)
		}
	}

	semanticChange := []byte(`{"install_path":"/tmp/root-b/pkg","semantic_path":"src/other.mo","target":"host"}`)
	if got := v.argHashFor("compile", semanticChange); got == want {
		t.Fatalf("semantic path change hash=%q, must differ from %q", got, want)
	}

	if got := v.argHashFor("other-tool", []byte(variants[0])); got == want {
		t.Fatalf("other tool hash=%q, must not inherit compile's schema declaration", got)
	}
}

func TestNonsemanticPathFieldsFailClosed(t *testing.T) {
	v := New(8)
	v.RegisterNonsemanticPathFields("compile", "install_path")

	cases := [][]byte{
		[]byte(`{"install_path":7,"target":"host"}`),
		[]byte(` { "install_path" : 7, "target" : "host" } `),
		[]byte(`{"install_path":"/tmp/pkg","target":`),
		[]byte(`[ "/tmp/pkg" ]`),
		[]byte(`null`),
	}
	for _, args := range cases {
		if got, want := v.argHashFor("compile", args), rawArgHash(args); got != want {
			t.Errorf("args=%s hash=%q, want exact-byte fallback %q", args, got, want)
		}
	}
}

func nonsemanticPathHashOracle(t *testing.T, args []byte, fields ...string) string {
	t.Helper()
	var object map[string]any
	if err := json.Unmarshal(args, &object); err != nil {
		t.Fatalf("oracle decode: %v", err)
	}
	for _, field := range fields {
		delete(object, field)
	}
	canonical, err := json.Marshal(object)
	if err != nil {
		t.Fatalf("oracle encode: %v", err)
	}
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:])[:24]
}
