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

// legacyArgHashFor simulates the pre-optimization decode/re-encode pipeline
// to verify exact hash identity across all combinations.
func legacyArgHashFor(v *VDSO, tool string, args []byte) string {
	v.regMu.RLock()
	declaration := v.nonsemanticPathFields[tool]
	fields := make([]string, 0, len(declaration))
	for field := range declaration {
		fields = append(fields, field)
	}
	v.regMu.RUnlock()

	var normalized []byte
	declared := len(fields) > 0
	ok := true
	if declared {
		var object map[string]any
		if err := json.Unmarshal(args, &object); err != nil || object == nil {
			ok = false
		} else {
			for _, field := range fields {
				value, present := object[field]
				if !present {
					continue
				}
				if _, isStr := value.(string); !isStr {
					ok = false
					break
				}
				delete(object, field)
			}
			if ok {
				var err error
				normalized, err = json.Marshal(object)
				if err != nil {
					ok = false
				}
			}
		}
	}

	if !ok {
		return rawArgHash(args)
	}
	if declared {
		args = normalized
	}
	if v.NearDupOf() {
		return legacyNearDupArgHash(args)
	}
	return argHash(args)
}

func TestArgHashFor_ExactHashEquivalence(t *testing.T) {
	cases := []struct {
		name string
		tool string
		args []byte
	}{
		{"declared_present_single", "compile", []byte(`{"install_path":"/tmp/pkg","target":"host"}`)},
		{"declared_present_formatting", "compile", []byte(`{"install_path":"/tmp/pkg","target":"  HOST  ","note":"Build  Task"}`)},
		{"declared_missing", "compile", []byte(`{"target":"host","debug":true}`)},
		{"declared_non_string_int", "compile", []byte(`{"install_path":123,"target":"host"}`)},
		{"declared_non_string_bool", "compile", []byte(`{"install_path":true,"target":"host"}`)},
		{"declared_non_string_array", "compile", []byte(`{"install_path":["/tmp/pkg"],"target":"host"}`)},
		{"declared_non_string_obj", "compile", []byte(`{"install_path":{"p":"/tmp"},"target":"host"}`)},
		{"declared_malformed", "compile", []byte(`{"install_path":"/tmp",`)},
		{"declared_array_root", "compile", []byte(`[1,2,3]`)},
		{"declared_null_root", "compile", []byte(`null`)},
		{"declared_empty_obj", "compile", []byte(`{}`)},
		{"declared_non_json", "compile", []byte(`not json at all`)},
		{"undeclared_standard", "search", []byte(`{"query":"hello world","limit":10}`)},
		{"undeclared_formatting", "search", []byte(`{"query":"  HELLO   WORLD  ","limit":10}`)},
		{"undeclared_malformed", "search", []byte(`{"query":`)},
		{"undeclared_non_json", "search", []byte(`plain text`)},
		{"undeclared_empty", "search", []byte(``)},
	}

	for _, nearDup := range []bool{false, true} {
		v := New(8)
		v.SetNearDup(nearDup)
		v.RegisterNonsemanticPathFields("compile", "install_path")

		for _, tc := range cases {
			got := v.argHashFor(tc.tool, tc.args)
			want := legacyArgHashFor(v, tc.tool, tc.args)
			if got != want {
				t.Errorf("nearDup=%v case=%s tool=%s got=%q, want legacy=%q", nearDup, tc.name, tc.tool, got, want)
			}
		}
	}
}

func BenchmarkArgHashFor_NonsemanticPath(b *testing.B) {
	v := New(8)
	v.RegisterNonsemanticPathFields("compile", "install_path")
	input := []byte(`{"install_path":"/tmp/root-a/pkg","semantic_path":"src/main.mo","target":"host","debug":true}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = v.argHashFor("compile", input)
	}
}

func BenchmarkArgHashFor_NonsemanticPathWithNearDup(b *testing.B) {
	v := New(8)
	v.SetNearDup(true)
	v.RegisterNonsemanticPathFields("compile", "install_path")
	input := []byte(`{"install_path":"/tmp/root-a/pkg","semantic_path":"src/main.mo","target":"host","note":"Build   Task"}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = v.argHashFor("compile", input)
	}
}
