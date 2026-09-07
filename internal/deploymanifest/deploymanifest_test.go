package deploymanifest

import (
	"errors"
	"reflect"
	"testing"
)

// TestFakTomlRoundTrip is the primary witness for #3421: the artifact `fak init`
// emits (Minimal) parses cleanly, and its declared values land on the typed
// Manifest — `fak init` emits → load round-trips.
func TestFakTomlRoundTrip(t *testing.T) {
	m, err := Parse(Minimal())
	if err != nil {
		t.Fatalf("Minimal() must parse clean, got: %v", err)
	}
	if !m.Runtimes.Gateway || !m.Runtimes.AgentRuntime {
		t.Errorf("minimal manifest should start both runtimes, got %+v", m.Runtimes)
	}
	if m.Runtimes.Model != "upstream" {
		t.Errorf("model = %q, want upstream", m.Runtimes.Model)
	}
	if m.Auth.RequireKeyEnv != "FAK_GATEWAY_KEY" {
		t.Errorf("require_key_env = %q, want FAK_GATEWAY_KEY", m.Auth.RequireKeyEnv)
	}
	if !m.Observability.Metrics || m.Observability.Bind != "127.0.0.1:9090" {
		t.Errorf("observability = %+v, want metrics on @127.0.0.1:9090", m.Observability)
	}
}

// TestFakTomlUnknownKeyRefuses is the load-time contract: a typo'd key must
// refuse at load with the named closed-vocabulary reason, NOT silently disable
// auth. This is the exact failure the contract exists to kill.
func TestFakTomlUnknownKeyRefuses(t *testing.T) {
	// `requre_key_env` (typo of require_key_env) must not quietly leave auth off.
	src := []byte("[auth]\nrequre_key_env = \"FAK_GATEWAY_KEY\"\n")
	_, err := Parse(src)
	if err == nil {
		t.Fatal("typo'd auth key must refuse at load, got nil (silent drift)")
	}
	var le *LoadError
	if !errors.As(err, &le) {
		t.Fatalf("want *LoadError, got %T: %v", err, err)
	}
	if le.Reason != ReasonUnknownKey {
		t.Errorf("reason = %q, want %q", le.Reason, ReasonUnknownKey)
	}
	if le.Section != "auth" || le.Key != "requre_key_env" {
		t.Errorf("locus = [%s] %q, want [auth] requre_key_env", le.Section, le.Key)
	}
}

// TestFakTomlUnknownSectionRefuses: an unknown [section] header refuses too.
func TestFakTomlUnknownSectionRefuses(t *testing.T) {
	_, err := Parse([]byte("[bogus]\nx = true\n"))
	var le *LoadError
	if !errors.As(err, &le) || le.Reason != ReasonUnknownSection {
		t.Fatalf("unknown section must refuse with UNKNOWN_SECTION, got %v", err)
	}
	if le.Section != "bogus" {
		t.Errorf("section = %q, want bogus", le.Section)
	}
}

// TestFakTomlBadValueRefuses: a wrong-typed / out-of-vocabulary value refuses.
func TestFakTomlBadValueRefuses(t *testing.T) {
	cases := []struct {
		name string
		src  string
	}{
		{"model-not-in-vocab", "[runtimes]\nmodel = \"gpu-cluster\"\n"},
		{"gateway-not-bool", "[runtimes]\ngateway = \"yes\"\n"},
		{"budget-not-int", "[budgets]\ndefault_tokens = \"lots\"\n"},
		{"retention-negative", "[audit]\nretention_days = -3\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Parse([]byte(tc.src))
			var le *LoadError
			if !errors.As(err, &le) || le.Reason != ReasonBadValue {
				t.Fatalf("want BAD_VALUE, got %v", err)
			}
		})
	}
}

// TestFakTomlDuplicateKeyRefuses: the same key twice in a section refuses.
func TestFakTomlDuplicateKeyRefuses(t *testing.T) {
	_, err := Parse([]byte("[auth]\nrequire_key_env = \"A\"\nrequire_key_env = \"B\"\n"))
	var le *LoadError
	if !errors.As(err, &le) || le.Reason != ReasonDuplicateKey {
		t.Fatalf("duplicate key must refuse with DUPLICATE_KEY, got %v", err)
	}
}

// TestFakTomlBareKeyRefuses: a key before any [section] refuses.
func TestFakTomlBareKeyRefuses(t *testing.T) {
	_, err := Parse([]byte("gateway = true\n"))
	var le *LoadError
	if !errors.As(err, &le) || le.Reason != ReasonBareKey {
		t.Fatalf("bare key must refuse with BARE_KEY, got %v", err)
	}
}

// TestFakTomlMalformedLineRefuses: a non-header line with no `=` refuses.
func TestFakTomlMalformedLineRefuses(t *testing.T) {
	_, err := Parse([]byte("[runtimes]\ngateway true\n"))
	var le *LoadError
	if !errors.As(err, &le) || le.Reason != ReasonMalformedLine {
		t.Fatalf("malformed line must refuse with MALFORMED_LINE, got %v", err)
	}
}

// TestFakTomlPrecedence is the flags > manifest > defaults contract. A field
// absent from the manifest keeps its default; a field set in the manifest
// overrides the default; an explicit flag override beats the manifest.
func TestFakTomlPrecedence(t *testing.T) {
	// Manifest overrides the default: default model is "upstream"; set in-kernel.
	m, err := Parse([]byte("[runtimes]\nmodel = \"in-kernel\"\n"))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if m.Runtimes.Model != "in-kernel" {
		t.Errorf("manifest should override default: model = %q, want in-kernel", m.Runtimes.Model)
	}
	// Field absent from manifest keeps the default.
	if m.Observability.Bind != "127.0.0.1:9090" {
		t.Errorf("absent field should keep default bind, got %q", m.Observability.Bind)
	}
	// Explicit flag override beats the manifest.
	flagModel := "upstream"
	off := false
	m2 := m.WithOverrides(Overrides{Model: &flagModel, Gateway: &off})
	if m2.Runtimes.Model != "upstream" {
		t.Errorf("flag should override manifest: model = %q, want upstream", m2.Runtimes.Model)
	}
	if m2.Runtimes.Gateway {
		t.Error("flag override gateway=false should win over default true")
	}
	// A nil override must not disturb the manifest value.
	m3 := m.WithOverrides(Overrides{})
	if m3.Runtimes.Model != "in-kernel" {
		t.Errorf("nil override must keep manifest value, got %q", m3.Runtimes.Model)
	}
}

// TestFakTomlDefaults pins the built-in defaults layer.
func TestFakTomlDefaults(t *testing.T) {
	d := Defaults()
	if !d.Runtimes.Gateway || !d.Runtimes.AgentRuntime || d.Runtimes.Model != "upstream" {
		t.Errorf("default runtimes = %+v", d.Runtimes)
	}
	if d.Auth.RequireKeyEnv != "" {
		t.Errorf("default auth must be empty (opt-in), got %q", d.Auth.RequireKeyEnv)
	}
}

func TestPresentDistinguishesDeclaredValuesFromDefaults(t *testing.T) {
	m, err := Parse([]byte("[observability]\nbind = \"127.0.0.1:9090\"\n"))
	if err != nil {
		t.Fatal(err)
	}
	if !m.Present("observability", "bind") {
		t.Fatal("declared observability.bind not recorded as present")
	}
	if m.Present("runtimes", "gateway") {
		t.Fatal("omitted default runtimes.gateway recorded as user-declared")
	}
}

func TestKnownAndDeclaredKeysAreStableAndComplete(t *testing.T) {
	keys := KnownKeys()
	if len(keys) != 33 {
		t.Fatalf("KnownKeys count = %d, want 33", len(keys))
	}
	for i := 1; i < len(keys); i++ {
		if keys[i-1].Dotted() >= keys[i].Dotted() {
			t.Fatalf("KnownKeys not strictly sorted at %q, %q", keys[i-1].Dotted(), keys[i].Dotted())
		}
	}
	m, err := Parse([]byte("[policy]\nfloor = \"policy.json\"\n[tenants]\nenabled = false\n"))
	if err != nil {
		t.Fatal(err)
	}
	declared := m.DeclaredKeys()
	got := []string{declared[0].Dotted(), declared[1].Dotted()}
	want := []string{"policy.floor", "tenants.enabled"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("DeclaredKeys = %v, want %v", got, want)
	}
}
