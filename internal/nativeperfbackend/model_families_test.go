package nativeperfbackend

import (
	"strings"
	"testing"
)

// TestModelFamilyAllowlistIsConfigDriven pins the registry seam: the
// allow-list comes from the embedded config, not from Go source literals.
func TestModelFamilyAllowlistIsConfigDriven(t *testing.T) {
	want := "Qwen3.8,other"
	got := strings.Join(ModelFamilies(), ",")
	if got != want {
		t.Fatalf("ModelFamilies() = %q, want %q", got, want)
	}
	s := populatedSnapshot()
	s.ModelFamily = "other"
	if err := Validate(s); err != nil {
		t.Fatalf("configured family 'other' rejected: %v", err)
	}
	s.ModelFamily = "gpt-9"
	if err := Validate(s); err == nil || !strings.Contains(err.Error(), "model_family must be one of") {
		t.Fatalf("unconfigured family accepted or wrong error: %v", err)
	}
}

// TestLoadModelFamiliesFailsClosed pins the registry seam's safety property:
// a malformed or foreign-schema registry admits no family at all, so a bad
// config can never widen the allow-list.
func TestLoadModelFamiliesFailsClosed(t *testing.T) {
	for _, raw := range []string{
		`{"schema":"fak-native-backend-model-families/1","allowed":`,
		`{"schema":"some-other/schema","allowed":["Qwen3.8"]}`,
	} {
		if got := loadModelFamilies(raw); len(got) != 0 {
			t.Fatalf("loadModelFamilies(%q) = %v, want empty fail-closed set", raw, got)
		}
	}
}
