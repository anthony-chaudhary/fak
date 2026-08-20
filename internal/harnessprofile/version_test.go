package harnessprofile

import "testing"

func TestBuiltinAdapterVersionPinsSemanticDigest(t *testing.T) {
	want := map[string]struct {
		version string
		digest  string
	}{
		"claude": {version: "1.0.0", digest: "sha256:39d19a2d76f30e7ce7d2b4f2953d90018ec40d6239a9d6715f9d5640b07471fa"},
		"codex":  {version: "1.0.0", digest: "sha256:3d51ee171eb08ce467ca9cff9e97084e39c8103d9b0a889fb245e06a23bf2e51"},
	}
	for _, profile := range Builtins() {
		expected, ok := want[profile.Name]
		if !ok {
			continue
		}
		if profile.AdapterVersion != expected.version {
			t.Fatalf("%s adapter version=%q want=%q", profile.Name, profile.AdapterVersion, expected.version)
		}
		digest, err := SemanticDigest(profile)
		if err != nil {
			t.Fatal(err)
		}
		if digest != expected.digest {
			t.Fatalf("%s@%s semantic digest=%q want=%q; bump adapter_version and add its new immutable snapshot", profile.Name, profile.AdapterVersion, digest, expected.digest)
		}
	}
}

func TestConfigRejectsMalformedAdapterVersion(t *testing.T) {
	_, err := Resolve([]byte(`{"harnesses":[{"names":["codex"],"adapter_version":"latest"}]}`))
	if err == nil {
		t.Fatal("malformed adapter version was accepted")
	}
}
