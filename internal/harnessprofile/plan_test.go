package harnessprofile

import (
	"strings"
	"testing"
)

func TestResolvedBindingUsesConfigDescriptorAndRefusesStaleSnapshot(t *testing.T) {
	profiles, err := Resolve([]byte(`{"harnesses":[{"name":"acme","adapter_version":"2.1.0","names":["acme-cli"],"wire":"openai","repoint":["env"],"credential":{"kind":"env-key","env_key":"ACME_API_KEY"},"identity":"env-key"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	binding, ok, err := ResolveBinding(profiles, `C:\bin\acme-cli.exe`)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("config-declared harness did not resolve")
	}
	if binding.Host != "acme" || binding.AdapterVersion != "2.1.0" || binding.Wire != WireOpenAI {
		t.Fatalf("binding=%+v", binding)
	}
	if len(binding.Repoint) != 1 || binding.Repoint[0] != RepointEnv {
		t.Fatalf("repoint=%v", binding.Repoint)
	}

	mutated := profiles[len(profiles)-1]
	mutated.Repoint = append(mutated.Repoint, RepointCLIConfig)
	if err := VerifyFresh(mutated, binding); err == nil || !strings.Contains(err.Error(), "stale") {
		t.Fatalf("mutated descriptor was not refused as stale: %v", err)
	}
}
