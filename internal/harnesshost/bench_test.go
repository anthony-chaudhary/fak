package harnesshost

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

func BenchmarkBuild(b *testing.B) {
	hosts := []string{"codex", "claude"}
	for _, host := range hosts {
		b.Run(host, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				artifacts, err := Build(host, "v1alpha1")
				if err != nil {
					b.Fatalf("Build(%q): %v", host, err)
				}
				if artifacts.Host == "" {
					b.Fatalf("empty host artifact")
				}
			}
		})
	}
}

func BenchmarkBuildResolved(b *testing.B) {
	profiles, err := harnessprofile.Resolve([]byte(`{"harnesses":[{"name":"acme","adapter_version":"2.1.0","names":["acme-cli"],"wire":"openai","repoint":["env"],"credential":{"kind":"env-key","env_key":"ACME_API_KEY"},"identity":"env-key"}]}`))
	if err != nil {
		b.Fatal(err)
	}
	binding, ok, err := harnessprofile.ResolveBinding(profiles, "acme-cli")
	if err != nil || !ok {
		b.Fatalf("resolve binding: ok=%v err=%v", ok, err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		artifacts, err := BuildResolved(binding, "fixture:harnessprofile", "fixture:harnessprofile/acme@2.1.0", "v1alpha1")
		if err != nil {
			b.Fatalf("BuildResolved: %v", err)
		}
		if artifacts.Host == "" {
			b.Fatalf("empty host artifact")
		}
	}
}
