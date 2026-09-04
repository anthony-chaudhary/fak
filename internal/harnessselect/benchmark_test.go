package harnessselect

import (
	"fmt"
	"testing"
)

var (
	benchSinkResult   Result
	benchSinkManifest Manifest
)

const benchEnterpriseManifestJSON = `{
	"schema": "fak.harness-selection/v1alpha1",
	"layers": [
		{
			"id": "enterprise-baseline",
			"scope": "company",
			"priority": 1,
			"capabilities": ["audit-logging", "telemetry", "base-governance"],
			"lock": ["audit-logging"]
		},
		{
			"id": "team-platform",
			"scope": "team",
			"priority": 10,
			"when": {"tags": ["infra", "backend"]},
			"capabilities": ["k8s-client", "grpc-tools", "cloud-secrets"]
		},
		{
			"id": "repo-fak",
			"scope": "repo",
			"priority": 20,
			"when": {"path_prefixes": ["src/fak", "services/fak"]},
			"capabilities": ["go-toolchain", "simd-accel", "metal-bridge"]
		},
		{
			"id": "domain-security",
			"scope": "domain",
			"priority": 30,
			"when": {"tags": ["security-review"]},
			"capabilities": ["policy-prover", "kernel-tracer"],
			"remove": ["base-governance"]
		},
		{
			"id": "task-override",
			"scope": "task",
			"priority": 50,
			"capabilities": ["ephemeral-sandbox"]
		}
	]
}`

func BenchmarkHarnessSelect(b *testing.B) {
	manifest, err := Parse([]byte(benchEnterpriseManifestJSON))
	if err != nil {
		b.Fatalf("failed to parse manifest: %v", err)
	}
	ctx := Context{
		Path: "services/fak/internal/engine",
		Tags: []string{"backend", "security-review"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Resolve(manifest, ctx)
		if err != nil {
			b.Fatalf("resolve failed: %v", err)
		}
		benchSinkResult = res
	}
}

func BenchmarkParse(b *testing.B) {
	raw := []byte(benchEnterpriseManifestJSON)

	b.ReportAllocs()
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m, err := Parse(raw)
		if err != nil {
			b.Fatalf("parse failed: %v", err)
		}
		benchSinkManifest = m
	}
}

func BenchmarkResolve(b *testing.B) {
	scales := []int{5, 25, 100}
	scopes := []string{"company", "team", "person", "repo", "project", "domain", "task"}

	for _, count := range scales {
		b.Run(fmt.Sprintf("%d_layers", count), func(b *testing.B) {
			layers := make([]Layer, count)
			for i := 0; i < count; i++ {
				scope := scopes[i%len(scopes)]
				layerID := fmt.Sprintf("layer-%03d", i)
				layers[i] = Layer{
					ID:       layerID,
					Scope:    scope,
					Priority: i,
					When: Match{
						PathPrefixes: []string{fmt.Sprintf("repo/sub-%d", i%5)},
						Tags:         []string{fmt.Sprintf("tag-%d", i%10)},
					},
					Capabilities: []string{fmt.Sprintf("cap-%d", i), fmt.Sprintf("shared-%d", i%4)},
				}
				if i == 0 {
					layers[i].Lock = []string{"cap-0"}
				}
			}
			m := Manifest{
				Schema: Schema,
				Layers: layers,
			}
			ctx := Context{
				Path: "repo/sub-2/pkg/core",
				Tags: []string{"tag-2", "tag-7"},
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				res, err := Resolve(m, ctx)
				if err != nil {
					b.Fatalf("resolve failed: %v", err)
				}
				benchSinkResult = res
			}
		})
	}
}

func BenchmarkPathNormalization(b *testing.B) {
	paths := []string{
		`C:\work\fak\internal\harnessselect\..\..\cmd\fak\`,
		`/var/data/workspace/./subpath/../pkg//module/`,
		`relative\path\to\service/subservice/`,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, p := range paths {
			_ = cleanPath(p)
		}
	}
}
