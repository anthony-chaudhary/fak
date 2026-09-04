package harnessdiscover

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

var benchResultSink Result

func setupBenchmarkEnv(tb testing.TB) Options {
	tb.Helper()
	root := tb.TempDir()

	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		tb.Fatalf("generate ed25519 key: %v", err)
	}

	writeManifest := func(relPath, scope, id string) []byte {
		raw := []byte(`{"schema":"fak.harness-selection/v1alpha1","layers":[{"id":"` + id + `","scope":"` + scope + `","capabilities":["` + id + `"]}]}`)
		fullPath := filepath.Join(root, relPath)
		if err := os.MkdirAll(filepath.Dir(fullPath), 0o755); err != nil {
			tb.Fatalf("mkdir %s: %v", filepath.Dir(fullPath), err)
		}
		if err := os.WriteFile(fullPath, raw, 0o600); err != nil {
			tb.Fatalf("write %s: %v", fullPath, err)
		}
		return raw
	}

	companyRaw := writeManifest("managed/company.json", "company", "acme-corp")
	teamRaw := writeManifest("managed/team.json", "team", "platform-infra")
	writeManifest("user/developer.json", "person", "engineer")
	writeManifest("projects/migration.json", "project", "v2-migration")
	writeManifest("repo/.fak/harness.json", "repo", "main-repo")

	sig := func(data []byte) string {
		return base64.StdEncoding.EncodeToString(ed25519.Sign(priv, data))
	}

	reg := Registry{
		Schema: Schema,
		TrustedSigners: map[string]string{
			"acme-security": base64.StdEncoding.EncodeToString(pub),
		},
		DiscoverRepo: true,
		Sources: []Source{
			{
				ID:            "company-policy",
				Scope:         "company",
				Owner:         "security-team",
				Root:          ".",
				Path:          "managed/company.json",
				Trust:         "managed",
				Signer:        "acme-security",
				Signature:     sig(companyRaw),
				RefreshPolicy: "immutable",
			},
			{
				ID:            "platform-team",
				Scope:         "team",
				Owner:         "platform-lead",
				Principals:    []string{"dev@company.test"},
				Root:          ".",
				Path:          "managed/team.json",
				Trust:         "managed",
				Signer:        "acme-security",
				Signature:     sig(teamRaw),
				RefreshPolicy: "session",
			},
			{
				ID:            "dev-override",
				Scope:         "person",
				Owner:         "engineer",
				Principals:    []string{"dev@company.test"},
				Root:          ".",
				Path:          "user/developer.json",
				Trust:         "local",
				RefreshPolicy: "manual",
			},
			{
				ID:            "migration-proj",
				Scope:         "project",
				Owner:         "migration-guild",
				Root:          ".",
				Path:          "projects/migration.json",
				Trust:         "local",
				RefreshPolicy: "session",
			},
		},
	}

	regPath := filepath.Join(root, "registry.json")
	regData, err := json.Marshal(reg)
	if err != nil {
		tb.Fatalf("marshal registry: %v", err)
	}
	if err := os.WriteFile(regPath, regData, 0o600); err != nil {
		tb.Fatalf("write registry: %v", err)
	}

	startDir := filepath.Join(root, "repo", "src", "services", "api", "v1")
	if err := os.MkdirAll(startDir, 0o755); err != nil {
		tb.Fatalf("mkdir startDir: %v", err)
	}

	return Options{
		RegistryPath: regPath,
		StartPath:    startDir,
		Principal:    "dev@company.test",
	}
}

// BenchmarkHarnessDiscover measures end-to-end harness discovery across all scopes
// (company, team, person, repo, project) including cryptographic signature verification,
// SHA-256 digest computation, directory climbing for repo manifests, and deterministic layer sorting.
func BenchmarkHarnessDiscover(b *testing.B) {
	opts := setupBenchmarkEnv(b)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Discover(opts)
		if err != nil {
			b.Fatalf("discover failed: %v", err)
		}
		if len(res.Candidates) != 5 {
			b.Fatalf("expected 5 candidates, got %d", len(res.Candidates))
		}
		benchResultSink = res
	}
}

// BenchmarkHarnessDiscoverLocal measures discovery throughput without cryptographic
// signature verification on purely local sources.
func BenchmarkHarnessDiscoverLocal(b *testing.B) {
	root := b.TempDir()
	writeManifest := func(relPath, scope, id string) {
		raw := []byte(`{"schema":"fak.harness-selection/v1alpha1","layers":[{"id":"` + id + `","scope":"` + scope + `"}]}`)
		fullPath := filepath.Join(root, relPath)
		if err := os.MkdirAll(filepath.Dir(fullPath), 0o755); err != nil {
			b.Fatal(err)
		}
		if err := os.WriteFile(fullPath, raw, 0o600); err != nil {
			b.Fatal(err)
		}
	}

	writeManifest("user.json", "person", "user-1")
	writeManifest("proj.json", "project", "proj-1")

	reg := Registry{
		Schema: Schema,
		Sources: []Source{
			{ID: "user-1", Scope: "person", Owner: "dev", Principals: []string{"alice"}, Root: ".", Path: "user.json", Trust: "local", RefreshPolicy: "manual"},
			{ID: "proj-1", Scope: "project", Owner: "lead", Root: ".", Path: "proj.json", Trust: "local", RefreshPolicy: "session"},
		},
	}
	regPath := filepath.Join(root, "registry.json")
	data, err := json.Marshal(reg)
	if err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(regPath, data, 0o600); err != nil {
		b.Fatal(err)
	}

	opts := Options{RegistryPath: regPath, Principal: "alice"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Discover(opts)
		if err != nil {
			b.Fatalf("discover local failed: %v", err)
		}
		benchResultSink = res
	}
}

// TestBenchmarkHarnessDiscoverSanity verifies that BenchmarkHarnessDiscover executes without error.
func TestBenchmarkHarnessDiscoverSanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkHarnessDiscover)
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}
