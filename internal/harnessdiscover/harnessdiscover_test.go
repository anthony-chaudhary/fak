package harnessdiscover

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestDiscoverAllScopesWithProvenance(t *testing.T) {
	root := t.TempDir()
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	write := func(name, scope, id string) []byte {
		raw := []byte(`{"schema":"fak.harness-selection/v1alpha1","layers":[{"id":"` + id + `","scope":"` + scope + `","capabilities":["` + id + `"]}]}`)
		path := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, raw, 0o600); err != nil {
			t.Fatal(err)
		}
		return raw
	}
	company := write("managed/company.json", "company", "acme-floor")
	team := write("managed/team.json", "team", "litigation-team")
	write("person.json", "person", "alice")
	write("projects/matter.json", "project", "matter-7")
	repoDir := filepath.Join(root, "repos", "legal", "briefs")
	write("repos/legal/.fak/harness.json", "repo", "legal-repo")

	sig := func(raw []byte) string { return base64.StdEncoding.EncodeToString(ed25519.Sign(priv, raw)) }
	reg := Registry{Schema: Schema, TrustedSigners: map[string]string{"acme": base64.StdEncoding.EncodeToString(pub)}, DiscoverRepo: true, Sources: []Source{
		{ID: "acme", Scope: "company", Owner: "acme/security", Root: ".", Path: "managed/company.json", Trust: "managed", Signer: "acme", Signature: sig(company), RefreshPolicy: "immutable"},
		{ID: "litigation", Scope: "team", Owner: "acme/legal", Principals: []string{"alice@acme.test"}, Root: ".", Path: "managed/team.json", Trust: "managed", Signer: "acme", Signature: sig(team), RefreshPolicy: "session"},
		{ID: "alice", Scope: "person", Owner: "alice", Principals: []string{"alice@acme.test"}, Root: ".", Path: "person.json", Trust: "local", RefreshPolicy: "manual"},
		{ID: "matter-7", Scope: "project", Owner: "legal/project-7", Root: ".", Path: "projects/matter.json", Trust: "local", RefreshPolicy: "session"},
	}}
	regPath := filepath.Join(root, "registry.json")
	raw, _ := json.Marshal(reg)
	if err := os.WriteFile(regPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(repoDir, 0o755); err != nil {
		t.Fatal(err)
	}

	got, err := Discover(Options{RegistryPath: regPath, StartPath: repoDir, Principal: "alice@acme.test"})
	if err != nil {
		t.Fatal(err)
	}
	wantScopes := []string{"company", "team", "person", "repo", "project"}
	var scopes []string
	for _, candidate := range got.Candidates {
		scopes = append(scopes, candidate.Scope)
		if !strings.HasPrefix(candidate.Digest, "sha256:") || candidate.Source == "" || candidate.Owner == "" {
			t.Fatalf("incomplete provenance: %#v", candidate)
		}
	}
	if !reflect.DeepEqual(scopes, wantScopes) {
		t.Fatalf("scopes=%v want=%v", scopes, wantScopes)
	}
	if len(got.Manifest.Layers) != 5 {
		t.Fatalf("layers=%d", len(got.Manifest.Layers))
	}
}

func TestDiscoverRejectsUnsignedManagedSource(t *testing.T) {
	root := t.TempDir()
	writeSelection(t, filepath.Join(root, "company.json"), "company", "floor")
	writeRegistry(t, root, Registry{Schema: Schema, Sources: []Source{{ID: "company", Scope: "company", Owner: "security", Root: ".", Path: "company.json", Trust: "managed", RefreshPolicy: "session"}}})
	_, err := Discover(Options{RegistryPath: filepath.Join(root, "registry.json")})
	if err == nil || !strings.Contains(err.Error(), "must declare signer and signature") {
		t.Fatalf("err=%v", err)
	}
}

func TestDiscoverRejectsRevokedDigest(t *testing.T) {
	root := t.TempDir()
	raw := writeSelection(t, filepath.Join(root, "person.json"), "person", "alice")
	sum := sha256.Sum256(raw)
	digest := "sha256:" + hex.EncodeToString(sum[:])
	writeRegistry(t, root, Registry{Schema: Schema, RevokedDigests: []string{digest}, Sources: []Source{{ID: "alice", Scope: "person", Owner: "alice", Principals: []string{"alice@acme.test"}, Root: ".", Path: "person.json", Trust: "local", RefreshPolicy: "manual"}}})
	_, err := Discover(Options{Principal: "alice@acme.test", RegistryPath: filepath.Join(root, "registry.json")})
	if err == nil || !strings.Contains(err.Error(), "is revoked") {
		t.Fatalf("err=%v", err)
	}
}

func TestDiscoverRejectsTraversalAndDuplicateIdentity(t *testing.T) {
	root := t.TempDir()
	outside := filepath.Join(filepath.Dir(root), "outside.json")
	writeSelection(t, outside, "person", "outside")
	writeRegistry(t, root, Registry{Schema: Schema, Sources: []Source{{ID: "alice", Scope: "person", Owner: "alice", Principals: []string{"alice"}, Root: ".", Path: "../outside.json", Trust: "local", RefreshPolicy: "manual"}}})
	if _, err := Discover(Options{Principal: "alice", RegistryPath: filepath.Join(root, "registry.json")}); err == nil || !strings.Contains(err.Error(), "confined relative path") {
		t.Fatalf("traversal err=%v", err)
	}

	writeSelection(t, filepath.Join(root, "person.json"), "person", "alice")
	writeRegistry(t, root, Registry{Schema: Schema, Sources: []Source{
		{ID: "alice", Scope: "person", Owner: "alice", Principals: []string{"alice"}, Root: ".", Path: "person.json", Trust: "local", RefreshPolicy: "manual"},
		{ID: "alice", Scope: "person", Owner: "alice-2", Principals: []string{"alice"}, Root: ".", Path: "person.json", Trust: "local", RefreshPolicy: "manual"},
	}})
	if _, err := Discover(Options{Principal: "alice", RegistryPath: filepath.Join(root, "registry.json")}); err == nil || !strings.Contains(err.Error(), "duplicate source identity") {
		t.Fatalf("duplicate err=%v", err)
	}
}

func TestDiscoverRequiresAdmittedPrincipal(t *testing.T) {
	root := t.TempDir()
	writeSelection(t, filepath.Join(root, "person.json"), "person", "alice")
	writeRegistry(t, root, Registry{Schema: Schema, Sources: []Source{{ID: "alice", Scope: "person", Owner: "alice", Principals: []string{"alice"}, Root: ".", Path: "person.json", Trust: "local", RefreshPolicy: "manual"}}})
	_, err := Discover(Options{RegistryPath: filepath.Join(root, "registry.json")})
	if err == nil || !strings.Contains(err.Error(), "requires an authenticated principal") {
		t.Fatalf("missing principal err=%v", err)
	}
	_, err = Discover(Options{RegistryPath: filepath.Join(root, "registry.json"), Principal: "mallory"})
	if err == nil || !strings.Contains(err.Error(), "is not admitted") {
		t.Fatalf("wrong principal err=%v", err)
	}
}

func writeSelection(t *testing.T, path, scope, id string) []byte {
	t.Helper()
	raw := []byte(`{"schema":"fak.harness-selection/v1alpha1","layers":[{"id":"` + id + `","scope":"` + scope + `"}]}`)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return raw
}

func writeRegistry(t *testing.T, root string, reg Registry) {
	t.Helper()
	raw, _ := json.Marshal(reg)
	if err := os.WriteFile(filepath.Join(root, "registry.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
}
