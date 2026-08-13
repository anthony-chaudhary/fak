package portability

import (
	"crypto/ed25519"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func registryFixture(t *testing.T, root, name, version string, seq uint64, deps []Dependency, permissions []string) (LocalRegistry, PublishRequest, RegistryTrust) {
	t.Helper()
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(2000000000, 0)
	pkg := Package{Schema: Schema, Objects: []Object{{ID: "skill:review", Kind: "skill", Name: "review", Payload: json.RawMessage(`{"title":{"sensitivity":"public","value":"review safely"}}`), Digest: "object-digest", Active: false}}}
	manifest := RegistryManifest{Schema: "fak.registry/v1", Namespace: "acme", Name: name, Version: version, Sequence: seq, IssuedAt: now.Add(-time.Minute).Unix(), ExpiresAt: now.Add(time.Hour).Unix(), Provenance: Provenance{"https://example.invalid/acme/review", "abc123", "fak-ci", "slsa-local"}, License: "Apache-2.0", Compatibility: []string{"fak>=1"}, Sensitivity: "public", Dependencies: deps, Permissions: permissions, Rollback: "restore previous lockfile", Signer: "acme-release"}
	reg := LocalRegistry{filepath.Join(root, "registry")}
	policy := RegistryTrust{Keys: map[string]ed25519.PublicKey{"acme-release": pub}, Retired: map[string]bool{}, Now: now, MaxMetadataAge: 24 * time.Hour, MinSequence: map[string]uint64{}, AllowedCompatibility: map[string]bool{"fak>=1": true}}
	return reg, PublishRequest{manifest, pkg, priv, false}, policy
}

func TestRegistryLifecycleJourney(t *testing.T) {
	root := t.TempDir()
	depReg, depReq, policy := registryFixture(t, root, "base", "1.0.0", 1, nil, []string{"read:workspace"})
	depReq.Commit = true
	if _, err := Publish(depReg, depReq); err != nil {
		t.Fatal(err)
	}
	depSP, depIn, err := Inspect(depReg, PackageRef{"acme", "base", "1.0.0"}, policy)
	if err != nil {
		t.Fatal(err)
	}
	_ = depSP
	reg, req, _ := registryFixture(t, root, "review", "1.0.0", 1, []Dependency{{"acme", "base", "1.0.0", depIn.Digest}}, []string{"read:workspace"})
	req.PrivateKey = depReq.PrivateKey
	req.Manifest.Signer = "acme-release"
	preview, err := Publish(reg, req)
	if err != nil {
		t.Fatal(err)
	}
	if preview.Action != "dry-run" {
		t.Fatalf("publish must dry-run: %#v", preview)
	}
	if _, err := os.Stat(filepath.Join(reg.Root, "acme", "review", "1.0.0.fakpkg.json")); !os.IsNotExist(err) {
		t.Fatal("dry-run wrote registry")
	}
	req.Commit = true
	if _, err = Publish(reg, req); err != nil {
		t.Fatal(err)
	}
	old, in, err := Inspect(reg, PackageRef{"acme", "review", "1.0.0"}, policy)
	if err != nil {
		t.Fatal(err)
	}
	if !in.Installable || len(in.Dependencies) != 1 || len(in.Permissions) != 1 {
		t.Fatalf("incomplete inspection: %#v", in)
	}
	lock, err := Resolve(reg, PackageRef{"acme", "review", "1.0.0"}, policy)
	if err != nil {
		t.Fatal(err)
	}
	if len(lock.Entries) != 2 || lock.Entries[0].Reference > lock.Entries[1].Reference {
		t.Fatalf("bad lock: %#v", lock)
	}
	installed := InstalledState{in.Reference, in.Digest, false, false, true}
	if installed.Active {
		t.Fatal("install must remain inactive")
	}
	installed.Active = true
	newReq := req
	newReq.Manifest.Version = "2.0.0"
	newReq.Manifest.Sequence = 2
	newReq.Manifest.BreakingChanges = []string{"renamed review mode"}
	newReq.Manifest.Migration = "map mode=brief to mode=concise"
	newReq.Manifest.Permissions = []string{"read:workspace", "write:report"}
	newReq.Package.Objects = append(newReq.Package.Objects, Object{ID: "workflow:triage", Kind: "workflow", Name: "triage", Payload: json.RawMessage(`{"title":{"sensitivity":"public","value":"triage"}}`)})
	if _, err = Publish(reg, newReq); err != nil {
		t.Fatal(err)
	}
	newSP, _, err := Inspect(reg, PackageRef{"acme", "review", "2.0.0"}, policy)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BuildUpdateDiff(old, newSP)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.RequiresApproval || len(plan.Breaking) == 0 || plan.Migration == "" || plan.Rollback == "" || len(plan.ObjectDiff) == 0 {
		t.Fatalf("unsafe update plan %#v", plan)
	}
	revoked := newSP.Manifest
	revoked.Revoked = "signing key compromise"
	installed.Reference = plan.To
	installed = EnforceSyncedStatus(installed, revoked, true)
	if installed.Active || !installed.Quarantined || !installed.EvidenceRetained {
		t.Fatalf("revocation not enforced %#v", installed)
	}
}

func TestRegistryAdversarialFailuresAndNoSecretLeak(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*SignedPackage, *RegistryTrust)
		want   string
	}{
		{"tamper", func(s *SignedPackage, p *RegistryTrust) { s.Package.Objects[0].Name = "evil" }, "TAMPERED"},
		{"stale", func(s *SignedPackage, p *RegistryTrust) { p.Now = time.Unix(s.Manifest.ExpiresAt+1, 0) }, "STALE_METADATA"},
		{"rotation", func(s *SignedPackage, p *RegistryTrust) { p.Retired[s.Manifest.Signer] = true }, "UNTRUSTED_SIGNER"},
		{"replay", func(s *SignedPackage, p *RegistryTrust) { p.MinSequence["acme/review"] = 2 }, "REPLAY_OR_DOWNGRADE"},
		{"namespace", func(s *SignedPackage, p *RegistryTrust) { s.Manifest.Namespace = "attacker" }, "NAMESPACE_TAKEOVER"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			reg, req, policy := registryFixture(t, root, "review", "1.0.0", 1, nil, nil)
			req.Commit = true
			if _, err := Publish(reg, req); err != nil {
				t.Fatal(err)
			}
			sp, _ := reg.Get(PackageRef{"acme", "review", "1.0.0"})
			tt.mutate(&sp, &policy)
			mem := memoryRegistry{sp}
			_, _, err := Inspect(mem, PackageRef{"acme", "review", "1.0.0"}, policy)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err=%v want %s", err, tt.want)
			}
		})
	}
	root := t.TempDir()
	reg, req, policy := registryFixture(t, root, "review", "1.0.0", 1, nil, nil)
	req.Package.Objects[0].Payload = json.RawMessage(`{"title":{"sensitivity":"public","value":"ok"},"token":"ghp_supersecret123456"}`)
	_, err := Publish(reg, req)
	if err == nil || !strings.Contains(err.Error(), "EGRESS_DENIED") {
		t.Fatalf("err=%v", err)
	}
	b, _ := json.Marshal(err.Error())
	if strings.Contains(string(b), "supersecret") {
		t.Fatal("JSON error leaked excluded content")
	}
	_ = policy
}

type memoryRegistry struct{ p SignedPackage }

func (m memoryRegistry) Put(SignedPackage) error               { return nil }
func (m memoryRegistry) Get(PackageRef) (SignedPackage, error) { return m.p, nil }

func TestLocalRegistryRejectsMaliciousMetadataAndImmutableOverwrite(t *testing.T) {
	root := t.TempDir()
	reg, req, _ := registryFixture(t, root, "review", "1.0.0", 1, nil, nil)
	req.Commit = true
	if _, err := Publish(reg, req); err != nil {
		t.Fatal(err)
	}
	req.Package.Objects[0].Name = "overwrite"
	if _, err := Publish(reg, req); err == nil || !strings.Contains(err.Error(), "IMMUTABLE_VERSION") {
		t.Fatalf("overwrite err=%v", err)
	}
	path := filepath.Join(reg.Root, "acme", "review", "9.0.0.fakpkg.json")
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`{"manifest":{},"package":{},"signature":"","surprise":"payload"}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := reg.Get(PackageRef{"acme", "review", "9.0.0"}); err == nil || !strings.Contains(err.Error(), "MALICIOUS_METADATA") {
		t.Fatalf("metadata err=%v", err)
	}
}
