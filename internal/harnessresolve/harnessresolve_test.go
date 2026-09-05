package harnessresolve

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/stackresolve"
)

func TestResolveMixedProductAndImmutableLock(t *testing.T) {
	m := validManifest()
	got, err := Resolve(context.Background(), m, []string{"company", "legal"}, Environment{OS: "linux", Arch: "amd64", Contract: "v1"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(got.Lock.ID, "sha256:") {
		t.Fatalf("id=%s", got.Lock.ID)
	}
	if got.Lock.Budget != (Budget{ContextTokens: 1500, MemoryMiB: 320, Workers: 1}) {
		t.Fatalf("budget=%#v", got.Lock.Budget)
	}
	want := []string{"kernel", "legal-pack", "search-provider"}
	var ids []string
	for _, c := range got.Lock.Components {
		ids = append(ids, c.ID)
		if c.Digest == "" || c.Source == "" || c.Reason == "" {
			t.Fatalf("incomplete lock component %#v", c)
		}
	}
	if !reflect.DeepEqual(ids, want) {
		t.Fatalf("ids=%v want=%v", ids, want)
	}
	if len(got.Lock.Assets) != 3 || len(got.Explain) < 3 {
		t.Fatalf("lock=%#v", got.Lock)
	}
}

func TestResolvePermutationInvariant(t *testing.T) {
	m := validManifest()
	a, err := Resolve(context.Background(), m, []string{"company", "legal"}, Environment{OS: "linux", Arch: "amd64", Contract: "v1"})
	if err != nil {
		t.Fatal(err)
	}
	for i, j := 0, len(m.Components)-1; i < j; i, j = i+1, j-1 {
		m.Components[i], m.Components[j] = m.Components[j], m.Components[i]
	}
	for i, j := 0, len(m.Assets.Layers)-1; i < j; i, j = i+1, j-1 {
		m.Assets.Layers[i], m.Assets.Layers[j] = m.Assets.Layers[j], m.Assets.Layers[i]
	}
	b, err := Resolve(context.Background(), m, []string{"company", "legal"}, Environment{OS: "linux", Arch: "amd64", Contract: "v1"})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("permutation changed result:\n%#v\n%#v", a, b)
	}
}

func TestResolveRejectsAmbiguousProvider(t *testing.T) {
	m := validManifest()
	m.Components = append(m.Components, component("search-alt", "1.1.0", []string{"search"}, nil))
	_, err := Resolve(context.Background(), m, []string{"company", "legal"}, Environment{OS: "linux", Arch: "amd64", Contract: "v1"})
	if err == nil || !strings.Contains(err.Error(), "ambiguous providers") {
		t.Fatalf("err=%v", err)
	}
}

func TestResolveRejectsMissingIncompatibleCycleAndBudget(t *testing.T) {
	tests := map[string]struct {
		mutate func(*Manifest)
		want   string
	}{
		"missing": {func(m *Manifest) { m.Components[1].Requires[0].Capability = "absent" }, "missing dependency"},
		"range":   {func(m *Manifest) { m.Components[1].Requires[0].Range = ">=2.0.0" }, "missing dependency"},
		"cycle": {func(m *Manifest) {
			m.Components[0].Requires = []Requirement{{Capability: "legal-pack", Range: ">=1.0.0"}}
		}, "dependency cycle"},
		"compatibility": {func(m *Manifest) { m.Components[1].Compatibility.Arch = []string{"arm64"} }, "incompatible arch"},
		"budget":        {func(m *Manifest) { m.Budget.ContextTokens = 100 }, "context budget exceeded"},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			m := validManifest()
			tc.mutate(&m)
			_, err := Resolve(context.Background(), m, []string{"company", "legal"}, Environment{OS: "linux", Arch: "amd64", Contract: "v1"})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v want=%s", err, tc.want)
			}
		})
	}
}

func TestResolveRejectsConflictAndPrivilegeWidening(t *testing.T) {
	t.Run("component conflict", func(t *testing.T) {
		m := validManifest()
		m.Components[1].Conflicts = []string{"kernel"}
		_, err := Resolve(context.Background(), m, []string{"company", "legal"}, Environment{OS: "linux", Arch: "amd64", Contract: "v1"})
		if err == nil || !strings.Contains(err.Error(), "dependency resolution refused") {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("asset widening", func(t *testing.T) {
		m := validManifest()
		m.Assets.Layers[1].Assets = append(m.Assets.Layers[1].Assets, harnesscompose.Asset{Kind: "policy", ID: "tools", Grants: []string{"shell"}})
		_, err := Resolve(context.Background(), m, []string{"company", "legal"}, Environment{OS: "linux", Arch: "amd64", Contract: "v1"})
		if err == nil || !strings.Contains(err.Error(), "cannot change locked asset") {
			t.Fatalf("err=%v", err)
		}
	})
}

func validManifest() Manifest {
	e := stackresolve.Evidence{Authority: "test", Source: "fixture", Freshness: "2026-08-15"}
	return Manifest{Schema: Schema, Roots: []string{"legal-pack"}, Compatibility: Compatibility{OS: []string{"linux"}, Arch: []string{"amd64"}, Contract: "v1"}, Budget: Budget{ContextTokens: 2000, MemoryMiB: 512, Workers: 2}, Components: []Component{
		{ID: "kernel", Version: "1.0.0", Digest: "sha256:kernel", Source: "registry/kernel", Provides: []string{"runtime"}, Cost: Budget{ContextTokens: 500, MemoryMiB: 256}, Evidence: e},
		{ID: "legal-pack", Version: "1.2.0", Digest: "sha256:legal", Source: "registry/legal", Adapters: []string{"instruction", "policy", "workflow"}, Requires: []Requirement{{Capability: "runtime", Range: ">=1.0.0"}, {Capability: "search", Range: ">=1.0.0"}}, Compatibility: Compatibility{Contract: "v1"}, Cost: Budget{ContextTokens: 1000, MemoryMiB: 64, Workers: 1}, Evidence: e},
		{ID: "search-provider", Version: "1.1.0", Digest: "sha256:search", Source: "registry/search", Adapters: []string{"tool"}, Provides: []string{"search"}, Cost: Budget{MemoryMiB: 0}, Evidence: e},
	}, Assets: harnesscompose.Manifest{Schema: harnesscompose.Schema, Layers: []harnesscompose.Layer{
		{ID: "company", Scope: "company", Assets: []harnesscompose.Asset{{Kind: "policy", ID: "tools", Grants: []string{"search"}, Denies: []string{"shell"}, Lock: true}, {Kind: "workflow", ID: "audit", Value: "record", Mandatory: true}}},
		{ID: "legal", Scope: "domain", Assets: []harnesscompose.Asset{{Kind: "instruction", ID: "citations", Value: "primary-only"}}},
	}}}
}
func component(id, version string, provides []string, requires []Requirement) Component {
	return Component{ID: id, Version: version, Digest: "sha256:" + id, Source: "registry/" + id, Provides: provides, Requires: requires, Evidence: stackresolve.Evidence{Authority: "test", Source: id, Freshness: "2026-08-15"}}
}

func TestVerifyLockRejectsForgedIdentity(t *testing.T) {
	lock := Lock{Schema: LockSchema, ID: "sha256:forged"}
	if err := VerifyLock(lock); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("VerifyLock error = %v", err)
	}
	lock.ID = ""
	id, err := lockID(lock)
	if err != nil {
		t.Fatal(err)
	}
	lock.ID = id
	if err := VerifyLock(lock); err != nil {
		t.Fatalf("valid lock refused: %v", err)
	}
}

func TestResolveRetainsMixSafetyEvidence(t *testing.T) {
	manifest := validManifest()
	manifest.Components[1].Conflicts = []string{"unsafe-shell"}
	manifest.Components[1].Compatibility = Compatibility{OS: []string{"linux"}, Arch: []string{"amd64"}, Contract: "v1"}
	result, err := Resolve(context.Background(), manifest, []string{"company"}, Environment{OS: "linux", Arch: "amd64", Contract: "v1"})
	if err != nil {
		t.Fatal(err)
	}
	var found *LockedComponent
	for i := range result.Lock.Components {
		if result.Lock.Components[i].ID == "legal-pack" {
			found = &result.Lock.Components[i]
		}
	}
	if found == nil {
		t.Fatal("legal-pack not selected")
	}
	if len(found.Requires) != 2 || len(found.Conflicts) != 1 || found.Compatibility.Contract != "v1" || found.Cost.ContextTokens != 1000 || len(found.Adapters) != 3 {
		t.Fatalf("evidence not retained: %+v", *found)
	}
	if err := VerifyLock(result.Lock); err != nil {
		t.Fatal(err)
	}
}

func TestMixableRefusesLegacyAndMissingEvidence(t *testing.T) {
	lock := Lock{Schema: LegacyLockSchema, Components: []LockedComponent{{ID: "x", Version: "1.0.0", Digest: "sha256:x", Source: "r", Reason: "selected", Provider: "x"}}}
	if err := ReidentifyLock(&lock); err != nil {
		t.Fatal(err)
	}
	if err := Mixable(lock); err == nil || !strings.Contains(err.Error(), "launchable but not mixable") {
		t.Fatalf("legacy err=%v", err)
	}
	lock.Schema = LockSchema
	if err := ReidentifyLock(&lock); err != nil {
		t.Fatal(err)
	}
	if err := Mixable(lock); err == nil || !strings.Contains(err.Error(), "compatibility contract") {
		t.Fatalf("compat err=%v", err)
	}
	lock.Components[0].Compatibility.Contract = "v1"
	if err := ReidentifyLock(&lock); err != nil {
		t.Fatal(err)
	}
	if err := Mixable(lock); err == nil || !strings.Contains(err.Error(), "adapter conformance") {
		t.Fatalf("adapter err=%v", err)
	}
}

func TestProductLockV2(t *testing.T) {
	t.Run("deterministic CRLF and LF canonical ID", TestProductLockV2DeterministicCRLFAndLFCanonicalID)
	t.Run("secret contracts", TestProductLockV2SecretContracts)
	t.Run("multi-platform compatibility", TestProductLockV2MultiPlatformCompatibility)
	t.Run("backward compatibility", TestProductLockV2BackwardCompatibility)
}

func TestProductLockV2DeterministicCRLFAndLFCanonicalID(t *testing.T) {
	lockCRLF := Lock{
		Schema: LockSchemaV2,
		Platforms: []PlatformRequirement{
			{OS: "linux", Arch: "amd64", Contract: "v1"},
		},
		Components: []LockedComponent{
			{
				ID:       "c",
				Version:  "1.0.0",
				Digest:   "sha256:c",
				Source:   "s",
				Reason:   "selected\r\nline2",
				Provider: "p",
			},
		},
		Assets: []harnesscompose.EffectiveAsset{
			{Kind: "instruction", ID: "inst", Source: "s", Value: "val1\r\nval2"},
		},
	}

	lockLF := Lock{
		Schema: LockSchemaV2,
		Platforms: []PlatformRequirement{
			{OS: "linux", Arch: "amd64", Contract: "v1"},
		},
		Components: []LockedComponent{
			{
				ID:       "c",
				Version:  "1.0.0",
				Digest:   "sha256:c",
				Source:   "s",
				Reason:   "selected\nline2",
				Provider: "p",
			},
		},
		Assets: []harnesscompose.EffectiveAsset{
			{Kind: "instruction", ID: "inst", Source: "s", Value: "val1\nval2"},
		},
	}

	idCRLF, err := lockID(lockCRLF)
	if err != nil {
		t.Fatal(err)
	}
	idLF, err := lockID(lockLF)
	if err != nil {
		t.Fatal(err)
	}
	if idCRLF != idLF {
		t.Fatalf("CRLF and LF produced different IDs: crlf=%s lf=%s", idCRLF, idLF)
	}

	lockCRLF.ID = idCRLF
	if err := VerifyLock(lockCRLF); err != nil {
		t.Fatalf("VerifyLock failed for CRLF lock: %v", err)
	}
	lockLF.ID = idLF
	if err := VerifyLock(lockLF); err != nil {
		t.Fatalf("VerifyLock failed for LF lock: %v", err)
	}
}

func TestProductLockV2SecretContracts(t *testing.T) {
	t.Run("secret plaintext leak in VerifyLock rejected", func(t *testing.T) {
		lock := Lock{
			Schema: LockSchemaV2,
			Assets: []harnesscompose.EffectiveAsset{
				{Kind: "secret", ID: "db-pass", Value: "plaintext123", Ref: "env:DB_PASS", Source: "s"},
			},
		}
		if err := ReidentifyLock(&lock); err != nil {
			t.Fatal(err)
		}
		err := VerifyLock(lock)
		if err == nil || !strings.Contains(err.Error(), ErrSecretPlaintextLeak) {
			t.Fatalf("expected error containing %q, got %v", ErrSecretPlaintextLeak, err)
		}
	})

	t.Run("secret plaintext leak in Resolve rejected", func(t *testing.T) {
		m := validManifest()
		m.Assets.Layers[1].Assets = append(m.Assets.Layers[1].Assets, harnesscompose.Asset{
			Kind:  "secret",
			ID:    "api-token",
			Value: "leaked-plaintext",
			Ref:   "vault:secret/token",
		})
		_, err := Resolve(context.Background(), m, []string{"company", "legal"}, Environment{OS: "linux", Arch: "amd64", Contract: "v1"})
		if err == nil || !strings.Contains(err.Error(), ErrSecretPlaintextLeak) {
			t.Fatalf("expected error containing %q, got %v", ErrSecretPlaintextLeak, err)
		}
	})

	validRefs := []string{
		"env:API_KEY",
		"file:/run/secrets/key",
		"vault:secret/data#token",
		"keyring:fak-agent",
	}
	for _, ref := range validRefs {
		t.Run("valid ref "+ref, func(t *testing.T) {
			lock := Lock{
				Schema: LockSchemaV2,
				Assets: []harnesscompose.EffectiveAsset{
					{Kind: "secret", ID: "secret-asset", Ref: ref, Source: "s"},
				},
			}
			if err := ReidentifyLock(&lock); err != nil {
				t.Fatal(err)
			}
			if err := VerifyLock(lock); err != nil {
				t.Fatalf("valid ref %q rejected: %v", ref, err)
			}
		})
	}

	invalidRefs := []string{
		"bare-string",
		":missing-scheme",
		"http://remote/secret",
		"env:",
		"",
	}
	for _, ref := range invalidRefs {
		t.Run("invalid ref "+ref, func(t *testing.T) {
			lock := Lock{
				Schema: LockSchemaV2,
				Assets: []harnesscompose.EffectiveAsset{
					{Kind: "secret", ID: "secret-asset", Ref: ref, Source: "s"},
				},
			}
			if err := ReidentifyLock(&lock); err != nil {
				t.Fatal(err)
			}
			if err := VerifyLock(lock); err == nil {
				t.Fatalf("expected error for invalid ref %q, got nil", ref)
			}
		})
	}
}

func TestProductLockV2MultiPlatformCompatibility(t *testing.T) {
	e := stackresolve.Evidence{Authority: "test", Source: "fixture", Freshness: "2026-08-15"}
	allPlatforms := []PlatformRequirement{
		{OS: "linux", Arch: "amd64", Contract: "v1"},
		{OS: "darwin", Arch: "arm64", Contract: "v1"},
		{OS: "windows", Arch: "amd64", Contract: "v1"},
	}

	m := Manifest{
		Schema:    Schema,
		Roots:     []string{"multi-pack"},
		Platforms: allPlatforms,
		Budget:    Budget{ContextTokens: 2000, MemoryMiB: 512, Workers: 2},
		Components: []Component{
			{
				ID:       "multi-pack",
				Version:  "1.0.0",
				Digest:   "sha256:multi",
				Source:   "registry/multi",
				Adapters: []string{"instruction"},
				Compatibility: Compatibility{
					OS:       []string{"linux", "darwin", "windows"},
					Arch:     []string{"amd64", "arm64"},
					Contract: "v1",
				},
				Evidence: e,
			},
		},
		Assets: harnesscompose.Manifest{
			Schema: harnesscompose.Schema,
			Layers: []harnesscompose.Layer{
				{ID: "company", Scope: "company", Assets: []harnesscompose.Asset{{Kind: "instruction", ID: "i", Value: "v"}}},
			},
		},
	}

	// 1. Resolve multi-platform manifest simultaneously supporting linux/amd64, darwin/arm64, windows/amd64
	res, err := Resolve(context.Background(), m, []string{"company"}, Environment{})
	if err != nil {
		t.Fatalf("multi-platform resolution failed: %v", err)
	}
	if res.Lock.Schema != LockSchemaV2 {
		t.Fatalf("expected schema %q, got %q", LockSchemaV2, res.Lock.Schema)
	}
	if len(res.Lock.Platforms) != 3 {
		t.Fatalf("expected 3 platforms, got %d", len(res.Lock.Platforms))
	}
	if err := VerifyLock(res.Lock); err != nil {
		t.Fatalf("multi-platform lock failed VerifyLock: %v", err)
	}

	// 2. CheckMultiPlatformCompatibility directly
	if err := CheckMultiPlatformCompatibility(m.Components, allPlatforms); err != nil {
		t.Fatalf("CheckMultiPlatformCompatibility failed: %v", err)
	}

	// 3. Negative case: add a component incompatible with darwin
	incompatibleComponents := append([]Component(nil), m.Components...)
	incompatibleComponents = append(incompatibleComponents, Component{
		ID:      "linux-only",
		Version: "1.0.0",
		Digest:  "sha256:linux-only",
		Source:  "registry/linux-only",
		Compatibility: Compatibility{
			OS:       []string{"linux"},
			Arch:     []string{"amd64"},
			Contract: "v1",
		},
		Evidence: e,
	})
	if err := CheckMultiPlatformCompatibility(incompatibleComponents, allPlatforms); err == nil || !strings.Contains(err.Error(), "incompatible OS \"darwin\"") {
		t.Fatalf("expected incompatible OS darwin error, got %v", err)
	}
}

func TestProductLockV2BackwardCompatibility(t *testing.T) {
	schemas := []string{
		LegacyLockSchema,
		LockSchema,
		LockSchemaV2,
	}
	for _, s := range schemas {
		t.Run("schema "+s, func(t *testing.T) {
			lock := Lock{
				Schema: s,
				Components: []LockedComponent{
					{ID: "c", Version: "1.0.0", Digest: "sha256:c", Source: "s", Reason: "r", Provider: "p"},
				},
				Assets: []harnesscompose.EffectiveAsset{
					{Kind: "instruction", ID: "i", Source: "s", Value: "hello"},
				},
			}
			if err := ReidentifyLock(&lock); err != nil {
				t.Fatal(err)
			}
			if err := VerifyLock(lock); err != nil {
				t.Fatalf("schema %q failed VerifyLock: %v", s, err)
			}
		})
	}
}
