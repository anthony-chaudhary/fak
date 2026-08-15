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
		{ID: "legal-pack", Version: "1.2.0", Digest: "sha256:legal", Source: "registry/legal", Requires: []Requirement{{Capability: "runtime", Range: ">=1.0.0"}, {Capability: "search", Range: ">=1.0.0"}}, Compatibility: Compatibility{Contract: "v1"}, Cost: Budget{ContextTokens: 1000, MemoryMiB: 64, Workers: 1}, Evidence: e},
		{ID: "search-provider", Version: "1.1.0", Digest: "sha256:search", Source: "registry/search", Provides: []string{"search"}, Cost: Budget{MemoryMiB: 0}, Evidence: e},
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
