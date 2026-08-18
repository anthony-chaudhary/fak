package harnessmix

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func TestMixCombinesVerifiedImportsAndDeduplicatesSharedSetup(t *testing.T) {
	shared := component("kernel", "sha256:kernel", []string{"runtime"}, harnessresolve.Budget{ContextTokens: 100, MemoryMiB: 64}, []string{"instruction"})
	a := lock(t, "support", []harnessresolve.LockedComponent{shared, component("support", "sha256:support", []string{"support"}, harnessresolve.Budget{ContextTokens: 200}, []string{"instruction"})}, []harnesscompose.EffectiveAsset{{Kind: "instruction", ID: "support", Value: "resolve tickets", Source: "support"}})
	b := lock(t, "research", []harnessresolve.LockedComponent{shared, component("research", "sha256:research", []string{"citations"}, harnessresolve.Budget{ContextTokens: 300}, []string{"instruction"})}, []harnesscompose.EffectiveAsset{{Kind: "instruction", ID: "citations", Value: "cite sources", Source: "research"}})
	result, err := Mix([]harnessresolve.Lock{b, a}, Limits{ContextTokens: 700, MemoryMiB: 128})
	if err != nil {
		t.Fatal(err)
	}
	if err := harnessresolve.VerifyLock(result.Lock); err != nil {
		t.Fatal(err)
	}
	if len(result.Lock.Components) != 3 || len(result.Lock.Assets) != 2 || result.Lock.Budget.ContextTokens != 600 || len(result.Receipt.Deduplicated) != 1 {
		t.Fatalf("result=%+v", result)
	}
	reverse, err := Mix([]harnessresolve.Lock{a, b}, Limits{ContextTokens: 700, MemoryMiB: 128})
	if err != nil {
		t.Fatal(err)
	}
	if reverse.Lock.ID != result.Lock.ID {
		t.Fatalf("nondeterministic %s %s", result.Lock.ID, reverse.Lock.ID)
	}
}

func TestMixRefusesSixConflictClasses(t *testing.T) {
	base := lock(t, "a", []harnessresolve.LockedComponent{component("a", "sha256:a", []string{"a"}, harnessresolve.Budget{ContextTokens: 100}, []string{"instruction"})}, []harnesscompose.EffectiveAsset{{Kind: "instruction", ID: "same", Value: "one", Source: "a"}})
	cases := []struct {
		name   string
		left   harnessresolve.Lock
		other  harnessresolve.Lock
		limits Limits
		want   string
	}{
		{"capability", base, lock(t, "b", []harnessresolve.LockedComponent{component("b", "sha256:b", []string{"b"}, harnessresolve.Budget{}, []string{"instruction"})}, []harnesscompose.EffectiveAsset{{Kind: "instruction", ID: "same", Value: "two", Source: "b"}}), Limits{}, "capability conflict"},
		{"environment", base, withEnv(lock(t, "b", []harnessresolve.LockedComponent{component("b", "sha256:b", []string{"b"}, harnessresolve.Budget{}, []string{"instruction"})}, nil), "windows"), Limits{}, "environment mismatch"},
		{"policy-floor", policyBase(t), policyCollision(t), Limits{}, "policy floor collision"},
		{"secret", secretBase(t), secretCollision(t), Limits{}, "duplicate secret boundary"},
		{"budget", base, lock(t, "b", []harnessresolve.LockedComponent{component("b", "sha256:b", []string{"b"}, harnessresolve.Budget{ContextTokens: 1000}, []string{"instruction"})}, nil), Limits{ContextTokens: 500}, "context budget exceeded"},
		{"missing-adapter", base, missingAdapter(t), Limits{}, "adapter conformance"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Mix([]harnessresolve.Lock{tc.left, tc.other}, tc.limits)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v want=%q", err, tc.want)
			}
		})
	}
}

func component(id, digest string, provides []string, cost harnessresolve.Budget, adapters []string) harnessresolve.LockedComponent {
	return harnessresolve.LockedComponent{ID: id, Version: "1.0.0", Digest: digest, Source: "registry/" + id, Reason: "selected import", Provider: id, Provides: provides, Compatibility: harnessresolve.Compatibility{OS: []string{"linux"}, Arch: []string{"amd64"}, Contract: "v1"}, Cost: cost, Adapters: adapters}
}
func lock(t *testing.T, name string, cs []harnessresolve.LockedComponent, assets []harnesscompose.EffectiveAsset) harnessresolve.Lock {
	t.Helper()
	l := harnessresolve.Lock{Schema: harnessresolve.LockSchema, Environment: harnessresolve.Environment{OS: "linux", Arch: "amd64", Contract: "v1"}, Components: cs, Assets: assets}
	if err := harnessresolve.ReidentifyLock(&l); err != nil {
		t.Fatal(err)
	}
	return l
}
func withEnv(l harnessresolve.Lock, os string) harnessresolve.Lock {
	l.Environment.OS = os
	_ = harnessresolve.ReidentifyLock(&l)
	return l
}
func secretPair(t *testing.T, base harnessresolve.Lock) harnessresolve.Lock {
	b := lock(t, "b", []harnessresolve.LockedComponent{component("b", "sha256:b", []string{"b"}, harnessresolve.Budget{}, []string{"secret"})}, []harnesscompose.EffectiveAsset{{Kind: "secret", ID: "api", Ref: "env:B", Boundary: "b", Source: "b"}})
	base.Assets = append(base.Assets, harnesscompose.EffectiveAsset{Kind: "secret", ID: "api", Ref: "env:A", Boundary: "a", Source: "a"})
	_ = harnessresolve.ReidentifyLock(&base)
	return b
}
func missingAdapter(t *testing.T) harnessresolve.Lock {
	c := component("b", "sha256:b", []string{"b"}, harnessresolve.Budget{}, nil)
	return lock(t, "b", []harnessresolve.LockedComponent{c}, nil)
}

func policyCollision(t *testing.T) harnessresolve.Lock {
	return lock(t, "b", []harnessresolve.LockedComponent{component("b", "sha256:b", []string{"b"}, harnessresolve.Budget{}, []string{"policy"})}, []harnesscompose.EffectiveAsset{{Kind: "instruction", ID: "same", Value: "one", Source: "a"}, {Kind: "policy", ID: "floor", Grants: []string{"shell"}, Source: "b"}})
}
func secretCollision(t *testing.T) harnessresolve.Lock {
	return lock(t, "b", []harnessresolve.LockedComponent{component("b", "sha256:b", []string{"b"}, harnessresolve.Budget{}, []string{"secret"})}, []harnesscompose.EffectiveAsset{{Kind: "instruction", ID: "same", Value: "one", Source: "a"}, {Kind: "secret", ID: "api", Ref: "env:B", Boundary: "b", Source: "b"}})
}

func policyBase(t *testing.T) harnessresolve.Lock {
	return lock(t, "pa", []harnessresolve.LockedComponent{component("pa", "sha256:pa", []string{"pa"}, harnessresolve.Budget{}, []string{"policy"})}, []harnesscompose.EffectiveAsset{{Kind: "policy", ID: "floor", Grants: []string{"search"}, Locked: true, Source: "pa"}})
}
func secretBase(t *testing.T) harnessresolve.Lock {
	return lock(t, "sa", []harnessresolve.LockedComponent{component("sa", "sha256:sa", []string{"sa"}, harnessresolve.Budget{}, []string{"secret"})}, []harnesscompose.EffectiveAsset{{Kind: "secret", ID: "api", Ref: "env:A", Boundary: "a", Source: "sa"}})
}
