package harnessderive

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func TestDeriveInstructionPreservesBaseAndProducesVerifiedLock(t *testing.T) {
	base := fixture(t, false, false)
	result, err := Derive(base, Request{Layer: "my-support", Deltas: []Delta{{Capability: "instruction:style", Operation: "replace", Value: "detailed"}}})
	if err != nil {
		t.Fatal(err)
	}
	if err := harnessresolve.VerifyLock(result.Lock); err != nil {
		t.Fatal(err)
	}
	if err := VerifyReceipt(result.Receipt); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Receipt.Rebuild, "--expect-base "+base.ID) {
		t.Fatalf("rebuild = %s", result.Receipt.Rebuild)
	}
	if result.Receipt.BaseID != base.ID || result.Receipt.ResultID != result.Lock.ID || result.Lock.ID == base.ID {
		t.Fatalf("bad lineage: %+v", result.Receipt)
	}
	if base.Assets[0].Value != "concise" {
		t.Fatal("base lock was mutated")
	}
	got := result.Lock.Assets[0]
	if got.Value != "detailed" || !strings.Contains(got.Source, "derive:my-support") || !strings.Contains(got.Source, "company:support") {
		t.Fatalf("asset = %+v", got)
	}
	if len(result.Lock.AssetTrace) != 1 || !strings.Contains(result.Lock.AssetTrace[0].Reason, base.ID) {
		t.Fatalf("trace = %+v", result.Lock.AssetTrace)
	}
}

func TestDerivePolicyCanOnlyNarrow(t *testing.T) {
	base := fixture(t, false, false)
	base.Assets = append(base.Assets, harnesscompose.EffectiveAsset{Kind: "policy", ID: "tools", Grants: []string{"search", "shell"}, Source: "company:support"})
	if err := harnessresolve.ReidentifyLock(&base); err != nil {
		t.Fatal(err)
	}
	result, err := Derive(base, Request{Deltas: []Delta{{Capability: "policy:tools", Operation: "deny", Denies: []string{"shell"}}}})
	if err != nil {
		t.Fatal(err)
	}
	policy := result.Lock.Assets[1]
	if strings.Join(policy.Grants, ",") != "search" || strings.Join(policy.Denies, ",") != "shell" {
		t.Fatalf("policy = %+v", policy)
	}
}

func TestDeriveRefusesGuaranteeViolations(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*harnessresolve.Lock)
		delta  Delta
		want   string
	}{
		{"unknown", func(*harnessresolve.Lock) {}, Delta{Capability: "instruction:nope", Operation: "replace", Value: "x"}, "not active"},
		{"locked", func(l *harnessresolve.Lock) { l.Assets[0].Locked = true }, Delta{Capability: "instruction:style", Operation: "replace", Value: "x"}, "locked by"},
		{"mandatory", func(l *harnessresolve.Lock) { l.Assets[0].Mandatory = true }, Delta{Capability: "instruction:style", Operation: "replace", Value: "x"}, "mandatory"},
		{"unsupported-adapter", func(*harnessresolve.Lock) {}, Delta{Capability: "instruction:style", Operation: "deny", Denies: []string{"x"}}, "requires a policy"},
		{"unsupported-operation", func(*harnessresolve.Lock) {}, Delta{Capability: "instruction:style", Operation: "append", Value: "x"}, "unsupported operation"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			lock := fixture(t, false, false)
			tc.mutate(&lock)
			if err := harnessresolve.ReidentifyLock(&lock); err != nil {
				t.Fatal(err)
			}
			_, err := Derive(lock, Request{Deltas: []Delta{tc.delta}})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v want %q", err, tc.want)
			}
		})
	}
	bad := fixture(t, false, false)
	bad.ID = "sha256:bad"
	if _, err := Derive(bad, Request{Deltas: []Delta{{Capability: "instruction:style", Operation: "replace", Value: "x"}}}); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("tamper err=%v", err)
	}
}

func TestDeriveIsDeterministicAcrossDeltaOrder(t *testing.T) {
	base := fixture(t, false, false)
	base.Assets = append(base.Assets, harnesscompose.EffectiveAsset{Kind: "instruction", ID: "tone", Value: "warm", Source: "company:support"})
	if err := harnessresolve.ReidentifyLock(&base); err != nil {
		t.Fatal(err)
	}
	a := []Delta{{Capability: "instruction:tone", Operation: "replace", Value: "direct"}, {Capability: "instruction:style", Operation: "replace", Value: "detailed"}}
	b := []Delta{a[1], a[0]}
	ra, err := Derive(base, Request{Deltas: a})
	if err != nil {
		t.Fatal(err)
	}
	rb, err := Derive(base, Request{Deltas: b})
	if err != nil {
		t.Fatal(err)
	}
	if ra.Lock.ID != rb.Lock.ID {
		t.Fatalf("ids differ: %s %s", ra.Lock.ID, rb.Lock.ID)
	}
}

func fixture(tb testing.TB, locked, mandatory bool) harnessresolve.Lock {
	tb.Helper()
	lock := harnessresolve.Lock{Schema: harnessresolve.LockSchema, Environment: harnessresolve.Environment{OS: "linux", Arch: "amd64", Contract: "v1"}, Components: []harnessresolve.LockedComponent{{ID: "support", Version: "1.0.0", Digest: "sha256:support", Source: "registry/support", Reason: "root", Provider: "support"}}, Assets: []harnesscompose.EffectiveAsset{{Kind: "instruction", ID: "style", Value: "concise", Source: "company:support", Locked: locked, Mandatory: mandatory}}}
	if err := harnessresolve.ReidentifyLock(&lock); err != nil {
		tb.Fatal(err)
	}
	return lock
}
