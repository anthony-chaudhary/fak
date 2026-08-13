package portabilitycontract

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func load(t *testing.T, name string) Package {
	t.Helper()
	b, e := os.ReadFile("testdata/" + name)
	if e != nil {
		t.Fatal(e)
	}
	var p Package
	if e = json.Unmarshal(b, &p); e != nil {
		t.Fatal(e)
	}
	return p
}
func TestRepresentativeGoldenContract(t *testing.T) {
	p := load(t, "representative.golden.json")
	if e := p.Validate(); e != nil {
		t.Fatal(e)
	}
	if got := len(p.Collections[0].Objects); got != 4 {
		t.Fatalf("objects=%d", got)
	}
	u := p.Collections[0].Objects[2]
	if u.Known() || u.Active() {
		t.Fatal("unknown object activated")
	}
	b, _ := os.ReadFile("testdata/representative.golden.json")
	rt, e := RoundTrip(b)
	if e != nil || !EqualJSON(b, rt) {
		t.Fatalf("inert round-trip: %v", e)
	}
	x := Explain(p)
	if !strings.Contains(x, "INERT (unknown type; preserved)") || !strings.Contains(x, "machine > user > project") {
		t.Fatal(x)
	}
}
func TestTransportIdentityExcludesLocalAndSecret(t *testing.T) {
	p := load(t, "representative.golden.json")
	base, _ := PackageIdentity(p)
	p.Local = map[string]any{"path": "C:/secret"}
	p.Signatures[0].Value = "changed"
	p.Collections[0].Objects[3].Payload = json.RawMessage(`{"provider":"vault","name":"changed"}`)
	got, _ := PackageIdentity(p)
	if got != base {
		t.Fatalf("local/secret changed identity %s != %s", got, base)
	}
	p.Collections[0].Objects[0].Payload = json.RawMessage(`{"allow":["read"]}`)
	got, _ = PackageIdentity(p)
	if got == base {
		t.Fatal("portable content did not change identity")
	}
}
func TestDeterministicScopePrecedence(t *testing.T) {
	os := []Object{{StableID: "z", Scope: ScopePublic}, {StableID: "b", Scope: ScopeUser}, {StableID: "a", Scope: ScopeUser}, {StableID: "m", Scope: ScopeMachine}}
	got := ResolvePrecedence(os)
	want := []string{"m", "a", "b", "z"}
	for i := range want {
		if got[i].StableID != want[i] {
			t.Fatalf("%d=%s", i, got[i].StableID)
		}
	}
}
func TestHostileAndInvalidFixturesRejected(t *testing.T) {
	for _, n := range []string{"hostile.unknown-critical.json", "invalid.schema.json"} {
		p := load(t, n)
		if e := p.Validate(); e == nil {
			t.Fatalf("%s accepted", n)
		}
	}
}
func TestPreviewApplyIdempotentAndRecoverable(t *testing.T) {
	tx := Transaction{Schema: Schema, StableID: "tx:1", Mode: "merge", IdempotencyKey: "request:1", ExpectedState: "state:old", Operations: []Operation{{Kind: "merge", CollectionID: "c", Strategy: "fail-on-conflict"}}, Recovery: "journal", Rollback: []Operation{{Kind: "restore", From: "receipt.before"}}}
	tx, e := Preview(tx, "state:old")
	if e != nil {
		t.Fatal(e)
	}
	s := Store{State: "state:old"}
	r, e := s.Apply(tx)
	if e != nil {
		t.Fatal(e)
	}
	again, e := s.Apply(tx)
	if e != nil || again.Status != "already-applied" {
		t.Fatalf("idempotency: %#v %v", again, e)
	}
	if e = s.Recover(r); e != nil || s.State != "state:old" {
		t.Fatalf("recover %s %v", s.State, e)
	}
}
func TestSemanticDegradationContract(t *testing.T) {
	r := TranslationReport{SourceHarness: "codex", TargetHarness: "other", Exact: false, Degradations: []Degradation{{ObjectID: "policy:x", Feature: "tool-argument-policy", Severity: "security", MeaningLost: "target cannot express argument predicate", Fallback: "deny tool"}}}
	b, _ := json.Marshal(r)
	if !strings.Contains(string(b), "meaning_lost") || r.Exact || len(r.Degradations) == 0 {
		t.Fatal(string(b))
	}
}
