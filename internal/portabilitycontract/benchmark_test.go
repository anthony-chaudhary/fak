package portabilitycontract

import (
	"encoding/json"
	"os"
	"testing"
)

var (
	benchPkgSink     Package
	benchStringSink  string
	benchReceiptSink Receipt
	benchBytesSink   []byte
	benchObjectsSink []Object
)

func loadGoldenPackage(b *testing.B, name string) Package {
	b.Helper()
	data, err := os.ReadFile("testdata/" + name)
	if err != nil {
		b.Fatalf("load golden package %s: %v", name, err)
	}
	var p Package
	if err := json.Unmarshal(data, &p); err != nil {
		b.Fatalf("unmarshal golden package %s: %v", name, err)
	}
	return p
}

// BenchmarkPortabilityContract measures the composite contract pipeline:
// package validation, portable sanitization, identity calculation, scope precedence resolution,
// transaction preview/apply/recovery, and human-readable explanation generation.
func BenchmarkPortabilityContract(b *testing.B) {
	pkg := loadGoldenPackage(b, "representative.golden.json")
	tx := Transaction{
		Schema:         Schema,
		StableID:       "tx:bench-portability",
		Mode:           "merge",
		IdempotencyKey: "bench-key-42",
		ExpectedState:  "state:initial",
		Operations: []Operation{
			{Kind: "merge", CollectionID: "collection:working-set", Strategy: "fail-on-conflict"},
		},
		Recovery: "journal",
		Rollback: []Operation{
			{Kind: "restore", From: "receipt.before"},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := pkg.Validate(); err != nil {
			b.Fatalf("validate failed: %v", err)
		}
		port := Portable(pkg)
		benchPkgSink = port

		id, err := PackageIdentity(pkg)
		if err != nil || id == "" {
			b.Fatalf("package identity failed: %v", err)
		}
		benchStringSink = id

		ordered := ResolvePrecedence(pkg.Collections[0].Objects)
		if len(ordered) == 0 {
			b.Fatal("unexpected empty precedence slice")
		}
		benchObjectsSink = ordered

		previewed, err := Preview(tx, "state:initial")
		if err != nil {
			b.Fatalf("preview failed: %v", err)
		}

		store := Store{State: "state:initial"}
		receipt, err := store.Apply(previewed)
		if err != nil {
			b.Fatalf("apply failed: %v", err)
		}
		benchReceiptSink = receipt

		if err := store.Recover(receipt); err != nil {
			b.Fatalf("recover failed: %v", err)
		}

		explanation := Explain(port)
		if len(explanation) == 0 {
			b.Fatal("unexpected empty explanation")
		}
		benchStringSink = explanation
	}
}

// BenchmarkPackageValidate measures schema, object integrity, and identity validation throughput.
func BenchmarkPackageValidate(b *testing.B) {
	pkg := loadGoldenPackage(b, "representative.golden.json")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := pkg.Validate(); err != nil {
			b.Fatalf("validate failed: %v", err)
		}
	}
}

// BenchmarkPackageIdentity measures deterministic SHA-256 content hashing for portable packages.
func BenchmarkPackageIdentity(b *testing.B) {
	pkg := loadGoldenPackage(b, "representative.golden.json")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id, err := PackageIdentity(pkg)
		if err != nil || id == "" {
			b.Fatalf("package identity failed: %v", err)
		}
		benchStringSink = id
	}
}

// BenchmarkPortableSanitization measures stripping of local paths, secrets, machine scopes, and signatures.
func BenchmarkPortableSanitization(b *testing.B) {
	pkg := loadGoldenPackage(b, "representative.golden.json")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchPkgSink = Portable(pkg)
	}
}

// BenchmarkResolvePrecedence measures sorting stability and scope rank precedence resolution.
func BenchmarkResolvePrecedence(b *testing.B) {
	objects := []Object{
		{StableID: "obj-pub", Scope: ScopePublic, Precedence: 0},
		{StableID: "obj-corp", Scope: ScopeCorporate, Precedence: 5},
		{StableID: "obj-team-1", Scope: ScopeTeam, Precedence: 10},
		{StableID: "obj-team-2", Scope: ScopeTeam, Precedence: 20},
		{StableID: "obj-proj-1", Scope: ScopeProject, Precedence: 10},
		{StableID: "obj-proj-2", Scope: ScopeProject, Precedence: 15},
		{StableID: "obj-user-1", Scope: ScopeUser, Precedence: 0},
		{StableID: "obj-user-2", Scope: ScopeUser, Precedence: 50},
		{StableID: "obj-mach", Scope: ScopeMachine, Precedence: 100},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchObjectsSink = ResolvePrecedence(objects)
	}
}

// BenchmarkCanonicalContentID measures payload unmarshaling, structure normalization, and SHA-256 hashing.
func BenchmarkCanonicalContentID(b *testing.B) {
	obj := Object{
		Schema:      Schema,
		Type:        "policy",
		TypeVersion: "1.0.0",
		StableID:    "policy:bench",
		Payload:     json.RawMessage(`{"allow":["read","build"],"timeout":30}`),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cid, err := obj.CanonicalContentID()
		if err != nil {
			b.Fatalf("canonical content id failed: %v", err)
		}
		benchStringSink = cid
	}
}

// BenchmarkTransactionLifecycle measures preview verification, state mutation apply, and rollback recovery.
func BenchmarkTransactionLifecycle(b *testing.B) {
	tx := Transaction{
		Schema:         Schema,
		StableID:       "tx:lifecycle-bench",
		Mode:           "apply",
		IdempotencyKey: "lifecycle-key-99",
		ExpectedState:  "state:v1",
		Operations: []Operation{
			{Kind: "apply", CollectionID: "col:main", Strategy: "fast"},
		},
		Recovery: "journal",
		Rollback: []Operation{
			{Kind: "restore", From: "receipt.before"},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		previewed, err := Preview(tx, "state:v1")
		if err != nil {
			b.Fatalf("preview failed: %v", err)
		}

		store := Store{State: "state:v1"}
		receipt, err := store.Apply(previewed)
		if err != nil {
			b.Fatalf("apply failed: %v", err)
		}
		benchReceiptSink = receipt

		if err := store.Recover(receipt); err != nil {
			b.Fatalf("recover failed: %v", err)
		}
	}
}

// BenchmarkExplain measures formatting overhead when generating diagnostics summaries.
func BenchmarkExplain(b *testing.B) {
	pkg := loadGoldenPackage(b, "representative.golden.json")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = Explain(pkg)
	}
}

// BenchmarkRoundTrip measures deserialization and indented serialization fidelity.
func BenchmarkRoundTrip(b *testing.B) {
	raw, err := os.ReadFile("testdata/representative.golden.json")
	if err != nil {
		b.Fatalf("read testdata: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := RoundTrip(raw)
		if err != nil {
			b.Fatalf("roundtrip failed: %v", err)
		}
		benchBytesSink = out
	}
}
