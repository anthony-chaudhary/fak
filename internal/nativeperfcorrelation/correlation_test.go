package nativeperfcorrelation

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"testing/fstest"
)

func TestIndexJoinsEvidenceDeterministicallyAndBoundsRetention(t *testing.T) {
	index, err := NewIndex(2)
	if err != nil {
		t.Fatal(err)
	}
	firstInput := testInput("request-one", "run-one")
	first, err := index.Add(firstInput)
	if err != nil {
		t.Fatal(err)
	}
	repeated, err := index.Add(firstInput)
	if err != nil {
		t.Fatal(err)
	}
	if repeated.Key != first.Key {
		t.Fatalf("same evidence produced keys %q and %q", first.Key, repeated.Key)
	}
	if first.Engine.Name != NativeEngine || first.ModuleAtRev != "internal/nativeperfcorrelation@r1+g0123456" {
		t.Fatalf("native identity lost: %+v", first)
	}
	if first.RequestFingerprint == firstInput.RequestID || first.RunFingerprint == firstInput.RunID {
		t.Fatalf("raw high-cardinality IDs escaped into record: %+v", first)
	}

	second, err := index.Add(testInput("request-two", "run-two"))
	if err != nil {
		t.Fatal(err)
	}
	third, err := index.Add(testInput("request-three", "run-three"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := index.Lookup(first.Key); !errors.Is(err, ErrNotFound) {
		t.Fatalf("oldest record lookup error = %v, want ErrNotFound", err)
	}
	snapshot := index.Snapshot()
	if len(snapshot) != 2 || snapshot[0].Key != second.Key || snapshot[1].Key != third.Key {
		t.Fatalf("bounded order = %#v", snapshot)
	}
	exemplar, err := index.Exemplar(third.Key)
	if err != nil {
		t.Fatal(err)
	}
	if exemplar.CorrelationKey != third.Key {
		t.Fatalf("exemplar = %+v", exemplar)
	}
}

func TestIndexRejectsKeyCollision(t *testing.T) {
	index, err := NewIndex(2, WithKeyFunc(func(Record) string {
		return "npc1_aaaaaaaaaaaaaaaa"
	}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := index.Add(testInput("request-one", "run-one")); err != nil {
		t.Fatal(err)
	}
	if _, err := index.Add(testInput("request-two", "run-two")); !errors.Is(err, ErrCollision) {
		t.Fatalf("collision error = %v, want ErrCollision", err)
	}
}

func TestVerifyArtifactReportsMissingAndMismatchedEvidence(t *testing.T) {
	index, err := NewIndex(1)
	if err != nil {
		t.Fatal(err)
	}
	record, err := index.Add(testInput("request", "run"))
	if err != nil {
		t.Fatal(err)
	}
	files := fstest.MapFS{
		"artifacts/receipt.json": &fstest.MapFile{Data: []byte("receipt")},
		"artifacts/trace.json":   &fstest.MapFile{Data: []byte("tampered")},
	}
	if err := index.VerifyArtifact(record.Key, ArtifactReceipt, files, 1024); err != nil {
		t.Fatalf("receipt verification: %v", err)
	}
	if err := index.VerifyArtifact(record.Key, ArtifactTrace, files, 1024); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("trace verification error = %v, want ErrDigestMismatch", err)
	}
	if err := index.VerifyArtifact(record.Key, ArtifactProfile, files, 1024); !errors.Is(err, ErrArtifactMissing) {
		t.Fatalf("profile verification error = %v, want ErrArtifactMissing", err)
	}
}

func TestRecordIsScrubbedAndPrivateLocatorsAreRejected(t *testing.T) {
	input := testInput("customer@example.com/request/secret", "ssh://private-host/run/token")
	record, err := scrub(input)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	text := string(encoded)
	for _, secret := range []string{input.RequestID, input.RunID, input.ReceiptID, input.TraceID, input.ProfileID} {
		if strings.Contains(text, secret) {
			t.Fatalf("serialized record leaked %q: %s", secret, text)
		}
	}

	private := testInput("request", "run")
	private.Artifacts[0].Locator = "/private/lab/receipt.json"
	if _, err := scrub(private); err == nil || !strings.Contains(err.Error(), "bounded relative") {
		t.Fatalf("absolute private locator error = %v", err)
	}
	traversal := testInput("request", "run")
	traversal.Artifacts[0].Locator = "artifacts/../private/receipt.json"
	if _, err := scrub(traversal); err == nil {
		t.Fatal("traversal locator accepted")
	}
	credential := testInput("request", "run")
	credential.Engine.Backend = "ssh://private-host/run"
	if _, err := scrub(credential); err == nil {
		t.Fatal("unscrubbed engine backend accepted")
	}
}

func testInput(requestID, runID string) Input {
	return Input{
		RequestID: requestID,
		RunID:     runID,
		CommitSHA: "0123456789abcdef0123456789abcdef01234567",
		ReceiptID: "receipt-private-id",
		TraceID:   "trace-private-id",
		ProfileID: "profile-private-id",
		Engine: EngineIdentity{
			Name:         NativeEngine,
			Backend:      "cuda-sm90",
			Model:        "qwen3.8-8b",
			Quantization: "q4_k_m",
		},
		ModuleAtRev: "internal/nativeperfcorrelation@r1+g0123456",
		Artifacts: []Artifact{
			{Kind: ArtifactReceipt, Locator: "artifacts/receipt.json", SHA256: digest("receipt")},
			{Kind: ArtifactTrace, Locator: "artifacts/trace.json", SHA256: digest("trace")},
			{Kind: ArtifactProfile, Locator: "artifacts/profile.pb.gz", SHA256: digest("profile")},
		},
	}
}

func digest(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

// BenchmarkIndexAdd evaluates throughput of record scrubbing, hashing, and insertion into the bounded index.
func BenchmarkIndexAdd(b *testing.B) {
	idx, err := NewIndex(1024)
	if err != nil {
		b.Fatal(err)
	}
	input := testInput("req-bench", "run-bench")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := idx.Add(input); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkIndexLookup measures key retrieval latency against an indexed correlation record.
func BenchmarkIndexLookup(b *testing.B) {
	idx, err := NewIndex(1024)
	if err != nil {
		b.Fatal(err)
	}
	record, err := idx.Add(testInput("req-lookup", "run-lookup"))
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := idx.Lookup(record.Key); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkVerifyArtifact evaluates streaming SHA-256 artifact integrity checking over simulated files.
func BenchmarkVerifyArtifact(b *testing.B) {
	idx, err := NewIndex(1024)
	if err != nil {
		b.Fatal(err)
	}
	record, err := idx.Add(testInput("req-verify", "run-verify"))
	if err != nil {
		b.Fatal(err)
	}
	files := fstest.MapFS{
		"artifacts/receipt.json": &fstest.MapFile{Data: []byte("receipt")},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := idx.VerifyArtifact(record.Key, ArtifactReceipt, files, 1024); err != nil {
			b.Fatal(err)
		}
	}
}
