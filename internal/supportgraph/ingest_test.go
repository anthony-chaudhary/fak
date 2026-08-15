package supportgraph

import (
	"strings"
	"testing"
	"time"
)

func TestIngestWitnessAndInvalidateChangedBaseline(t *testing.T) {
	observed := time.Date(2026, 8, 15, 20, 0, 0, 0, time.UTC)
	oldTuple := Tuple{Artifact: "a", Architecture: "arch", Quant: "q", Layout: "l", Backend: "b", Kernel: "k", Runtime: "cuda@12.8", Hardware: "h100-driver@570"}
	graph := Graph{Schema: Schema, Edges: []Edge{{Tuple: oldTuple, Evidence: []Evidence{{ID: "old", State: Supported, Tier: Witnessed, Authority: "lab", Source: "old", Expires: observed.Add(24 * time.Hour)}}}}}
	witness := fixtureRecord(observed)
	result, err := Ingest(graph, witness)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Inserted || result.StaleEdges != 1 {
		t.Fatalf("result=%+v", result)
	}
	old := Query(result.Graph, oldTuple, observed.Add(time.Second))
	if old.State != Stale {
		t.Fatalf("old=%+v", old)
	}
	current := Query(result.Graph, witness.Tuple, observed.Add(time.Second))
	if current.State != Supported || current.Decisive[0].ArtifactDigest != witness.ArtifactDigest {
		t.Fatalf("current=%+v", current)
	}
	again, err := Ingest(result.Graph, witness)
	if err != nil || again.Inserted {
		t.Fatalf("duplicate=%+v err=%v", again, err)
	}
}

func TestWitnessRejectsDigestAndProvenanceMismatch(t *testing.T) {
	witness := fixtureRecord(time.Now().UTC())
	witness.PayloadSHA256 = strings.Repeat("0", 64)
	if _, err := Ingest(Graph{Schema: Schema}, witness); err == nil || !strings.Contains(err.Error(), "payload_sha256") {
		t.Fatalf("err=%v", err)
	}
	witness = fixtureRecord(time.Now().UTC())
	witness.ArtifactDigest = "sha256:not-hex"
	witness.PayloadSHA256 = WitnessDigest(witness)
	if _, err := Ingest(Graph{Schema: Schema}, witness); err == nil || !strings.Contains(err.Error(), "artifact_digest") {
		t.Fatalf("err=%v", err)
	}
}

func fixtureRecord(observed time.Time) Witness {
	w := Witness{Schema: WitnessSchema, ID: "gpu-run-1", Tuple: Tuple{Artifact: "a", Architecture: "arch", Quant: "q", Layout: "l", Backend: "b", Kernel: "k", Runtime: "cuda@12.9", Hardware: "h100-driver@580"}, State: Supported, Tier: Witnessed, Authority: "fak-lab", ArtifactDigest: "sha256:" + strings.Repeat("a", 64), Environment: "gpu=h100,cuda=12.9,driver=580", Reproduce: "go test -tags cuda ./internal/gpu", ObservedAt: observed, Expires: observed.Add(30 * 24 * time.Hour), Required: []string{"sm>=90"}, SourceCommit: "abc123"}
	w.PayloadSHA256 = WitnessDigest(w)
	return w
}
