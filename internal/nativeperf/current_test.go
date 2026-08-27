package nativeperf

import (
	"strings"
	"testing"
)

func TestReadyLeversReturnsEveryDependencyReadyArm(t *testing.T) {
	ready, err := ReadyLevers(ActiveGraph())
	if err != nil {
		t.Fatal(err)
	}
	got := make([]string, 0, len(ready))
	for _, lever := range ready {
		got = append(got, lever.ID)
	}
	want := []string{
		"metal.command-buffer-amortization",
		"metal.paged-kv",
		"metal.chunked-prefill",
		"cuda.q8_1-activation-quant",
	}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("ready levers = %v, want %v", got, want)
	}

	first, err := NextLever(ActiveGraph())
	if err != nil {
		t.Fatal(err)
	}
	if first == nil || first.ID != want[0] {
		t.Fatalf("backward-compatible first ready lever = %+v", first)
	}
}

func TestCurrentSnapshotCoversConstraintsWavesAndOSSWalk(t *testing.T) {
	graph := ActiveGraph()
	snapshot, err := BuildCurrentSnapshot(graph)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Schema != CurrentSchema || snapshot.AsOf != "2026-08-27" {
		t.Fatalf("unexpected snapshot identity: %+v", snapshot)
	}
	constraintIDs := map[string]bool{}
	for _, constraint := range snapshot.Constraints {
		constraintIDs[constraint.ID] = true
	}
	for _, want := range []string{"measurement-control-loop", "metal-resident-decode", "cuda-cold-decode", "cuda-cache-correctness", "laptop-placement", "native-serving-stack"} {
		if !constraintIDs[want] {
			t.Errorf("missing current constraint %q", want)
		}
	}
	if len(snapshot.ReadyWaves) != 2 || snapshot.ReadyWaves[0].ID != "metal" || snapshot.ReadyWaves[1].ID != "cuda" {
		t.Fatalf("unexpected independent waves: %+v", snapshot.ReadyWaves)
	}
	if len(snapshot.Collisions) < 2 || len(snapshot.OSSWalk) != 6 {
		t.Fatalf("current snapshot lacks collision or OSS walk coverage: collisions=%+v walk=%+v", snapshot.Collisions, snapshot.OSSWalk)
	}
	chain := make([]string, 0, len(snapshot.OSSWalk))
	for _, step := range snapshot.OSSWalk {
		chain = append(chain, step.Name)
	}
	if strings.Join(chain, " -> ") != "source -> seam -> measured constraint -> deduped issue -> A/B -> keep/reject" {
		t.Fatalf("OSS walk changed: %s", strings.Join(chain, " -> "))
	}
}

func TestValidateCurrentSnapshotRejectsStaleAndMissingReadyWork(t *testing.T) {
	graph := ActiveGraph()
	snapshot, err := BuildCurrentSnapshot(graph)
	if err != nil {
		t.Fatal(err)
	}

	stale := snapshot
	stale.Constraints = append([]CurrentConstraint(nil), snapshot.Constraints...)
	stale.Constraints[0].ReviewBy = "2026-08-26"
	if err := ValidateCurrentSnapshot(graph, stale); err == nil || !strings.Contains(err.Error(), "invalid observed/review dates") {
		t.Fatalf("stale snapshot error = %v", err)
	}

	missing := snapshot
	missing.ReadyWaves = append([]ReadyWave(nil), snapshot.ReadyWaves...)
	missing.ReadyWaves[0].ReadyLeverIDs = missing.ReadyWaves[0].ReadyLeverIDs[1:]
	if err := ValidateCurrentSnapshot(graph, missing); err == nil || !strings.Contains(err.Error(), "missing from current waves") {
		t.Fatalf("missing ready work error = %v", err)
	}
}

func TestRenderCurrentMarkdownIsDeterministicAndExplicit(t *testing.T) {
	snapshot, err := BuildCurrentSnapshot(ActiveGraph())
	if err != nil {
		t.Fatal(err)
	}
	first := RenderCurrentMarkdown(snapshot)
	second := RenderCurrentMarkdown(snapshot)
	if first != second {
		t.Fatal("current markdown changed across identical renders")
	}
	for _, want := range []string{
		"# Current native-performance constraints",
		"measurement-control-loop",
		"cuda-cache-correctness",
		"36 GiB laptop placement",
		"metal.command-buffer-amortization",
		"metal.paged-kv",
		"metal.chunked-prefill",
		"cuda.q8_1-activation-quant",
		"source -> seam -> measured constraint -> deduped issue -> matched A/B -> keep/reject",
		"mapped backlog is not performance-closed",
	} {
		if !strings.Contains(first, want) {
			t.Errorf("rendered markdown missing %q", want)
		}
	}
}
