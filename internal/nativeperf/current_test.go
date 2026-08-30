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
	for _, want := range []string{"active-native-lane-collision", "measurement-control-loop", "metal-startup-capacity", "metal-resident-decode", "cuda-cold-decode", "cuda-cache-correctness", "laptop-placement", "native-serving-stack"} {
		if !constraintIDs[want] {
			t.Errorf("missing current constraint %q", want)
		}
	}
	if len(snapshot.Programs) != 4 {
		t.Fatalf("execution programs = %d, want 4", len(snapshot.Programs))
	}
	packetByID := map[string]WorkPacket{}
	for _, packet := range snapshot.WorkPackets {
		packetByID[packet.ID] = packet
	}
	for _, want := range []string{"cuda.cache-weight-residency", "mac.m1-streamed-q4k-no-copy", "mac.m2-whole-sequence-prefill", "mac.m10-parity-reconvergence", "cuda.q8_1-numerical-gate", "profile.returned-receipt-gate"} {
		if _, ok := packetByID[want]; !ok {
			t.Errorf("missing current work packet %q", want)
		}
	}
	if packetByID["cuda.cache-weight-residency"].State != WorkRunning || packetByID["mac.m1-streamed-q4k-no-copy"].State != WorkReady || packetByID["mac.m2-whole-sequence-prefill"].State != WorkWaitingDependency {
		t.Fatalf("live dispatch states are not explicit: cache=%q m1=%q m2=%q", packetByID["cuda.cache-weight-residency"].State, packetByID["mac.m1-streamed-q4k-no-copy"].State, packetByID["mac.m2-whole-sequence-prefill"].State)
	}
	if packetByID["mac.m1-streamed-q4k-no-copy"].Issue != 8325 || packetByID["mac.m2-whole-sequence-prefill"].Issue != 9230 {
		t.Fatalf("corrected Mac packet owners are stale: m1=#%d m2=#%d", packetByID["mac.m1-streamed-q4k-no-copy"].Issue, packetByID["mac.m2-whole-sequence-prefill"].Issue)
	}
	if packetByID["profile.real-metal"].Issue != 9495 || packetByID["profile.real-cuda"].Issue != 9497 || packetByID["profile.returned-receipt-gate"].Issue != 9498 {
		t.Fatalf("profile-control-loop owners are stale: metal=#%d cuda=#%d gate=#%d", packetByID["profile.real-metal"].Issue, packetByID["profile.real-cuda"].Issue, packetByID["profile.returned-receipt-gate"].Issue)
	}
	if len(snapshot.ReadyWaves) != 2 || snapshot.ReadyWaves[0].ID != "metal" || snapshot.ReadyWaves[1].ID != "cuda" {
		t.Fatalf("unexpected independent waves: %+v", snapshot.ReadyWaves)
	}
	if len(snapshot.Collisions) < 2 {
		t.Fatalf("current snapshot lacks collision coverage: %+v", snapshot.Collisions)
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

	badDependency := snapshot
	badDependency.WorkPackets = append([]WorkPacket(nil), snapshot.WorkPackets...)
	badDependency.WorkPackets[0].BlockedByIDs = []string{"not-a-packet"}
	if err := ValidateCurrentSnapshot(graph, badDependency); err == nil || !strings.Contains(err.Error(), "invalid dependency/blocker") {
		t.Fatalf("bad packet dependency error = %v", err)
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
		"active-native-lane-collision",
		"55.73 GiB startup",
		"cuda-cache-correctness",
		"on `A100-SXM4-40GB`",
		"36 GiB laptop placement",
		"Divide-and-conquer execution",
		"mac.m1-streamed-q4k-no-copy",
		"cuda.cache-weight-residency",
		"waiting-coordination",
		"Graph-dependency-ready arms",
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
