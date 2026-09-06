package wipref

import (
	"fmt"
	"strings"
	"testing"
)

var benchSink any

// BenchmarkEncodeStamp measures serializing checkpoint metadata stamps into commit message lines.
func BenchmarkEncodeStamp(b *testing.B) {
	st := Stamp{
		SessionID:      "sess-worker-alpha-001",
		StartSHA:       "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
		Leaves:         []string{"internal/wipref", "internal/leaseref", "cmd/fak"},
		Scope:          []string{"internal/wipref/wipref.go", "internal/wipref/sync.go"},
		Buildable:      true,
		CheckpointedAt: 1725000000,
		Host:           "node-us-central1-c-04",
		DeltaBytes:     1024 * 64,
		MetadataOnly:   false,
		DeltaObject:    "f0e1d2c3b4a5968778695a4b3c2d1e0f12345678",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		msg, err := EncodeStamp(st)
		if err != nil {
			b.Fatalf("EncodeStamp failed: %v", err)
		}
		benchSink = msg
	}
}

// BenchmarkDecodeStamp measures extracting and parsing JSON metadata stamps from commit messages.
func BenchmarkDecodeStamp(b *testing.B) {
	st := Stamp{
		SessionID:      "sess-worker-alpha-001",
		StartSHA:       "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
		Leaves:         []string{"internal/wipref", "internal/leaseref", "cmd/fak"},
		Scope:          []string{"internal/wipref/wipref.go", "internal/wipref/sync.go"},
		Buildable:      true,
		CheckpointedAt: 1725000000,
		Host:           "node-us-central1-c-04",
		DeltaBytes:     1024 * 64,
		MetadataOnly:   false,
		DeltaObject:    "f0e1d2c3b4a5968778695a4b3c2d1e0f12345678",
	}
	encoded, err := EncodeStamp(st)
	if err != nil {
		b.Fatalf("EncodeStamp failed: %v", err)
	}
	msg := fmt.Sprintf("commit subject\n\nsome git trailer or message\n%s\nmore details\n", encoded)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decoded, ok := DecodeStamp(msg)
		if !ok {
			b.Fatal("DecodeStamp failed to decode stamp")
		}
		benchSink = decoded
	}
}

// BenchmarkFoldStatus measures projecting live checkpoint ref records into a sorted status report.
func BenchmarkFoldStatus(b *testing.B) {
	const count = 100
	recs := make([]RefRecord, count)
	now := int64(1725000000)
	for i := 0; i < count; i++ {
		sess := fmt.Sprintf("session-%04d", i)
		recs[i] = RefRecord{
			Ref:    SessionRef(sess),
			Object: fmt.Sprintf("obj-%04d-%036x", i, i*17),
			Stamp: Stamp{
				SessionID:      sess,
				StartSHA:       "commit-base-0001",
				Leaves:         []string{"internal/wipref", "cmd/fak"},
				Buildable:      i%2 == 0,
				CheckpointedAt: now - int64(i*30),
			},
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := Fold(recs, now)
		benchSink = rep
	}
}

// BenchmarkFoldWithMirror measures status projection when resolving replication state against a mirror index.
func BenchmarkFoldWithMirror(b *testing.B) {
	const count = 100
	recs := make([]RefRecord, count)
	mirror := make(map[string]string, count)
	now := int64(1725000000)
	for i := 0; i < count; i++ {
		sess := fmt.Sprintf("session-%04d", i)
		obj := fmt.Sprintf("obj-%04d-%036x", i, i*17)
		recs[i] = RefRecord{
			Ref:    SessionRef(sess),
			Object: obj,
			Stamp: Stamp{
				SessionID:      sess,
				StartSHA:       "commit-base-0001",
				Leaves:         []string{"internal/wipref", "cmd/fak"},
				Buildable:      i%2 == 0,
				CheckpointedAt: now - int64(i*30),
			},
		}
		if i%3 == 0 {
			mirror[sess] = obj
		} else if i%3 == 1 {
			mirror[sess] = "older-obj-hash"
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := FoldWithMirror(recs, mirror, now)
		benchSink = rep
	}
}

// BenchmarkReconcile measures last-writer-wins compare-and-swap convergence between racing checkpoints.
func BenchmarkReconcile(b *testing.B) {
	current := RefRecord{
		Ref:    SessionRef("sess-rec-01"),
		Object: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Stamp: Stamp{
			SessionID:      "sess-rec-01",
			CheckpointedAt: 1725000000,
		},
	}
	candidates := []RefRecord{
		{
			Ref:    SessionRef("sess-rec-01"),
			Object: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
			Stamp:  Stamp{SessionID: "sess-rec-01", CheckpointedAt: 1725000050},
		},
		{
			Ref:    SessionRef("sess-rec-01"),
			Object: "cccccccccccccccccccccccccccccccccccccccc",
			Stamp:  Stamp{SessionID: "sess-rec-01", CheckpointedAt: 1724999950},
		},
		{
			Ref:    SessionRef("sess-rec-01"),
			Object: "ffffffffffffffffffffffffffffffffffffffff",
			Stamp:  Stamp{SessionID: "sess-rec-01", CheckpointedAt: 1725000000},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cand := candidates[i%len(candidates)]
		winner, changed := Reconcile(current, cand)
		_ = changed
		benchSink = winner
	}
}

// BenchmarkReapDecisions measures retention evaluation folding live refs against session owner states.
func BenchmarkReapDecisions(b *testing.B) {
	const count = 100
	recs := make([]RefRecord, count)
	owners := make(map[string]OwnerState, count)
	states := []OwnerState{OwnerLive, OwnerLanded, OwnerClosedClean, OwnerClosedDirty, OwnerUnknown}

	for i := 0; i < count; i++ {
		sess := fmt.Sprintf("sess-reap-%04d", i)
		recs[i] = RefRecord{
			Ref:    SessionRef(sess),
			Object: fmt.Sprintf("obj-reap-%04d", i),
			Stamp:  Stamp{SessionID: sess, CheckpointedAt: 1725000000 - int64(i*60)},
		}
		owners[sess] = states[i%len(states)]
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		verdicts := Reap(recs, owners)
		benchSink = verdicts
	}
}

// BenchmarkCensusClassification measures per-ref fact classification across closed census classes.
func BenchmarkCensusClassification(b *testing.B) {
	factsCases := []CensusFacts{
		{Live: true, Landed: false},
		{Live: false, Landed: true},
		{PayloadRead: false},
		{PayloadRead: true, PayloadAbsent: 2, PayloadFiles: 5},
		{PayloadRead: true, PayloadDiverged: 1, PayloadFiles: 5},
		{PayloadRead: true, PayloadFiles: 5},
		{PayloadRead: true, Resolved: false},
		{PayloadRead: true, Resolved: true, DeltaEmpty: true},
		{PayloadRead: true, Resolved: true, Subsumed: true},
		{PayloadRead: true, Resolved: true, DeltaEmpty: false, Subsumed: false},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := factsCases[i%len(factsCases)]
		c := Classify(f)
		benchSink = c
	}
}

// BenchmarkBuildCensus measures aggregate report generation from census verdicts.
func BenchmarkBuildCensus(b *testing.B) {
	const count = 100
	verdicts := make([]CensusVerdict, count)
	classes := []CensusClass{
		CensusLive,
		CensusLanded,
		CensusClosedCleanEstimate,
		CensusClosedDirtyRecoverable,
		CensusDiverged,
		CensusUnknown,
	}

	for i := 0; i < count; i++ {
		c := classes[i%len(classes)]
		verdicts[i] = CensusVerdict{
			Session:       fmt.Sprintf("session-%04d", i),
			Ref:           SessionRef(fmt.Sprintf("session-%04d", i)),
			Object:        fmt.Sprintf("obj-%04d", i),
			Class:         c,
			Reason:        CensusReason(c),
			PayloadRead:   true,
			Payload:       3,
			Absent:        1,
			Diverged:      1,
			Landed:        1,
			AbsentPaths:   []string{"path/new.go"},
			DivergedPaths: []string{"path/mod.go"},
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := BuildCensus(verdicts)
		benchSink = rep
	}
}

// BenchmarkDeltaSubsumed measures line-set subsumption testing added and removed delta lines against HEAD.
func BenchmarkDeltaSubsumed(b *testing.B) {
	added := map[string][]string{
		"fileA.go": {"func New() *A {", "    return &A{}", "}"},
		"fileB.go": {"const Version = \"v1.0.0\""},
	}
	removed := map[string][]string{
		"fileA.go": {"// old comment to remove"},
	}
	head := map[string]map[string]bool{
		"fileA.go": {
			"func New() *A {": true,
			"    return &A{}": true,
			"}":               true,
			"package filea":   true,
		},
		"fileB.go": {
			"const Version = \"v1.0.0\"": true,
			"package fileb":              true,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ok := DeltaSubsumed(added, removed, head)
		benchSink = ok
	}
}

// BenchmarkParseNameStatus measures parsing two-dot git diff status output into absent and diverged paths.
func BenchmarkParseNameStatus(b *testing.B) {
	diffOutput := `A	internal/wipref/benchmark_test.go
M	internal/wipref/wipref.go
M	internal/wipref/census.go
D	internal/wipref/obsolete.go
R100	old/path.go	new/path.go
C080	template.go	copy.go
T	symlink.txt
A	cmd/fak/newcmd.go
`

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		absent, diverged := ParseNameStatus(diffOutput)
		_ = absent
		benchSink = diverged
	}
}

// BenchmarkThreeWayConflictMarkers measures detecting git merge conflict markers in file buffers.
func BenchmarkThreeWayConflictMarkers(b *testing.B) {
	cleanData := []byte(strings.Repeat("package main\n\nfunc main() {\n    println(\"hello world\")\n}\n", 20))
	conflictedData := []byte(`package main

func main() {
<<<<<<< HEAD
    println("hello from HEAD")
=======
    println("hello from patch")
>>>>>>> candidate
}
`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if i%2 == 0 {
			benchSink = HasConflictMarkers(cleanData)
		} else {
			benchSink = HasConflictMarkers(conflictedData)
		}
	}
}

// BenchmarkOwnerOfPath measures ownership arbitration of untracked paths across recent session claims.
func BenchmarkOwnerOfPath(b *testing.B) {
	now := int64(1725000000)
	claims := []Claim{
		{Session: "sess-01", CheckpointedAt: now - 300, Live: true},
		{Session: "sess-02", CheckpointedAt: now - 1800, Live: false},
		{Session: "sess-03", CheckpointedAt: now - 7200, Live: false},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		own := OwnerOfPath("internal/wipref/wipref.go", claims, now, DefaultClaimTTL)
		benchSink = own
	}
}

// BenchmarkBuildOwnerReport measures generating an ownership report for multiple created paths.
func BenchmarkBuildOwnerReport(b *testing.B) {
	const pathCount = 50
	now := int64(1725000000)
	paths := make([]string, pathCount)
	claimsMap := make(map[string][]Claim, pathCount)

	for i := 0; i < pathCount; i++ {
		p := fmt.Sprintf("internal/pkg%d/file%d.go", i/5, i)
		paths[i] = p
		claimsMap[p] = []Claim{
			{Session: fmt.Sprintf("sess-%d", i), CheckpointedAt: now - int64(i*60), Live: i%2 == 0},
			{Session: fmt.Sprintf("sess-prior-%d", i), CheckpointedAt: now - int64(i*300), Live: false},
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := BuildOwnerReport(paths, claimsMap, now, DefaultClaimTTL)
		benchSink = rep
	}
}

// BenchmarkExtractTargetPaths measures parsing and normalizing target file paths from tool invocations.
func BenchmarkExtractTargetPaths(b *testing.B) {
	tools := []string{"write_file", "edit", "str_replace_editor", "fak_patch_file"}
	argsJSON := `{"filePath": "C:\\work\\fak\\internal\\wipref\\wipref.go", "file_paths": ["cmd/fak/main.go", "./internal/leaseref/lease.go"]}`

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tool := tools[i%len(tools)]
		paths := ExtractTargetPaths(tool, argsJSON)
		benchSink = paths
	}
}

// BenchmarkFleetPlanPublish measures size-gated publication planning across local checkpoint records.
func BenchmarkFleetPlanPublish(b *testing.B) {
	const count = 100
	recs := make([]RefRecord, count)
	for i := 0; i < count; i++ {
		sess := fmt.Sprintf("session-%04d", i)
		var deltaBytes int64
		if i%4 == 0 {
			deltaBytes = DefaultMaxDeltaBytes + 1024
		} else {
			deltaBytes = 50 * 1024
		}
		recs[i] = RefRecord{
			Ref:    SessionRef(sess),
			Object: fmt.Sprintf("obj-%04d", i),
			Stamp: Stamp{
				SessionID:      sess,
				DeltaBytes:     deltaBytes,
				CheckpointedAt: 1725000000,
			},
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		plan := PlanPublish(recs, DefaultMaxDeltaBytes)
		benchSink = plan
	}
}

// BenchmarkFoldFleet measures compiling mirrored peer checkpoints into a host-grouped fleet report.
func BenchmarkFoldFleet(b *testing.B) {
	const count = 100
	recs := make([]RefRecord, count)
	localSessions := map[string]bool{"session-0001": true, "session-0005": true}
	objectPresent := make(map[string]bool, count)
	now := int64(1725000000)

	for i := 0; i < count; i++ {
		sess := fmt.Sprintf("peer-sess-%04d", i)
		obj := fmt.Sprintf("obj-peer-%04d", i)
		host := fmt.Sprintf("host-node-%02d", i%5)
		recs[i] = RefRecord{
			Ref:    fmt.Sprintf("%sorigin/%s", RemoteMirrorNamespace, sess),
			Object: obj,
			Stamp: Stamp{
				SessionID:      sess,
				Host:           host,
				CheckpointedAt: now - int64(i*120),
				DeltaBytes:     2048,
			},
		}
		objectPresent[obj] = i%3 != 0
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := FoldFleet("origin", recs, localSessions, objectPresent, now)
		benchSink = rep
	}
}

// TestBenchmarkOperationsSanity ensures benchmark fixtures and operations execute without error.
func TestBenchmarkOperationsSanity(t *testing.T) {
	st := Stamp{
		SessionID:      "sess-test-01",
		StartSHA:       "deadbeef",
		Leaves:         []string{"internal/wipref"},
		Buildable:      true,
		CheckpointedAt: 1725000000,
	}
	enc, err := EncodeStamp(st)
	if err != nil {
		t.Fatalf("EncodeStamp failed: %v", err)
	}
	dec, ok := DecodeStamp(enc)
	if !ok || dec.SessionID != st.SessionID {
		t.Fatalf("DecodeStamp roundtrip failed: got %+v, ok=%v", dec, ok)
	}

	recs := []RefRecord{
		{
			Ref:    SessionRef("sess-test-01"),
			Object: "obj-01",
			Stamp:  st,
		},
	}
	rep := Fold(recs, 1725000000)
	if rep.Count != 1 {
		t.Fatalf("Fold count = %d, want 1", rep.Count)
	}

	if !HasConflictMarkers([]byte("<<<<<<<\n=======\n>>>>>>>\n")) {
		t.Fatal("expected conflict markers to be detected")
	}
}
