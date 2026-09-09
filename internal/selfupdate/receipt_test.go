package selfupdate

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

func TestReceiptJSONSchemaAndFieldNames(t *testing.T) {
	oldRev := "old-sha"
	newRev := "new-sha"
	receipt := Receipt{
		Schema:          ReceiptSchema,
		SchemaVersion:   ReceiptSchemaVersion,
		CorrelationID:   "corr-test-1",
		Status:          "updated",
		OldRevision:     &oldRev,
		NewRevision:     &newRev,
		Targets:         []ReceiptTarget{{Role: "primary", Path: "bin/fak"}},
		Attempted:       1,
		Changed:         1,
		RollbackStatus:  "not_attempted",
		RollbackErrors:  []string{},
		RestartRequired: false,
		NextCommand:     "fak version",
		Detail:          "installed cleanly",
		BuildProvenance: &BuildProvenance{
			SourceCommit:         "src-commit",
			ArtifactSourceCommit: "art-commit",
			BuildInputDigest:     "input-digest",
			BuildEnvelope:        map[string]string{"GOOS": "darwin"},
			ArtifactDigest:       "art-digest",
			ArtifactSize:         1024,
			AppVersion:           "1.0.0",
			Reused:               false,
		},
		Transfer: &TransferReceipt{
			ChosenPath:     "delta",
			DeltaBytes:     256,
			FullBytes:      1024,
			TotalMS:        50,
			Verification:   "verified",
			FallbackReason: "",
			FallbackBytes:  0,
			FallbackMS:     0,
		},
		Handoff: &HandoffReceipt{
			State:             selfinstall.HandoffState("completed"),
			SessionID:         "sess-123",
			SuccessorRevision: "new-sha",
			Detail:            "",
		},
		CandidateCache: &CandidateCacheDisposition{
			State:  CandidateCacheHit,
			Reason: "",
		},
		TotalMS: 150,
		PhaseMS: PhaseMS{
			Check:     10,
			Lock:      5,
			Cleanup:   2,
			Prepare:   15,
			Companion: 3,
			Build:     60,
			Vet:       20,
			Smoke:     15,
			Install:   10,
			Verify:    5,
			Handoff:   5,
		},
	}

	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("Unmarshal raw failed: %v", err)
	}

	expectedKeys := []string{
		"schema", "schema_version", "correlation_id", "status",
		"old_revision", "new_revision", "targets", "attempted",
		"changed", "rollback_status", "rollback_errors",
		"restart_required", "next_command", "detail",
		"build_provenance", "transfer", "handoff", "candidate_cache",
		"total_ms", "phase_ms",
	}

	for _, k := range expectedKeys {
		if _, ok := raw[k]; !ok {
			t.Errorf("missing expected JSON key %q in receipt", k)
		}
	}

	// Verify unmarshaling back into Receipt struct
	var roundTrip Receipt
	if err := json.Unmarshal(data, &roundTrip); err != nil {
		t.Fatalf("Unmarshal back to Receipt failed: %v", err)
	}

	if roundTrip.Schema != ReceiptSchema || roundTrip.SchemaVersion != ReceiptSchemaVersion {
		t.Fatalf("schema mismatch: got %s v%d, want %s v%d", roundTrip.Schema, roundTrip.SchemaVersion, ReceiptSchema, ReceiptSchemaVersion)
	}
	if roundTrip.CorrelationID != "corr-test-1" || roundTrip.Status != "updated" {
		t.Fatalf("fields mismatch: %+v", roundTrip)
	}
	if *roundTrip.OldRevision != oldRev || *roundTrip.NewRevision != newRev {
		t.Fatalf("revision mismatch: old=%v, new=%v", roundTrip.OldRevision, roundTrip.NewRevision)
	}
	if len(roundTrip.Targets) != 1 || roundTrip.Targets[0].Role != "primary" {
		t.Fatalf("targets mismatch: %+v", roundTrip.Targets)
	}
	if roundTrip.BuildProvenance == nil || roundTrip.BuildProvenance.SourceCommit != "src-commit" {
		t.Fatalf("build provenance mismatch: %+v", roundTrip.BuildProvenance)
	}
	if roundTrip.Transfer == nil || roundTrip.Transfer.ChosenPath != "delta" {
		t.Fatalf("transfer receipt mismatch: %+v", roundTrip.Transfer)
	}
	if roundTrip.Handoff == nil || roundTrip.Handoff.SessionID != "sess-123" {
		t.Fatalf("handoff receipt mismatch: %+v", roundTrip.Handoff)
	}
	if roundTrip.CandidateCache == nil || roundTrip.CandidateCache.State != CandidateCacheHit {
		t.Fatalf("candidate cache mismatch: %+v", roundTrip.CandidateCache)
	}
}

func TestReceiptJSONOmitempty(t *testing.T) {
	receipt := Receipt{
		Schema:         ReceiptSchema,
		SchemaVersion:  ReceiptSchemaVersion,
		CorrelationID:  "corr-minimal",
		Status:         "current",
		Targets:        []ReceiptTarget{},
		RollbackErrors: []string{},
		NextCommand:    "fak version",
		RollbackStatus: "not_attempted",
	}

	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	rawJSON := string(data)
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	omittedKeys := []string{
		"detail",
		"build_provenance",
		"transfer",
		"handoff",
		"candidate_cache",
	}
	for _, key := range omittedKeys {
		if _, ok := raw[key]; ok {
			t.Errorf("expected key %q to be omitted, but found in JSON: %s", key, string(data))
		}
	}

	// Check that targets and rollback_errors are serialized as empty arrays []
	if !strings.Contains(rawJSON, `"targets":[]`) {
		t.Errorf("expected targets to be empty array, got: %s", rawJSON)
	}
	if !strings.Contains(rawJSON, `"rollback_errors":[]`) {
		t.Errorf("expected rollback_errors to be empty array, got: %s", rawJSON)
	}
}

func TestBuilderDefaultsAndReset(t *testing.T) {
	b := NewBuilder()
	if b.targets == nil || len(b.targets) != 0 {
		t.Fatalf("NewBuilder targets = %v, want empty slice", b.targets)
	}
	if b.rollbackErrors == nil || len(b.rollbackErrors) != 0 {
		t.Fatalf("NewBuilder rollbackErrors = %v, want empty slice", b.rollbackErrors)
	}

	b.SetOldRevision("old").
		SetNewRevision("new").
		AddTarget(ReceiptTarget{Role: "primary", Path: "bin/fak"}).
		SetAttempted(2).
		SetChanged(1).
		SetDetail("some detail").
		SetCorrelationID("custom-id")

	r1 := b.Build(OutcomeInstalled, "", "")
	if r1.CorrelationID != "custom-id" || r1.Status != "updated" || len(r1.Targets) != 1 {
		t.Fatalf("unexpected built receipt: %+v", r1)
	}

	b.Reset()
	if b.oldRevision != "" || b.newRevision != "" || len(b.targets) != 0 || b.attempted != 0 || b.changed != 0 || b.detail != "" || b.correlationID != "" {
		t.Fatalf("Reset did not clear fields: %+v", b)
	}
}

func TestBuilderSettersAndAccumulation(t *testing.T) {
	prov := &BuildProvenance{SourceCommit: "src"}
	trans := &TransferReceipt{ChosenPath: "full"}
	handoff := &HandoffReceipt{SessionID: "s1"}
	cache := &CandidateCacheDisposition{State: CandidateCacheMiss}

	b := NewBuilder().
		SetRevisions("rev-old", "rev-new").
		SetCounts(3, 2).
		SetTargets([]ReceiptTarget{{Role: "w1", Path: "bin/w1"}}).
		AddTarget(ReceiptTarget{Role: "w2", Path: "bin/w2"}).
		SetRollbackStatus("custom-rollback").
		SetRollbackErrors([]string{"err1", "err2"}).
		SetRestartRequired(true).
		SetNextCommand("custom-cmd").
		SetDetail("builder detail").
		SetBuildProvenance(prov).
		SetTransfer(trans).
		SetHandoff(handoff).
		SetCandidateCache(cache).
		SetTiming(1234, PhaseMS{Check: 100})

	r := b.Build(OutcomeInstalled, "", "")

	if *r.OldRevision != "rev-old" || *r.NewRevision != "rev-new" {
		t.Errorf("revisions mismatch: old=%v, new=%v", r.OldRevision, r.NewRevision)
	}
	if r.Attempted != 3 || r.Changed != 2 {
		t.Errorf("counts mismatch: attempted=%d, changed=%d", r.Attempted, r.Changed)
	}
	if len(r.Targets) != 2 || r.Targets[0].Role != "w1" || r.Targets[1].Role != "w2" {
		t.Errorf("targets mismatch: %+v", r.Targets)
	}
	if r.RollbackStatus != "custom-rollback" {
		t.Errorf("rollback status mismatch: %s", r.RollbackStatus)
	}
	if !reflect.DeepEqual(r.RollbackErrors, []string{"err1", "err2"}) {
		t.Errorf("rollback errors mismatch: %v", r.RollbackErrors)
	}
	if !r.RestartRequired {
		t.Errorf("restart required = false, want true")
	}
	if r.NextCommand != "custom-cmd" {
		t.Errorf("next command = %s, want custom-cmd", r.NextCommand)
	}
	if r.Detail != "builder detail" {
		t.Errorf("detail = %s, want builder detail", r.Detail)
	}
	if r.BuildProvenance != prov || r.Transfer != trans || r.Handoff != handoff || r.CandidateCache != cache {
		t.Errorf("pointer fields mismatch")
	}
	if r.TotalMS != 1234 || r.PhaseMS.Check != 100 {
		t.Errorf("timing mismatch: total=%d, check=%d", r.TotalMS, r.PhaseMS.Check)
	}
}

func TestBuilderNilSliceHandling(t *testing.T) {
	b := NewBuilder()
	b.SetTargets(nil)
	b.SetRollbackErrors(nil)

	r := b.Build(OutcomeMetadataOnly, "", "")
	if r.Targets == nil {
		t.Errorf("Targets is nil, want non-nil empty slice")
	}
	if r.RollbackErrors == nil {
		t.Errorf("RollbackErrors is nil, want non-nil empty slice")
	}
}

func TestClassifyOutcome(t *testing.T) {
	cases := []struct {
		cause          Outcome
		oldRev         string
		newRev         string
		detail         string
		wantStatus     string
		wantRollback   string
		wantRestart    bool
		wantNext       string
		wantErrorCount int
	}{
		{
			cause:        OutcomeInstalled,
			wantStatus:   "updated",
			wantRollback: "not_attempted",
			wantNext:     "fak version",
		},
		{
			cause:        OutcomeMetadataOnly,
			wantStatus:   "current",
			wantRollback: "not_attempted",
			wantNext:     "fak version",
		},
		{
			cause:        OutcomeGateFailed,
			wantStatus:   "gate_failed",
			wantRollback: "not_attempted",
			wantNext:     "fak self-update",
		},
		{
			cause:        OutcomePrepareFailed,
			wantStatus:   "prepare_failed",
			wantRollback: "not_attempted",
			wantNext:     "fak self-update",
		},
		{
			cause:        OutcomePinSkew,
			wantStatus:   "pin_skew",
			wantRollback: "not_attempted",
			wantNext:     "fak self-update --check",
		},
		{
			cause:        OutcomeRolledBack,
			wantStatus:   "rolled_back",
			wantRollback: "succeeded",
			wantNext:     "fak self-update",
		},
		{
			cause:          OutcomeRollbackFailed,
			detail:         "permission denied while restoring binary",
			wantStatus:     "rollback_failed",
			wantRollback:   "failed",
			wantNext:       "fak self-update --check",
			wantErrorCount: 1,
		},
		{
			cause:        OutcomeBusy,
			wantStatus:   "busy",
			wantRollback: "not_attempted",
			wantNext:     "fak self-update",
		},
		{
			cause:        OutcomeCheckOnly,
			oldRev:       "rev-a",
			newRev:       "rev-b",
			wantStatus:   "stale",
			wantRollback: "not_attempted",
			wantNext:     "fak self-update",
		},
		{
			cause:        OutcomeCheckOnly,
			oldRev:       "rev-same",
			newRev:       "rev-same",
			wantStatus:   "current",
			wantRollback: "not_attempted",
			wantNext:     "fak version",
		},
		{
			cause:        OutcomeCheckOnly,
			oldRev:       "",
			newRev:       "rev-b",
			wantStatus:   "current",
			wantRollback: "not_attempted",
			wantNext:     "fak version",
		},
		{
			cause:        OutcomeHotCopyDivergent,
			wantStatus:   "divergent",
			wantRollback: "not_attempted",
			wantNext:     "fak self-update",
		},
		{
			cause:        OutcomeHandoffRefused,
			wantStatus:   "handoff_refused",
			wantRollback: "not_attempted",
			wantNext:     "fak self-update --check",
		},
		{
			cause:        OutcomeRestartRequired,
			wantStatus:   "restart_required",
			wantRollback: "not_attempted",
			wantRestart:  true,
			wantNext:     "fak self-update --check",
		},
	}

	for _, tc := range cases {
		t.Run(string(tc.cause), func(t *testing.T) {
			status, rollbackStatus, restartRequired, nextCommand, rollbackErrors := ClassifyOutcome(tc.cause, tc.oldRev, tc.newRev, tc.detail)
			if status != tc.wantStatus {
				t.Errorf("status = %q, want %q", status, tc.wantStatus)
			}
			if rollbackStatus != tc.wantRollback {
				t.Errorf("rollbackStatus = %q, want %q", rollbackStatus, tc.wantRollback)
			}
			if restartRequired != tc.wantRestart {
				t.Errorf("restartRequired = %v, want %v", restartRequired, tc.wantRestart)
			}
			if nextCommand != tc.wantNext {
				t.Errorf("nextCommand = %q, want %q", nextCommand, tc.wantNext)
			}
			if len(rollbackErrors) != tc.wantErrorCount {
				t.Errorf("rollbackErrors count = %d, want %d", len(rollbackErrors), tc.wantErrorCount)
			}
		})
	}
}

func TestPhaseMSAndPhaseOrder(t *testing.T) {
	expectedOrder := []Phase{
		PhaseCheck,
		PhaseLock,
		PhaseCleanup,
		PhasePrepare,
		PhaseCompanion,
		PhaseBuild,
		PhaseVet,
		PhaseSmoke,
		PhaseInstall,
		PhaseVerify,
		PhaseHandoff,
	}

	if len(PhaseOrder) != len(expectedOrder) {
		t.Fatalf("PhaseOrder len = %d, want %d", len(PhaseOrder), len(expectedOrder))
	}
	for i, phase := range PhaseOrder {
		if phase != expectedOrder[i] {
			t.Errorf("PhaseOrder[%d] = %q, want %q", i, phase, expectedOrder[i])
		}
	}

	var p PhaseMS
	for i, phase := range PhaseOrder {
		val := int64((i + 1) * 10)
		p.Set(phase, val)
	}

	if p.Check != 10 || p.Lock != 20 || p.Cleanup != 30 || p.Prepare != 40 ||
		p.Companion != 50 || p.Build != 60 || p.Vet != 70 || p.Smoke != 80 ||
		p.Install != 90 || p.Verify != 100 || p.Handoff != 110 {
		t.Fatalf("PhaseMS fields unexpected: %+v", p)
	}
}

func TestOptionalRevision(t *testing.T) {
	if got := OptionalRevision(""); got != nil {
		t.Errorf("OptionalRevision(\"\") = %v, want nil", got)
	}
	if got := OptionalRevision("   \t\n "); got != nil {
		t.Errorf("OptionalRevision(whitespace) = %v, want nil", got)
	}
	rev := "abc1234"
	got := OptionalRevision(rev)
	if got == nil || *got != rev {
		t.Errorf("OptionalRevision(%q) = %v, want %q", rev, got, rev)
	}
}

func TestNormalizeTargets(t *testing.T) {
	// Empty targets and no fallback
	res := NormalizeTargets(nil, "")
	if res == nil || len(res) != 0 {
		t.Errorf("NormalizeTargets(nil, \"\") = %v, want empty non-nil slice", res)
	}

	// Empty targets with <self> fallback
	res = NormalizeTargets(nil, "<self>")
	if len(res) != 0 {
		t.Errorf("NormalizeTargets(nil, \"<self>\") = %v, want empty slice", res)
	}

	// Empty targets with valid fallback path
	res = NormalizeTargets(nil, filepath.Join("path", "to", "fak"))
	if len(res) != 1 || res[0].Role != "primary" || res[0].Path != "path/to/fak" {
		t.Errorf("NormalizeTargets fallback = %+v, want primary path/to/fak", res)
	}

	// Non-empty targets ignore fallback
	existing := []ReceiptTarget{{Role: "secondary", Path: "other/path"}}
	res = NormalizeTargets(existing, "fallback/fak")
	if len(res) != 1 || res[0].Role != "secondary" {
		t.Errorf("NormalizeTargets preserved = %+v, want secondary", res)
	}
}
