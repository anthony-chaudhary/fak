package sessionrecovery

import (
	"reflect"
	"testing"
	"time"
)

func TestMergeCodexHostCohortUsesExactStateUUIDAndExecResume(t *testing.T) {
	const id = "91000001-0000-4000-8000-000000000001"
	report := InventoryReport{Sessions: []Session{{
		Thread:     &Thread{ID: id, Source: "cli", CWD: `C:\work\fak`, CreatedAt: "2026-08-24T01:00:01Z"},
		LatestTurn: &Turn{ID: "turn-1", Status: "inProgress", StartedAt: "2026-08-24T01:01:00Z"},
	}}}
	guards := []HostEvidenceRow{{Handle: "guard-handle", TraceID: "trace", PID: 42, StartedAt: "2026-08-24T01:00:00Z", CWD: `C:\work\fak`, Command: []string{"codex"}}}
	cohort := []HostCohortEntry{{Handle: "guard-handle", PID: 42, StartedAt: "2026-08-24T01:00:00Z"}}
	got := MergeCodexHostCohort(report, guards, cohort, map[string]string{"trace": id})
	if len(got.Sessions) != 1 || got.Sessions[0].Thread.ID != id || got.Sessions[0].Category != CategorySubstantive || got.Sessions[0].Action != ActionRecover || got.Sessions[0].IdentityProvenance != IdentityTraceLedger {
		t.Fatalf("merged=%+v", got.Sessions)
	}
	reqs := Select(got, Options{ManagerBin: "fak", CodexBin: "codex", Limit: 1, ReceiptDir: t.TempDir()})
	want := []string{"fak", "guard", "--", "codex", "exec", "--cd", `C:\work\fak`, "resume", id}
	if len(reqs) != 1 || reqs[0].IdentityProvenance != IdentityTraceLedger || !reflect.DeepEqual(reqs[0].Argv, want) {
		t.Fatalf("requests=%+v want=%q", reqs, want)
	}
}

func TestMergeCodexHostCohortNormalizesExtendedWindowsCWDAndRejectsOlderThread(t *testing.T) {
	const oldID = "92500001-0000-4000-8000-000000000001"
	report := InventoryReport{Sessions: []Session{{Thread: &Thread{ID: oldID, CWD: `\\?\C:\Work\Same\`, CreatedAt: "2026-08-24T00:59:59Z"}}}}
	guard := HostEvidenceRow{Handle: "guard", PID: 7, StartedAt: "2026-08-24T01:00:00Z", CWD: `c:/work/same`, Command: []string{"codex"}}
	got := MergeCodexHostCohort(report, []HostEvidenceRow{guard}, []HostCohortEntry{{Handle: "guard", PID: 7, StartedAt: guard.StartedAt}}, nil)
	blocked := got.Sessions[len(got.Sessions)-1]
	if blocked.Thread.ID != "guard" || blocked.Category != CategoryIdentityBlocked || blocked.Reason != "exact_identity_missing" {
		t.Fatalf("pre-existing thread was selected by fallback: %+v", got.Sessions)
	}
	if normalizedCWD(`\\?\C:\Work\Same\`) != normalizedCWD(`c:/work/same`) {
		t.Fatalf("extended Windows cwd did not normalize: %q vs %q", normalizedCWD(`\\?\C:\Work\Same\`), normalizedCWD(`c:/work/same`))
	}
}

func TestMergeHostCohortAccountsOpenCodeAndCoalescesClaudeTrace(t *testing.T) {
	const claudeID = "92600001-0000-4000-8000-000000000001"
	report := InventoryReport{Sessions: []Session{{
		Thread: &Thread{ID: claudeID, Source: "claude_transcript", CWD: `C:\work`}, Provider: ProviderClaude,
		Category: CategoryProbe, Action: ActionExcludeProbe, Reason: "semantic_probe",
	}}}
	guards := []HostEvidenceRow{
		{Handle: "claude-h", TraceID: "claude-trace", PID: 11, StartedAt: "2026-08-24T01:00:00Z", CWD: `C:\work`, Command: []string{"claude"}},
		{Handle: "open-h", PID: 12, StartedAt: "2026-08-24T01:00:00Z", CWD: `C:\work`, Command: []string{"opencode"}},
	}
	cohort := []HostCohortEntry{{Handle: "claude-h", PID: 11, StartedAt: guards[0].StartedAt}, {Handle: "open-h", PID: 12, StartedAt: guards[1].StartedAt}}
	got := MergeCodexHostCohort(report, guards, cohort, map[string]string{"claude-trace": claudeID})
	if len(got.Sessions) != 2 {
		t.Fatalf("host cohort member silently omitted: %+v", got.Sessions)
	}
	claude := got.Sessions[0]
	if claude.HostHandle != "claude-h" || claude.Category != CategoryProbe || claude.Action != ActionExcludeProbe || claude.IdentityProvenance != IdentityTraceLedger {
		t.Fatalf("Claude trace did not coalesce safely: %+v", claude)
	}
	blocked := got.Sessions[1]
	if blocked.Provider != "opencode" || blocked.Category != CategoryIdentityBlocked || blocked.Action != ActionResolveIdentity || blocked.Reason != "exact_resume_provider_blocked:opencode" {
		t.Fatalf("OpenCode cohort row not accounted: %+v", blocked)
	}
}

func TestMergeCodexHostCohortPersistsArgvIdentityProvenance(t *testing.T) {
	const id = "92700001-0000-4000-8000-000000000001"
	report := InventoryReport{Sessions: []Session{{Thread: &Thread{ID: id, Source: "cli", CWD: `C:\work`}}}}
	guard := HostEvidenceRow{Handle: "g", PID: 14, StartedAt: "2026-08-24T01:00:00Z", CWD: `C:\work`, Command: []string{"codex", "exec", "resume", id}}
	got := MergeCodexHostCohort(report, []HostEvidenceRow{guard}, []HostCohortEntry{{Handle: "g", PID: 14, StartedAt: guard.StartedAt}}, nil)
	if got.Sessions[0].IdentityProvenance != IdentityArgvExact {
		t.Fatalf("identity provenance=%q", got.Sessions[0].IdentityProvenance)
	}
}

func TestMergeCodexHostCohortAmbiguousCWDStartIsIdentityBlocked(t *testing.T) {
	report := InventoryReport{Sessions: []Session{
		{Thread: &Thread{ID: "92000001-0000-4000-8000-000000000001", CWD: `C:\work\same`, CreatedAt: "2026-08-24T01:01:00Z"}},
		{Thread: &Thread{ID: "92000002-0000-4000-8000-000000000002", CWD: `C:\work\same`, CreatedAt: "2026-08-24T01:01:00Z"}},
	}}
	guards := []HostEvidenceRow{{Handle: "ambiguous", PID: 7, StartedAt: "2026-08-24T01:00:00Z", CWD: `C:\work\same`, Command: []string{"codex"}}}
	cohort := []HostCohortEntry{{Handle: "ambiguous", PID: 7, StartedAt: "2026-08-24T01:00:00Z"}}
	got := MergeCodexHostCohort(report, guards, cohort, nil)
	blocked := got.Sessions[len(got.Sessions)-1]
	if blocked.Category != CategoryIdentityBlocked || blocked.Action != ActionResolveIdentity || blocked.Reason != "exact_identity_missing" {
		t.Fatalf("blocked=%+v", blocked)
	}
	reqs := Select(got, Options{Limit: 5})
	for _, req := range reqs {
		if req.ThreadID == "ambiguous" && (req.Status != "identity_blocked" || len(req.Argv) != 0) {
			t.Fatalf("blocked row became actionable: %+v", req)
		}
	}
}

func TestMergeCodexHostCohortRejectsPoisonedIdentityNotInState(t *testing.T) {
	const absent = "93000001-0000-4000-8000-000000000001"
	guards := []HostEvidenceRow{{Handle: "g", TraceID: "trace", PID: 9, StartedAt: "2026-08-24T01:00:00Z", CWD: `C:\x`, Command: []string{"codex", "resume", absent}}}
	cohort := []HostCohortEntry{{Handle: "g", PID: 9, StartedAt: "2026-08-24T01:00:00Z"}}
	got := MergeCodexHostCohort(InventoryReport{}, guards, cohort, map[string]string{"trace": absent})
	if len(got.Sessions) != 1 || got.Sessions[0].Category != CategoryIdentityBlocked || got.Sessions[0].Reason != "exact_uuid_not_in_state_5" {
		t.Fatalf("got=%+v", got.Sessions)
	}
}

func TestMergeHostCohortAccountsMissingAndMismatchedGuardEvidence(t *testing.T) {
	cohort := []HostCohortEntry{
		{Handle: "missing", PID: 4, StartedAt: "2026-08-24T01:00:00Z"},
		{Handle: "mismatch", PID: 5, StartedAt: "2026-08-24T01:00:00Z"},
	}
	guards := []HostEvidenceRow{{Handle: "mismatch", PID: 99, StartedAt: cohort[1].StartedAt, Command: []string{"codex"}}}
	got := MergeCodexHostCohort(InventoryReport{}, guards, cohort, nil)
	if len(got.Sessions) != 2 {
		t.Fatalf("cohort members disappeared: %+v", got.Sessions)
	}
	byHandle := map[string]Session{}
	for _, row := range got.Sessions {
		byHandle[row.HostHandle] = row
	}
	if byHandle["missing"].Reason != "host_evidence_missing" || byHandle["mismatch"].Reason != "host_evidence_pid_start_mismatch" {
		t.Fatalf("blocked accounting=%+v", byHandle)
	}
}

func TestMergeHostCohortCoalescesMultipleHandlesForExactUUID(t *testing.T) {
	const id = "92800001-0000-4000-8000-000000000001"
	report := InventoryReport{Sessions: []Session{{Thread: &Thread{ID: id, Source: "cli", CWD: `C:\work`}}}}
	guards := []HostEvidenceRow{
		{Handle: "g1", TraceID: "t1", PID: 1, StartedAt: "2026-08-24T01:00:00Z", Command: []string{"codex"}},
		{Handle: "g2", TraceID: "t2", PID: 2, StartedAt: "2026-08-24T01:00:01Z", Command: []string{"codex"}},
	}
	cohort := []HostCohortEntry{{Handle: "g1", PID: 1, StartedAt: guards[0].StartedAt}, {Handle: "g2", PID: 2, StartedAt: guards[1].StartedAt}}
	got := MergeCodexHostCohort(report, guards, cohort, map[string]string{"t1": id, "t2": id})
	if len(got.Sessions) != 1 || !reflect.DeepEqual(got.Sessions[0].HostHandles, []string{"g1", "g2"}) {
		t.Fatalf("duplicate exact identity was not coalesced: %+v", got.Sessions)
	}
	reqs := Select(got, Options{Limit: 10, ReceiptDir: t.TempDir()})
	if len(reqs) != 1 || reqs[0].ThreadID != id || !reflect.DeepEqual(reqs[0].HostHandles, []string{"g1", "g2"}) {
		t.Fatalf("duplicate launch candidates: %+v", reqs)
	}
	summary := NewSummary("preview", got, reqs, time.Unix(1, 0))
	if len(summary.Results) != 1 || !reflect.DeepEqual(summary.Results[0].HostHandles, []string{"g1", "g2"}) {
		t.Fatalf("host-handle accounting missing from witness: %+v", summary.Results)
	}
}
