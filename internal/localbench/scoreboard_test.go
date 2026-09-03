package localbench

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestScoreboardIntakeValidation(t *testing.T) {
	sb := NewScoreboard(nil)
	r := sampleReceipt(t)

	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}

	// 1. Valid receipt intake
	entry, isDup, err := sb.Intake(raw, time.Now().UTC())
	if err != nil {
		t.Fatalf("Intake failed: %v", err)
	}
	if isDup {
		t.Fatal("expected isDup=false on initial intake")
	}
	if entry.State != ModerationPending {
		t.Fatalf("expected state=pending, got %s", entry.State)
	}
	if entry.TrustStatus != TrustUnsigned {
		t.Fatalf("expected TrustUnsigned, got %s", entry.TrustStatus)
	}

	// 2. Tampered receipt fails closed
	tampered := strings.Replace(string(raw), `"exit_status":0`, `"exit_status":1`, 1)
	if tampered == string(raw) {
		tampered = strings.Replace(string(raw), `"exit_status": 0`, `"exit_status": 1`, 1)
	}
	_, _, err = sb.Intake([]byte(tampered), time.Now().UTC())
	if err == nil || !strings.Contains(err.Error(), "integrity verification failed") {
		t.Fatalf("expected integrity error, got %v", err)
	}

	// 3. Unknown schema fails closed
	badSchema := strings.Replace(string(raw), receiptSchema, "fak.unknown.v99", 1)
	_, _, err = sb.Intake([]byte(badSchema), time.Now().UTC())
	if err == nil || !strings.Contains(err.Error(), "unsupported receipt schema") {
		t.Fatalf("expected unsupported schema error, got %v", err)
	}

	// 4. Oversized payload fails closed
	sb.SetMaxPayloadBytes(100)
	_, _, err = sb.Intake(raw, time.Now().UTC())
	if err == nil || !strings.Contains(err.Error(), "exceeds limit") {
		t.Fatalf("expected exceeds limit error, got %v", err)
	}
}

func TestScoreboardAttestedIntake(t *testing.T) {
	r := sampleReceipt(t)
	pub, priv, err := GenerateSigningKey()
	if err != nil {
		t.Fatal(err)
	}

	bindings := sampleBindings(r.Integrity.SHA256)
	createdAt := time.Date(2026, 8, 31, 12, 0, 0, 0, time.UTC)
	env, err := SignReceipt(r, bindings, priv, "operator-key-alpha", createdAt, time.Time{})
	if err != nil {
		t.Fatal(err)
	}

	store := NewTrustStore()
	store.AddTrustedKey("operator-key-alpha", pub)

	sb := NewScoreboard(store)
	rawEnv, err := json.Marshal(env)
	if err != nil {
		t.Fatal(err)
	}

	entry, isDup, err := sb.Intake(rawEnv, createdAt)
	if err != nil {
		t.Fatalf("attested intake: %v", err)
	}
	if isDup {
		t.Fatal("unexpected dup")
	}
	if entry.TrustStatus != TrustVerified {
		t.Fatalf("expected TrustVerified, got %s", entry.TrustStatus)
	}
	if entry.Attestation == nil {
		t.Fatal("expected attestation to be populated")
	}

	proj, err := sb.Project(entry.ID)
	if err != nil {
		t.Fatal(err)
	}
	if proj.Backend != "metal" || proj.ModelName != "unsloth/Qwen3.8-27B-GGUF" || !proj.QualityPass {
		t.Fatalf("unexpected projection: %+v", proj)
	}
}

func TestScoreboardModerationTransitions(t *testing.T) {
	sb := NewScoreboard(nil)
	r := sampleReceipt(t)
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}

	entry, _, err := sb.Intake(raw, time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}

	// 1. Pending -> Accepted
	entry, err = sb.Moderate(entry.ID, ModerationAccepted, "moderator@example.com", "verified hardware and numbers", time.Now().UTC())
	if err != nil {
		t.Fatalf("Moderate to accepted: %v", err)
	}
	if entry.State != ModerationAccepted {
		t.Fatalf("expected accepted, got %s", entry.State)
	}
	if len(entry.History) != 2 {
		t.Fatalf("history length %d, want 2", len(entry.History))
	}

	// 2. Accepted -> Quarantined
	entry, err = sb.Moderate(entry.ID, ModerationQuarantined, "lead-mod@example.com", "community report under review", time.Now().UTC())
	if err != nil {
		t.Fatalf("Moderate to quarantined: %v", err)
	}
	if entry.State != ModerationQuarantined {
		t.Fatalf("expected quarantined, got %s", entry.State)
	}

	// 3. Quarantined -> Rejected
	entry, err = sb.Moderate(entry.ID, ModerationRejected, "lead-mod@example.com", "repro failed on matched box", time.Now().UTC())
	if err != nil {
		t.Fatalf("Moderate to rejected: %v", err)
	}
	if entry.State != ModerationRejected {
		t.Fatalf("expected rejected, got %s", entry.State)
	}

	// 4. Invalid transition: cannot transition to same state
	_, err = sb.Moderate(entry.ID, ModerationRejected, "mod", "again", time.Now().UTC())
	if err == nil {
		t.Fatal("expected error on transition to same state")
	}

	// 5. Invalid transition: cannot transition with empty actor or reason
	_, err = sb.Moderate(entry.ID, ModerationAccepted, "", "reason", time.Now().UTC())
	if err == nil {
		t.Fatal("expected error on empty actor")
	}
	_, err = sb.Moderate(entry.ID, ModerationAccepted, "mod", "", time.Now().UTC())
	if err == nil {
		t.Fatal("expected error on empty reason")
	}
}

func TestScoreboardDeterministicDedup(t *testing.T) {
	sb := NewScoreboard(nil)
	r1 := sampleReceipt(t)
	raw1, _ := json.Marshal(r1)

	// Ingest first time
	entry1, isDup1, err := sb.Intake(raw1, time.Now().UTC())
	if err != nil || isDup1 {
		t.Fatalf("first intake: err=%v isDup=%v", err, isDup1)
	}

	// Ingest exact same receipt second time
	entry2, isDup2, err := sb.Intake(raw1, time.Now().UTC())
	if err != nil {
		t.Fatalf("second intake: %v", err)
	}
	if !isDup2 {
		t.Fatal("expected isDup=true on re-submitting exact same receipt")
	}
	if entry1.ID != entry2.ID {
		t.Fatalf("dedup ID mismatch: %s vs %s", entry1.ID, entry2.ID)
	}

	// A different run produces a distinct receipt with a distinct dedup key
	r2 := sampleReceipt(t)
	r2.Output = "different output bytes"
	r2.OutputSHA256 = strings.Repeat("b", 64)
	if err := seal(&r2); err != nil {
		t.Fatal(err)
	}
	raw2, _ := json.Marshal(r2)

	entry3, isDup3, err := sb.Intake(raw2, time.Now().UTC())
	if err != nil || isDup3 {
		t.Fatalf("third intake: err=%v isDup=%v", err, isDup3)
	}
	if entry3.ID == entry1.ID {
		t.Fatal("distinct runs should produce distinct dedup keys")
	}
}

func TestScoreboardPrivacySafeProjection(t *testing.T) {
	sb := NewScoreboard(nil)
	r := sampleReceipt(t)
	r.Output = "SUPER_SECRET_TOKEN=xyz123\nC:\\Users\\alice\\private\\data"
	r.Hardware.CPU = "<script>alert('xss')</script> Apple M3"
	if err := seal(&r); err != nil {
		t.Fatal(err)
	}

	raw, _ := json.Marshal(r)
	entry, _, err := sb.Intake(raw, time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}

	proj, err := sb.Project(entry.ID)
	if err != nil {
		t.Fatal(err)
	}

	projBytes, _ := json.Marshal(proj)
	projStr := string(projBytes)

	// Raw output must be completely absent from projection
	if strings.Contains(projStr, "SUPER_SECRET_TOKEN") || strings.Contains(projStr, "xyz123") {
		t.Fatalf("projection leaked secret: %s", projStr)
	}
	if strings.Contains(projStr, "alice") {
		t.Fatalf("projection leaked user path: %s", projStr)
	}
	// HTML characters must be escaped
	if strings.Contains(proj.HardwareCPU, "<script>") {
		t.Fatalf("projection leaked raw HTML/script tag: %s", proj.HardwareCPU)
	}
	if !strings.Contains(proj.HardwareCPU, "&lt;script&gt;") {
		t.Fatalf("projection did not escape script tag: %s", proj.HardwareCPU)
	}
}

func TestScoreboardFiltering(t *testing.T) {
	sb := NewScoreboard(nil)
	r := sampleReceipt(t)
	raw, _ := json.Marshal(r)
	entry, _, err := sb.Intake(raw, time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}

	// Unfiltered list returns the entry
	all := sb.ListProjections(FilterCriteria{})
	if len(all) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(all))
	}

	// Filter by state accepted (currently pending) -> 0
	accepted := sb.ListProjections(FilterCriteria{State: ModerationAccepted})
	if len(accepted) != 0 {
		t.Fatalf("expected 0 accepted entries, got %d", len(accepted))
	}

	// Moderate to accepted
	if _, err := sb.Moderate(entry.ID, ModerationAccepted, "mod", "ok", time.Now().UTC()); err != nil {
		t.Fatal(err)
	}

	// Filter by state accepted -> 1
	accepted = sb.ListProjections(FilterCriteria{State: ModerationAccepted})
	if len(accepted) != 1 {
		t.Fatalf("expected 1 accepted entry, got %d", len(accepted))
	}

	// Filter by matching OS -> 1
	byOS := sb.ListProjections(FilterCriteria{HardwareOS: "darwin"})
	if len(byOS) != 1 {
		t.Fatalf("expected 1 entry for darwin, got %d", len(byOS))
	}

	// Filter by mismatched OS -> 0
	byWindows := sb.ListProjections(FilterCriteria{HardwareOS: "windows"})
	if len(byWindows) != 0 {
		t.Fatalf("expected 0 entries for windows, got %d", len(byWindows))
	}
}
