package experiments

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func number(v float64) *float64 { return &v }

func testDigest(value string) string {
	sum := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func measuredReceipt(id string, verdict Verdict) Receipt {
	return Receipt{
		Schema:            ReceiptSchema,
		ID:                id,
		RecordedAt:        "2026-08-27T12:00:00Z",
		EvidenceClass:     EvidenceClassScreening,
		Hypothesis:        "candidate improves throughput",
		Verdict:           verdict,
		Baseline:          "sequential",
		Candidate:         "compatibility batch",
		Metric:            Metric{Name: "aggregate rate", Unit: "token_steps/s", BaselineValue: number(5.858), CandidateValue: number(3.159)},
		Revision:          "git:abc123",
		Environment:       "test-host CPU-reference",
		EnvironmentDigest: DigestEnvironment("test-host CPU-reference"),
		ArtifactDigest:    testDigest("artifact fixture"),
		Scope:             "CPU-reference Qwen2.5-0.5B Q8_0 mixed-length fixture",
		NextAction:        "retry with length-aware buckets on native CUDA",
	}
}

func identityFor(r Receipt) ReceiptIdentity {
	return ReceiptIdentity{
		Hypothesis: r.Hypothesis, Revision: r.Revision,
		Environment:       r.Environment,
		EnvironmentDigest: r.EnvironmentDigest, ArtifactDigest: r.ArtifactDigest,
	}
}

func TestLookupReceiptExactMatchAndIdentityMismatch(t *testing.T) {
	receipt := measuredReceipt("r1", VerdictLost)
	exact := LookupReceipt([]Receipt{receipt}, identityFor(receipt))
	if exact.Status != LookupExact || exact.Receipt == nil || exact.Receipt.ID != "r1" || !exact.MeasuredLoss || exact.ClaimEligible {
		t.Fatalf("exact lookup = %#v", exact)
	}

	query := identityFor(receipt)
	query.Environment = "different-host"
	query.EnvironmentDigest = DigestEnvironment(query.Environment)
	mismatch := LookupReceipt([]Receipt{receipt}, query)
	if mismatch.Status != LookupIdentityMismatch || mismatch.Receipt != nil || mismatch.MeasuredLoss {
		t.Fatalf("identity mismatch = %#v", mismatch)
	}
}

func TestInconclusiveAndInvalidAreRecordableButNeverMeasuredLosses(t *testing.T) {
	for _, verdict := range []Verdict{VerdictInconclusive, VerdictInvalid} {
		receipt := measuredReceipt("r-"+string(verdict), verdict)
		receipt.Baseline = ""
		receipt.Candidate = ""
		receipt.Metric = Metric{}
		receipt.Reason = "measurement interrupted before comparison completed"
		if err := receipt.Validate(); err != nil {
			t.Fatalf("%s should be recordable: %v", verdict, err)
		}
		result := LookupReceipt([]Receipt{receipt}, identityFor(receipt))
		if result.Status != LookupExact || result.MeasuredLoss {
			t.Fatalf("%s lookup = %#v", verdict, result)
		}
	}
}

func TestReceiptValidationKeepsMeasuredVerdictsRigorous(t *testing.T) {
	receipt := measuredReceipt("r1", VerdictLost)
	receipt.Metric.Unit = ""
	receipt.ArtifactDigest = ""
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "metric.unit") || !strings.Contains(err.Error(), "artifact_digest") {
		t.Fatalf("expected missing comparison identity fields, got %v", err)
	}

	receipt = measuredReceipt("r2", VerdictInvalid)
	receipt.Reason = ""
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "reason") {
		t.Fatalf("expected invalid receipt reason error, got %v", err)
	}
}

func TestLookupReceiptResolvesSupersedingReceipt(t *testing.T) {
	first := measuredReceipt("r1", VerdictLost)
	superseding := measuredReceipt("r2", VerdictInconclusive)
	superseding.Supersedes = "r1"
	superseding.Reason = "fixture identity was later found incomplete"
	result := LookupReceipt([]Receipt{first, superseding}, identityFor(first))
	if result.Receipt == nil || result.Receipt.ID != "r2" || result.MeasuredLoss {
		t.Fatalf("superseding receipt should replace prior loss without rewriting it: %#v", result)
	}
}

func TestParseReceiptLedgerRejectsInvalidLine(t *testing.T) {
	_, err := ParseReceiptLedger(`{"schema":"fak-experiment-receipt/1","id":"bad"}`)
	if err == nil || !strings.Contains(err.Error(), "line 1") {
		t.Fatalf("expected line-scoped validation error, got %v", err)
	}
}

func TestReceiptValidationRejectsMalformedTimestampAndDigests(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Receipt)
		want   string
	}{
		{"timestamp", func(r *Receipt) { r.RecordedAt = "yesterday" }, "RFC3339"},
		{"artifact digest", func(r *Receipt) { r.ArtifactDigest = "sha256:not-a-digest" }, "artifact_digest"},
		{"uppercase digest", func(r *Receipt) { r.ArtifactDigest = "sha256:" + strings.Repeat("A", 64) }, "artifact_digest"},
		{"environment digest", func(r *Receipt) { r.EnvironmentDigest = testDigest("different environment") }, "does not match"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := measuredReceipt("bad", VerdictLost)
			tt.mutate(&r)
			if err := r.Validate(); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestParseReceiptLedgerRejectsUnknownFields(t *testing.T) {
	b, err := json.Marshal(measuredReceipt("r1", VerdictLost))
	if err != nil {
		t.Fatal(err)
	}
	line := strings.TrimSuffix(string(b), "}") + `,"unknown":true}`
	if _, err := ParseReceiptLedger(line); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("expected unknown-field rejection, got %v", err)
	}
}

func TestAppendReceiptRejectsDuplicateAndDanglingSupersedes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipts.jsonl")
	first := measuredReceipt("r1", VerdictLost)
	if err := AppendReceipt(path, first); err != nil {
		t.Fatal(err)
	}
	if err := AppendReceipt(path, first); err == nil || !strings.Contains(err.Error(), "duplicate receipt id") {
		t.Fatalf("expected duplicate rejection, got %v", err)
	}
	dangling := measuredReceipt("r2", VerdictInconclusive)
	dangling.Reason = "follow-up could not run"
	dangling.Supersedes = "absent"
	if err := AppendReceipt(path, dangling); err == nil || !strings.Contains(err.Error(), "does not exist or is not active") {
		t.Fatalf("expected dangling supersedes rejection, got %v", err)
	}
}

func TestAppendReceiptRejectsInactiveSupersedesTarget(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipts.jsonl")
	first := measuredReceipt("r1", VerdictLost)
	second := measuredReceipt("r2", VerdictInconclusive)
	second.Reason = "identity invalidated"
	second.Supersedes = "r1"
	if err := AppendReceipt(path, first); err != nil {
		t.Fatal(err)
	}
	if err := AppendReceipt(path, second); err != nil {
		t.Fatal(err)
	}
	third := measuredReceipt("r3", VerdictInvalid)
	third.Reason = "duplicate invalidation"
	third.Supersedes = "r1"
	if err := AppendReceipt(path, third); err == nil || !strings.Contains(err.Error(), "not active") {
		t.Fatalf("expected inactive target rejection, got %v", err)
	}
}

func TestAppendReceiptRejectsMissingTerminalNewline(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipts.jsonl")
	b, err := json.Marshal(measuredReceipt("r1", VerdictLost))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := AppendReceipt(path, measuredReceipt("r2", VerdictWon)); err == nil || !strings.Contains(err.Error(), "terminal newline") {
		t.Fatalf("expected terminal-newline rejection, got %v", err)
	}
}

func TestAppendReceiptReportsLockContention(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipts.jsonl")
	lockPath := path + ".lock"
	lock, err := acquireLedgerLock(lockPath, time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}
	defer lock.release()
	err = AppendReceipt(path, measuredReceipt("r1", VerdictLost))
	var busy *LedgerBusyError
	if !errors.As(err, &busy) || busy.LockPath != lockPath {
		t.Fatalf("expected typed busy error for %s, got %v", lockPath, err)
	}
}

func TestReceiptLineSizeBoundaryAndOversize(t *testing.T) {
	receipt := measuredReceipt("limit", VerdictLost)
	receipt.Scope = ""
	base, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	receipt.Scope = strings.Repeat("x", MaxReceiptLineBytes-len(base))
	line, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if len(line) != MaxReceiptLineBytes {
		t.Fatalf("boundary receipt length = %d, want %d", len(line), MaxReceiptLineBytes)
	}
	path := filepath.Join(t.TempDir(), "boundary.jsonl")
	if err := AppendReceipt(path, receipt); err != nil {
		t.Fatalf("boundary append: %v", err)
	}
	if rows, err := ReadReceiptLedger(path); err != nil || len(rows) != 1 {
		t.Fatalf("boundary read rows=%d err=%v", len(rows), err)
	}

	receipt.Scope += "x"
	oversizePath := filepath.Join(t.TempDir(), "oversize.jsonl")
	if err := AppendReceipt(oversizePath, receipt); err == nil || !strings.Contains(err.Error(), "maximum") {
		t.Fatalf("expected oversized append rejection, got %v", err)
	}
	oversizeLine, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(oversizePath, append(oversizeLine, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadReceiptLedger(oversizePath); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("expected oversized read rejection, got %v", err)
	}
}

func TestReadReceiptLedgerRejectsCorruptHistory(t *testing.T) {
	first := measuredReceipt("r1", VerdictLost)
	second := measuredReceipt("r2", VerdictInconclusive)
	second.Reason = "superseding probe"
	second.Supersedes = "r1"
	third := measuredReceipt("r3", VerdictInvalid)
	third.Reason = "second supersession"
	third.Supersedes = "r1"
	dangling := measuredReceipt("dangling", VerdictInvalid)
	dangling.Reason = "missing target"
	dangling.Supersedes = "absent"
	tests := []struct {
		name string
		rows []Receipt
		want string
	}{
		{"duplicate id", []Receipt{first, first}, "duplicate receipt id"},
		{"dangling supersedes", []Receipt{dangling}, "does not exist or is not active"},
		{"inactive supersedes", []Receipt{first, second, third}, "does not exist or is not active"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "receipts.jsonl")
			var content strings.Builder
			for _, receipt := range tt.rows {
				b, err := json.Marshal(receipt)
				if err != nil {
					t.Fatal(err)
				}
				content.Write(b)
				content.WriteByte('\n')
			}
			if err := os.WriteFile(path, []byte(content.String()), 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := ReadReceiptLedger(path); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("ReadReceiptLedger() error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestAppendReceiptRecoversExpiredOwnerLease(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipts.jsonl")
	lockPath := path + ".lock"
	old := time.Now().UTC().Add(-2 * ledgerLockLease)
	metadata := ledgerLockMetadata{
		Owner: strings.Repeat("a", 32), PID: 1, Host: "test",
		CreatedAt: old.Format(time.RFC3339Nano), ExpiresAt: old.Add(ledgerLockLease).Format(time.RFC3339Nano),
	}
	b, err := json.Marshal(metadata)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(lockPath, append(b, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := AppendReceipt(path, measuredReceipt("recovered", VerdictLost)); err != nil {
		t.Fatalf("stale lease recovery: %v", err)
	}
	if _, err := os.Stat(lockPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("ownership-aware unlock left lock behind: %v", err)
	}
}

func TestAppendReceiptRecoversPartialPublication(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipts.jsonl")
	lockPath := path + ".lock"
	orphan := lockPath + ".owner.partial.tmp"
	if err := os.WriteFile(orphan, []byte("partial"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := AppendReceipt(path, measuredReceipt("recovered-partial", VerdictLost)); err != nil {
		t.Fatalf("unpublished partial owner must not wedge acquisition: %v", err)
	}
}

func TestFutureDatedMetadataCannotWedge(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipts.jsonl")
	lockPath := path + ".lock"
	future := time.Now().UTC().Add(time.Hour)
	metadata, err := newLedgerLockMetadata(future, ledgerLockLease)
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(metadata)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(lockPath, append(b, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	if snapshot, err := readLedgerLockSnapshot(lockPath, time.Now().UTC()); err != nil || snapshot.owner != "" {
		t.Fatalf("future metadata must be invalid, snapshot=%#v err=%v", snapshot, err)
	}
	if err := AppendReceipt(path, measuredReceipt("future-recovered", VerdictLost)); err != nil {
		t.Fatalf("free OS guard must replace future metadata: %v", err)
	}
}

func TestLedgerMetadataTimeMustMatchPublicationMtime(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipts.jsonl.lock")
	created := time.Now().UTC().Add(-time.Minute)
	metadata, err := newLedgerLockMetadata(created, ledgerLockLease)
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(metadata)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	snapshot, err := readLedgerLockSnapshot(path, time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.owner != "" {
		t.Fatalf("mtime-mismatched metadata accepted as owner: %#v", snapshot)
	}
}

func TestAppendReceiptCrossProcessContention(t *testing.T) {
	if os.Getenv("FAK_RECEIPT_LOCK_HELPER") == "holder" {
		runReceiptLockHolder(t)
		return
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "receipts.jsonl")
	ready := filepath.Join(dir, "ready")
	release := filepath.Join(dir, "release")
	cmd := exec.Command(os.Args[0], "-test.run=^TestAppendReceiptCrossProcessContention$", "-test.v")
	cmd.Env = append(os.Environ(),
		"FAK_RECEIPT_LOCK_HELPER=holder",
		"FAK_RECEIPT_LEDGER="+path,
		"FAK_RECEIPT_READY="+ready,
		"FAK_RECEIPT_RELEASE="+release,
	)
	var childOutput bytes.Buffer
	cmd.Stdout, cmd.Stderr = &childOutput, &childOutput
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waitForReceiptTestFile(t, ready)
	// The helper publishes a short nominal lease but retains the OS lock. Passing
	// the lease deadline must not make a live or suspended writer reclaimable.
	time.Sleep(100 * time.Millisecond)
	err := AppendReceipt(path, measuredReceipt("contender", VerdictWon))
	var busy *LedgerBusyError
	if !errors.As(err, &busy) {
		t.Fatalf("contender should observe child owner, got %v", err)
	}
	if err := os.WriteFile(release, []byte("release"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err != nil {
		t.Fatalf("lock holder failed: %v\n%s", err, childOutput.String())
	}
	rows, err := ReadReceiptLedger(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].ID != "holder" {
		t.Fatalf("contended ledger = %#v, want exactly holder", rows)
	}
}

func TestAppendReceiptCrashRecovery(t *testing.T) {
	if os.Getenv("FAK_RECEIPT_LOCK_HELPER") == "crash-holder" {
		runReceiptCrashHolder(t)
		return
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "receipts.jsonl")
	ready := filepath.Join(dir, "crash-ready")
	cmd := exec.Command(os.Args[0], "-test.run=^TestAppendReceiptCrashRecovery$", "-test.v")
	cmd.Env = append(os.Environ(),
		"FAK_RECEIPT_LOCK_HELPER=crash-holder",
		"FAK_RECEIPT_LEDGER="+path,
		"FAK_RECEIPT_READY="+ready,
	)
	var childOutput bytes.Buffer
	cmd.Stdout, cmd.Stderr = &childOutput, &childOutput
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waitForReceiptTestFile(t, ready)
	if err := cmd.Process.Kill(); err != nil {
		t.Fatalf("terminate lock holder: %v", err)
	}
	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	select {
	case <-exited:
	case <-time.After(5 * time.Second):
		t.Fatalf("terminated lock holder did not exit within 5s\n%s", childOutput.String())
	}

	if err := AppendReceipt(path, measuredReceipt("after-crash", VerdictLost)); err != nil {
		t.Fatalf("kernel-released guard should allow append without cleanup: %v\n%s", err, childOutput.String())
	}
	rows, err := ReadReceiptLedger(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].ID != "after-crash" {
		t.Fatalf("crash-recovered ledger = %#v, want exactly after-crash", rows)
	}
	if _, err := os.Stat(path + ".lock"); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("recovery writer left owner metadata behind: %v", err)
	}
}

func runReceiptCrashHolder(t *testing.T) {
	path := os.Getenv("FAK_RECEIPT_LEDGER")
	lock, err := acquireLedgerLock(path+".lock", time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(os.Getenv("FAK_RECEIPT_READY"), []byte(lock.owner), 0o644); err != nil {
		t.Fatal(err)
	}
	// The parent terminates this process. Keeping the reference live documents
	// that no cooperative release or metadata cleanup participates in recovery.
	_ = lock
	select {}
}

func runReceiptLockHolder(t *testing.T) {
	path := os.Getenv("FAK_RECEIPT_LEDGER")
	lock, err := acquireLedgerLockWithLease(path+".lock", time.Now().UTC(), 50*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(os.Getenv("FAK_RECEIPT_READY"), []byte("ready"), 0o644); err != nil {
		t.Fatal(err)
	}
	waitForReceiptTestFile(t, os.Getenv("FAK_RECEIPT_RELEASE"))
	receipt := measuredReceipt("holder", VerdictLost)
	line, err := json.Marshal(receipt)
	if err == nil {
		err = appendReceiptUnderLock(path, line, receipt)
	}
	if releaseErr := lock.release(); err == nil {
		err = releaseErr
	}
	if err != nil {
		t.Fatal(err)
	}
}

func TestLedgerLockReleaseCannotRemoveSuccessorMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipts.jsonl.lock")
	owner, err := acquireLedgerLock(path, time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}
	releaseDone := make(chan error, 1)
	acquireDone := make(chan struct {
		lock *ledgerLock
		err  error
	}, 1)
	go func() { releaseDone <- owner.release() }()
	go func() {
		lock, err := acquireLedgerLock(path, time.Now().UTC())
		acquireDone <- struct {
			lock *ledgerLock
			err  error
		}{lock, err}
	}()
	if err := <-releaseDone; err != nil {
		t.Fatal(err)
	}
	result := <-acquireDone
	if result.err != nil {
		var busy *LedgerBusyError
		if !errors.As(result.err, &busy) {
			t.Fatal(result.err)
		}
		result.lock, result.err = acquireLedgerLock(path, time.Now().UTC())
	}
	if result.err != nil {
		t.Fatal(result.err)
	}
	snapshot, err := readLedgerLockSnapshot(path, time.Now().UTC())
	if err != nil || snapshot.owner != result.lock.owner {
		t.Fatalf("successor metadata removed by prior release: snapshot=%#v err=%v", snapshot, err)
	}
	if err := result.lock.release(); err != nil {
		t.Fatal(err)
	}
}

func waitForReceiptTestFile(t *testing.T, path string) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(path); err == nil {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", path)
}

func TestMigratedCompatibilityBatchReceiptIsScopedScreeningEvidence(t *testing.T) {
	b, err := os.ReadFile(filepath.Join(repoRoot(t), ReceiptLedgerRel))
	if err != nil {
		t.Fatal(err)
	}
	receipts, err := ParseReceiptLedger(string(b))
	if err != nil {
		t.Fatal(err)
	}
	if len(receipts) != 1 {
		t.Fatalf("migrated receipts = %d, want 1", len(receipts))
	}
	r := receipts[0]
	if r.Verdict != VerdictLost || r.EvidenceClass != EvidenceClassScreening || !strings.Contains(r.Scope, "no CUDA") {
		t.Fatalf("migrated receipt broadened or mistyped: %#v", r)
	}
	if r.Environment != "GCP-sanctioned-L4-controlled-node CPU-reference" || r.EnvironmentDigest != DigestEnvironment(r.Environment) {
		t.Fatalf("migrated environment identity is not derivable: %q %q", r.Environment, r.EnvironmentDigest)
	}
	result := LookupReceipt(receipts, identityFor(r))
	if result.Status != LookupExact || !result.MeasuredLoss || result.ClaimEligible {
		t.Fatalf("migrated lookup = %#v", result)
	}
}
