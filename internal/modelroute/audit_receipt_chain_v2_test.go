package modelroute

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

func makeAuditReceiptV2(t *testing.T, subject string, verdict CrossAuditVerdict) IssueAuditReceipt {
	t.Helper()
	policyDigest := hashString("policy:v2")
	r := IssueAuditReceipt{
		Schema: CrossAuditReceiptSchema,
		Subject: IssueAuditSubject{
			IssueNumber: 42, IssueURL: "https://github.com/example/repo/issues/42", IssueState: "CLOSED",
			Title: subject, CommitSHA: "deadbeef", DiffSHA256: hashString("diff:" + subject), Digest: hashString("subject:" + subject),
		},
		Author: AuditIdentity{
			Harness: "claude-code", Provider: "anthropic", Family: "claude", Model: "claude-opus-4-6",
			WeightsRevision: "claude-w46", AccountClass: "subscription", EndpointClass: "hosted",
			ReasoningPosture: "high", ProvenanceSource: "session:author",
		},
		Auditor: AuditIdentity{
			Harness: "codex", Provider: "openai", Family: "gpt", Model: "gpt-5.6-sol",
			WeightsRevision: "gpt-w56", AccountClass: "subscription", EndpointClass: "hosted",
			ReasoningPosture: "xhigh", ProvenanceSource: "session:auditor",
		},
		Independence: IndependenceDecision{
			Admitted: true, Verdict: AuditIndependenceAdmit, Rule: AuditIndependencePolicyVersion,
			PolicyDigest: policyDigest, Reason: string(AuditReasonAdmitIndependent),
		},
		PolicyVersion: AuditIndependencePolicyVersion,
		PolicyDigest:  policyDigest,
		PromptVersion: CrossAuditPromptVersion,
		PromptDigest:  hashString("prompt:" + subject),
		Verdict:       verdict,
		Severity:      NormalizeAuditFindingSeverity(verdict, AuditSeverityHigh),
		Reason:        "evidence-bound " + strings.ToLower(string(verdict)),
		EvidenceRefs:  []EvidenceRef{{Kind: "commit", Ref: "deadbeef"}, {Kind: "test", Ref: "Test" + subject}},
		Timing:        &AuditTiming{StartedAtUnixNano: 100, CompletedAtUnixNano: 150, DurationNanos: 50},
		Usage: &AuditTokenCost{
			Measured: true, InputTokens: 80, OutputTokens: 20, TotalTokens: 100,
			CostMicrosUSD: 25, Currency: "USD", Basis: "provider-reported",
		},
	}
	r.AuditKey = AuditReceiptKey(r.Subject.Digest, r.PolicyVersion, r.PolicyDigest)
	r.IdentityDigest = auditReceiptIdentityDigest(r)
	r.EvidenceDigest = hashJSON(r.EvidenceRefs)
	r.ReceiptDigest = r.recomputeDigest()
	if err := r.Verify(); err != nil {
		t.Fatalf("make v2 receipt: %v", err)
	}
	return r
}

// TestAuditReceiptChainDetectsMutation is the named #3850 mutation-chain
// witness. It captures a clean two-row verification and a deliberate one-byte
// verdict mutation that the same verifier rejects.
func TestAuditReceiptChainDetectsMutation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit-receipts.jsonl")
	first, err := AppendAuditReceiptLedger(path, makeAuditReceiptV2(t, "alpha", CrossAuditPass))
	if err != nil || !first.Appended {
		t.Fatalf("append first: result=%+v err=%v", first, err)
	}
	second, err := AppendAuditReceiptLedger(path, makeAuditReceiptV2(t, "beta", CrossAuditRefute))
	if err != nil || !second.Appended || second.Row.Seq != 2 || second.Row.PrevHash != first.Row.Hash {
		t.Fatalf("append second: result=%+v err=%v", second, err)
	}
	verification, err := VerifyAuditReceiptLedger(path)
	if err != nil || verification.Rows != 2 || verification.HeadHash != second.Row.Hash {
		t.Fatalf("verify two-row ledger: verification=%+v err=%v", verification, err)
	}
	t.Logf("CLEAN_VERIFY exit=0 rows=%d head=%s", verification.Rows, verification.HeadHash)
	if extended, err := VerifyAuditReceiptLedgerAtCursor(path, first.Cursor); err != nil || extended.Rows != 2 {
		t.Fatalf("saved cursor rejected a valid extension: verification=%+v err=%v", extended, err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	mutations := []struct {
		name, old, replacement string
	}{
		{"identity", "claude-opus-4-6", "claude-opus-4-5"},
		{"evidence", "Testalpha", "Xestalpha"},
		{"verdict", `"verdict":"REFUTE"`, `"verdict":"XEFUTE"`},
	}
	var mutationErr error
	for _, mutation := range mutations {
		t.Run("mutated-"+mutation.name, func(t *testing.T) {
			mutated := bytes.Replace(data, []byte(mutation.old), []byte(mutation.replacement), 1)
			if bytes.Equal(mutated, data) || len(mutated) != len(data) {
				t.Fatalf("%s fixture mutation was not exactly one byte", mutation.name)
			}
			_, mutationErr = VerifyAuditReceiptLedgerReader(bytes.NewReader(mutated))
			if mutationErr == nil {
				t.Fatalf("one-byte %s mutation verified", mutation.name)
			}
		})
	}
	t.Logf("MUTATED_VERIFY exit=1 error=%v", mutationErr)

	lastLine := bytes.LastIndex(bytes.TrimSuffix(data, []byte{'\n'}), []byte{'\n'})
	if lastLine < 0 {
		t.Fatal("two-row ledger has no row boundary")
	}
	if err := os.WriteFile(path, data[:lastLine+1], 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyAuditReceiptLedgerAtCursor(path, second.Cursor); err == nil {
		t.Fatalf("saved cursor did not detect valid tail deletion: %v", err)
	}
}

func TestAuditFoldNeverTreatsNonPassAsPass(t *testing.T) {
	if fold, err := FoldAuditReceiptLedger(nil); err != nil || fold.CleanPass || fold.Verdict != CrossAuditUnavailable {
		t.Fatalf("empty fold = %+v err=%v", fold, err)
	}
	for _, verdict := range []CrossAuditVerdict{CrossAuditRefute, CrossAuditInconclusive, CrossAuditUnavailable} {
		t.Run(string(verdict), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "ledger.jsonl")
			appended, err := AppendAuditReceiptLedger(path, makeAuditReceiptV2(t, string(verdict), verdict))
			if err != nil {
				t.Fatal(err)
			}
			fold, err := FoldAuditReceiptLedger([]AuditReceiptLedgerRow{appended.Row})
			if err != nil {
				t.Fatal(err)
			}
			entry := fold.ByAuditKey[appended.Row.AuditKey]
			if fold.CleanPass || fold.Verdict == CrossAuditPass || entry.Passed {
				t.Fatalf("non-PASS folded as PASS: fold=%+v entry=%+v", fold, entry)
			}
		})
	}

	path := filepath.Join(t.TempDir(), "mixed.jsonl")
	pass, _ := AppendAuditReceiptLedger(path, makeAuditReceiptV2(t, "pass", CrossAuditPass))
	refute, _ := AppendAuditReceiptLedger(path, makeAuditReceiptV2(t, "refute", CrossAuditRefute))
	fold, err := FoldAuditReceiptLedger([]AuditReceiptLedgerRow{pass.Row, refute.Row})
	if err != nil || fold.CleanPass || fold.Verdict != CrossAuditRefute {
		t.Fatalf("REFUTE did not dominate mixed fold: %+v err=%v", fold, err)
	}
}

func TestAuditReceiptLedgerAppendDuplicateConflictAndTornTail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ledger.jsonl")
	receipt := makeAuditReceiptV2(t, "same-key", CrossAuditPass)
	first, err := AppendAuditReceiptLedger(path, receipt)
	if err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(path)
	duplicate, err := AppendAuditReceiptLedger(path, receipt)
	if err != nil || !duplicate.Duplicate || duplicate.Appended || duplicate.Cursor != first.Cursor {
		t.Fatalf("idempotent duplicate = %+v err=%v", duplicate, err)
	}
	afterDuplicate, _ := os.ReadFile(path)
	if !bytes.Equal(before, afterDuplicate) {
		t.Fatal("exact duplicate changed ledger bytes")
	}

	conflict := makeAuditReceiptV2(t, "same-key", CrossAuditRefute)
	if _, err := AppendAuditReceiptLedger(path, conflict); !errors.Is(err, ErrAuditReceiptKeyConflict) {
		t.Fatalf("same-key conflict error = %v", err)
	}
	afterConflict, _ := os.ReadFile(path)
	if !bytes.Equal(before, afterConflict) {
		t.Fatal("conflicting duplicate changed ledger bytes")
	}

	second, err := AppendAuditReceiptLedger(path, makeAuditReceiptV2(t, "other-key", CrossAuditInconclusive))
	if err != nil || second.Row.Seq != 2 {
		t.Fatalf("append distinct receipt = %+v err=%v", second, err)
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f.WriteString(`{"schema":"torn"}`)
	_ = f.Close()

	data, _ := os.ReadFile(path)
	prefix, integ, err := ParseAuditReceiptLedgerPrefix(bytes.NewReader(data))
	if err != nil || !integ.Broken || !integ.TornTail || len(prefix) != 2 {
		t.Fatalf("torn prefix = rows=%d integrity=%+v err=%v", len(prefix), integ, err)
	}
	if _, err := ParseAuditReceiptLedger(bytes.NewReader(data)); err == nil {
		t.Fatal("strict parser accepted torn tail")
	}
	beforeRefusal := append([]byte(nil), data...)
	if _, err := AppendAuditReceiptLedger(path, makeAuditReceiptV2(t, "after-torn", CrossAuditPass)); err == nil {
		t.Fatal("append extended a torn ledger")
	}
	afterRefusal, _ := os.ReadFile(path)
	if !bytes.Equal(beforeRefusal, afterRefusal) {
		t.Fatal("torn-tail append refusal changed ledger")
	}

	corruptPath := filepath.Join(t.TempDir(), "corrupt.jsonl")
	if _, err := AppendAuditReceiptLedger(corruptPath, makeAuditReceiptV2(t, "before-corrupt", CrossAuditPass)); err != nil {
		t.Fatal(err)
	}
	corruptFile, _ := os.OpenFile(corruptPath, os.O_WRONLY|os.O_APPEND, 0o600)
	_, _ = corruptFile.WriteString("not-json\n")
	_ = corruptFile.Close()
	corruptData, _ := os.ReadFile(corruptPath)
	_, corruptIntegrity, err := ParseAuditReceiptLedgerPrefix(bytes.NewReader(corruptData))
	if err != nil || !corruptIntegrity.Broken || corruptIntegrity.TornTail {
		t.Fatalf("newline corruption classification = %+v err=%v", corruptIntegrity, err)
	}
}

func TestAuditReceiptLedgerConcurrentAppendAndLockContention(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ledger.jsonl")
	const workers = 8
	errCh := make(chan error, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := AppendAuditReceiptLedger(path, makeAuditReceiptV2(t, fmt.Sprintf("concurrent-%d", i), CrossAuditPass))
			errCh <- err
		}(i)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			t.Fatalf("concurrent append: %v", err)
		}
	}
	verification, err := VerifyAuditReceiptLedger(path)
	if err != nil || verification.Rows != workers {
		t.Fatalf("concurrent ledger = %+v err=%v", verification, err)
	}

	const processes = 4
	commands := make([]*exec.Cmd, processes)
	outputs := make([]bytes.Buffer, processes)
	for i := range commands {
		commands[i] = exec.Command(os.Args[0], "-test.run=^TestAuditReceiptLedgerProcessHelper$")
		commands[i].Env = append(os.Environ(),
			"FAK_AUDIT_RECEIPT_PROCESS_HELPER=1",
			"FAK_AUDIT_RECEIPT_PROCESS_PATH="+path,
			fmt.Sprintf("FAK_AUDIT_RECEIPT_PROCESS_SUBJECT=process-%d", i),
		)
		commands[i].Stdout = &outputs[i]
		commands[i].Stderr = &outputs[i]
		if err := commands[i].Start(); err != nil {
			t.Fatalf("start append process %d: %v", i, err)
		}
	}
	for i, command := range commands {
		if err := command.Wait(); err != nil {
			t.Fatalf("append process %d: %v\n%s", i, err, outputs[i].String())
		}
	}
	verification, err = VerifyAuditReceiptLedger(path)
	if err != nil || verification.Rows != workers+processes {
		t.Fatalf("cross-process ledger = %+v err=%v", verification, err)
	}

	lockPath := filepath.Join(t.TempDir(), "blocked.jsonl")
	lockFile, err := os.OpenFile(lockPath+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer lockFile.Close()
	if err := flock.TryLock(lockFile); err != nil {
		t.Fatalf("take test lock: %v", err)
	}
	defer flock.Unlock(lockFile)
	priorWait := auditReceiptLedgerLockWait
	auditReceiptLedgerLockWait = 25 * time.Millisecond
	t.Cleanup(func() { auditReceiptLedgerLockWait = priorWait })
	if _, err := AppendAuditReceiptLedger(lockPath, makeAuditReceiptV2(t, "blocked", CrossAuditPass)); !errors.Is(err, ErrAuditReceiptLedgerBusy) {
		t.Fatalf("contended append error = %v", err)
	}
}

func TestAuditReceiptLedgerProcessHelper(t *testing.T) {
	if os.Getenv("FAK_AUDIT_RECEIPT_PROCESS_HELPER") != "1" {
		t.Skip("subprocess helper")
	}
	path := os.Getenv("FAK_AUDIT_RECEIPT_PROCESS_PATH")
	subject := os.Getenv("FAK_AUDIT_RECEIPT_PROCESS_SUBJECT")
	if _, err := AppendAuditReceiptLedger(path, makeAuditReceiptV2(t, subject, CrossAuditPass)); err != nil {
		t.Fatal(err)
	}
}

func TestAuditReceiptV1PublishedFixtureVerifiesUnchanged(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("testdata", "crossaudit_receipt_v1_3847.json"))
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := ParseIssueAuditReceipt(b)
	if err != nil {
		t.Fatalf("parse published #3847 v1 receipt: %v", err)
	}
	const publishedDigest = "sha256:3f864426cba6e23275fd6f5a5a483589b2474a8a6a4a9b4e2341eb0a41188d73"
	if receipt.Schema != CrossAuditReceiptSchemaV1 || receipt.ReceiptDigest != publishedDigest || receipt.recomputeDigest() != publishedDigest {
		t.Fatalf("published v1 changed: schema=%s stamped=%s recomputed=%s", receipt.Schema, receipt.ReceiptDigest, receipt.recomputeDigest())
	}
	path := filepath.Join(t.TempDir(), "legacy-ledger.jsonl")
	appended, err := AppendAuditReceiptLedger(path, receipt)
	if err != nil {
		t.Fatalf("append unchanged v1 receipt: %v", err)
	}
	wantKey := LegacyAuditReceiptLedgerKey(receipt.Subject.Digest, receipt.PolicyVersion)
	if appended.Row.AuditKey != wantKey || appended.Row.Receipt.ReceiptDigest != publishedDigest {
		t.Fatalf("legacy row rewrote receipt/key: %+v", appended.Row)
	}
	if _, err := VerifyAuditReceiptLedger(path); err != nil {
		t.Fatalf("verify legacy receipt ledger: %v", err)
	}
}

func TestAuditReceiptStrictParsersRejectUnknownFields(t *testing.T) {
	receipt := makeAuditReceiptV2(t, "strict", CrossAuditPass)
	b, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	unknownReceipt := bytes.Replace(b, []byte(`{"schema":`), []byte(`{"unknown":true,"schema":`), 1)
	if _, err := ParseIssueAuditReceipt(unknownReceipt); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("receipt unknown-field error = %v", err)
	}

	path := filepath.Join(t.TempDir(), "strict.jsonl")
	if _, err := AppendAuditReceiptLedger(path, receipt); err != nil {
		t.Fatal(err)
	}
	rowBytes, _ := os.ReadFile(path)
	unknownRow := bytes.Replace(rowBytes, []byte(`{"schema":`), []byte(`{"unknown":true,"schema":`), 1)
	if _, err := ParseAuditReceiptLedger(bytes.NewReader(unknownRow)); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("row unknown-field error = %v", err)
	}
}

func TestAuditReceiptV2RejectsHashedButInvalidIdentityAndUsage(t *testing.T) {
	t.Run("empty-identity", func(t *testing.T) {
		receipt := makeAuditReceiptV2(t, "empty-identity", CrossAuditPass)
		receipt.Author = AuditIdentity{}
		receipt.IdentityDigest = auditReceiptIdentityDigest(receipt)
		receipt.ReceiptDigest = receipt.recomputeDigest()
		if err := receipt.Verify(); err == nil || !strings.Contains(err.Error(), "identity axes are incomplete") {
			t.Fatalf("empty but re-hashed identity error = %v", err)
		}
	})
	t.Run("same-family", func(t *testing.T) {
		receipt := makeAuditReceiptV2(t, "same-family", CrossAuditPass)
		receipt.Auditor.Family = receipt.Author.Family
		receipt.IdentityDigest = auditReceiptIdentityDigest(receipt)
		receipt.ReceiptDigest = receipt.recomputeDigest()
		if err := receipt.Verify(); err == nil || !strings.Contains(err.Error(), "not independent") {
			t.Fatalf("same-family re-hashed identity error = %v", err)
		}
	})
	t.Run("inconsistent-usage", func(t *testing.T) {
		receipt := makeAuditReceiptV2(t, "bad-usage", CrossAuditPass)
		receipt.Usage.TotalTokens++
		receipt.ReceiptDigest = receipt.recomputeDigest()
		if err := receipt.Verify(); err == nil || !strings.Contains(err.Error(), "total tokens") {
			t.Fatalf("inconsistent usage error = %v", err)
		}
	})
}
