package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

type pendingAdmissionProofJSON struct {
	CompanionCommit    string `json:"companion_commit"`
	CompanionTree      string `json:"companion_tree"`
	GoExecutableDigest string `json:"go_executable_digest"`
	ToolchainDigest    string `json:"toolchain_digest"`
	TestEnvDigest      string `json:"test_env_digest"`
	WorkspaceDigest    string `json:"workspace_digest"`
	VerifierDigest     string `json:"verifier_digest"`
}

type pendingAdmissionBindingJSON struct {
	TargetRef       string                    `json:"target_ref"`
	ParentCommit    string                    `json:"parent_commit"`
	ParentTree      string                    `json:"parent_tree"`
	CandidateCommit string                    `json:"candidate_commit"`
	CandidateTree   string                    `json:"candidate_tree"`
	RecoveryRef     string                    `json:"recovery_ref"`
	Paths           []string                  `json:"paths"`
	PathsDigest     string                    `json:"paths_digest"`
	ProofContext    pendingAdmissionProofJSON `json:"proof_context"`
	ContextDigest   string                    `json:"context_digest"`
}

type pendingAdmissionReceiptJSON struct {
	ReceiptID string                      `json:"receipt_id"`
	State     string                      `json:"state"`
	Binding   pendingAdmissionBindingJSON `json:"binding"`
}

type pendingAdmissionObligationJSON struct {
	Kind   string `json:"kind"`
	ID     string `json:"id"`
	Digest string `json:"digest"`
}

type pendingAdmissionAckJSON struct {
	Schema            string                           `json:"schema"`
	Contract          string                           `json:"contract"`
	AckID             string                           `json:"ack_id"`
	State             string                           `json:"state"`
	PreparedReceiptID string                           `json:"prepared_receipt_id"`
	Binding           pendingAdmissionBindingJSON      `json:"binding"`
	RecipeDigest      string                           `json:"recipe_digest"`
	Obligations       []pendingAdmissionObligationJSON `json:"obligations"`
}

func pendingAdmissionDigest(seed string) string {
	sum := sha256.Sum256([]byte(seed))
	return hex.EncodeToString(sum[:])
}

func pendingAdmissionProofFixture() pendingAdmissionProofJSON {
	return pendingAdmissionProofJSON{
		CompanionCommit:    strings.Repeat("1", 40),
		CompanionTree:      strings.Repeat("2", 40),
		GoExecutableDigest: pendingAdmissionDigest("go"),
		ToolchainDigest:    pendingAdmissionDigest("toolchain"),
		TestEnvDigest:      pendingAdmissionDigest("test-env"),
		WorkspaceDigest:    pendingAdmissionDigest("workspace"),
		VerifierDigest:     pendingAdmissionDigest("verifier"),
	}
}

func writePendingAdmissionJSON(t *testing.T, name string, value any) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func pendingAdmissionArgs(mode, repo, worktree, base string, paths []string, contextFile string) []string {
	args := []string{mode, "--root", repo, "--worktree", worktree, "--base-sha", base}
	for _, path := range paths {
		args = append(args, "--paths", path)
	}
	if contextFile != "" {
		args = append(args, "--pending-context-file", contextFile)
	}
	return args
}

func runPendingAdmissionCLI(t *testing.T, args []string) (workerworktree.Result, pendingAdmissionReceiptJSON, int, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	result, code := runWorktreeWorkerLand(&stdout, &stderr, args)
	var envelope struct {
		PendingReceipt pendingAdmissionReceiptJSON `json:"pending_receipt"`
	}
	if stdout.Len() > 0 {
		if err := json.Unmarshal(stdout.Bytes(), &envelope); err != nil {
			t.Fatalf("decode pending land output: %v; stdout=%s stderr=%s", err, stdout.String(), stderr.String())
		}
	}
	return result, envelope.PendingReceipt, code, stderr.String()
}

func pendingAdmissionAckFile(t *testing.T, receipt pendingAdmissionReceiptJSON, recipe string, mutate func(*pendingAdmissionAckJSON)) string {
	t.Helper()
	ack := pendingAdmissionAckJSON{
		Schema:            "fak.pending-debt-ack/v1",
		Contract:          "durable-pending-obligations/v1",
		State:             "pending",
		PreparedReceiptID: receipt.ReceiptID,
		Binding:           receipt.Binding,
		RecipeDigest:      recipe,
		Obligations: []pendingAdmissionObligationJSON{{
			Kind: "release", ID: pendingAdmissionDigest("release-id"), Digest: pendingAdmissionDigest("release-record"),
		}},
	}
	if mutate != nil {
		mutate(&ack)
	}
	raw, err := json.Marshal(ack)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	ack.AckID = hex.EncodeToString(sum[:])
	return writePendingAdmissionJSON(t, "debt-ack.json", ack)
}

func preparePendingAdmissionFixture(t *testing.T) (repo, worktree, base string, paths []string, receipt pendingAdmissionReceiptJSON) {
	t.Helper()
	repo, worktree, base, paths = newPreparedCLIWorkerFixture(t, false)
	if err := workerworktree.SaveIntent(worktree, base, "feat(calc): extend calculation (fak calc)", paths); err != nil {
		t.Fatalf("save durable coordinator intent: %v", err)
	}
	contextFile := writePendingAdmissionJSON(t, "proof-context.json", pendingAdmissionProofFixture())
	before := strings.TrimSpace(worktreeWorkerTestGit(t, repo, "rev-parse", "refs/heads/main"))
	result, receipt, code, stderr := runPendingAdmissionCLI(t, pendingAdmissionArgs("prepare-pending", repo, worktree, base, paths, contextFile))
	if code != 0 || !result.OK || receipt.ReceiptID == "" || receipt.State != "pending" {
		t.Fatalf("prepare-pending result=%+v receipt=%+v code=%d stderr=%s", result, receipt, code, stderr)
	}
	if receipt.Binding.ParentCommit != before || receipt.Binding.CandidateCommit == "" || receipt.Binding.CandidateCommit == before {
		t.Fatalf("prepared binding does not seal exact parent/candidate: %+v", receipt.Binding)
	}
	if got := strings.TrimSpace(worktreeWorkerTestGit(t, repo, "rev-parse", receipt.Binding.CandidateCommit+"^{tree}")); got != receipt.Binding.CandidateTree {
		t.Fatalf("candidate tree=%s, binding=%s", got, receipt.Binding.CandidateTree)
	}
	if got := strings.TrimSpace(worktreeWorkerTestGit(t, repo, "rev-parse", "--verify", receipt.Binding.RecoveryRef+"^{commit}")); got != receipt.Binding.CandidateCommit {
		t.Fatalf("recovery ref=%s, candidate=%s", got, receipt.Binding.CandidateCommit)
	}
	if after := strings.TrimSpace(worktreeWorkerTestGit(t, repo, "rev-parse", "refs/heads/main")); after != before {
		t.Fatalf("prepare-pending moved target: %s -> %s", before, after)
	}
	return repo, worktree, base, paths, receipt
}

// fak-test:runtime slow est=60s lane=default
func TestWorktreeWorkerPendingAdmissionAcceptsOnlyDurablyAcknowledgedCandidate(t *testing.T) {
	repo, worktree, base, paths, receipt := preparePendingAdmissionFixture(t)
	recipe := pendingAdmissionDigest("recipe")
	ackFile := pendingAdmissionAckFile(t, receipt, recipe, nil)
	args := pendingAdmissionArgs("accept-pending", repo, worktree, base, paths, "")
	args = append(args, "--receipt-id", receipt.ReceiptID, "--debt-ack-file", ackFile, "--expected-recipe-digest", recipe)
	result, _, code, stderr := runPendingAdmissionCLI(t, args)
	if code != 0 || !result.OK || !result.Committed || result.CommitSHA != receipt.Binding.CandidateCommit {
		t.Fatalf("accept-pending result=%+v code=%d stderr=%s", result, code, stderr)
	}
	if head := strings.TrimSpace(worktreeWorkerTestGit(t, repo, "rev-parse", "refs/heads/main")); head != receipt.Binding.CandidateCommit {
		t.Fatalf("target=%s, want sealed candidate %s", head, receipt.Binding.CandidateCommit)
	}
	outcome := result.Code + " " + result.Reason + " " + result.Detail
	lowerOutcome := strings.ToLower(outcome)
	if !strings.Contains(outcome, "LANDED_PENDING") || strings.Contains(lowerOutcome, "verified") || strings.Contains(lowerOutcome, "qualified") {
		t.Fatalf("pending acceptance claimed the wrong state: %+v", result)
	}
}

// fak-test:runtime slow est=120s lane=default
func TestWorktreeWorkerPendingAdmissionRefusesMissingInvalidMismatchedAndCrossModeProofs(t *testing.T) {
	repo, worktree, base, paths, receipt := preparePendingAdmissionFixture(t)
	recipe := pendingAdmissionDigest("recipe")
	before := strings.TrimSpace(worktreeWorkerTestGit(t, repo, "rev-parse", "refs/heads/main"))
	baseArgs := pendingAdmissionArgs("accept-pending", repo, worktree, base, paths, "")
	baseArgs = append(baseArgs, "--receipt-id", receipt.ReceiptID, "--expected-recipe-digest", recipe)

	invalidFile := filepath.Join(t.TempDir(), "invalid-ack.json")
	if err := os.WriteFile(invalidFile, []byte(`{"schema":`), 0o600); err != nil {
		t.Fatal(err)
	}
	mismatchFile := pendingAdmissionAckFile(t, receipt, recipe, func(ack *pendingAdmissionAckJSON) {
		ack.PreparedReceiptID = pendingAdmissionDigest("other-receipt")
	})
	for _, tc := range []struct {
		name string
		args []string
	}{
		{name: "missing ack", args: baseArgs},
		{name: "invalid ack", args: append(append([]string(nil), baseArgs...), "--debt-ack-file", invalidFile)},
		{name: "mismatched ack", args: append(append([]string(nil), baseArgs...), "--debt-ack-file", mismatchFile)},
		{name: "pending receipt in verified accept", args: append(preparedCLIArgs("accept", repo, worktree, base, paths), "--receipt-id", receipt.ReceiptID)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, _, code, _ := runPendingAdmissionCLI(t, tc.args)
			if code == 0 || result.OK || result.Committed {
				t.Fatalf("%s accepted: %+v code=%d", tc.name, result, code)
			}
			if head := strings.TrimSpace(worktreeWorkerTestGit(t, repo, "rev-parse", "refs/heads/main")); head != before {
				t.Fatalf("%s mutated target: %s -> %s", tc.name, before, head)
			}
			if got := strings.TrimSpace(worktreeWorkerTestGit(t, repo, "rev-parse", "--verify", receipt.Binding.RecoveryRef+"^{commit}")); got != receipt.Binding.CandidateCommit {
				t.Fatalf("%s lost candidate recovery ref: %s", tc.name, got)
			}
		})
	}

	verifiedRepo, verifiedWorktree, verifiedBase, verifiedPaths := newPreparedCLIWorkerFixture(t, false)
	verifiedResult, verifiedReceipt, verifiedCode, verifiedStderr := runPreparedCLI(t, preparedCLIArgs("prepare", verifiedRepo, verifiedWorktree, verifiedBase, verifiedPaths))
	if verifiedCode != 0 || !verifiedResult.OK || verifiedReceipt == nil {
		t.Fatalf("verified prepare result=%+v receipt=%+v code=%d stderr=%s", verifiedResult, verifiedReceipt, verifiedCode, verifiedStderr)
	}
	verifiedAck := pendingAdmissionAckFile(t, receipt, recipe, func(ack *pendingAdmissionAckJSON) {
		ack.PreparedReceiptID = verifiedReceipt.ReceiptID
	})
	verifiedBefore := strings.TrimSpace(worktreeWorkerTestGit(t, verifiedRepo, "rev-parse", "refs/heads/main"))
	verifiedArgs := pendingAdmissionArgs("accept-pending", verifiedRepo, verifiedWorktree, verifiedBase, verifiedPaths, "")
	verifiedArgs = append(verifiedArgs, "--receipt-id", verifiedReceipt.ReceiptID, "--debt-ack-file", verifiedAck, "--expected-recipe-digest", recipe)
	if result, _, code, _ := runPendingAdmissionCLI(t, verifiedArgs); code == 0 || result.OK || result.Committed {
		t.Fatalf("verified receipt crossed into pending accept: %+v code=%d", result, code)
	}
	if after := strings.TrimSpace(worktreeWorkerTestGit(t, verifiedRepo, "rev-parse", "refs/heads/main")); after != verifiedBefore {
		t.Fatalf("cross-mode refusal moved target: %s -> %s", verifiedBefore, after)
	}
}

// fak-test:runtime slow est=75s lane=default
func TestWorktreeWorkerPendingAdmissionPreservesCandidateWhenTargetOrIntentChanges(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(t *testing.T, repo, worktree string)
	}{
		{name: "target moved before CAS", change: func(t *testing.T, repo, _ string) {
			if err := os.WriteFile(filepath.Join(repo, "peer.txt"), []byte("peer\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			worktreeWorkerTestGit(t, repo, "add", "peer.txt")
			worktreeWorkerTestGit(t, repo, "commit", "-qm", "peer advance")
		}},
		{name: "prepared worktree intent changed", change: func(t *testing.T, _, worktree string) {
			if err := os.WriteFile(filepath.Join(worktree, "pkg", "calc.go"), []byte("package pkg\n\nfunc Calc(int) int { return 99 }\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			repo, worktree, base, paths, receipt := preparePendingAdmissionFixture(t)
			recipe := pendingAdmissionDigest("recipe")
			ackFile := pendingAdmissionAckFile(t, receipt, recipe, nil)
			tc.change(t, repo, worktree)
			before := strings.TrimSpace(worktreeWorkerTestGit(t, repo, "rev-parse", "refs/heads/main"))
			args := pendingAdmissionArgs("accept-pending", repo, worktree, base, paths, "")
			args = append(args, "--receipt-id", receipt.ReceiptID, "--debt-ack-file", ackFile, "--expected-recipe-digest", recipe)
			result, _, code, _ := runPendingAdmissionCLI(t, args)
			if code == 0 || result.OK || result.Committed {
				t.Fatalf("changed state accepted: %+v code=%d", result, code)
			}
			if after := strings.TrimSpace(worktreeWorkerTestGit(t, repo, "rev-parse", "refs/heads/main")); after != before {
				t.Fatalf("refusal moved target: %s -> %s", before, after)
			}
			if got := strings.TrimSpace(worktreeWorkerTestGit(t, repo, "rev-parse", "--verify", receipt.Binding.RecoveryRef+"^{commit}")); got != receipt.Binding.CandidateCommit {
				t.Fatalf("refusal lost candidate: got %s want %s", got, receipt.Binding.CandidateCommit)
			}
		})
	}
}
