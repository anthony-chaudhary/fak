package pendingadmission

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"
)

func pendingTestDigest(seed string) string {
	sum := sha256.Sum256([]byte(seed))
	return hex.EncodeToString(sum[:])
}

func pendingTestProof() ProofContext {
	return ProofContext{
		CompanionCommit:    strings.Repeat("1", 40),
		CompanionTree:      strings.Repeat("2", 40),
		GoExecutableDigest: pendingTestDigest("go"),
		ToolchainDigest:    pendingTestDigest("toolchain"),
		TestEnvDigest:      pendingTestDigest("test-env"),
		WorkspaceDigest:    pendingTestDigest("workspace"),
		VerifierDigest:     pendingTestDigest("verifier"),
	}
}

func pendingTestBinding(t *testing.T) Binding {
	t.Helper()
	paths := []string{"cmd/fak/main.go", "pkg/example/example.go"}
	proof := pendingTestProof()
	pathJSON, err := json.Marshal(paths)
	if err != nil {
		t.Fatal(err)
	}
	proofJSON, err := json.Marshal(proof)
	if err != nil {
		t.Fatal(err)
	}
	return Binding{
		TargetRef:       "refs/heads/main",
		ParentCommit:    strings.Repeat("3", 40),
		ParentTree:      strings.Repeat("4", 40),
		CandidateCommit: strings.Repeat("5", 40),
		CandidateTree:   strings.Repeat("6", 40),
		RecoveryRef:     "refs/fak/recovery/pending-candidate",
		Paths:           paths,
		PathsDigest:     pendingTestDigestBytes(pathJSON),
		ProofContext:    proof,
		ContextDigest:   pendingTestDigestBytes(proofJSON),
	}
}

func pendingTestDigestBytes(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func pendingTestAck(t *testing.T) DebtAck {
	t.Helper()
	ack := DebtAck{
		Schema:            AckSchema,
		Contract:          AckContract,
		State:             "pending",
		PreparedReceiptID: pendingTestDigest("receipt"),
		Binding:           pendingTestBinding(t),
		RecipeDigest:      pendingTestDigest("recipe"),
		Obligations: []Obligation{
			{Kind: "benchmark", ID: pendingTestDigest("benchmark-id"), Digest: pendingTestDigest("benchmark")},
			{Kind: "release", ID: pendingTestDigest("release-id"), Digest: pendingTestDigest("release")},
		},
	}
	id, err := ack.ID()
	if err != nil {
		t.Fatalf("DebtAck.ID: %v", err)
	}
	ack.AckID = id
	return ack
}

// fak-test:runtime fast est=100ms lane=default
func TestPendingAdmissionAckBindsExactReceiptCandidateAndRecipe(t *testing.T) {
	t.Parallel()
	ack := pendingTestAck(t)
	if err := ack.Validate(); err != nil {
		t.Fatalf("valid acknowledgement refused: %v", err)
	}
	if got, err := ack.ID(); err != nil || got != ack.AckID {
		t.Fatalf("DebtAck.ID = %q, %v; want stable %q", got, err, ack.AckID)
	}
	if err := ack.Matches(ack.PreparedReceiptID, ack.Binding, ack.RecipeDigest); err != nil {
		t.Fatalf("exact receipt binding did not match: %v", err)
	}

	for _, tc := range []struct {
		name    string
		receipt string
		binding Binding
		recipe  string
	}{
		{name: "other receipt", receipt: pendingTestDigest("other-receipt"), binding: ack.Binding, recipe: ack.RecipeDigest},
		{name: "other candidate", receipt: ack.PreparedReceiptID, binding: func() Binding { b := ack.Binding; b.CandidateCommit = strings.Repeat("7", 40); return b }(), recipe: ack.RecipeDigest},
		{name: "other recipe", receipt: ack.PreparedReceiptID, binding: ack.Binding, recipe: pendingTestDigest("other-recipe")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := ack.Matches(tc.receipt, tc.binding, tc.recipe); err == nil {
				t.Fatal("mismatched acknowledgement binding accepted")
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestPendingAdmissionAckRejectsNonCanonicalOrTamperedData(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		edit func(*DebtAck)
	}{
		{name: "tampered ID", edit: func(a *DebtAck) { a.AckID = pendingTestDigest("tampered") }},
		{name: "verified state", edit: func(a *DebtAck) { a.State = "verified" }},
		{name: "unsorted paths", edit: func(a *DebtAck) { a.Binding.Paths[0], a.Binding.Paths[1] = a.Binding.Paths[1], a.Binding.Paths[0] }},
		{name: "noncanonical path", edit: func(a *DebtAck) { a.Binding.Paths[0] = "../main.go" }},
		{name: "drive absolute path", edit: func(a *DebtAck) { a.Binding.Paths[0] = "C:/outside/main.go" }},
		{name: "mutable target", edit: func(a *DebtAck) { a.Binding.TargetRef = "main" }},
		{name: "same candidate and parent", edit: func(a *DebtAck) { a.Binding.CandidateCommit = a.Binding.ParentCommit }},
		{name: "wrong context digest", edit: func(a *DebtAck) { a.Binding.ContextDigest = pendingTestDigest("wrong") }},
		{name: "duplicate obligation kind", edit: func(a *DebtAck) { a.Obligations[1].Kind = a.Obligations[0].Kind }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ack := pendingTestAck(t)
			tc.edit(&ack)
			if err := ack.Validate(); err == nil {
				t.Fatal("noncanonical or tampered acknowledgement accepted")
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestPendingAdmissionDecodeIsStrictAndBounded(t *testing.T) {
	t.Parallel()
	ack := pendingTestAck(t)
	raw, err := json.Marshal(ack)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeDebtAck(bytes.NewReader(raw))
	if err != nil || decoded.AckID != ack.AckID {
		t.Fatalf("DecodeDebtAck = %+v, %v", decoded, err)
	}
	if _, err := DecodeDebtAck(strings.NewReader(strings.TrimSuffix(string(raw), "}") + `,"unknown":true}`)); err == nil {
		t.Fatal("unknown acknowledgement field accepted")
	}
	if _, err := DecodeDebtAck(strings.NewReader(string(raw) + `{}`)); err == nil {
		t.Fatal("trailing acknowledgement object accepted")
	}
	if _, err := DecodeDebtAck(strings.NewReader(strings.Repeat(" ", 256<<10) + string(raw))); err == nil {
		t.Fatal("oversize acknowledgement accepted")
	}

	proofRaw, err := json.Marshal(pendingTestProof())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeProofContext(strings.NewReader(strings.TrimSuffix(string(proofRaw), "}") + `,"unknown":true}`)); err == nil {
		t.Fatal("unknown proof-context field accepted")
	}
}
