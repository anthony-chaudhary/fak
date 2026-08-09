package rsiloop

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ExtensionKind is the closed set of agent-authored extension proposals.
type ExtensionKind string

const (
	ExtensionLinter ExtensionKind = "linter"
	ExtensionSkill  ExtensionKind = "skill"
	ExtensionPrompt ExtensionKind = "prompt-policy"
	ExtensionCode   ExtensionKind = "code"
)

// ExtensionProposal is the immutable envelope presented to an independent witness.
type ExtensionProposal struct {
	Kind           ExtensionKind `json:"kind"`
	Provenance     string        `json:"provenance"`
	ClaimedMetric  string        `json:"claimed_metric"`
	Scope          []string      `json:"scope"`
	ArtifactDigest string        `json:"artifact_digest"`
	RollbackRecipe string        `json:"rollback_recipe"`
	CandidatePaths []string      `json:"candidate_paths"`
}

type PromotionEvidence struct {
	IsolationRef    string  `json:"isolation_ref"`
	WitnessRef      string  `json:"witness_ref"`
	TestsPassed     bool    `json:"tests_passed"`
	TruthPassed     bool    `json:"truth_passed"`
	MetricMeasured  bool    `json:"metric_measured"`
	TunedBaseline   float64 `json:"tuned_baseline"`
	CandidateMetric float64 `json:"candidate_metric"`
}

type PromotionVerdict string

var protectedPromotionPaths = []string{
	"internal/rsiloop/judge",
	"internal/rsiloop/verifier",
	"internal/rsiloop/metric",
	"internal/rsiloop/policy",
}

const (
	PromotionKeep   PromotionVerdict = "KEEP"
	PromotionRevert PromotionVerdict = "REVERT"
	PromotionRefuse PromotionVerdict = "REFUSE"
)

type PromotionReceipt struct {
	Schema     string            `json:"schema"`
	State      string            `json:"state"`
	Proposal   ExtensionProposal `json:"proposal"`
	Evidence   PromotionEvidence `json:"evidence"`
	Verdict    PromotionVerdict  `json:"verdict"`
	Reason     string            `json:"reason"`
	RecordedAt time.Time         `json:"recorded_at"`
}

// EvaluateExtensionProposal fails closed. protectedPaths must name the judge,
// verifier, metric, and keep/revert policy implementation owned by the witness.
func EvaluateExtensionProposal(p ExtensionProposal, e PromotionEvidence, protectedPaths []string) (PromotionVerdict, string) {
	if err := validateExtensionProposal(p); err != nil {
		return PromotionRefuse, err.Error()
	}
	guards := append(append([]string(nil), protectedPromotionPaths...), protectedPaths...)
	for _, candidate := range p.CandidatePaths {
		for _, protected := range guards {
			if pathsOverlap(candidate, protected) {
				return PromotionRefuse, "candidate overlaps witness-controlled judge/verifier/metric/policy: " + protected
			}
		}
	}
	if strings.TrimSpace(e.IsolationRef) == "" {
		return PromotionRevert, "missing isolated-application witness"
	}
	if strings.TrimSpace(e.WitnessRef) == "" {
		return PromotionRevert, "missing independent witness"
	}
	if !e.TestsPassed {
		return PromotionRevert, "test witness failed"
	}
	if !e.TruthPassed {
		return PromotionRevert, "truth witness failed"
	}
	if !e.MetricMeasured {
		return PromotionRevert, "metric witness missing"
	}
	if e.CandidateMetric <= e.TunedBaseline {
		return PromotionRevert, "no strict net-true gain against tuned baseline"
	}
	return PromotionKeep, "independent witnesses confirm a strict net-true gain"
}

func validateExtensionProposal(p ExtensionProposal) error {
	switch p.Kind {
	case ExtensionLinter, ExtensionSkill, ExtensionPrompt, ExtensionCode:
	default:
		return errors.New("unknown extension kind")
	}
	if strings.TrimSpace(p.Provenance) == "" || strings.TrimSpace(p.ClaimedMetric) == "" || len(p.Scope) == 0 || strings.TrimSpace(p.RollbackRecipe) == "" || len(p.CandidatePaths) == 0 {
		return errors.New("proposal envelope is incomplete")
	}
	raw, err := hex.DecodeString(p.ArtifactDigest)
	if err != nil || len(raw) != sha256.Size {
		return errors.New("artifact_digest must be a SHA-256 hex digest")
	}
	return nil
}

func pathsOverlap(a, b string) bool {
	clean := func(s string) string { return strings.Trim(strings.ReplaceAll(filepath.Clean(s), "\\", "/"), "/") }
	a, b = clean(a), clean(b)
	if a == "" || b == "" {
		return false
	}
	return a == b || strings.HasPrefix(a, b+"/") || strings.HasPrefix(b, a+"/")
}

// RunExtensionPromotion writes PREPARED before invoking the independent witness.
// A crash can therefore leave only an unkept receipt, never an implicit KEEP.
func RunExtensionPromotion(dir string, p ExtensionProposal, protected []string, witness func(ExtensionProposal) PromotionEvidence) (PromotionReceipt, error) {
	if err := validateExtensionProposal(p); err != nil {
		return PromotionReceipt{}, err
	}
	receipt := PromotionReceipt{Schema: "fak.rsiloop-extension-promotion/1", State: "PREPARED", Proposal: p, RecordedAt: time.Now().UTC()}
	path := filepath.Join(dir, p.ArtifactDigest+".json")
	if err := writePromotionReceipt(path, receipt); err != nil {
		return PromotionReceipt{}, err
	}
	if witness == nil {
		receipt.Verdict, receipt.Reason = PromotionRevert, "missing independent witness"
	} else {
		receipt.Evidence = witness(p)
		receipt.Verdict, receipt.Reason = EvaluateExtensionProposal(p, receipt.Evidence, protected)
	}
	receipt.State = "FINAL"
	receipt.RecordedAt = time.Now().UTC()
	if err := writePromotionReceipt(path, receipt); err != nil {
		return PromotionReceipt{}, err
	}
	return receipt, nil
}

func writePromotionReceipt(path string, r PromotionReceipt) error {
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	tmp := fmt.Sprintf("%s.tmp-%d", path, os.Getpid())
	if err = os.WriteFile(tmp, b, 0600); err != nil {
		return err
	}
	if err = os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}
