package agenticbench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/armbench"
)

const (
	ALEExperimentIdentitySchema = "fak.ale-experiment-identity.v1"
	ALEResumeReceiptSchema      = "fak.ale-resume-preflight.v1"
	ALEIdentitySidecar          = "fak-experiment-identity.json"
	ALEOutputIdentityFile       = "fak-output-identity.json"
)

const (
	ReasonALEIdentityInvalid         = "ALE_IDENTITY_INVALID"
	ReasonALEOutputIdentityMissing   = "ALE_OUTPUT_IDENTITY_MISSING"
	ReasonALEOutputIdentityMismatch  = "ALE_OUTPUT_IDENTITY_MISMATCH"
	ReasonALEResumeIdentityMissing   = "ALE_RESUME_IDENTITY_MISSING"
	ReasonALEResumeIdentityMismatch  = "ALE_RESUME_IDENTITY_MISMATCH"
	ReasonALEResumeSourceAmbiguous   = "ALE_RESUME_SOURCE_AMBIGUOUS"
	ReasonALEResumeReadFailed        = "ALE_RESUME_READ_FAILED"
	ReasonALECrossoverResumeUnscoped = "ALE_CROSSOVER_RESUME_UNSCOPED"
)

type ALEResumeDisposition string

const (
	ALEResumeFresh   ALEResumeDisposition = "fresh"
	ALEResumed       ALEResumeDisposition = "resumed"
	ALEResumeRefused ALEResumeDisposition = "refused"
)

// ALEExperimentIdentity pins every experiment term that can make two native
// ALE run units look equal while making their measurements incomparable.
type ALEExperimentIdentity struct {
	TaskPath           string `json:"task_path"`
	SourceRepo         string `json:"source_repo"`
	SourceSHA          string `json:"source_sha"`
	Harness            string `json:"harness"`
	AgentID            string `json:"agent_id"`
	Model              string `json:"model"`
	Effort             string `json:"effort"`
	Endpoint           string `json:"endpoint"`
	Arm                string `json:"arm"`
	Snapshot           string `json:"snapshot"`
	BudgetSeconds      int    `json:"budget_seconds"`
	MaxTokens          int    `json:"max_tokens"`
	RetryPolicy        string `json:"retry_policy"`
	PromptSuffixSHA256 string `json:"prompt_suffix_sha256"`
	Repetition         int    `json:"repetition"`
	Variant            int    `json:"variant"`
}

// ALEIdentityRecord is written beside an ALE run before launch. Keeping the
// full projection, not only its digest, makes a refusal independently auditable.
type ALEIdentityRecord struct {
	Schema   string                `json:"schema"`
	Identity ALEExperimentIdentity `json:"identity"`
}

// ALEOutputRoot binds an arm/repetition-specific root to the same immutable
// identity as the launch. A blank ManifestIdentity denotes a shared ALE root.
type ALEOutputRoot struct {
	Path             string `json:"path"`
	ManifestIdentity string `json:"manifest_identity,omitempty"`
}

type ALELaunchSpec struct {
	Identity      ALEExperimentIdentity `json:"identity"`
	Output        ALEOutputRoot         `json:"output_root"`
	Crossover     bool                  `json:"crossover"`
	DisableResume bool                  `json:"disable_resume"`
}

// ALEResumeReceipt is embedded unchanged in the comparison manifest. Refusals
// are receipts too, so a skipped launch cannot disappear from the experiment.
type ALEResumeReceipt struct {
	Schema           string               `json:"schema"`
	Decision         ALEResumeDisposition `json:"decision"`
	ManifestIdentity string               `json:"manifest_identity,omitempty"`
	OutputRoot       string               `json:"output_root"`
	SourceRunID      string               `json:"source_run_id,omitempty"`
	SourceRunPath    string               `json:"source_run_path,omitempty"`
	Detail           string               `json:"detail"`
}

type ALEAliasError struct {
	Reason string
	Detail string
}

func (e *ALEAliasError) Error() string { return e.Reason + ": " + e.Detail }

func IsALEAliasReason(err error, reason string) bool {
	var target *ALEAliasError
	return errors.As(err, &target) && target.Reason == reason
}

// ALEManifestIdentity projects ALE's wider launch contract into a valid
// ArmBench manifest. The source content hash binds the complete ALE projection,
// while ArmBench supplies the canonical immutable manifest identity.
func ALEManifestIdentity(identity ALEExperimentIdentity) (string, error) {
	if err := validateALEIdentity(identity); err != nil {
		return "", err
	}
	encoded, err := json.Marshal(identity)
	if err != nil {
		return "", &ALEAliasError{Reason: ReasonALEIdentityInvalid, Detail: err.Error()}
	}
	projectionHash := digest(encoded)
	judgeHash := digest([]byte("ale-official-evaluator@" + identity.SourceSHA))
	controlHash := digest([]byte("ale-identity-control@" + projectionHash))
	manifest := &armbench.Manifest{
		Schema: armbench.ManifestSchema,
		ID:     "ale-resume-preflight",
		Sources: []armbench.Source{{
			Name: "agents-last-exam", Repo: identity.SourceRepo, SHA: identity.SourceSHA,
			Path: identity.TaskPath, ContentHash: projectionHash,
		}},
		Model: armbench.Model{
			Provider: identity.Endpoint, Snapshot: identity.Model + "@" + identity.Snapshot,
			Region: "ale-run", Sampling: armbench.Sampling{Temperature: 0, TopP: 1}, MaxTokens: identity.MaxTokens,
		},
		Corpus: armbench.Corpus{ID: identity.TaskPath, Hash: projectionHash, TaskCount: 1},
		Judge:  armbench.Judge{ID: "ale-official", Hash: judgeHash, Kind: "external"},
		Trials: armbench.Trials{Count: 1, Seed: int64(identity.Repetition), Order: armbench.OrderCounterbalanced, Concurrency: 1},
		Environment: armbench.Environment{
			OS: "ale-vm", Arch: identity.Harness, HostClass: identity.Effort,
			FakVersion: identity.RetryPolicy, PricingDate: "1970-01-01",
		},
		Arms: []armbench.Arm{
			{ID: "ale-identity-control", Kind: armbench.ArmBaseline, PromptHash: controlHash},
			{ID: identity.Arm, Kind: armbench.ArmUpstreamTreatment, PromptHash: identity.PromptSuffixSHA256, SourceName: "agents-last-exam"},
		},
	}
	if err := manifest.Validate(); err != nil {
		return "", &ALEAliasError{Reason: ReasonALEIdentityInvalid, Detail: fmt.Sprintf("ArmBench projection: %v", err)}
	}
	return manifest.Identity(), nil
}

// CheckALEResumeAlias runs before ALE's native resume filter. It mirrors the
// native on-disk lookup, then requires every terminal collision to carry the
// exact full identity before allowing ALE to skip the unit.
func CheckALEResumeAlias(spec ALELaunchSpec) (ALEResumeReceipt, error) {
	receipt := ALEResumeReceipt{
		Schema: ALEResumeReceiptSchema, Decision: ALEResumeRefused,
		OutputRoot: filepath.Clean(spec.Output.Path),
	}
	manifestIdentity, err := ALEManifestIdentity(spec.Identity)
	if err != nil {
		return refuseALEResume(receipt, reasonOfALEError(err, ReasonALEIdentityInvalid), "%v", err)
	}
	receipt.ManifestIdentity = manifestIdentity
	if strings.TrimSpace(spec.Output.Path) == "" {
		return refuseALEResume(receipt, ReasonALEIdentityInvalid, "output root is empty")
	}
	if spec.Output.ManifestIdentity != "" && spec.Output.ManifestIdentity != manifestIdentity {
		return refuseALEResume(receipt, ReasonALEOutputIdentityMismatch,
			"output root declares identity %s but launch identity is %s", spec.Output.ManifestIdentity, manifestIdentity)
	}
	if spec.Output.ManifestIdentity != "" {
		record, readErr := readALEIdentityRecord(filepath.Join(spec.Output.Path, ALEOutputIdentityFile))
		if readErr != nil {
			return refuseALEResume(receipt, ReasonALEOutputIdentityMissing,
				"identity-scoped output root has no readable %s: %v", ALEOutputIdentityFile, readErr)
		}
		rootIdentity, identityErr := ALEManifestIdentity(record.Identity)
		if identityErr != nil || rootIdentity != manifestIdentity {
			return refuseALEResume(receipt, ReasonALEOutputIdentityMismatch,
				"output-root marker identity is %s, launch identity is %s (marker error: %v)", rootIdentity, manifestIdentity, identityErr)
		}
	}
	if spec.DisableResume {
		receipt.Decision = ALEResumeFresh
		receipt.Detail = "fresh launch: ALE --disable-resume is required and present"
		return receipt, nil
	}

	candidates, err := terminalALERuns(spec.Output.Path, spec.Identity)
	if err != nil {
		return refuseALEResume(receipt, ReasonALEResumeReadFailed, "%v", err)
	}
	exact := make([]aleTerminalRun, 0, len(candidates))
	for _, candidate := range candidates {
		record, readErr := readALEIdentityRecord(filepath.Join(candidate.Path, ALEIdentitySidecar))
		if readErr != nil {
			receipt.SourceRunID = candidate.RunID
			receipt.SourceRunPath = filepath.ToSlash(candidate.Path)
			return refuseALEResume(receipt, ReasonALEResumeIdentityMissing,
				"terminal ALE run %q has no readable full identity: %v", candidate.RunID, readErr)
		}
		priorIdentity, identityErr := ALEManifestIdentity(record.Identity)
		if identityErr != nil {
			receipt.SourceRunID = candidate.RunID
			receipt.SourceRunPath = filepath.ToSlash(candidate.Path)
			return refuseALEResume(receipt, ReasonALEResumeIdentityMissing,
				"terminal ALE run %q carries an invalid identity: %v", candidate.RunID, identityErr)
		}
		if priorIdentity != manifestIdentity {
			receipt.SourceRunID = candidate.RunID
			receipt.SourceRunPath = filepath.ToSlash(candidate.Path)
			return refuseALEResume(receipt, ReasonALEResumeIdentityMismatch,
				"native ALE key collides with run %q under identity %s; planned identity is %s", candidate.RunID, priorIdentity, manifestIdentity)
		}
		exact = append(exact, candidate)
	}
	if spec.Crossover && spec.Output.ManifestIdentity != manifestIdentity {
		return refuseALEResume(receipt, ReasonALECrossoverResumeUnscoped,
			"crossover auto-resume requires an arm/repetition output root carrying identity %s; otherwise pass --disable-resume", manifestIdentity)
	}
	if len(exact) > 1 {
		return refuseALEResume(receipt, ReasonALEResumeSourceAmbiguous,
			"%d exact terminal runs share the native ALE key; source run identity is ambiguous", len(exact))
	}
	if len(exact) == 1 {
		receipt.Decision = ALEResumed
		receipt.SourceRunID = exact[0].RunID
		receipt.SourceRunPath = filepath.ToSlash(exact[0].Path)
		receipt.Detail = "exact full-identity terminal run admitted for resume"
		return receipt, nil
	}
	receipt.Decision = ALEResumeFresh
	receipt.Detail = "fresh launch: no terminal run collides with the native ALE key"
	return receipt, nil
}

type aleTerminalRun struct {
	RunID string
	Path  string
}

func terminalALERuns(root string, identity ALEExperimentIdentity) ([]aleTerminalRun, error) {
	variantDir := filepath.Join(root, slugALEAgent(identity.AgentID), slugALEModel(identity.Model), slugALETask(identity.TaskPath), fmt.Sprintf("v%d", identity.Variant))
	entries, err := os.ReadDir(variantDir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read native ALE key directory: %w", err)
	}
	var runs []aleTerminalRun
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		path := filepath.Join(variantDir, entry.Name())
		b, readErr := os.ReadFile(filepath.Join(path, "run.json"))
		if readErr != nil {
			continue
		}
		var run struct {
			RunID  string `json:"run_id"`
			Status string `json:"status"`
		}
		if json.Unmarshal(b, &run) != nil || (run.Status != "completed" && run.Status != "timeout") {
			continue
		}
		if strings.TrimSpace(run.RunID) == "" {
			return nil, fmt.Errorf("terminal run at %s has no run_id", filepath.ToSlash(path))
		}
		runs = append(runs, aleTerminalRun{RunID: run.RunID, Path: path})
	}
	return runs, nil
}

func readALEIdentityRecord(path string) (ALEIdentityRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return ALEIdentityRecord{}, err
	}
	defer f.Close()
	decoder := json.NewDecoder(f)
	decoder.DisallowUnknownFields()
	var record ALEIdentityRecord
	if err := decoder.Decode(&record); err != nil {
		return ALEIdentityRecord{}, err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
		return ALEIdentityRecord{}, err
	}
	if record.Schema != ALEExperimentIdentitySchema {
		return ALEIdentityRecord{}, fmt.Errorf("schema %q, want %q", record.Schema, ALEExperimentIdentitySchema)
	}
	return record, nil
}

func validateALEIdentity(identity ALEExperimentIdentity) error {
	fields := []struct {
		name  string
		value string
	}{
		{"task_path", identity.TaskPath}, {"source_repo", identity.SourceRepo}, {"source_sha", identity.SourceSHA},
		{"harness", identity.Harness}, {"agent_id", identity.AgentID}, {"model", identity.Model},
		{"effort", identity.Effort}, {"endpoint", identity.Endpoint}, {"arm", identity.Arm},
		{"snapshot", identity.Snapshot}, {"retry_policy", identity.RetryPolicy}, {"prompt_suffix_sha256", identity.PromptSuffixSHA256},
	}
	for _, field := range fields {
		if strings.TrimSpace(field.value) == "" || strings.TrimSpace(field.value) != field.value {
			return &ALEAliasError{Reason: ReasonALEIdentityInvalid, Detail: fmt.Sprintf("%s must be non-empty with no surrounding whitespace", field.name)}
		}
	}
	if identity.Arm == "ale-identity-control" {
		return &ALEAliasError{Reason: ReasonALEIdentityInvalid, Detail: "arm uses the reserved ArmBench projection control id"}
	}
	if !validLowerHex(identity.SourceSHA, 40) {
		return &ALEAliasError{Reason: ReasonALEIdentityInvalid, Detail: "source_sha must be a full lowercase 40-hex commit"}
	}
	if !strings.HasPrefix(identity.PromptSuffixSHA256, "sha256:") || !validLowerHex(strings.TrimPrefix(identity.PromptSuffixSHA256, "sha256:"), 64) {
		return &ALEAliasError{Reason: ReasonALEIdentityInvalid, Detail: "prompt_suffix_sha256 must be sha256:<64 lowercase hex>"}
	}
	if identity.BudgetSeconds <= 0 || identity.MaxTokens <= 0 {
		return &ALEAliasError{Reason: ReasonALEIdentityInvalid, Detail: "budget_seconds and max_tokens must both be > 0"}
	}
	if identity.Repetition < 0 || identity.Variant < 0 {
		return &ALEAliasError{Reason: ReasonALEIdentityInvalid, Detail: "repetition and variant must both be >= 0"}
	}
	return nil
}

func refuseALEResume(receipt ALEResumeReceipt, reason, format string, args ...any) (ALEResumeReceipt, error) {
	receipt.Decision = ALEResumeRefused
	receipt.Detail = fmt.Sprintf(format, args...)
	return receipt, &ALEAliasError{Reason: reason, Detail: receipt.Detail}
}

func reasonOfALEError(err error, fallback string) string {
	var target *ALEAliasError
	if errors.As(err, &target) {
		return target.Reason
	}
	return fallback
}

func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func validLowerHex(s string, size int) bool {
	if len(s) != size || strings.ToLower(s) != s {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

// These slugs adapt ALE's native run layout at
// ale_run/orchestration/run_writer.py@75a3f866535946b67f9a57e4f158eb30ad50be8a
// (Apache-2.0), so the preflight inspects the same key ALE will resume from.
var aleSlugModelRE = regexp.MustCompile(`[^a-z0-9-]+`)
var aleSlugAgentRE = regexp.MustCompile(`[^a-z0-9_]+`)

func slugALEModel(model string) string {
	if model == "" {
		return "unknown-model"
	}
	s := strings.ToLower(model)
	s = strings.NewReplacer(".", "-", "/", "-", "_", "-").Replace(s)
	s = strings.Trim(aleSlugModelRE.ReplaceAllString(s, "-"), "-")
	if s == "" {
		return "unknown-model"
	}
	return s
}

func slugALETask(task string) string {
	return strings.ReplaceAll(strings.Trim(task, "/"), "/", "__")
}

func slugALEAgent(agent string) string {
	s := strings.ToLower(strings.ReplaceAll(agent, "-", "_"))
	s = strings.Trim(aleSlugAgentRE.ReplaceAllString(s, "_"), "_")
	if s == "" {
		return "unknown"
	}
	return s
}
