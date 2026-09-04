// Package verattest provides commit-bound pre-PR verification attestation
// schemas and deterministic, read-only validation primitives (#10463, parent #2391).
//
// In agentic workflows, autonomous workers can author, build, test, and inspect
// changes before submitting a pull request or landing on trunk. Traditional CI
// assumes the first trustworthy execution happens only after submission, which
// turns remote verification queues into redundant re-discovery bottlenecks.
//
// verattest defines the standard-library-only envelope and deterministic verifier
// that allows an intake gate or CI workflow to check structured pre-submission
// evidence against:
//  1. Commit identity (commit_sha matches the candidate tip);
//  2. Verification configuration integrity (verification_config_digest matches);
//  3. Completeness (all required verification steps are present and unskipped);
//  4. Freshness (not expired or stale);
//  5. Step results (all steps reported pass, none failed);
//  6. Structural validity (well-formed JSON, known schema, non-empty mandatory fields).
//
// The verifier emits a typed, closed verdict:
//   - VERIFIED: all bindings, freshness, completeness, and passing outcomes hold.
//   - COMMIT_MISMATCH: attestation was generated for a different commit SHA.
//   - CONFIG_MISMATCH: attestation was generated for a different verification configuration.
//   - INCOMPLETE: one or more required verification steps are missing or were skipped.
//   - STALE: attestation has expired or exceeded its allowed age.
//   - RESULT_FAILED: one or more verification steps reported a failure.
//   - MALFORMED: missing mandatory fields, invalid schema, duplicate steps, or unparseable JSON.
//
// Fences and invariants:
//   - Standard library only: no external or non-standard dependencies.
//   - Read-only and deterministic: no command execution, network access, or side effects.
//   - Fail-closed: missing, malformed, stale, mismatched, or failed evidence always refuses.
//   - Cache hint, not self-trust: an unsigned attestation is an accelerator for selective rerun,
//     never an excuse to bypass authoritative gate verification.
package verattest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// Schema is the canonical schema identifier for verification attestations.
const Schema = "fak-verification-attestation/v1"

// Verdict is the closed result of validating a verification attestation.
type Verdict string

const (
	// VerdictVerified indicates all bindings, completeness, freshness, and passing controls hold.
	VerdictVerified Verdict = "VERIFIED"

	// VerdictCommitMismatch indicates the attestation was generated for a different commit SHA.
	VerdictCommitMismatch Verdict = "COMMIT_MISMATCH"

	// VerdictConfigMismatch indicates the attestation was generated under a different configuration digest.
	VerdictConfigMismatch Verdict = "CONFIG_MISMATCH"

	// VerdictIncomplete indicates one or more required verification steps are missing or were skipped.
	VerdictIncomplete Verdict = "INCOMPLETE"

	// VerdictStale indicates the attestation has expired or exceeded its allowed age.
	VerdictStale Verdict = "STALE"

	// VerdictResultFailed indicates one or more verification steps reported a failure.
	VerdictResultFailed Verdict = "RESULT_FAILED"

	// VerdictMalformed indicates the attestation is syntactically invalid or missing mandatory fields.
	VerdictMalformed Verdict = "MALFORMED"
)

// StepStatus represents the outcome of a single verification step.
type StepStatus string

const (
	// StatusPass indicates the verification step passed successfully.
	StatusPass StepStatus = "pass"

	// StatusFail indicates the verification step failed.
	StatusFail StepStatus = "fail"

	// StatusSkip indicates the verification step was skipped.
	StatusSkip StepStatus = "skip"
)

// Producer identifies the tool and version that generated the attestation.
type Producer struct {
	Tool    string `json:"tool"`
	Version string `json:"version,omitempty"`
}

// Step records the execution outcome and duration of a named verification control.
type Step struct {
	Name      string     `json:"name"`
	Status    StepStatus `json:"status"`
	ElapsedMS int64      `json:"elapsed_ms"`
	Detail    string     `json:"detail,omitempty"`
}

// Attestation is the commit-bound verification evidence envelope.
type Attestation struct {
	Schema                   string            `json:"schema"`
	CommitSHA                string            `json:"commit_sha"`
	VerificationConfigDigest string            `json:"verification_config_digest"`
	Producer                 Producer          `json:"producer"`
	GeneratedAt              time.Time         `json:"generated_at"`
	ExpiresAt                *time.Time        `json:"expires_at,omitempty"`
	Steps                    []Step            `json:"steps"`
	Metadata                 map[string]string `json:"metadata,omitempty"`
}

// Expected specifies the required bindings and control policies against which an
// attestation is evaluated.
type Expected struct {
	// CommitSHA is the expected candidate commit hash (case-insensitive hex match).
	// If empty, commit binding is not enforced.
	CommitSHA string `json:"commit_sha,omitempty"`

	// VerificationConfigDigest is the expected configuration hash (e.g. "sha256:...").
	// If empty, config binding is not enforced.
	VerificationConfigDigest string `json:"verification_config_digest,omitempty"`

	// RequiredSteps names the verification steps that must be present with StatusPass.
	RequiredSteps []string `json:"required_steps,omitempty"`

	// MaxAge, if positive, caps how long ago GeneratedAt may be relative to Now.
	MaxAge time.Duration `json:"max_age,omitempty"`

	// Now is the reference time used to evaluate freshness. If zero, time.Now().UTC() is used.
	Now time.Time `json:"-"`
}

// ValidationResult carries the typed Verdict alongside human/machine diagnostic reasons.
type ValidationResult struct {
	Verdict    Verdict  `json:"verdict"`
	Reason     string   `json:"reason,omitempty"`
	Violations []string `json:"violations,omitempty"`
}

// Verified reports whether the validation result is VerdictVerified.
func (r ValidationResult) Verified() bool {
	return r.Verdict == VerdictVerified
}

// ComputeConfigDigest calculates the canonical "sha256:<hex>" digest over raw configuration bytes.
func ComputeConfigDigest(config []byte) string {
	sum := sha256.Sum256(config)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// New constructs an Attestation with the standard schema tag and validated fields.
func New(commitSHA, configDigest string, producer Producer, steps []Step, generatedAt time.Time, expiresAt *time.Time) Attestation {
	return Attestation{
		Schema:                   Schema,
		CommitSHA:                strings.TrimSpace(commitSHA),
		VerificationConfigDigest: strings.TrimSpace(configDigest),
		Producer:                 producer,
		GeneratedAt:              generatedAt.UTC(),
		ExpiresAt:                expiresAt,
		Steps:                    steps,
	}
}

// JSON serializes the attestation as indented JSON.
func (a Attestation) JSON() ([]byte, error) {
	return json.MarshalIndent(a, "", "  ")
}

// FromJSON deserializes an Attestation from JSON bytes.
func FromJSON(raw []byte) (Attestation, error) {
	var att Attestation
	if err := json.Unmarshal(raw, &att); err != nil {
		return Attestation{}, err
	}
	return att, nil
}

// Check evaluates an Attestation against an Expected specification and returns only the closed Verdict.
func Check(att Attestation, exp Expected) Verdict {
	return Verify(att, exp).Verdict
}

// CheckBytes parses and evaluates raw attestation JSON against an Expected specification.
func CheckBytes(raw []byte, exp Expected) Verdict {
	return VerifyBytes(raw, exp).Verdict
}

// Verify evaluates an Attestation against an Expected specification and returns a typed ValidationResult.
func Verify(att Attestation, exp Expected) ValidationResult {
	// 1. Structural / MALFORMED checks.
	if att.Schema != Schema {
		return ValidationResult{
			Verdict: VerdictMalformed,
			Reason:  fmt.Sprintf("schema mismatch: got %q, want %q", att.Schema, Schema),
		}
	}
	if strings.TrimSpace(att.CommitSHA) == "" {
		return ValidationResult{
			Verdict: VerdictMalformed,
			Reason:  "commit_sha must not be empty",
		}
	}
	if strings.TrimSpace(att.VerificationConfigDigest) == "" {
		return ValidationResult{
			Verdict: VerdictMalformed,
			Reason:  "verification_config_digest must not be empty",
		}
	}
	if strings.TrimSpace(att.Producer.Tool) == "" {
		return ValidationResult{
			Verdict: VerdictMalformed,
			Reason:  "producer.tool must not be empty",
		}
	}
	if att.GeneratedAt.IsZero() {
		return ValidationResult{
			Verdict: VerdictMalformed,
			Reason:  "generated_at must be a valid non-zero timestamp",
		}
	}
	if len(att.Steps) == 0 {
		return ValidationResult{
			Verdict: VerdictMalformed,
			Reason:  "steps list must contain at least one step",
		}
	}

	stepMap := make(map[string]Step, len(att.Steps))
	for i, step := range att.Steps {
		name := strings.TrimSpace(step.Name)
		if name == "" {
			return ValidationResult{
				Verdict: VerdictMalformed,
				Reason:  fmt.Sprintf("step[%d] has empty name", i),
			}
		}
		switch step.Status {
		case StatusPass, StatusFail, StatusSkip:
			// valid status
		default:
			return ValidationResult{
				Verdict: VerdictMalformed,
				Reason:  fmt.Sprintf("step %q has invalid status %q", name, step.Status),
			}
		}
		if _, exists := stepMap[name]; exists {
			return ValidationResult{
				Verdict: VerdictMalformed,
				Reason:  fmt.Sprintf("duplicate step name %q", name),
			}
		}
		stepMap[name] = step
	}

	if att.ExpiresAt != nil && !att.ExpiresAt.IsZero() && att.ExpiresAt.Before(att.GeneratedAt) {
		return ValidationResult{
			Verdict: VerdictMalformed,
			Reason:  fmt.Sprintf("expires_at (%s) precedes generated_at (%s)", att.ExpiresAt.Format(time.RFC3339), att.GeneratedAt.Format(time.RFC3339)),
		}
	}

	// 2. COMMIT_MISMATCH
	expectedCommit := strings.TrimSpace(exp.CommitSHA)
	if expectedCommit != "" && !strings.EqualFold(strings.TrimSpace(att.CommitSHA), expectedCommit) {
		return ValidationResult{
			Verdict: VerdictCommitMismatch,
			Reason:  fmt.Sprintf("commit SHA mismatch: attestation has %s, expected %s", att.CommitSHA, expectedCommit),
		}
	}

	// 3. CONFIG_MISMATCH
	expectedConfig := strings.TrimSpace(exp.VerificationConfigDigest)
	if expectedConfig != "" && strings.TrimSpace(att.VerificationConfigDigest) != expectedConfig {
		return ValidationResult{
			Verdict: VerdictConfigMismatch,
			Reason:  fmt.Sprintf("verification config digest mismatch: attestation has %s, expected %s", att.VerificationConfigDigest, expectedConfig),
		}
	}

	// 4. STALE
	now := exp.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}
	if att.ExpiresAt != nil && !att.ExpiresAt.IsZero() {
		if !now.Before(*att.ExpiresAt) {
			return ValidationResult{
				Verdict: VerdictStale,
				Reason:  fmt.Sprintf("attestation expired at %s (now: %s)", att.ExpiresAt.Format(time.RFC3339), now.Format(time.RFC3339)),
			}
		}
	}
	if exp.MaxAge > 0 {
		age := now.Sub(att.GeneratedAt)
		if age > exp.MaxAge {
			return ValidationResult{
				Verdict: VerdictStale,
				Reason:  fmt.Sprintf("attestation age %s exceeds allowed max_age %s", age.Round(time.Second), exp.MaxAge),
			}
		}
	}

	// 5. INCOMPLETE: check that all required steps are present and not skipped
	var missingSteps []string
	var skippedRequired []string
	for _, req := range exp.RequiredSteps {
		reqName := strings.TrimSpace(req)
		if reqName == "" {
			continue
		}
		step, found := stepMap[reqName]
		if !found {
			missingSteps = append(missingSteps, reqName)
		} else if step.Status == StatusSkip {
			skippedRequired = append(skippedRequired, reqName)
		}
	}
	if len(missingSteps) > 0 || len(skippedRequired) > 0 {
		var msgs []string
		if len(missingSteps) > 0 {
			msgs = append(msgs, fmt.Sprintf("missing required steps: [%s]", strings.Join(missingSteps, ", ")))
		}
		if len(skippedRequired) > 0 {
			msgs = append(msgs, fmt.Sprintf("skipped required steps: [%s]", strings.Join(skippedRequired, ", ")))
		}
		return ValidationResult{
			Verdict:    VerdictIncomplete,
			Reason:     strings.Join(msgs, "; "),
			Violations: append(missingSteps, skippedRequired...),
		}
	}

	// 6. RESULT_FAILED: check if any step failed
	var failedSteps []string
	for _, step := range att.Steps {
		if step.Status == StatusFail {
			failedSteps = append(failedSteps, step.Name)
		}
	}
	if len(failedSteps) > 0 {
		return ValidationResult{
			Verdict:    VerdictResultFailed,
			Reason:     fmt.Sprintf("one or more steps failed: [%s]", strings.Join(failedSteps, ", ")),
			Violations: failedSteps,
		}
	}

	// 7. All checks passed
	return ValidationResult{
		Verdict: VerdictVerified,
		Reason:  "all verification controls passed and bindings matched",
	}
}

// VerifyBytes parses raw JSON bytes and evaluates the attestation against an Expected specification.
func VerifyBytes(raw []byte, exp Expected) ValidationResult {
	if len(strings.TrimSpace(string(raw))) == 0 {
		return ValidationResult{
			Verdict: VerdictMalformed,
			Reason:  "attestation JSON is empty",
		}
	}
	att, err := FromJSON(raw)
	if err != nil {
		return ValidationResult{
			Verdict: VerdictMalformed,
			Reason:  fmt.Sprintf("invalid attestation JSON: %v", err),
		}
	}
	return Verify(att, exp)
}
