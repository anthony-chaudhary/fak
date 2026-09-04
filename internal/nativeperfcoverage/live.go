package nativeperfcoverage

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

// PublicLiveProofSchema identifies the schema version for public live verification evidence proofs.
const PublicLiveProofSchema = "fak-native-performance-live-proof/v1"

// LiveState separates an honest unavailable witness from observed execution.
// LIVE_PENDING is valid evidence of an attempted-but-unavailable route, never a
// claim that the issue's live execution requirement is complete.
type LiveState string

const (
	// LivePending indicates that live hardware execution is currently pending or awaiting route attempts.
	LivePending LiveState = "LIVE_PENDING"
	// LiveProven indicates that verified native execution has been observed and recorded.
	LiveProven LiveState = "LIVE_PROVEN"
)

type executionIdentity struct {
	Engine         string `json:"engine"`
	RuntimeEngine  string `json:"runtime_engine"`
	Planner        string `json:"planner"`
	Model          string `json:"model"`
	ModelOwner     string `json:"model_owner"`
	FallbackCount  int    `json:"fallback_count"`
	FallbackActive bool   `json:"fallback_active"`
	LlamaCPPUsed   bool   `json:"llama_cpp_used"`
	Completed      bool   `json:"completed"`
	OutputTokens   int    `json:"output_tokens"`
	CorrelationKey string `json:"correlation_key"`
}

type liveProofJSON struct {
	Schema         string `json:"schema"`
	Status         string `json:"status"`
	CapturedAtUTC  string `json:"captured_at_utc"`
	CompletedAtUTC string `json:"completed_at_utc"`
	Model          struct {
		Alias  string `json:"alias"`
		Family string `json:"family"`
	} `json:"model"`
	RequiredExecution executionIdentity `json:"required_execution"`
	ObservedExecution executionIdentity `json:"observed_execution"`
	Execution         executionIdentity `json:"execution"`
	Attempts          []struct {
		Route  string `json:"route"`
		Result string `json:"result"`
	} `json:"attempts"`
	Jobs []struct {
		Name   string `json:"name"`
		Status string `json:"status"`
	} `json:"jobs"`
	LiveExecutionObtained       bool `json:"live_execution_obtained"`
	RawLogsCommitted            bool `json:"raw_logs_committed"`
	PrivateIdentifiersCommitted bool `json:"private_identifiers_committed"`
}

// LiveOptions controls freshness and any supervised jobs required by the live
// operator route. Prometheus scrape-job presence is validated separately by
// Validate against tools/grafana/prometheus.yml.
type LiveOptions struct {
	Now          time.Time
	MaxAge       time.Duration
	RequiredJobs []string
}

// ValidateLiveEvidence validates a scrubbed public receipt. An unavailable
// receipt returns LIVE_PENDING with nil error only when it is explicit and
// internally honest. Observed/success receipts must prove the native execution.
func ValidateLiveEvidence(raw []byte, options LiveOptions) (LiveState, error) {
	var proof liveProofJSON
	decoder := json.NewDecoder(bytes.NewReader(raw))
	if err := decoder.Decode(&proof); err != nil {
		return "", err
	}
	if proof.Schema != PublicLiveProofSchema {
		return "", fmt.Errorf("live proof schema %q, want %q", proof.Schema, PublicLiveProofSchema)
	}
	if proof.RawLogsCommitted || proof.PrivateIdentifiersCommitted {
		return "", errors.New("public live proof must not commit raw logs or private identifiers")
	}
	if proof.Model.Family != Qwen38Prefix {
		return "", fmt.Errorf("live proof model family %q, want exact Qwen3.8", proof.Model.Family)
	}
	if err := validateExecutionIdentity(proof.RequiredExecution, false); err != nil {
		return "", fmt.Errorf("required execution: %w", err)
	}

	switch proof.Status {
	case "unavailable":
		if proof.LiveExecutionObtained {
			return "", errors.New("unavailable proof claims live execution was obtained")
		}
		if len(proof.Attempts) == 0 {
			return "", errors.New("unavailable proof has no sanctioned route attempts")
		}
		for _, attempt := range proof.Attempts {
			result := strings.ToLower(attempt.Result)
			if attempt.Route == "" || result == "" || strings.Contains(result, "success") || strings.Contains(result, "observed") {
				return "", fmt.Errorf("unavailable proof has dishonest attempt route=%q result=%q", attempt.Route, attempt.Result)
			}
		}
		if _, err := time.Parse(time.RFC3339, proof.CapturedAtUTC); err != nil {
			return "", fmt.Errorf("captured_at_utc: %w", err)
		}
		return LivePending, nil
	case "observed", "success":
		if !proof.LiveExecutionObtained {
			return "", errors.New("observed proof does not affirm live execution")
		}
		actual := proof.ObservedExecution
		if actual.Engine == "" {
			actual = proof.Execution
		}
		if err := validateExecutionIdentity(actual, true); err != nil {
			return "", fmt.Errorf("observed execution: %w", err)
		}
		if !strings.HasPrefix(actual.Model, Qwen38Prefix) {
			return "", fmt.Errorf("observed model %q is not Qwen3.8", actual.Model)
		}
		completed := proof.CompletedAtUTC
		if completed == "" {
			completed = proof.CapturedAtUTC
		}
		completedAt, err := time.Parse(time.RFC3339, completed)
		if err != nil {
			return "", fmt.Errorf("completed_at_utc: %w", err)
		}
		now := options.Now
		if now.IsZero() {
			now = time.Now().UTC()
		}
		maxAge := options.MaxAge
		if maxAge == 0 {
			maxAge = 15 * time.Minute
		}
		age := now.Sub(completedAt)
		if age < 0 || age > maxAge {
			return "", fmt.Errorf("live proof is stale or future-dated: age=%s maximum=%s", age, maxAge)
		}
		if err := validateJobs(proof, options.RequiredJobs); err != nil {
			return "", err
		}
		return LiveProven, nil
	default:
		return "", fmt.Errorf("unsupported live proof status %q", proof.Status)
	}
}

func validateExecutionIdentity(identity executionIdentity, observed bool) error {
	if identity.Engine != NativeEngine {
		return fmt.Errorf("engine=%q, want exact fak-native", identity.Engine)
	}
	if identity.RuntimeEngine != "inkernel" || identity.Planner != "inkernel" {
		return fmt.Errorf("runtime_engine=%q planner=%q, want inkernel/inkernel", identity.RuntimeEngine, identity.Planner)
	}
	if identity.ModelOwner != "fak" {
		return fmt.Errorf("model_owner=%q, want fak", identity.ModelOwner)
	}
	if identity.FallbackCount != 0 || identity.FallbackActive || identity.LlamaCPPUsed {
		return errors.New("fallback or llama.cpp use is forbidden")
	}
	if observed {
		if !identity.Completed || identity.OutputTokens <= 0 {
			return errors.New("execution did not complete with positive output tokens")
		}
		if identity.CorrelationKey == "" || !correlationKeyRE.MatchString(identity.CorrelationKey) {
			return fmt.Errorf("correlation_key %q is not bounded", identity.CorrelationKey)
		}
	}
	return nil
}

func validateJobs(proof liveProofJSON, required []string) error {
	statuses := make(map[string]string)
	for _, job := range proof.Jobs {
		statuses[job.Name] = job.Status
	}
	for _, name := range required {
		if statuses[name] != "succeeded" {
			return fmt.Errorf("required supervised job %q is absent or not succeeded", name)
		}
	}
	return nil
}
