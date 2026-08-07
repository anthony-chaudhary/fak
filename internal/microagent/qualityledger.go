package microagent

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
)

const QualityLedgerSchema = "fak-microcontext-quality-ledger/1"

type SourceRun struct {
	Schema           string  `json:"schema"`
	Verdict          string  `json:"verdict"`
	LogicalShards    int     `json:"logical_shards"`
	PhysicalWorkers  int     `json:"physical_workers"`
	Completed        int     `json:"completed"`
	Failed           int     `json:"failed"`
	TurnCount        int64   `json:"turn_count"`
	ElapsedMS        int64   `json:"elapsed_ms"`
	Mode             string  `json:"mode"`
	Provider         string  `json:"provider"`
	Model            string  `json:"model"`
	PromptTokens     int64   `json:"prompt_tokens"`
	CompletionTokens int64   `json:"completion_tokens"`
	UsageResponses   int     `json:"usage_responses"`
	TTFTP95MS        float64 `json:"ttft_p95_ms"`
	FirstFailure     string  `json:"first_failure"`
}

type VerificationSummary struct {
	Checked int `json:"checked"`
	Passed  int `json:"passed"`
	Failed  int `json:"failed"`
}

type ClaimFamilies struct {
	Orchestration struct {
		SubmittedTotal  int `json:"submitted_total"`
		PhysicalWorkers int `json:"physical_workers"`
	} `json:"orchestration"`
	Inference struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		UsageResponses   int   `json:"usage_responses"`
	} `json:"inference"`
	UsefulWork struct {
		VerifiedCompletions int     `json:"verified_completions"`
		PerWallSecond       float64 `json:"per_wall_second"`
	} `json:"useful_work"`
}

type QualityLedger struct {
	Schema        string              `json:"schema"`
	RunID         string              `json:"run_id"`
	BaseID        string              `json:"base_id"`
	CheckContract string              `json:"check_contract"`
	Mode          string              `json:"mode"`
	Provider      string              `json:"provider"`
	Model         string              `json:"model"`
	Submitted     int                 `json:"submitted"`
	Retired       int                 `json:"retired"`
	Failed        int                 `json:"failed"`
	Cancelled     int                 `json:"cancelled"`
	Retried       int                 `json:"retried"`
	ElapsedMS     int64               `json:"elapsed_ms"`
	Verification  VerificationSummary `json:"verification"`
	ClaimFamilies ClaimFamilies       `json:"claim_families"`
	ErrorClasses  map[string]int      `json:"error_classes"`
	Outcomes      map[string]int      `json:"outcomes"`
	SampleIDs     []string            `json:"sample_ids"`
	SampleLimit   int                 `json:"sample_limit"`
}

type OutcomeCheck interface{ Verify(contextID string) error }
type OutcomeCheckFunc func(string) error

func (f OutcomeCheckFunc) Verify(id string) error { return f(id) }

func IngestSourceRun(data []byte, runID, baseID string, verifier OutcomeCheck, sampleLimit int) (QualityLedger, error) {
	var w SourceRun
	if err := json.Unmarshal(data, &w); err != nil {
		return QualityLedger{}, err
	}
	if w.LogicalShards <= 0 || w.Completed < 0 || w.Failed < 0 || w.Completed+w.Failed != w.LogicalShards || w.TurnCount != int64(w.Completed) || w.ElapsedMS <= 0 {
		return QualityLedger{}, errors.New("microagent: witness accounting does not reconcile")
	}
	if runID == "" || baseID == "" || verifier == nil {
		return QualityLedger{}, errors.New("microagent: run, base, and independent verifier are required")
	}
	if sampleLimit < 0 {
		return QualityLedger{}, errors.New("microagent: sample limit must be nonnegative")
	}
	l := QualityLedger{Schema: QualityLedgerSchema, RunID: runID, BaseID: baseID, CheckContract: "caller-supplied-independent-outcome-check", Mode: w.Mode, Provider: w.Provider, Model: w.Model, Submitted: w.LogicalShards, Retired: w.Completed, Failed: w.Failed, ElapsedMS: w.ElapsedMS, SampleLimit: sampleLimit, ErrorClasses: map[string]int{}}
	l.Outcomes = map[string]int{"success": w.Completed, "error": w.Failed, "refusal": 0}
	l.ClaimFamilies.Orchestration.SubmittedTotal, l.ClaimFamilies.Orchestration.PhysicalWorkers = w.LogicalShards, w.PhysicalWorkers
	l.ClaimFamilies.Inference.PromptTokens, l.ClaimFamilies.Inference.CompletionTokens, l.ClaimFamilies.Inference.UsageResponses = w.PromptTokens, w.CompletionTokens, w.UsageResponses
	for i := 0; i < w.Completed; i++ {
		id := fmt.Sprintf("ctx-%08d", i)
		l.Verification.Checked++
		if err := verifier.Verify(id); err != nil {
			l.Verification.Failed++
			l.ErrorClasses["verifier"]++
		} else {
			l.Verification.Passed++
		}
		if len(l.SampleIDs) < sampleLimit {
			l.SampleIDs = append(l.SampleIDs, id)
		}
	}
	l.Verification.Failed += w.Failed
	if w.Failed > 0 {
		l.ErrorClasses["run"] += w.Failed
	}
	l.ClaimFamilies.UsefulWork.VerifiedCompletions = l.Verification.Passed
	l.ClaimFamilies.UsefulWork.PerWallSecond = float64(l.Verification.Passed) / (float64(w.ElapsedMS) / 1000)
	return l, VerifyQualityLedger(l)
}

func VerifyQualityLedger(l QualityLedger) error {
	if l.Schema != QualityLedgerSchema || l.RunID == "" || l.BaseID == "" || l.CheckContract == "" {
		return errors.New("microagent: ledger identity missing")
	}
	if l.Submitted != l.Retired+l.Failed+l.Cancelled || l.Verification.Checked != l.Retired || l.Verification.Passed+l.Verification.Failed != l.Submitted {
		return errors.New("microagent: ledger totals do not reconcile")
	}
	if l.Outcomes == nil || l.Outcomes["success"] != l.Retired || l.Outcomes["error"] != l.Failed || l.Outcomes["refusal"] != l.Cancelled {
		return errors.New("microagent: outcome counters do not reconcile")
	}
	if l.ClaimFamilies.Orchestration.SubmittedTotal != l.Submitted {
		return errors.New("microagent: orchestration family mismatch")
	}
	if l.ClaimFamilies.UsefulWork.VerifiedCompletions != l.Verification.Passed {
		return errors.New("microagent: useful-work denominator mismatch")
	}
	if len(l.SampleIDs) > l.SampleLimit {
		return errors.New("microagent: sample cardinality exceeded")
	}
	if !sort.StringsAreSorted(l.SampleIDs) {
		return errors.New("microagent: sample context IDs are unstable")
	}
	return nil
}
