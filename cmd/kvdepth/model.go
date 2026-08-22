package main

import (
	"errors"
	"fmt"
	"slices"
	"strings"
)

const (
	manifestSchema    = "fak.kv-prefix-depth-campaign/v1"
	observationSchema = "fak.kv-prefix-depth-observation/v1"
	depthReportSchema = "fak.kv-prefix-depth-report/v1"
	selfcheckSchema   = "fak.kv-prefix-depth-selfcheck/v1"
)

type Manifest struct {
	Schema         string           `json:"schema"`
	CampaignID     string           `json:"campaign_id"`
	Pins           Pins             `json:"pins"`
	Tokenization   Tokenization     `json:"tokenization"`
	Axes           Axes             `json:"axes"`
	ReferenceArm   ArmCoordinates   `json:"reference_arm"`
	Ordering       OrderingContract `json:"ordering"`
	Reset          ResetContract    `json:"reset"`
	Confidence     Confidence       `json:"confidence"`
	UsefulWorkRule string           `json:"useful_work_rule"`
}

type Pins struct {
	Backend         string `json:"backend"`
	BackendRevision string `json:"backend_revision"`
	RuntimeRevision string `json:"runtime_revision"`
	ModelID         string `json:"model_id"`
	ModelRevision   string `json:"model_revision"`
	FakRevision     string `json:"fak_revision"`
}

type Tokenization struct {
	TokenizerID          string `json:"tokenizer_id"`
	TokenizerRevision    string `json:"tokenizer_revision"`
	PromptTemplateSHA256 string `json:"prompt_template_sha256"`
	Unit                 string `json:"unit"`
}

type Axes struct {
	PrefixDepthTokens []int           `json:"prefix_depth_tokens"`
	SuffixPatterns    []SuffixPattern `json:"suffix_patterns"`
	TurnCounts        []int           `json:"turn_counts"`
	Concurrency       []int           `json:"concurrency"`
	PressureArms      []PressureArm   `json:"pressure_arms"`
	Repetitions       int             `json:"repetitions"`
}

type SuffixPattern struct {
	ID            string  `json:"id"`
	Description   string  `json:"description"`
	ChurnFraction float64 `json:"churn_fraction"`
}

type PressureArm struct {
	ID                string `json:"id"`
	Phase             string `json:"phase"`
	CompetingTokens   int64  `json:"competing_tokens"`
	RemovedBeforeNext bool   `json:"removed_before_next"`
}

type ArmCoordinates struct {
	TurnCount     int    `json:"turn_count"`
	Concurrency   int    `json:"concurrency"`
	PressurePhase string `json:"pressure_phase"`
}

type OrderingContract struct {
	Strategy     string   `json:"strategy"`
	Seed         int64    `json:"seed"`
	ThermalOrder []string `json:"cold_warm_order"`
	RequestOrder []string `json:"request_order"`
}

type ResetContract struct {
	BeforeCampaign string `json:"before_campaign"`
	BeforeColdArm  string `json:"before_cold_arm"`
	BeforeWarmArm  string `json:"before_warm_arm"`
	AfterPressure  string `json:"after_pressure"`
}

type Confidence struct {
	Level              float64 `json:"level"`
	ReliableReuseRatio float64 `json:"reliable_reuse_ratio"`
	MinimumSamples     int     `json:"minimum_samples"`
}

type Observation struct {
	Schema              string       `json:"schema"`
	CampaignID          string       `json:"campaign_id"`
	RequestID           string       `json:"request_id"`
	ArmID               string       `json:"arm_id"`
	PrefixDepthTokens   int          `json:"prefix_depth_tokens"`
	SuffixPattern       string       `json:"suffix_pattern"`
	TurnCount           int          `json:"turn_count"`
	Concurrency         int          `json:"concurrency"`
	PressurePhase       string       `json:"pressure_phase"`
	Repetition          int          `json:"repetition"`
	ThermalState        string       `json:"thermal_state"`
	OrderIndex          int          `json:"order_index"`
	ResetProcedure      string       `json:"reset_procedure"`
	Pins                Pins         `json:"pins"`
	Tokenization        Tokenization `json:"tokenization"`
	PromptTokens        int64        `json:"prompt_tokens"`
	TTFTMillis          float64      `json:"ttft_ms"`
	UsefulWorkCompleted bool         `json:"useful_work_completed"`
	SemanticPromptEqual bool         `json:"semantic_prompt_equal"`
	TokenPrefixEqual    bool         `json:"token_prefix_equal"`
	KV                  *KVSignals   `json:"backend_kv,omitempty"`
}

// KVSignals keeps backend evidence optional. Unsupported metrics stay absent
// instead of being converted into zeros that could fabricate a boundary.
type KVSignals struct {
	Admitted          *bool    `json:"admitted,omitempty"`
	CachedInputTokens *int64   `json:"cached_input_tokens,omitempty"`
	ResidentTokens    *int64   `json:"resident_tokens,omitempty"`
	OccupancyRatio    *float64 `json:"occupancy_ratio,omitempty"`
	Evictions         *int64   `json:"evictions,omitempty"`
	Preemptions       *int64   `json:"preemptions,omitempty"`
}

func (m Manifest) Validate() error {
	if m.Schema != manifestSchema {
		return fmt.Errorf("schema: got %q, want %q", m.Schema, manifestSchema)
	}
	if strings.TrimSpace(m.CampaignID) == "" {
		return errors.New("campaign_id is required")
	}
	if err := validatePins(m.Pins); err != nil {
		return err
	}
	if err := validateTokenization(m.Tokenization); err != nil {
		return err
	}
	if len(m.Axes.PrefixDepthTokens) < 6 || !positiveUniqueSorted(m.Axes.PrefixDepthTokens) {
		return errors.New("prefix_depth_tokens must contain at least six positive, unique, ascending depths")
	}
	if len(m.Axes.SuffixPatterns) < 2 {
		return errors.New("at least two suffix_patterns are required")
	}
	suffixIDs := map[string]bool{}
	for _, pattern := range m.Axes.SuffixPatterns {
		if strings.TrimSpace(pattern.ID) == "" || strings.TrimSpace(pattern.Description) == "" || pattern.ChurnFraction < 0 || pattern.ChurnFraction > 1 || suffixIDs[pattern.ID] {
			return errors.New("suffix_patterns require unique ids, descriptions, and churn_fraction in [0,1]")
		}
		suffixIDs[pattern.ID] = true
	}
	if len(m.Axes.TurnCounts) < 2 || !positiveUnique(m.Axes.TurnCounts) {
		return errors.New("turn_counts must contain at least two positive values")
	}
	if len(m.Axes.Concurrency) < 2 || !positiveUnique(m.Axes.Concurrency) {
		return errors.New("concurrency must contain at least two positive values")
	}
	if m.Axes.Repetitions < 3 {
		return errors.New("repetitions must be at least three")
	}
	if !slices.Contains(m.Axes.TurnCounts, m.ReferenceArm.TurnCount) || !slices.Contains(m.Axes.Concurrency, m.ReferenceArm.Concurrency) {
		return errors.New("reference_arm must select declared turn_count and concurrency values")
	}
	phases := map[string]bool{}
	pressureSeen := false
	for _, arm := range m.Axes.PressureArms {
		if arm.ID == "" || !oneOf(arm.Phase, "baseline", "pressure", "recovery") {
			return errors.New("pressure_arms require ids and baseline|pressure|recovery phases")
		}
		phases[arm.Phase] = true
		if arm.Phase == "pressure" && arm.CompetingTokens > 0 && arm.RemovedBeforeNext {
			pressureSeen = true
		}
	}
	if !phases["baseline"] || !phases["pressure"] || !phases["recovery"] || !pressureSeen || !phases[m.ReferenceArm.PressurePhase] {
		return errors.New("pressure arms must declare baseline, positive pressure, and recovery after removal")
	}
	strategy := strings.ToLower(m.Ordering.Strategy)
	if !strings.Contains(strategy, "counterbalanced") && !strings.Contains(strategy, "randomized") {
		return errors.New("ordering strategy must be counterbalanced or randomized")
	}
	if !sameStrings(m.Ordering.ThermalOrder, []string{"cold", "warm"}) || len(m.Ordering.RequestOrder) < 2 {
		return errors.New("ordering must declare cold/warm and an explicit request order")
	}
	if strings.TrimSpace(m.Reset.BeforeCampaign) == "" || strings.TrimSpace(m.Reset.BeforeColdArm) == "" || strings.TrimSpace(m.Reset.BeforeWarmArm) == "" || strings.TrimSpace(m.Reset.AfterPressure) == "" {
		return errors.New("all reset procedures are required")
	}
	if m.Confidence.Level != .95 || m.Confidence.ReliableReuseRatio <= 0 || m.Confidence.ReliableReuseRatio > 1 || m.Confidence.MinimumSamples < 3 {
		return errors.New("confidence requires the supported 0.95 level, reuse ratio in (0,1], and at least three samples")
	}
	if strings.TrimSpace(m.UsefulWorkRule) == "" {
		return errors.New("useful_work_rule is required")
	}
	return nil
}

func validatePins(p Pins) error {
	for name, value := range map[string]string{
		"backend": p.Backend, "backend_revision": p.BackendRevision,
		"runtime_revision": p.RuntimeRevision, "model_id": p.ModelID,
		"model_revision": p.ModelRevision, "fak_revision": p.FakRevision,
	} {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("pins.%s is required", name)
		}
	}
	return nil
}

func validateTokenization(t Tokenization) error {
	if strings.TrimSpace(t.TokenizerID) == "" || strings.TrimSpace(t.TokenizerRevision) == "" || strings.TrimSpace(t.PromptTemplateSHA256) == "" {
		return errors.New("tokenization must pin tokenizer id, revision, and prompt template sha256")
	}
	if t.Unit != "tokens" {
		return errors.New("tokenization.unit must be tokens")
	}
	return nil
}

func positiveUniqueSorted(values []int) bool {
	if !positiveUnique(values) {
		return false
	}
	return slices.IsSorted(values)
}

func positiveUnique(values []int) bool {
	seen := map[int]bool{}
	for _, value := range values {
		if value <= 0 || seen[value] {
			return false
		}
		seen[value] = true
	}
	return true
}

func sameStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func oneOf(value string, values ...string) bool { return slices.Contains(values, value) }
