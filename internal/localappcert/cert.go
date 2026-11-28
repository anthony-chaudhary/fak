// Package localappcert validates the v1 Apple-Silicon certification matrix.
// A supported envelope is publishable only when every required lifecycle scenario
// carries an external artifact and a fak-native, fallback-free runtime receipt.
//
// The six-gate serving acceptance contract (load -> chat -> reasoning -> tools ->
// context -> speed) adapts the ordered gating and evidence discipline from
// sybil-solutions/local-ai-registry (lab/lab.py:37,400-488 at
/*
MIT License

Copyright (c) 0xSero / sybil-solutions/local-ai-registry contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
package localappcert

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
)

const Schema = "fak.local-app-certification/1"

// SixGateSchema identifies a captured six-gate serving acceptance report.
const SixGateSchema = "fak.local-app-certification-six-gate/1"

const (
	SourceTokenizer = "tokenizer"
	SourceChars     = "chars"

	DefaultSpeedFloorTokensPerSecond    = 15.0
	DefaultContextDepthRatio            = 0.85
	DefaultContextPromptTokenFloorRatio = 0.60
	DefaultSpeedWindow                  = 30 * time.Second
)

type Status string

const (
	StatusPass        Status = "PASS"
	StatusFail        Status = "FAIL"
	StatusUnsupported Status = "UNSUPPORTED"
)

type GateName string

const (
	GateLoad      GateName = "load"
	GateChat      GateName = "chat"
	GateReasoning GateName = "reasoning"
	GateTools     GateName = "tools"
	GateContext   GateName = "context"
	GateSpeed     GateName = "speed"
)

// SixGates defines the mandatory ordered serving acceptance sequence.
var SixGates = []GateName{
	GateLoad,
	GateChat,
	GateReasoning,
	GateTools,
	GateContext,
	GateSpeed,
}

const (
	ReasonModelNotListed        = "MODEL_NOT_LISTED"
	ReasonLoadFailed            = "LOAD_FAILED"
	ReasonRequestFailed         = "REQUEST_FAILED"
	ReasonEmptyAnswer           = "EMPTY_ANSWER"
	ReasonAnswerCapped          = "ANSWER_CAPPED"
	ReasonFinishReasonNotStop   = "FINISH_REASON_NOT_STOP"
	ReasonReasoningFieldMissing = "REASONING_FIELD_MISSING"
	ReasonReasoningMismatch     = "REASONING_MISMATCH"
	ReasonToolNotCalled         = "TOOL_NOT_CALLED"
	ReasonToolArgMismatch       = "TOOL_ARG_MISMATCH"
	ReasonToolNonceMissing      = "TOOL_NONCE_MISSING"
	ReasonContextCodeMissing    = "CONTEXT_CODE_MISSING"
	ReasonContextTokensLow      = "CONTEXT_TOKENS_LOW"
	ReasonSpeedSourceInvalid    = "SPEED_SOURCE_NOT_TOKENIZER"
	ReasonSpeedBelowFloor       = "SPEED_BELOW_FLOOR"
	ReasonSpeedDraftsCounted    = "SPEED_SPECULATIVE_DRAFTS_COUNTED"
	ReasonSpeedZeroDuration     = "SPEED_ZERO_DURATION"
)

var RequiredScenarios = []string{
	"clean-install", "cold-start", "warm-start", "long-context",
	"concurrent-foreground-load", "cancel-restart", "memory-pressure",
	"low-power-thermal-downshift", "offline-start", "interrupted-update",
	"canary-reject", "rollback", "uninstall-cleanup",
}

type Receipt struct {
	Engine                string  `json:"engine"`
	Fallback              string  `json:"fallback"`
	Artifact              string  `json:"artifact"`
	RuntimeRevision       string  `json:"runtime_revision"`
	TTFTMilliseconds      float64 `json:"ttft_milliseconds,omitempty"`
	DecodeTokensPerSecond float64 `json:"decode_tokens_per_second,omitempty"`
	PeakMemoryBytes       int64   `json:"peak_memory_bytes,omitempty"`
	DiskBytes             int64   `json:"disk_bytes,omitempty"`
}

type Scenario struct {
	Name           string   `json:"name"`
	Status         Status   `json:"status"`
	Evidence       string   `json:"evidence,omitempty"`
	EvidenceSHA256 string   `json:"evidence_sha256,omitempty"`
	EvidenceBytes  int64    `json:"evidence_bytes"`
	Reason         string   `json:"reason,omitempty"`
	Receipt        *Receipt `json:"receipt,omitempty"`
}

type SpeculationSpec struct {
	Enabled     bool   `json:"enabled"`
	Method      string `json:"method,omitempty"`
	DraftModel  string `json:"draft_model,omitempty"`
	DraftTokens int    `json:"draft_tokens,omitempty"`
}

type SixGateSpec struct {
	Model                     string           `json:"model"`
	Endpoint                  string           `json:"endpoint,omitempty"`
	Artifact                  string           `json:"artifact,omitempty"`
	RuntimeRevision           string           `json:"runtime_revision,omitempty"`
	ContextWindow             int              `json:"context_window"`
	AnswerCapped              bool             `json:"answer_capped"`
	SpeedFloorTokensPerSecond float64          `json:"speed_floor_tokens_per_second,omitempty"`
	SpeedWindowSeconds        float64          `json:"speed_window_seconds,omitempty"`
	ChatPrompt                string           `json:"chat_prompt,omitempty"`
	ReasoningPrompt           string           `json:"reasoning_prompt,omitempty"`
	ReasoningExpectedInt      *int64           `json:"reasoning_expected_int,omitempty"`
	ToolName                  string           `json:"tool_name,omitempty"`
	ToolArgKey                string           `json:"tool_arg_key,omitempty"`
	ToolExpectedArg           string           `json:"tool_expected_arg,omitempty"`
	Speculation               *SpeculationSpec `json:"speculation,omitempty"`
}

type GateResult struct {
	Gate                     GateName  `json:"gate"`
	Status                   Status    `json:"status"`
	BlewUp                   bool      `json:"blew_up,omitempty"`
	Reason                   string    `json:"reason,omitempty"`
	DurationSeconds          float64   `json:"duration_seconds"`
	Evidence                 string    `json:"evidence,omitempty"`
	EvidenceSHA256           string    `json:"evidence_sha256,omitempty"`
	EvidenceBytes            int64     `json:"evidence_bytes"`
	AnswerCapped             bool      `json:"answer_capped,omitempty"`
	FinishReason             string    `json:"finish_reason,omitempty"`
	ListedModels             []string  `json:"listed_models,omitempty"`
	SeparateReasoning        bool      `json:"separate_reasoning,omitempty"`
	ReasoningText            string    `json:"reasoning_text,omitempty"`
	ExpectedInteger          *int64    `json:"expected_integer,omitempty"`
	ExtractedIntegers        []int64   `json:"extracted_integers,omitempty"`
	ToolName                 string    `json:"tool_name,omitempty"`
	ToolArgs                 string    `json:"tool_args,omitempty"`
	ToolNonce                string    `json:"tool_nonce,omitempty"`
	NonceMatched             bool      `json:"nonce_matched,omitempty"`
	TargetPromptTokens       int       `json:"target_prompt_tokens,omitempty"`
	MinPromptTokens          int       `json:"min_prompt_tokens,omitempty"`
	PromptTokens             int       `json:"prompt_tokens,omitempty"`
	CodeRecovered            bool      `json:"code_recovered,omitempty"`
	DropSeconds              []float64 `json:"drop_seconds,omitempty"`
	CachedTokens             int       `json:"cached_tokens,omitempty"`
	PrefixCached             bool      `json:"prefix_cached,omitempty"`
	PrefillTokensPerSecond   *float64  `json:"prefill,omitempty"`
	PrefillSuppressed        bool      `json:"prefill_suppressed,omitempty"`
	PrefillSuppressionReason string    `json:"prefill_suppression_reason,omitempty"`
	DecodeTokensPerSecond    float64   `json:"decode_tokens_per_second,omitempty"`
	FloorTokensPerSecond     float64   `json:"floor_tokens_per_second,omitempty"`
	Source                   string    `json:"source,omitempty"`
	DraftTokens              int       `json:"draft_tokens,omitempty"`
	AcceptedTokens           int       `json:"accepted_tokens,omitempty"`
	RejectedTokens           int       `json:"rejected_tokens,omitempty"`
	CountedTokens            int       `json:"counted_tokens,omitempty"`
}

type SixGateProof struct {
	Gates                  string   `json:"gates"`
	DecodeTokensPerSecond  *float64 `json:"tps,omitempty"`
	PrefillTokensPerSecond *float64 `json:"prefill,omitempty"`
	Served                 string   `json:"served,omitempty"`
	LogSHA256              string   `json:"log,omitempty"`
}

type SixGateReport struct {
	Schema         string       `json:"schema"`
	Spec           SixGateSpec  `json:"spec"`
	Gates          []GateResult `json:"gates"`
	Passed         bool         `json:"passed"`
	AbortedGate    GateName     `json:"aborted_gate,omitempty"`
	Proof          SixGateProof `json:"proof"`
	Evidence       string       `json:"evidence,omitempty"`
	EvidenceSHA256 string       `json:"evidence_sha256,omitempty"`
	EvidenceBytes  int64        `json:"evidence_bytes"`
	Receipt        *Receipt     `json:"receipt,omitempty"`
}

type Envelope struct {
	ID              string         `json:"id"`
	Chip            string         `json:"chip"`
	MemoryBytes     int64          `json:"memory_bytes"`
	MacOS           string         `json:"macos"`
	Power           string         `json:"power"`
	Thermal         string         `json:"thermal"`
	DiskFreeBytes   int64          `json:"disk_free_bytes"`
	PackRevision    string         `json:"pack_revision"`
	RuntimeRevision string         `json:"runtime_revision"`
	Supported       bool           `json:"supported"`
	Reason          string         `json:"reason,omitempty"`
	Scenarios       []Scenario     `json:"scenarios,omitempty"`
	SixGate         *SixGateReport `json:"six_gate,omitempty"`
}

type Matrix struct {
	Schema      string     `json:"schema"`
	GeneratedAt string     `json:"generated_at"`
	Envelopes   []Envelope `json:"envelopes"`
}

func Load(path string) (Matrix, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Matrix{}, err
	}
	var m Matrix
	if err := json.Unmarshal(b, &m); err != nil {
		return Matrix{}, err
	}
	return m, nil
}

func Validate(m Matrix) error {
	if m.Schema != Schema {
		return fmt.Errorf("localappcert: schema %q, want %q", m.Schema, Schema)
	}
	if len(m.Envelopes) == 0 {
		return errors.New("localappcert: matrix has no envelopes")
	}
	ids := map[string]bool{}
	supported := 0
	for _, e := range m.Envelopes {
		if strings.TrimSpace(e.ID) == "" || ids[e.ID] {
			return fmt.Errorf("localappcert: missing or duplicate envelope id %q", e.ID)
		}
		ids[e.ID] = true
		if e.Chip == "" || e.MemoryBytes <= 0 || e.MacOS == "" || e.Power == "" || e.Thermal == "" || e.PackRevision == "" || e.RuntimeRevision == "" {
			return fmt.Errorf("localappcert: envelope %s has incomplete identity", e.ID)
		}
		if !e.Supported {
			if strings.TrimSpace(e.Reason) == "" {
				return fmt.Errorf("localappcert: unsupported envelope %s has no typed reason", e.ID)
			}
			if len(e.Scenarios) != 0 {
				return fmt.Errorf("localappcert: unsupported envelope %s must not carry passing scenarios", e.ID)
			}
			if e.SixGate != nil {
				if err := ValidateSixGateReport(*e.SixGate); err != nil {
					return fmt.Errorf("localappcert: unsupported envelope %s six-gate report: %w", e.ID, err)
				}
				if e.SixGate.Passed {
					return fmt.Errorf("localappcert: unsupported envelope %s must not carry passing six-gate report", e.ID)
				}
			}
			continue
		}
		supported++
		if e.Reason != "" {
			return fmt.Errorf("localappcert: supported envelope %s carries a rejection reason", e.ID)
		}
		got := map[string]Scenario{}
		for _, s := range e.Scenarios {
			if _, ok := got[s.Name]; ok {
				return fmt.Errorf("localappcert: envelope %s duplicates scenario %s", e.ID, s.Name)
			}
			got[s.Name] = s
		}
		for _, name := range RequiredScenarios {
			s, ok := got[name]
			if !ok {
				return fmt.Errorf("localappcert: envelope %s missing scenario %s", e.ID, name)
			}
			if s.Status != StatusPass {
				return fmt.Errorf("localappcert: supported envelope %s scenario %s is %s", e.ID, name, s.Status)
			}
			if strings.TrimSpace(s.Evidence) == "" {
				return fmt.Errorf("localappcert: envelope %s scenario %s has no evidence", e.ID, name)
			}
			if !hasValidNativeReceipt(s.Receipt, e.RuntimeRevision) {
				return fmt.Errorf("localappcert: envelope %s scenario %s lacks exact fak-native fallback-free receipt", e.ID, name)
			}
		}
		if len(got) != len(RequiredScenarios) {
			extra := make([]string, 0)
			required := map[string]bool{}
			for _, n := range RequiredScenarios {
				required[n] = true
			}
			for n := range got {
				if !required[n] {
					extra = append(extra, n)
				}
			}
			sort.Strings(extra)
			return fmt.Errorf("localappcert: envelope %s has unknown scenarios %v", e.ID, extra)
		}
		if e.SixGate != nil {
			if err := ValidateSixGateReport(*e.SixGate); err != nil {
				return fmt.Errorf("localappcert: envelope %s six-gate report: %w", e.ID, err)
			}
			if !e.SixGate.Passed {
				return fmt.Errorf("localappcert: supported envelope %s failed six-gate acceptance", e.ID)
			}
			if e.SixGate.Spec.RuntimeRevision != "" && e.SixGate.Spec.RuntimeRevision != e.RuntimeRevision {
				return fmt.Errorf("localappcert: envelope %s six-gate runtime revision %q, want %q", e.ID, e.SixGate.Spec.RuntimeRevision, e.RuntimeRevision)
			}
			if e.SixGate.Receipt != nil && !hasValidNativeReceipt(e.SixGate.Receipt, e.RuntimeRevision) {
				return fmt.Errorf("localappcert: envelope %s six-gate lacks exact fak-native fallback-free receipt", e.ID)
			}
		}
	}
	if supported == 0 {
		return errors.New("localappcert: matrix certifies no supported envelope")
	}
	return nil
}

func hasValidNativeReceipt(r *Receipt, wantRuntimeRevision string) bool {
	if r == nil || r.Engine != "fak-native" || r.Fallback != "none" || strings.TrimSpace(r.Artifact) == "" || strings.TrimSpace(r.RuntimeRevision) == "" {
		return false
	}
	if wantRuntimeRevision != "" && r.RuntimeRevision != wantRuntimeRevision {
		return false
	}
	return true
}

func effectiveSpeedFloor(floor float64) float64 {
	if floor == 0 {
		return DefaultSpeedFloorTokensPerSecond
	}
	return floor
}

func minContextPromptTokens(ctx int) int {
	if ctx <= 0 {
		return 0
	}
	return (ctx*3 + 4) / 5
}

// ValidateSixGateReport verifies the structural and soundness invariants of a
// six-gate serving acceptance report, whether passed or failed.
func ValidateSixGateReport(r SixGateReport) error {
	if r.Schema != SixGateSchema {
		return fmt.Errorf("localappcert: six-gate schema %q, want %q", r.Schema, SixGateSchema)
	}
	if strings.TrimSpace(r.Spec.Model) == "" {
		return errors.New("localappcert: six-gate spec model is required")
	}
	if r.Spec.ContextWindow <= 0 {
		return errors.New("localappcert: six-gate spec context_window must be positive")
	}
	if r.Spec.AnswerCapped {
		return errors.New("localappcert: six-gate spec must never cap answers")
	}
	if r.Spec.SpeedFloorTokensPerSecond < 0 || math.IsNaN(r.Spec.SpeedFloorTokensPerSecond) || math.IsInf(r.Spec.SpeedFloorTokensPerSecond, 0) {
		return errors.New("localappcert: six-gate spec speed floor must be finite and non-negative")
	}
	if r.Receipt != nil && !hasValidNativeReceipt(r.Receipt, r.Spec.RuntimeRevision) {
		return errors.New("localappcert: six-gate receipt lacks exact fak-native fallback-free identity")
	}
	if len(r.Gates) == 0 || len(r.Gates) > len(SixGates) {
		return fmt.Errorf("localappcert: six-gate gate count %d out of range [1,%d]", len(r.Gates), len(SixGates))
	}

	floor := effectiveSpeedFloor(r.Spec.SpeedFloorTokensPerSecond)
	var passedNames []string
	var sawBlowUp bool
	var abortedGate GateName
	var hasFail bool
	var contextSuppressed bool

	for index, g := range r.Gates {
		if g.Gate != SixGates[index] {
			return fmt.Errorf("localappcert: six-gate index %d has gate %q, want %q", index, g.Gate, SixGates[index])
		}
		if sawBlowUp {
			return fmt.Errorf("localappcert: six-gate %q executed after non-answering failure at %q", g.Gate, abortedGate)
		}
		if g.AnswerCapped {
			return fmt.Errorf("localappcert: six-gate %q reports capped answer", g.Gate)
		}
		if strings.TrimSpace(g.Evidence) == "" && strings.TrimSpace(g.EvidenceSHA256) == "" {
			return fmt.Errorf("localappcert: six-gate %q has no evidence", g.Gate)
		}
		switch g.Status {
		case StatusPass:
			passedNames = append(passedNames, string(g.Gate))
		case StatusFail:
			hasFail = true
		default:
			return fmt.Errorf("localappcert: six-gate %q has invalid status %q", g.Gate, g.Status)
		}

		if err := validateGateResult(g, r.Spec.ContextWindow, floor); err != nil {
			return err
		}
		if g.Gate == GateContext && (g.PrefixCached || g.CachedTokens > 0 || g.PrefillSuppressed) {
			contextSuppressed = true
		}
		if g.BlewUp {
			if g.Status != StatusFail {
				return fmt.Errorf("localappcert: six-gate %q marked blew_up with status %s", g.Gate, g.Status)
			}
			sawBlowUp = true
			abortedGate = g.Gate
		}
	}

	if contextSuppressed && r.Proof.PrefillTokensPerSecond != nil {
		return errors.New("localappcert: six-gate proof must suppress prefill rate when answered from prefix cache")
	}

	expectedProofGates := strings.Join(passedNames, ",")
	if r.Proof.Gates != "" && r.Proof.Gates != expectedProofGates {
		return fmt.Errorf("localappcert: six-gate proof gates %q, want %q", r.Proof.Gates, expectedProofGates)
	}

	if r.Passed {
		if hasFail || sawBlowUp || r.AbortedGate != "" || len(r.Gates) != len(SixGates) {
			return errors.New("localappcert: six-gate passed report must pass all six gates without abort")
		}
		if r.Proof.Gates != expectedProofGates {
			return fmt.Errorf("localappcert: six-gate passed proof gates %q, want %q", r.Proof.Gates, expectedProofGates)
		}
		if r.Proof.DecodeTokensPerSecond == nil || *r.Proof.DecodeTokensPerSecond < floor {
			return errors.New("localappcert: six-gate passed proof lacks tokenizer decode rate meeting floor")
		}
		return nil
	}

	if !hasFail {
		return errors.New("localappcert: six-gate failed report has no failing gate")
	}
	if sawBlowUp && r.AbortedGate != "" && r.AbortedGate != abortedGate {
		return fmt.Errorf("localappcert: six-gate aborted_gate %q, want %q", r.AbortedGate, abortedGate)
	}
	return nil
}

func validateGateResult(g GateResult, contextWindow int, floor float64) error {
	switch g.Gate {
	case GateChat:
		if g.Status == StatusPass && g.FinishReason != "stop" {
			return fmt.Errorf("localappcert: chat gate passed with finish_reason %q", g.FinishReason)
		}
	case GateReasoning:
		if g.Status == StatusPass && !g.SeparateReasoning {
			return errors.New("localappcert: reasoning gate passed without separate reasoning field")
		}
	case GateTools:
		if g.Status == StatusPass && (!g.NonceMatched || strings.TrimSpace(g.ToolNonce) == "") {
			return errors.New("localappcert: tools gate passed without fresh injected nonce verification")
		}
	case GateContext:
		if (g.PrefixCached || g.CachedTokens > 0 || g.PrefillSuppressed) && g.PrefillTokensPerSecond != nil {
			return errors.New("localappcert: context gate must suppress prefill rate when answered from prefix cache")
		}
		if g.Status == StatusPass {
			if !g.CodeRecovered {
				return errors.New("localappcert: context gate passed without planted code recovery")
			}
			if g.PromptTokens < minContextPromptTokens(contextWindow) {
				return fmt.Errorf("localappcert: context gate prompt_tokens %d below floor %d", g.PromptTokens, minContextPromptTokens(contextWindow))
			}
		}
	case GateSpeed:
		if g.DraftTokens > 0 {
			if g.CountedTokens == g.DraftTokens && g.AcceptedTokens < g.DraftTokens {
				return errors.New("localappcert: speed gate counted speculative drafts as accepted")
			}
			if g.RejectedTokens > 0 && g.CountedTokens == g.AcceptedTokens+g.RejectedTokens {
				return errors.New("localappcert: speed gate counted rejected speculative drafts as accepted")
			}
			if g.RejectedTokens == 0 && g.AcceptedTokens > 0 && g.CountedTokens > g.AcceptedTokens && g.CountedTokens >= g.DraftTokens {
				return errors.New("localappcert: speed gate counted unverified speculative drafts as accepted")
			}
		}
		if g.Status == StatusPass {
			if g.Source != SourceTokenizer {
				return fmt.Errorf("localappcert: speed gate source %q, want %q", g.Source, SourceTokenizer)
			}
			if g.DecodeTokensPerSecond < floor || math.IsNaN(g.DecodeTokensPerSecond) || math.IsInf(g.DecodeTokensPerSecond, 0) {
				return fmt.Errorf("localappcert: speed gate decode rate %.3f below floor %.3f", g.DecodeTokensPerSecond, floor)
			}
			if g.DraftTokens > 0 && (g.AcceptedTokens <= 0 || g.CountedTokens <= 0) {
				return errors.New("localappcert: speed gate passed with unverified speculative draft counts")
			}
		}
	}
	return nil
}

// ExtractExactIntegers returns the set of standalone base-10 integers present in
// text. Tokens attached to letters, decimals, underscores, or thousand-grouped
// commas (e.g. "3910", "391st", "391.5", "1,391") are excluded from matching 391.
func ExtractExactIntegers(text string) map[int64]bool {
	out := make(map[int64]bool)
	runes := []rune(text)
	for i := 0; i < len(runes); {
		if !isASCIIDigit(runes[i]) {
			i++
			continue
		}
		start := i
		for i < len(runes) && isASCIIDigit(runes[i]) {
			i++
		}
		end := i

		if start > 0 {
			prev := runes[start-1]
			if isWordRune(prev) || prev == '.' {
				continue
			}
			if prev == ',' && start >= 2 && isASCIIDigit(runes[start-2]) {
				continue
			}
		}
		if end < len(runes) {
			next := runes[end]
			if isWordRune(next) {
				continue
			}
			if next == '.' && end+1 < len(runes) && isASCIIDigit(runes[end+1]) {
				continue
			}
			if next == ',' && end+1 < len(runes) && isASCIIDigit(runes[end+1]) {
				continue
			}
		}

		token := string(runes[start:end])
		if start > 0 && runes[start-1] == '-' {
			if start >= 2 && isWordRune(runes[start-2]) {
				continue
			}
			token = "-" + token
		}
		if val, err := strconv.ParseInt(token, 10, 64); err == nil {
			out[val] = true
		}
	}
	return out
}

func isASCIIDigit(r rune) bool {
	return r >= '0' && r <= '9'
}

func isWordRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_'
}
