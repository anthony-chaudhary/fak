package ultracodetokenizer

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
)

const Schema = "fak.ultracode_tokenizer_portability.v1"

var ErrAbstain = errors.New("tokenizer portability comparison abstained")

type CanonicalInput struct {
	Campaign        string          `json:"campaign"`
	Task            string          `json:"task"`
	FullMessages    json.RawMessage `json:"full_messages"`
	ScopedMessages  json.RawMessage `json:"scoped_messages"`
	AcceptedOutcome string          `json:"accepted_outcome"`
}

type TokenizerIdentity struct {
	Family   string `json:"family"`
	Model    string `json:"model"`
	Revision string `json:"revision"`
	Source   string `json:"source"`
}

type Measurement struct {
	Tokenizer               TokenizerIdentity `json:"tokenizer"`
	CanonicalDigest         string            `json:"canonical_digest"`
	AcceptedOutcome         string            `json:"accepted_outcome"`
	FullInputTokens         int               `json:"full_input_tokens"`
	ScopedInputTokens       int               `json:"scoped_input_tokens"`
	RuntimePrefixReadTokens int               `json:"runtime_prefix_read_tokens"`
}

type Provenance struct {
	OmittedBytes            int `json:"omitted_bytes"`
	OmittedMessages         int `json:"omitted_messages"`
	ModelInputTokens        int `json:"model_input_tokens"`
	RuntimePrefixReadTokens int `json:"runtime_prefix_read_tokens"`
}

type Result struct {
	Tokenizer          TokenizerIdentity `json:"tokenizer"`
	Provenance         Provenance        `json:"provenance"`
	ApparentScopeShare float64           `json:"apparent_scope_share"`
}

type Report struct {
	Schema                          string   `json:"schema"`
	CanonicalDigest                 string   `json:"canonical_digest"`
	Results                         []Result `json:"results"`
	TokenizerOnlyScopeShareMovement float64  `json:"tokenizer_only_scope_share_movement"`
	PromotionEvidence               string   `json:"promotion_evidence"`
	DemotionEvidence                string   `json:"demotion_evidence"`
	InvalidatingAssumption          string   `json:"invalidating_assumption"`
}

func Digest(in CanonicalInput) string {
	h := sha256.New()
	h.Write(in.FullMessages)
	h.Write([]byte{0})
	h.Write(in.ScopedMessages)
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

func Evaluate(in CanonicalInput, measurements []Measurement) (Report, error) {
	fullCount, err := messageCount(in.FullMessages)
	if err != nil {
		return Report{}, fmt.Errorf("%w: invalid full messages: %v", ErrAbstain, err)
	}
	scopedCount, err := messageCount(in.ScopedMessages)
	if err != nil {
		return Report{}, fmt.Errorf("%w: invalid scoped messages: %v", ErrAbstain, err)
	}
	if in.Campaign == "" || in.Task == "" || in.AcceptedOutcome == "" || scopedCount >= fullCount {
		return Report{}, fmt.Errorf("%w: incomplete or non-omitting canonical input", ErrAbstain)
	}
	if len(measurements) < 3 {
		return Report{}, fmt.Errorf("%w: need at least three tokenizer families", ErrAbstain)
	}
	digest := Digest(in)
	seen := map[string]bool{}
	report := Report{Schema: Schema, CanonicalDigest: digest}
	minShare, maxShare := math.Inf(1), math.Inf(-1)
	for _, m := range measurements {
		id := m.Tokenizer
		if id.Family == "" || id.Model == "" || id.Revision == "" || id.Source == "" || seen[id.Family] {
			return Report{}, fmt.Errorf("%w: tokenizer identities must be complete and family-distinct", ErrAbstain)
		}
		seen[id.Family] = true
		if m.CanonicalDigest != digest || m.AcceptedOutcome != in.AcceptedOutcome {
			return Report{}, fmt.Errorf("%w: canonical input or accepted outcome differs for %s", ErrAbstain, id.Family)
		}
		if m.FullInputTokens <= m.ScopedInputTokens || m.ScopedInputTokens < 0 || m.RuntimePrefixReadTokens < 0 {
			return Report{}, fmt.Errorf("%w: invalid token receipt for %s", ErrAbstain, id.Family)
		}
		omittedTokens := m.FullInputTokens - m.ScopedInputTokens
		share := float64(omittedTokens) / float64(omittedTokens+m.RuntimePrefixReadTokens)
		minShare = math.Min(minShare, share)
		maxShare = math.Max(maxShare, share)
		report.Results = append(report.Results, Result{Tokenizer: id, Provenance: Provenance{
			OmittedBytes: len(in.FullMessages) - len(in.ScopedMessages), OmittedMessages: fullCount - scopedCount,
			ModelInputTokens: m.ScopedInputTokens, RuntimePrefixReadTokens: m.RuntimePrefixReadTokens,
		}, ApparentScopeShare: share})
	}
	report.TokenizerOnlyScopeShareMovement = maxShare - minShare
	report.PromotionEvidence = "promote toward gen/now after replay reproduces all tokenizer receipts and an integrated evaluator accepts equal outcomes"
	report.DemotionEvidence = "demote or retire when tokenizer receipts cannot be reproduced or outcome parity abstains"
	report.InvalidatingAssumption = "runtime cache-token telemetry is authoritative and comparable while only the tokenizer is substituted"
	return report, nil
}

func messageCount(raw json.RawMessage) (int, error) {
	var messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &messages); err != nil {
		return 0, err
	}
	if len(messages) == 0 {
		return 0, errors.New("empty message list")
	}
	for _, m := range messages {
		if m.Role == "" {
			return 0, errors.New("message role is empty")
		}
	}
	return len(messages), nil
}
