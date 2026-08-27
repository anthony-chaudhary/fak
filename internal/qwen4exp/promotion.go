package qwen4exp

import (
	"errors"
	"sort"
	"strings"
	"time"
)

const (
	CanonicalModel = "Qwen3.8-Flash-Next"
	RegistryKey    = "qwen4_exp"
)

type PromotionState string

const (
	PromotionReady    PromotionState = "READY"
	PromotionHold     PromotionState = "HOLD"
	PromotionRollback PromotionState = "ROLLBACK"
)

type EnvelopeEvidence struct {
	Backend         string    `json:"backend"`
	Artifact        string    `json:"artifact"`
	Engine          string    `json:"engine"`
	Fallback        string    `json:"fallback"`
	Quality         bool      `json:"quality"`
	Text            bool      `json:"text"`
	StructuredJSON  bool      `json:"structured_json"`
	CorrelatedTools bool      `json:"correlated_tools"`
	Continues       bool      `json:"continues"`
	CapturedAt      time.Time `json:"captured_at"`
	ExpiresAt       time.Time `json:"expires_at"`
}

type PromotionInput struct {
	ExpectedArtifact string             `json:"expected_artifact"`
	Now              time.Time          `json:"now"`
	Envelopes        []EnvelopeEvidence `json:"envelopes"`
}
type PromotionVerdict struct {
	Model       string         `json:"model"`
	RegistryKey string         `json:"registry_key"`
	Aliases     []string       `json:"aliases"`
	State       PromotionState `json:"state"`
	Reasons     []string       `json:"reasons,omitempty"`
}

func EvaluatePromotion(in PromotionInput) PromotionVerdict {
	v := PromotionVerdict{Model: CanonicalModel, RegistryKey: RegistryKey, Aliases: []string{CanonicalModel, RegistryKey, "Qwen3.8-Flash-Next", "qwen4_exp"}}
	if in.Now.IsZero() {
		in.Now = time.Now().UTC()
	}
	required := map[string]bool{"cuda": false, "metal": false}
	rollback := false
	for _, e := range in.Envelopes {
		backend := strings.ToLower(strings.TrimSpace(e.Backend))
		if _, ok := required[backend]; !ok {
			continue
		}
		if e.Artifact != in.ExpectedArtifact {
			v.Reasons = append(v.Reasons, backend+": artifact mismatch")
			rollback = true
			continue
		}
		if e.Engine != "fak-native" || e.Fallback != "none" {
			v.Reasons = append(v.Reasons, backend+": non-native or fallback evidence")
			rollback = true
			continue
		}
		if !e.ExpiresAt.After(in.Now) || e.CapturedAt.After(in.Now) {
			v.Reasons = append(v.Reasons, backend+": stale or future evidence")
			rollback = true
			continue
		}
		if !(e.Quality && e.Text && e.StructuredJSON && e.CorrelatedTools && e.Continues) {
			v.Reasons = append(v.Reasons, backend+": incomplete capability witness")
			continue
		}
		required[backend] = true
	}
	for _, b := range []string{"cuda", "metal"} {
		if !required[b] {
			v.Reasons = append(v.Reasons, "missing green "+b+" envelope")
		}
	}
	sort.Strings(v.Reasons)
	if rollback {
		v.State = PromotionRollback
	} else if required["cuda"] && required["metal"] {
		v.State = PromotionReady
	} else {
		v.State = PromotionHold
	}
	return v
}

func (v PromotionVerdict) ValidateReady() error {
	if v.State != PromotionReady {
		return errors.New("qwen4exp: model is not READY")
	}
	return nil
}
