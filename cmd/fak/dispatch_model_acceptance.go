package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelaccept"
)

const dispatchAcceptanceMaxAge = 30 * 24 * time.Hour

var readDispatchAcceptanceArtifact = os.ReadFile

type dispatchAcceptanceDecision struct {
	Enabled       bool   `json:"enabled"`
	Allowed       bool   `json:"allowed"`
	Override      bool   `json:"override,omitempty"`
	Reason        string `json:"reason"`
	Model         string `json:"model"`
	RequiredTier  int    `json:"required_tier"`
	WitnessedTier *int   `json:"witnessed_tier,omitempty"`
	CorpusID      string `json:"corpus_id,omitempty"`
	Verdict       string `json:"verdict"`
	Artifact      string `json:"artifact,omitempty"`
}

func dispatchRequiredModelTier(labels []string) int {
	required := 2
	for _, raw := range labels {
		s := strings.ToLower(strings.TrimSpace(raw))
		switch {
		case strings.HasPrefix(s, "tier/t0-") || s == "tier/t0":
			return 0
		case strings.HasPrefix(s, "tier/t1-") || s == "tier/t1":
			if required > 1 {
				required = 1
			}
		}
	}
	return required
}

func evaluateDispatchModelAcceptance(path, model string, labels []string, now time.Time, overrideReason string) dispatchAcceptanceDecision {
	d := dispatchAcceptanceDecision{
		Enabled:      strings.TrimSpace(path) != "",
		Model:        strings.TrimSpace(model),
		RequiredTier: dispatchRequiredModelTier(labels),
		Artifact:     strings.TrimSpace(path),
		Verdict:      string(modelaccept.Hold),
	}
	if !d.Enabled {
		d.Allowed = true
		d.Reason = "acceptance artifact not configured"
		return d
	}

	hold := func(reason string) dispatchAcceptanceDecision {
		d.Reason = reason
		if strings.TrimSpace(overrideReason) != "" {
			d.Allowed = true
			d.Override = true
			d.Reason = reason + "; operator override: " + strings.TrimSpace(overrideReason)
		}
		return d
	}
	if d.Model == "" {
		return hold("selected model ID is empty")
	}
	b, err := readDispatchAcceptanceArtifact(d.Artifact)
	if err != nil {
		return hold("acceptance artifact unreadable: " + err.Error())
	}
	var in modelaccept.Input
	if err := json.Unmarshal(b, &in); err != nil {
		return hold("acceptance artifact malformed: " + err.Error())
	}
	decision := modelaccept.Evaluate(in)
	d.CorpusID = decision.CorpusID
	if decision.Verdict != modelaccept.Pass {
		return hold("acceptance artifact verdict is HOLD")
	}
	latest := time.Time{}
	for _, run := range in.Runs {
		observed, err := time.Parse(time.RFC3339, run.ObservedAt)
		if err == nil && observed.After(latest) {
			latest = observed
		}
	}
	if latest.IsZero() || now.Sub(latest) > dispatchAcceptanceMaxAge {
		return hold("acceptance evidence is stale")
	}
	for _, md := range decision.Models {
		if md.Model != d.Model {
			continue
		}
		witnessed := md.RequestedTier
		d.WitnessedTier = &witnessed
		if md.Verdict != modelaccept.Pass {
			return hold("exact model acceptance verdict is HOLD")
		}
		if witnessed > d.RequiredTier {
			return hold(fmt.Sprintf("witnessed tier %d is below required tier %d", witnessed, d.RequiredTier))
		}
		d.Allowed = true
		d.Verdict = string(modelaccept.Pass)
		d.Reason = "exact model is witnessed for required tier"
		return d
	}
	return hold("exact model ID is absent from acceptance artifact")
}

func applyDispatchModelAcceptance(path, model string, labels []string, now time.Time, overrideReason string, payload map[string]any) bool {
	d := evaluateDispatchModelAcceptance(path, model, labels, now, overrideReason)
	if !d.Enabled {
		return true
	}
	payload["model_acceptance"] = d
	if d.Allowed {
		return true
	}
	payload["ok"] = false
	payload["action"] = "model_acceptance_hold"
	payload["verdict"] = "HOLD"
	payload["reason"] = d.Reason
	return false
}
