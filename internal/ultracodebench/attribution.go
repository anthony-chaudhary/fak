package ultracodebench

import (
	"fmt"
	"strings"
)

const AttributionSchema = "fak.ultracode_attribution.v1"

const (
	AttributionVerified   = "activation_verified"
	AttributionUnverified = "activation_unverified"
)

type UsageEvidence struct {
	BilledTokens int64   `json:"billed_tokens"`
	SpendUSD     float64 `json:"spend_usd"`
}

// DownstreamEvidence is the privacy-bounded join surface shared by workflow
// outcome, usage, and cross-harness trajectory audit consumers.
type DownstreamEvidence struct {
	RunID            string        `json:"run_id"`
	ChildID          string        `json:"child_id"`
	Outcome          string        `json:"outcome"`
	Usage            UsageEvidence `json:"usage"`
	TrajectoryDigest string        `json:"trajectory_digest"`
}

type ActivationAttribution struct {
	Schema           string             `json:"schema"`
	RunID            string             `json:"run_id"`
	ChildID          string             `json:"child_id"`
	ActivationState  ActivationState    `json:"activation_state"`
	Attribution      string             `json:"attribution"`
	Activation       *ActivationReceipt `json:"activation,omitempty"`
	Outcome          string             `json:"outcome"`
	Usage            UsageEvidence      `json:"usage"`
	TrajectoryDigest string             `json:"trajectory_digest"`
}

func JoinActivation(receipts []ActivationReceipt, evidence []DownstreamEvidence) ([]ActivationAttribution, error) {
	byKey := make(map[string]ActivationReceipt, len(receipts))
	for _, r := range receipts {
		if err := r.Validate(); err != nil {
			return nil, err
		}
		if _, exists := byKey[r.key()]; exists {
			return nil, fmt.Errorf("duplicate activation identity %s/%s", r.RunID, r.ChildID)
		}
		byKey[r.key()] = r
	}
	out := make([]ActivationAttribution, 0, len(evidence))
	for _, ev := range evidence {
		if !activationToken(ev.RunID) || !activationToken(ev.ChildID) || !activationToken(ev.Outcome) {
			return nil, fmt.Errorf("downstream evidence identity and outcome must be opaque tokens")
		}
		if ev.Usage.BilledTokens < 0 || ev.Usage.SpendUSD < 0 {
			return nil, fmt.Errorf("downstream usage cannot be negative")
		}
		if !validTrajectoryDigest(ev.TrajectoryDigest) {
			return nil, fmt.Errorf("trajectory evidence must be a sha256 digest, not a path or transcript")
		}
		row := ActivationAttribution{
			Schema: AttributionSchema, RunID: ev.RunID, ChildID: ev.ChildID,
			ActivationState: ActivationUnknown, Attribution: AttributionUnverified,
			Outcome: ev.Outcome, Usage: ev.Usage, TrajectoryDigest: ev.TrajectoryDigest,
		}
		if r, ok := byKey[ev.RunID+"\x00"+ev.ChildID]; ok {
			copy := r
			row.Activation = &copy
			row.ActivationState = r.State()
			if row.ActivationState == ActivationActive || row.ActivationState == ActivationInactive {
				row.Attribution = AttributionVerified
			}
		}
		out = append(out, row)
	}
	return out, nil
}

func validTrajectoryDigest(s string) bool {
	if !strings.HasPrefix(s, "sha256:") || len(s) != len("sha256:")+64 {
		return false
	}
	for _, r := range strings.TrimPrefix(s, "sha256:") {
		if (r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') {
			continue
		}
		return false
	}
	return true
}
