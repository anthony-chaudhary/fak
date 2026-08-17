package toolcallcontrol

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Mode selects whether runtime verdicts are disabled, observed, or applied.
type Mode string

const (
	ModeOff     Mode = "off"
	ModeShadow  Mode = "shadow"
	ModeEnforce Mode = "enforce"
)

// ParseMode is fail-safe for optimization: malformed values disable control.
func ParseMode(raw string) Mode {
	switch Mode(strings.ToLower(strings.TrimSpace(raw))) {
	case ModeShadow:
		return ModeShadow
	case ModeEnforce:
		return ModeEnforce
	default:
		return ModeOff
	}
}

// RuntimeState is the bounded per-session knowledge used by the live hook.
// Epoch advances after any potentially state-changing tool, invalidating reuse.
type RuntimeState struct {
	Epoch        uint64               `json:"epoch"`
	Observations []RuntimeObservation `json:"observations,omitempty"`
}

// RuntimeObservation records a completed read at one mutation epoch.
type RuntimeObservation struct {
	Fingerprint string `json:"fingerprint"`
	Epoch       uint64 `json:"epoch"`
	ResultRef   string `json:"result_ref,omitempty"`
}

// RuntimeInput is the provider-neutral subset available at a pre/post tool seam.
type RuntimeInput struct {
	SessionID   string
	CallID      string
	Tool        string
	Args        json.RawMessage
	ReadOnly    bool
	Succeeded   bool
	ResultRef   string
	PromptUnits int64
}

// RuntimeVerdict is emitted for every shadow/enforce pre-tool evaluation.
type RuntimeVerdict struct {
	Mode               Mode   `json:"mode"`
	Action             Action `json:"action"`
	Reason             string `json:"reason"`
	Fingerprint        string `json:"fingerprint,omitempty"`
	ResultRef          string `json:"result_ref,omitempty"`
	Epoch              uint64 `json:"epoch"`
	Applied            bool   `json:"applied"`
	ReplayUnitsSaved   int64  `json:"replay_units_saved"`
	ReplaySquaredSaved string `json:"replay_squared_saved,omitempty"`
	PromptBucket       string `json:"prompt_bucket,omitempty"`
}

// Before applies exact-fresh reuse against state. Shadow records the same
// counterfactual without changing execution.
func Before(mode Mode, state RuntimeState, in RuntimeInput) RuntimeVerdict {
	v := RuntimeVerdict{Mode: mode, Action: Allow, Reason: "mode_off", Epoch: state.Epoch}
	if mode == ModeOff {
		return v
	}
	if strings.TrimSpace(in.Tool) == "" || len(in.Args) == 0 {
		v.Reason = "malformed_fail_open"
		return v
	}
	if !in.ReadOnly {
		v.Reason = "mutation_not_suppressed"
		return v
	}
	v.Fingerprint = runtimeFingerprint(in.Tool, in.Args, state.Epoch)
	v.Reason = "novel_at_epoch"
	for i := len(state.Observations) - 1; i >= 0; i-- {
		o := state.Observations[i]
		if o.Epoch == state.Epoch && o.Fingerprint == v.Fingerprint {
			v.Action = Reuse
			v.Reason = "exact_fresh_result"
			v.ResultRef = o.ResultRef
			v.Applied = mode == ModeEnforce
			if in.PromptUnits > 0 {
				v.ReplayUnitsSaved = in.PromptUnits
				v.ReplaySquaredSaved = squareDecimal(in.PromptUnits)
				v.PromptBucket = promptBucket(in.PromptUnits)
			}
			return v
		}
	}
	return v
}

// After learns successful reads and invalidates all reads after any mutation.
// The observation bound prevents session state from growing with context length.
func After(state RuntimeState, in RuntimeInput, maxObservations int) RuntimeState {
	if maxObservations <= 0 {
		maxObservations = 128
	}
	if !in.ReadOnly {
		state.Epoch++
		state.Observations = nil
		return state
	}
	if !in.Succeeded || strings.TrimSpace(in.Tool) == "" || len(in.Args) == 0 {
		return state
	}
	o := RuntimeObservation{
		Fingerprint: runtimeFingerprint(in.Tool, in.Args, state.Epoch),
		Epoch:       state.Epoch,
		ResultRef:   firstNonEmpty(in.ResultRef, in.CallID),
	}
	for i := range state.Observations {
		if state.Observations[i].Fingerprint == o.Fingerprint && state.Observations[i].Epoch == o.Epoch {
			state.Observations[i] = o
			return state
		}
	}
	state.Observations = append(state.Observations, o)
	if len(state.Observations) > maxObservations {
		state.Observations = append([]RuntimeObservation(nil), state.Observations[len(state.Observations)-maxObservations:]...)
	}
	return state
}

func runtimeFingerprint(tool string, args json.RawMessage, epoch uint64) string {
	return fingerprint(tool, string(args), fmt.Sprintf("epoch:%d", epoch))
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func promptBucket(units int64) string {
	switch {
	case units < 16_000:
		return "lt16k"
	case units < 64_000:
		return "16k_64k"
	case units < 128_000:
		return "64k_128k"
	default:
		return "gte128k"
	}
}
