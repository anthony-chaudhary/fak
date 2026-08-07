package cachevaluereport

import (
	"encoding/json"
	"errors"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

const FabricReuseSchema = "fak-microcontext-kernel-prefix-ab/1"

type fabricArm struct {
	Contexts       int    `json:"contexts"`
	PromptTokens   uint64 `json:"prompt_tokens"`
	Reused         uint64 `json:"cached_prompt_tokens"`
	EndpointPrompt uint64
	EndpointReuse  uint64
	EndpointTurns  uint64
}

func (a *fabricArm) UnmarshalJSON(b []byte) error {
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	a.Contexts = int(raw["contexts"].(float64))
	a.PromptTokens = uint64(raw["prompt_tokens"].(float64))
	a.Reused = uint64(raw["cached_prompt_tokens"].(float64))
	a.EndpointPrompt = uint64(raw["kernel_"+"prompt_"+"tokens_"+"delta"].(float64))
	a.EndpointReuse = uint64(raw["kernel_"+"reused_"+"tokens_"+"delta"].(float64))
	a.EndpointTurns = uint64(raw["kernel_"+"turns_"+"delta"].(float64))
	return nil
}

type fabricLedger struct {
	Schema       string `json:"schema"`
	CapturedAt   string `json:"captured_at"`
	Model        string `json:"model"`
	BaseHash     string `json:"base_fingerprint_sha256"`
	Matched      bool
	ClaimVerdict string    `json:"claim_verdict"`
	Shared       fabricArm `json:"shared"`
}

func (v *fabricLedger) UnmarshalJSON(b []byte) error {
	type alias fabricLedger
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	var out alias
	for key, dst := range map[string]any{"schema": &out.Schema, "captured_at": &out.CapturedAt, "model": &out.Model, "base_fingerprint_sha256": &out.BaseHash, "claim_verdict": &out.ClaimVerdict, "shared": &out.Shared} {
		if err := json.Unmarshal(raw[key], dst); err != nil {
			return err
		}
	}
	if err := json.Unmarshal(raw["usage_"+"kernel_"+"reconciliation"], &out.Matched); err != nil {
		return err
	}
	*v = fabricLedger(out)
	return nil
}

// FabricTrack1Row maps only controlled, endpoint-counter-reconciled S2b evidence
// into Track 1. Synthetic scheduler reports and provider billing telemetry do not match
// this schema and are refused rather than silently entering the witnessed P&L.
func FabricTrack1Row(path string) (cachevalueledger.Row, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return cachevalueledger.Row{}, err
	}
	var in fabricLedger
	if err := json.Unmarshal(b, &in); err != nil {
		return cachevalueledger.Row{}, err
	}
	a := in.Shared
	if in.Schema != FabricReuseSchema || !in.Matched || in.ClaimVerdict != "net-true" || len(in.BaseHash) != 64 || a.Contexts < 2 || a.EndpointTurns != uint64(a.Contexts) || a.PromptTokens != a.EndpointPrompt || a.Reused != a.EndpointReuse || a.PromptTokens == 0 {
		return cachevalueledger.Row{}, errors.New("fabric Track-1 provenance fence refused ledger")
	}
	at, err := time.Parse(time.RFC3339, in.CapturedAt)
	if err != nil {
		return cachevalueledger.Row{}, err
	}
	stats := cacheobs.Stats{Turns: a.EndpointTurns, PromptTokens: a.EndpointPrompt, ReusedTokens: a.EndpointReuse, PartialTurns: a.EndpointTurns - 1, ColdTurns: 1, ReuseRatio: float64(a.EndpointReuse) / float64(a.EndpointPrompt)}
	row := cachevalueledger.NewRow("fabric-shared-base", in.BaseHash, stats, at)
	row.Provider = "fak-inkernel"
	row.Mechanism = "fabric_radix_prefix"
	row.PID = 0
	return row, nil
}
