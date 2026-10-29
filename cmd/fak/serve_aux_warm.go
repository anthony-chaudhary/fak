package main

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// serve_aux_warm.go — the serve/up caller of InKernelPlanner.SetAuxWarmConfig (umem
// TICKET-03). The startup agent KV warm (CW-08/CW-09) already runs WarmPrefix in the
// background after listen; WarmPrefix runs the optional expert-profile and V4.1 Engram
// layers LAST, but only when an aux config is installed. Before this file no cmd path
// installed one, so Model.WarmExpertProfile and WarmV41EngramPrefix never ran on a real
// serve. This arms that config, bounded, before the background warm starts.
//
// Default ON. FAK_SERVE_STARTUP_WARM=off (or 0/false/no/disabled) keeps the historical
// behaviour: SetAuxWarmConfig is never called and WarmReceipt.Aux stays nil.

const (
	serveStartupWarmEnv          = "FAK_SERVE_STARTUP_WARM"
	serveStartupWarmBytesEnv     = "FAK_SERVE_STARTUP_WARM_BYTES"
	serveStartupWarmHistogramEnv = "FAK_SERVE_STARTUP_WARM_HISTOGRAM"
	// serveStartupWarmDefaultPercent of the sized expert residency pool is the default
	// reserve when FAK_SERVE_STARTUP_WARM_BYTES is unset.
	serveStartupWarmDefaultPercent = 25
	// serveStartupWarmHistogramSuffix names the default histogram beside the model file.
	// The v1 histogram carries no model identity, so binding by sibling path (or the
	// explicit env) is what ties it to this checkpoint.
	serveStartupWarmHistogramSuffix = ".expert-usage.json"
)

// Closed reasons for the arm decision.
const (
	serveAuxWarmArmed           = "armed"
	serveAuxWarmOff             = "off"
	serveAuxWarmNoPlanner       = "no_planner"
	serveAuxWarmNoCapability    = "no_aux_capability"
	serveAuxWarmNoReserve       = "no_reserve"
	serveAuxWarmBadReserveBytes = "bad_reserve_bytes"
)

type serveAuxWarmConfigurer interface {
	SetAuxWarmConfig(agent.AuxWarmConfig)
}

type serveAuxWarmEncoder interface {
	EncodePrompt(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (agent.PromptEncoding, error)
}

// serveAuxWarmFacts are the live startup facts the arm decision reads; split from the
// model so the witness drives it without a loaded checkpoint.
type serveAuxWarmFacts struct {
	ModelPath          string
	IsV41              bool
	Tier               fakmodel.ExpertCheckpointStats
	HostAvailableBytes int64
	Inputs             agent.WarmPrefixInputs
	HaveInputs         bool
}

type serveAuxWarmDecision struct {
	Installed        bool
	Reason           string
	Config           agent.AuxWarmConfig
	HistogramPath    string
	HistogramEntries int
	HistogramError   string
}

func serveStartupWarmEnabled(getenv func(string) string) bool {
	switch strings.ToLower(strings.TrimSpace(getenv(serveStartupWarmEnv))) {
	case "off", "0", "false", "no", "disabled":
		return false
	}
	return true
}

// serveAuxWarmReserve is min(FAK_SERVE_STARTUP_WARM_BYTES or 25% of the sized expert
// pool, the pool's remaining budget, host MemAvailable). A V4.1 model without a tier has
// no expert pool; its base is host-available (the expert layer then reports unsupported
// without reading and only the bounded Engram prefix runs).
func serveAuxWarmReserve(facts serveAuxWarmFacts, getenv func(string) string) (int64, string) {
	var reserve int64
	if raw := strings.TrimSpace(getenv(serveStartupWarmBytesEnv)); raw != "" {
		n, err := strconv.ParseInt(raw, 10, 64)
		if err != nil || n <= 0 {
			return 0, serveAuxWarmBadReserveBytes
		}
		reserve = n
	} else {
		base := facts.Tier.BudgetBytes
		if !facts.Tier.Enabled {
			base = facts.HostAvailableBytes
		}
		reserve = base / 100 * serveStartupWarmDefaultPercent
	}
	if facts.Tier.Enabled {
		if remaining := facts.Tier.BudgetBytes - facts.Tier.ResidentBytes; remaining < reserve {
			reserve = remaining
		}
	}
	if facts.HostAvailableBytes > 0 && facts.HostAvailableBytes < reserve {
		reserve = facts.HostAvailableBytes
	}
	if reserve <= 0 {
		return 0, serveAuxWarmNoReserve
	}
	return reserve, ""
}

func serveAuxWarmHistogramPath(modelPath string, getenv func(string) string) string {
	if p := strings.TrimSpace(getenv(serveStartupWarmHistogramEnv)); p != "" {
		return p
	}
	if strings.TrimSpace(modelPath) == "" {
		return ""
	}
	return modelPath + serveStartupWarmHistogramSuffix
}

// planServeAuxWarm decides the aux config. It reads only the histogram file and writes
// nothing. A missing/corrupt histogram yields an empty profile, which the model reports
// as no_profile in the receipt rather than guessing.
func planServeAuxWarm(facts serveAuxWarmFacts, getenv func(string) string, encoder serveAuxWarmEncoder) serveAuxWarmDecision {
	if !serveStartupWarmEnabled(getenv) {
		return serveAuxWarmDecision{Reason: serveAuxWarmOff}
	}
	if !facts.IsV41 && !facts.Tier.Enabled {
		return serveAuxWarmDecision{Reason: serveAuxWarmNoCapability}
	}
	reserve, why := serveAuxWarmReserve(facts, getenv)
	if why != "" {
		return serveAuxWarmDecision{Reason: why}
	}
	d := serveAuxWarmDecision{Reason: serveAuxWarmArmed, Installed: true}
	identity := fakmodel.ExpertWarmProfileIdentity{
		Checkpoint: filepath.Base(facts.ModelPath),
		Layout:     fakmodel.ExpertUsageHistogramSchema,
	}
	profile := fakmodel.ExpertWarmProfile{Identity: identity}
	if facts.Tier.Enabled {
		d.HistogramPath = serveAuxWarmHistogramPath(facts.ModelPath, getenv)
		if d.HistogramPath != "" {
			hist, err := fakmodel.LoadExpertUsageHistogram(d.HistogramPath)
			if err != nil {
				d.HistogramError = err.Error()
			} else {
				for _, c := range hist.HotSet(hist.Len()) {
					profile.Entries = append(profile.Entries, fakmodel.ExpertWarmDemandEntry{Layer: c.Layer, Expert: c.Expert, Count: c.Count})
				}
			}
		}
		d.HistogramEntries = len(profile.Entries)
	}
	cfg := agent.AuxWarmConfig{
		ExpertProfile:  profile,
		ExpertIdentity: identity,
		ReserveBytes:   reserve,
	}
	if facts.IsV41 && facts.HaveInputs && encoder != nil {
		msgs := make([]agent.Message, 0, len(facts.Inputs.SystemBlocks)+1)
		if len(facts.Inputs.Instructions) > 0 {
			msgs = append(msgs, agent.Message{Role: agent.RoleSystem, Content: string(facts.Inputs.Instructions)})
		}
		for _, block := range facts.Inputs.SystemBlocks {
			if len(block) > 0 {
				msgs = append(msgs, agent.Message{Role: agent.RoleSystem, Content: string(block)})
			}
		}
		if enc, err := encoder.EncodePrompt(context.Background(), msgs, facts.Inputs.Tools); err == nil {
			cfg.EngramTokens = enc.TokenIDs
		}
	}
	d.Config = cfg
	return d
}

// installServeAuxWarm arms the planner's aux warm when the decision allows it. It must
// run before the background WarmPrefix; the warm itself stays off the readiness path.
func installServeAuxWarm(planner agent.Planner, facts serveAuxWarmFacts, getenv func(string) string, log io.Writer, who string) serveAuxWarmDecision {
	configurer, ok := planner.(serveAuxWarmConfigurer)
	if !ok || planner == nil {
		if !serveStartupWarmEnabled(getenv) {
			return serveAuxWarmDecision{Reason: serveAuxWarmOff}
		}
		return serveAuxWarmDecision{Reason: serveAuxWarmNoPlanner}
	}
	encoder, _ := planner.(serveAuxWarmEncoder)
	d := planServeAuxWarm(facts, getenv, encoder)
	if d.Installed {
		configurer.SetAuxWarmConfig(d.Config)
	}
	if log != nil {
		if d.Installed {
			fmt.Fprintf(log, "fak %s: startup expert/engram warm %s reserve_bytes=%d histogram=%q entries=%d engram_tokens=%d\n",
				who, d.Reason, d.Config.ReserveBytes, d.HistogramPath, d.HistogramEntries, len(d.Config.EngramTokens))
			if d.HistogramError != "" {
				fmt.Fprintf(log, "fak %s: startup expert warm histogram unreadable (empty profile): %s\n", who, d.HistogramError)
			}
		} else {
			fmt.Fprintf(log, "fak %s: startup expert/engram warm not armed (%s)\n", who, d.Reason)
		}
	}
	return d
}

// serveAuxWarmFactsFor reads the live facts from a loaded model.
func serveAuxWarmFactsFor(m *fakmodel.Model, modelPath, workspace string) serveAuxWarmFacts {
	facts := serveAuxWarmFacts{ModelPath: modelPath}
	if m != nil {
		facts.IsV41 = m.Cfg.IsDeepSeekV41()
		facts.Tier = m.ExpertCheckpointStats()
	}
	if _, free, known := compute.HostSystemMemoryInfo(); known {
		facts.HostAvailableBytes = free
	}
	facts.Inputs, facts.HaveInputs = deriveServeAgentWarmInputs(workspace)
	return facts
}

// logServeAuxWarmReceipt prints the aux layer receipt once the background warm finishes.
func logServeAuxWarmReceipt(log io.Writer, who string, aux *agent.AuxWarmReceipt) {
	if log == nil || aux == nil {
		return
	}
	fmt.Fprintf(log, "fak %s: startup aux warm receipt reserve_bytes=%d expert=%v/%s retained=%d retained_bytes=%d read_bytes=%d engram=%v/%s rows=%d read_bytes=%d\n",
		who, aux.ReserveBytes,
		aux.Expert.Status, aux.Expert.Reason, aux.Expert.Retained, aux.Expert.BytesRetained, aux.Expert.BytesRead,
		aux.Engram.Status, aux.Engram.Reason, aux.Engram.Retained, aux.Engram.BytesRead)
}
