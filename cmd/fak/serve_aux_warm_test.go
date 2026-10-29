package main

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// serve_aux_warm_test.go — umem TICKET-03: serve/up startup arms SetAuxWarmConfig with a
// bounded reserve by default, and never when FAK_SERVE_STARTUP_WARM=off.

// The real planner the serve/up hooks hand in must satisfy both seams, or the hook
// would silently report no_planner on every real serve.
var (
	_ serveAuxWarmConfigurer = (*agent.InKernelPlanner)(nil)
	_ serveAuxWarmEncoder    = (*agent.InKernelPlanner)(nil)
)

type auxWarmFakePlanner struct {
	agent.Planner
	calls int
	last  agent.AuxWarmConfig
}

func (f *auxWarmFakePlanner) SetAuxWarmConfig(cfg agent.AuxWarmConfig) {
	f.calls++
	f.last = cfg
}

func (f *auxWarmFakePlanner) EncodePrompt(_ context.Context, msgs []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (agent.PromptEncoding, error) {
	return agent.PromptEncoding{TokenIDs: []int{1, 2, 3, len(msgs)}}, nil
}

func auxWarmEnv(kv map[string]string) func(string) string {
	return func(k string) string { return kv[k] }
}

func auxWarmTierFacts(t *testing.T) serveAuxWarmFacts {
	t.Helper()
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "v41.gguf")
	h := fakmodel.NewExpertUsageHistogram()
	h.Observe(3, 7, 5)
	h.Observe(1, 2, 9)
	if err := h.Persist(modelPath + serveStartupWarmHistogramSuffix); err != nil {
		t.Fatal(err)
	}
	return serveAuxWarmFacts{
		ModelPath:          modelPath,
		IsV41:              true,
		Tier:               fakmodel.ExpertCheckpointStats{Enabled: true, BudgetBytes: 8 << 30, ResidentBytes: 1 << 30},
		HostAvailableBytes: 64 << 30,
		Inputs:             agent.WarmPrefixInputs{Instructions: []byte("agents")},
		HaveInputs:         true,
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestServeAuxWarmArmsBoundedReserveByDefault(t *testing.T) {
	facts := auxWarmTierFacts(t)
	p := &auxWarmFakePlanner{}
	d := installServeAuxWarm(p, facts, auxWarmEnv(nil), nil, "serve")
	if !d.Installed || d.Reason != serveAuxWarmArmed || p.calls != 1 {
		t.Fatalf("default must arm once: decision=%+v calls=%d", d, p.calls)
	}
	want := int64(8<<30) / 100 * serveStartupWarmDefaultPercent
	if p.last.ReserveBytes != want {
		t.Fatalf("reserve=%d want 25%% of pool %d", p.last.ReserveBytes, want)
	}
	if len(p.last.ExpertProfile.Entries) != 2 || p.last.ExpertProfile.Entries[0].Layer != 1 {
		t.Fatalf("histogram not loaded hottest-first: %+v", p.last.ExpertProfile.Entries)
	}
	if !p.last.ExpertProfile.Identity.Equal(p.last.ExpertIdentity) {
		t.Fatalf("profile identity must match presented identity")
	}
	if len(p.last.EngramTokens) == 0 {
		t.Fatalf("V4.1 with stable inputs must carry Engram prefix tokens")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestServeAuxWarmOffArmNeverCalls(t *testing.T) {
	for _, v := range []string{"off", "0", "false", "OFF"} {
		p := &auxWarmFakePlanner{}
		d := installServeAuxWarm(p, auxWarmTierFacts(t), auxWarmEnv(map[string]string{serveStartupWarmEnv: v}), nil, "serve")
		if d.Installed || d.Reason != serveAuxWarmOff || p.calls != 0 {
			t.Fatalf("%s=%q must not call SetAuxWarmConfig: decision=%+v calls=%d", serveStartupWarmEnv, v, d, p.calls)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestServeAuxWarmReserveClamps(t *testing.T) {
	facts := auxWarmTierFacts(t)
	// Explicit bytes above the pool's remaining budget clamp to remaining.
	p := &auxWarmFakePlanner{}
	installServeAuxWarm(p, facts, auxWarmEnv(map[string]string{serveStartupWarmBytesEnv: "100000000000"}), nil, "serve")
	if got, want := p.last.ReserveBytes, int64(7<<30); got != want {
		t.Fatalf("explicit reserve=%d want clamp to remaining %d", got, want)
	}
	// Host headroom is the tighter bound.
	facts.HostAvailableBytes = 1 << 20
	p = &auxWarmFakePlanner{}
	installServeAuxWarm(p, facts, auxWarmEnv(nil), nil, "serve")
	if p.last.ReserveBytes != 1<<20 {
		t.Fatalf("reserve=%d want host headroom %d", p.last.ReserveBytes, 1<<20)
	}
	// A full pool leaves no reserve: not armed.
	facts = auxWarmTierFacts(t)
	facts.Tier.ResidentBytes = facts.Tier.BudgetBytes
	p = &auxWarmFakePlanner{}
	if d := installServeAuxWarm(p, facts, auxWarmEnv(nil), nil, "serve"); d.Installed || d.Reason != serveAuxWarmNoReserve || p.calls != 0 {
		t.Fatalf("full pool must not arm: %+v calls=%d", d, p.calls)
	}
	// A bad explicit value is refused, not guessed.
	p = &auxWarmFakePlanner{}
	if d := installServeAuxWarm(p, auxWarmTierFacts(t), auxWarmEnv(map[string]string{serveStartupWarmBytesEnv: "lots"}), nil, "serve"); d.Reason != serveAuxWarmBadReserveBytes || p.calls != 0 {
		t.Fatalf("bad bytes must not arm: %+v", d)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestServeAuxWarmSkipsModelsWithoutAuxCapability(t *testing.T) {
	p := &auxWarmFakePlanner{}
	facts := serveAuxWarmFacts{ModelPath: "dense.gguf", HostAvailableBytes: 64 << 30}
	if d := installServeAuxWarm(p, facts, auxWarmEnv(nil), nil, "serve"); d.Reason != serveAuxWarmNoCapability || p.calls != 0 {
		t.Fatalf("dense model must not arm: %+v", d)
	}
	// Missing histogram still arms with an empty profile (model reports no_profile).
	facts = serveAuxWarmFacts{ModelPath: filepath.Join(t.TempDir(), "m.gguf"), Tier: fakmodel.ExpertCheckpointStats{Enabled: true, BudgetBytes: 1 << 30}}
	if d := installServeAuxWarm(p, facts, auxWarmEnv(nil), nil, "up"); !d.Installed || len(p.last.ExpertProfile.Entries) != 0 || len(p.last.EngramTokens) != 0 {
		t.Fatalf("missing histogram: %+v cfg=%+v", d, p.last)
	}
}
