//go:build cuda && cgo

package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestCUDAPromptPanelFencedOpProfile is the profile-ON harness behind prompt-panel perf
// receipts. It is opt-in: FAK_PROMPT_PANEL_PROFILE_HF names an HF snapshot loaded lean (Q8_0),
// FAK_PROMPT_PANEL_PROFILE_P lists prompt lengths (default 64,256) and
// FAK_PROMPT_PANEL_PROFILE_OUT receives the JSON. A wrapper drains the stream (a 1-float Read)
// before and after every HAL op of one warm panel prefill and subtracts the median empty-drain
// cost, so each row is launch plus GPU time of that op; the drains serialize the stream, so a
// profiled prefill is never a token-time arm.
func TestCUDAPromptPanelFencedOpProfile(t *testing.T) {
	dir := os.Getenv("FAK_PROMPT_PANEL_PROFILE_HF")
	if dir == "" {
		t.Skip("set FAK_PROMPT_PANEL_PROFILE_HF to an HF snapshot to profile")
	}
	be := promptPanelCUDA(t)
	var cfg Config
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.HeadDim == 0 && cfg.NumHeads != 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	m, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseWeights()
	lengths := []int{64, 256}
	if v := os.Getenv("FAK_PROMPT_PANEL_PROFILE_P"); v != "" {
		lengths = lengths[:0]
		for _, f := range strings.Split(v, ",") {
			p, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil || p < 2 {
				t.Fatalf("bad FAK_PROMPT_PANEL_PROFILE_P entry %q", f)
			}
			lengths = append(lengths, p)
		}
	}
	report := map[string]any{"model": filepath.Base(dir), "backend": be.Name(), "tier": be.Tier()}
	for _, P := range lengths {
		prof := newPromptPanelFencedProfile(be, m.Cfg)
		ids := promptPanelIDs(P, m.Cfg.VocabSize)
		warm := m.NewBackendSession(prof)
		warm.Quant = true
		warm.Prefill(ids)
		warm.Close()
		s := m.NewBackendSession(prof)
		s.Quant = true
		prof.on = true
		started := time.Now()
		s.Prefill(ids)
		wall := time.Since(started)
		prof.on = false
		s.Close()
		report[fmt.Sprintf("P%d", P)] = prof.rows(P, wall, s.promptPanelChunkTokens())
	}
	if out := os.Getenv("FAK_PROMPT_PANEL_PROFILE_OUT"); out != "" {
		data, _ := json.MarshalIndent(report, "", "  ")
		if err := os.WriteFile(out, append(data, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

type promptPanelFencedProfile struct {
	compute.Backend
	panel     promptPanelBackend
	cfg       Config
	drain     compute.Tensor
	drainCost time.Duration
	on        bool
	times     map[string]time.Duration
	calls     map[string]int
}

func newPromptPanelFencedProfile(be compute.Backend, cfg Config) *promptPanelFencedProfile {
	p := &promptPanelFencedProfile{Backend: be, panel: be.(promptPanelBackend), cfg: cfg,
		times: map[string]time.Duration{}, calls: map[string]int{}}
	p.drain = be.Upload(compute.NewF32(compute.Default(), []int{1}, []float32{0}), compute.F32)
	costs := make([]time.Duration, 0, 200)
	for i := 0; i < 200; i++ {
		p.fence()
		t0 := time.Now()
		p.fence()
		costs = append(costs, time.Since(t0))
	}
	sort.Slice(costs, func(i, j int) bool { return costs[i] < costs[j] })
	p.drainCost = costs[len(costs)/2]
	return p
}

func (p *promptPanelFencedProfile) fence() { p.Backend.Read(p.drain) }

func (p *promptPanelFencedProfile) timed(op string, f func()) {
	if !p.on {
		f()
		return
	}
	p.fence()
	t0 := time.Now()
	f()
	p.fence()
	p.times[op] += max(time.Since(t0)-p.drainCost, 0)
	p.calls[op]++
}

func (p *promptPanelFencedProfile) rows(P int, wall time.Duration, chunk int) map[string]any {
	var total time.Duration
	names := make([]string, 0, len(p.times))
	for op, d := range p.times {
		total += d
		names = append(names, op)
	}
	sort.Slice(names, func(i, j int) bool { return p.times[names[i]] > p.times[names[j]] })
	ops := make([]map[string]any, 0, len(names))
	for _, op := range names {
		ms := float64(p.times[op].Microseconds()) / 1000
		ops = append(ops, map[string]any{"op": op, "calls": p.calls[op], "ms": ms,
			"ms_per_prompt_token": ms / float64(P), "share_of_timed": float64(p.times[op]) / float64(total)})
	}
	return map[string]any{"prompt_tokens": P, "chunk_tokens": chunk,
		"wall_ms_profiled": float64(wall.Microseconds()) / 1000, "timed_ms": float64(total.Microseconds()) / 1000,
		"drain_cost_us": float64(p.drainCost.Nanoseconds()) / 1000, "ops": ops}
}

func (p *promptPanelFencedProfile) projection(w compute.Tensor) string {
	c := p.cfg
	out, in := w.Shape[0], w.Shape[1]
	switch {
	case out == c.NumHeads*c.HeadDim && in == c.HiddenSize:
		return "q_proj/o_proj"
	case out == c.NumKVHeads*c.HeadDim:
		return "k_proj/v_proj"
	case out == c.IntermediateSize:
		return "gate_proj/up_proj"
	case in == c.IntermediateSize:
		return "down_proj"
	case out == c.VocabSize:
		return "lm_head"
	}
	return fmt.Sprintf("%dx%d", out, in)
}

func (p *promptPanelFencedProfile) MatMul(w, x compute.Tensor) (y compute.Tensor) {
	p.timed("gemv "+p.projection(w), func() { y = p.Backend.MatMul(w, x) })
	return y
}
func (p *promptPanelFencedProfile) RMSNorm(x, w compute.Tensor, eps float32) (y compute.Tensor) {
	p.timed("rmsnorm", func() { y = p.Backend.RMSNorm(x, w, eps) })
	return y
}
func (p *promptPanelFencedProfile) AddBias(d, b compute.Tensor) {
	p.timed("add_bias", func() { p.Backend.AddBias(d, b) })
}
func (p *promptPanelFencedProfile) AddInPlace(d, s compute.Tensor) {
	p.timed("residual_add", func() { p.Backend.AddInPlace(d, s) })
}
func (p *promptPanelFencedProfile) SwiGLU(g, u compute.Tensor) (y compute.Tensor) {
	p.timed("swiglu", func() { y = p.Backend.SwiGLU(g, u) })
	return y
}
func (p *promptPanelFencedProfile) UploadClass(t compute.Tensor, as compute.Dtype, c compute.MemoryClass, site string) (y compute.Tensor) {
	p.timed("upload", func() { y = p.Backend.(classedUploadBackend).UploadClass(t, as, c, site) })
	return y
}
func (p *promptPanelFencedProfile) Recycle() {
	if r, ok := p.Backend.(interface{ Recycle() }); ok {
		r.Recycle()
	}
}
func (p *promptPanelFencedProfile) PromptPanelRowExact(dt compute.Dtype) bool {
	return p.panel.PromptPanelRowExact(dt)
}
func (p *promptPanelFencedProfile) PromptPanelBegin()         { p.panel.PromptPanelBegin() }
func (p *promptPanelFencedProfile) PromptPanelChunk(rows int) { p.panel.PromptPanelChunk(rows) }
func (p *promptPanelFencedProfile) PromptPanelEnd()           { p.panel.PromptPanelEnd() }
func (p *promptPanelFencedProfile) PromptPanelMatMul(w, x compute.Tensor, tokens int) (y compute.Tensor) {
	p.timed("q8_rows_gemm "+p.projection(w), func() { y = p.panel.PromptPanelMatMul(w, x, tokens) })
	return y
}
func (p *promptPanelFencedProfile) PromptPanelRoPE(x compute.Tensor, tokens, start, nh, hd int, th float64) (y compute.Tensor) {
	p.timed("rope_panel", func() { y = p.panel.PromptPanelRoPE(x, tokens, start, nh, hd, th) })
	return y
}
func (p *promptPanelFencedProfile) PromptPanelAppendKV(kv compute.KVStore, l int, kr, k, v compute.Tensor, start, tokens int) {
	p.timed("kv_append_panel", func() { p.panel.PromptPanelAppendKV(kv, l, kr, k, v, start, tokens) })
}
func (p *promptPanelFencedProfile) PromptPanelAttention(q compute.Tensor, kv compute.KVStore, l, tokens, start, g int, s float32) (y compute.Tensor) {
	p.timed("causal_attention_panel", func() { y = p.panel.PromptPanelAttention(q, kv, l, tokens, start, g, s) })
	return y
}
func (p *promptPanelFencedProfile) PromptPanelRow(x compute.Tensor, row, w int) (y compute.Tensor) {
	p.timed("panel_row", func() { y = p.panel.PromptPanelRow(x, row, w) })
	return y
}
