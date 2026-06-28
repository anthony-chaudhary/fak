package model

// profile.go — modular bottleneck-observability for the in-kernel forward pass.
//
// This is the "self-inspection" tool: it attributes, per operation class
// (qkv-projection, RoPE, attention, output-projection, MLP, RMSNorm, LM-head),
// the EXACT multiply-accumulates and weight-bytes-streamed (computed analytically
// from the shapes, not sampled) and the MEASURED wall time, then derives each
// class's arithmetic intensity and a roofline verdict (memory- vs compute-bound)
// against this machine's MEASURED memory bandwidth.
//
// It is an instrumented TWIN of the proven decode path (Session.tokenHidden +
// head): the proven hot path stays pristine (zero instrumentation overhead, no
// clutter), and TestProfileMatchesProven pins this twin to it bit-for-bit, so the
// profiler can never silently drift from the code it claims to measure — the same
// witness discipline the whole model lane uses.
//
// Why this matters for the fusion thesis: it turns "decode is memory-bandwidth
// bound; prefill is GEMV-per-token" from prose into a measured, reproducible
// artifact — the kernel inspecting its own attention-state operations.

import (
	"sort"
	"time"
)

// Op-class labels (stable keys for the JSON the cmd emits).
const (
	opQKVProj = "qkv_proj" // q/k/v input projections
	opRoPE    = "rope"     // rotary position embedding on q,k
	opAttn    = "attn"     // scores (Q·K) + softmax + weighted sum over V (reads KV cache)
	opOProj   = "o_proj"   // attention output projection
	opMLP     = "mlp"      // SwiGLU gate/up/down
	opNorm    = "norm"     // RMSNorm (input, post-attn, final)
	opHead    = "head"     // tied LM head (vocab × hidden) — the single largest weight
)

// OpStat accumulates one op-class's cost across a profiled run.
type OpStat struct {
	Class    string  `json:"class"`
	Calls    int     `json:"calls"`
	MACs     int64   `json:"macs"`     // exact multiply-accumulates
	Bytes    int64   `json:"bytes"`    // weight (+ KV-cache, for attn) bytes streamed
	Nanos    int64   `json:"nanos"`    // measured wall time
	TimePct  float64 `json:"time_pct"` // share of profiled compute time
	GFLOPs   float64 `json:"gflops"`   // achieved 2*MACs/sec
	GBps     float64 `json:"gbps"`     // achieved bytes/sec
	IntensFB float64 `json:"intensity_flops_per_byte"`
	Verdict  string  `json:"verdict"` // "memory-bound" | "compute-bound"
}

// Profile is one observability run's full attribution.
type Profile struct {
	Mode          string   `json:"mode"` // "decode" | "prefill"
	Model         string   `json:"model"`
	PromptTokens  int      `json:"prompt_tokens"`
	Steps         int      `json:"steps"` // decode steps OR prefill length
	TotalNanos    int64    `json:"total_nanos"`
	TotalMACs     int64    `json:"total_macs"`
	TotalBytes    int64    `json:"total_bytes"`
	PerTokenMS    float64  `json:"per_token_ms"`
	AchievedGFLOP float64  `json:"achieved_gflops"`
	AchievedGBps  float64  `json:"achieved_gbps"`
	MemBWGBps     float64  `json:"machine_mem_bw_gbps"`       // measured STREAM-triad ceiling
	BWUtilPct     float64  `json:"bandwidth_utilization_pct"` // achieved / machine ceiling
	RidgeFB       float64  `json:"roofline_ridge_flops_per_byte"`
	Stats         []OpStat `json:"ops"`
	Bottleneck    string   `json:"bottleneck"` // dominant op-class by time
	Summary       string   `json:"summary"`
}

// CleanDecode is an uninstrumented decode throughput measurement over the proven
// Session.Step path. It is emitted alongside ProfileDecode so modelprof's roofline
// attribution is not mistaken for achievable clean decode latency.
type CleanDecode struct {
	Mode         string  `json:"mode"`
	Model        string  `json:"model"`
	PromptTokens int     `json:"prompt_tokens"`
	Steps        int     `json:"steps"`
	TotalNanos   int64   `json:"total_nanos"`
	PerTokenMS   float64 `json:"per_token_ms"`
	Summary      string  `json:"summary"`
}

// PhaseStat is a coarse wall-time bucket from an opt-in Session phase profile.
// Unlike Profile's analytic roofline stats, this is intentionally lightweight:
// it answers "which real phase ate this Qwen3.6 run?" for the same Session.Prefill
// and Session.Step calls modelbench measures.
type PhaseStat struct {
	Phase   string  `json:"phase"`
	Calls   int     `json:"calls"`
	Nanos   int64   `json:"nanos"`
	MS      float64 `json:"ms"`
	TimePct float64 `json:"time_pct"`
}

// PhaseProfile is a machine-readable phase split for one measured prefill/decode run.
type PhaseProfile struct {
	Mode       string      `json:"mode"`
	Tokens     int         `json:"tokens,omitempty"`
	Steps      int         `json:"steps,omitempty"`
	TotalNanos int64       `json:"total_nanos"`
	TotalMS    float64     `json:"total_ms"`
	PerTokenMS float64     `json:"per_token_ms,omitempty"`
	Phases     []PhaseStat `json:"phases"`
	Bottleneck string      `json:"bottleneck"`
}

// PhaseProfiler records coarse phase timings when attached to Session.PhaseProfiler.
// It is not goroutine-safe; model sessions are already single-owner during generation.
type PhaseProfiler struct {
	stat  map[string]*PhaseStat
	order []string
}

// NewPhaseProfiler creates an empty opt-in phase profiler.
func NewPhaseProfiler() *PhaseProfiler {
	return &PhaseProfiler{stat: map[string]*PhaseStat{}}
}

func (p *PhaseProfiler) record(phase string, nanos int64) {
	if p == nil {
		return
	}
	s := p.stat[phase]
	if s == nil {
		s = &PhaseStat{Phase: phase}
		p.stat[phase] = s
		p.order = append(p.order, phase)
	}
	s.Calls++
	s.Nanos += nanos
}

// Reset clears accumulated phase timings so a caller can profile decode after an
// unprofiled prefill on the same Session.
func (p *PhaseProfiler) Reset() {
	if p == nil {
		return
	}
	for k := range p.stat {
		delete(p.stat, k)
	}
	p.order = p.order[:0]
}

// Snapshot returns a sorted copy of the current timings. totalNanos should be the
// externally measured wall time for the whole operation; if omitted, percentages use
// the sum of recorded phase nanos.
func (p *PhaseProfiler) Snapshot(mode string, tokens, steps int, totalNanos int64) *PhaseProfile {
	if p == nil {
		return nil
	}
	denom := totalNanos
	if denom <= 0 {
		for _, s := range p.stat {
			denom += s.Nanos
		}
	}
	pr := &PhaseProfile{
		Mode:       mode,
		Tokens:     tokens,
		Steps:      steps,
		TotalNanos: totalNanos,
		TotalMS:    float64(totalNanos) / 1e6,
	}
	if steps > 0 {
		pr.PerTokenMS = float64(totalNanos) / 1e6 / float64(steps)
	} else if tokens > 0 {
		pr.PerTokenMS = float64(totalNanos) / 1e6 / float64(tokens)
	}
	for _, key := range p.order {
		src := p.stat[key]
		st := *src
		st.MS = float64(st.Nanos) / 1e6
		if denom > 0 {
			st.TimePct = 100 * float64(st.Nanos) / float64(denom)
		}
		pr.Phases = append(pr.Phases, st)
	}
	sort.Slice(pr.Phases, func(i, j int) bool { return pr.Phases[i].Nanos > pr.Phases[j].Nanos })
	if len(pr.Phases) > 0 {
		pr.Bottleneck = pr.Phases[0].Phase
	}
	return pr
}

func (s *Session) phaseStart() time.Time {
	if s == nil || s.PhaseProfiler == nil {
		return time.Time{}
	}
	return time.Now()
}

// headLogitsBuf returns the reused, grown logits buffer for one decode head together with a
// fresh phase timer — the shared setup of headQ / headQ4 / headQ4K / headGPTQ.
func (s *Session) headLogitsBuf() ([]float32, time.Time) {
	if s.qDecode == nil {
		s.qDecode = &qDecodeBuf{}
	}
	y := grow(s.qDecode.Logits, s.M.Cfg.VocabSize)
	s.qDecode.Logits = y
	return y, s.phaseStart()
}

func (s *Session) phaseEnd(phase string, start time.Time) {
	if s == nil || s.PhaseProfiler == nil || start.IsZero() {
		return
	}
	s.PhaseProfiler.record(phase, time.Since(start).Nanoseconds())
}

// profiler accumulates op-class costs during a profiled twin run.
type profiler struct {
	stat  map[string]*OpStat
	order []string
}

func newProfiler() *profiler { return &profiler{stat: map[string]*OpStat{}} }

func (p *profiler) rec(class string, macs, bytes, nanos int64) {
	s := p.stat[class]
	if s == nil {
		s = &OpStat{Class: class}
		p.stat[class] = s
		p.order = append(p.order, class)
	}
	s.Calls++
	s.MACs += macs
	s.Bytes += bytes
	s.Nanos += nanos
}

// profToken is the instrumented twin of tokenHidden+head. It MUST stay numerically
// identical to the proven path; TestProfileMatchesProven enforces that. withHead
// mirrors the prefill optimization (head only on the consumed position).
//
// It drives the projections through parMatRows — the SAME row-parallel matmul kernel
// tokenHidden/head use — not the serial matRows. parMatRows is bit-identical to matRows
// (same fdot, same inner i-order; only the assignment of output rows to cores differs),
// so the numbers TestProfileMatchesProven pins are unchanged, but the per-op WALL TIME
// now reflects the parallel decode the proven path actually runs. Using serial matRows
// here was the root cause of issue #31: decode is memory-bandwidth-bound and one core
// taps ~41% of aggregate bandwidth, so a serial profiler reported ~2.7x the per-token
// latency of the parallel Session.Step path that modelbench/q8bench measure.
func (s *Session) profToken(p *profiler, id, pos int, withHead bool) []float32 {
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	cos, sin := ropeRow(cfg, pos)

	embed := m.embedRows()
	x := append([]float32(nil), embed[id*H:(id+1)*H]...)

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }

		t := time.Now()
		xn := rmsnorm(x, m.tensor(lp("input_layernorm.weight")), eps)
		p.rec(opNorm, int64(H), int64(H*4), time.Since(t).Nanoseconds())

		t = time.Now()
		q := parMatRows(m.tensor(lp("self_attn.q_proj.weight")), xn, nH*hd, H)
		kk := parMatRows(m.tensor(lp("self_attn.k_proj.weight")), xn, w, H)
		vv := parMatRows(m.tensor(lp("self_attn.v_proj.weight")), xn, w, H)
		p.rec(opQKVProj, int64((nH*hd+2*w)*H), int64((nH*hd+2*w)*H*4), time.Since(t).Nanoseconds())

		t = time.Now()
		// stash PRE-RoPE k, then rotate q/k through the shared single-row builder.
		s.Cache.Kraw[l] = append(s.Cache.Kraw[l], kk...)
		ropeRowQKInto(q, kk, cos, sin, hd, nH, nKV)
		p.rec(opRoPE, int64((nH+nKV)*hd), 0, time.Since(t).Nanoseconds())

		s.Cache.K[l] = append(s.Cache.K[l], kk...)
		s.Cache.V[l] = append(s.Cache.V[l], vv...)
		nPos := len(s.Cache.K[l]) / w

		t = time.Now()
		attnOut := make([]float32, nH*hd)
		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := q[h*hd : (h+1)*hd]
			scores := make([]float32, nPos)
			for j := 0; j < nPos; j++ {
				kh := s.Cache.K[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
				scores[j] = dot(qh, kh) * scale
			}
			softmaxInPlace(scores)
			out := attnOut[h*hd : (h+1)*hd]
			for j := 0; j < nPos; j++ {
				vh := s.Cache.V[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
				wj := scores[j]
				for d := 0; d < hd; d++ {
					out[d] += wj * vh[d]
				}
			}
		}
		// attn LOADS K and V from the cache once per QUERY head (nH), not per KV head:
		// with GQA the grp=nH/nKV query heads in a group each re-read their shared KV
		// head's K/V column, so the loads the code actually issues are 2*nH*nPos*hd, not
		// the 2*nKV minimal footprint. We charge loads-issued (intensity 0.5, consistent
		// with every other GEMV) to keep this "what the pass streams" not "optimal-kernel
		// lower bound". NB the KV cache at these nPos is L2-resident, so attn's *DRAM*
		// traffic is far below this — which is exactly why its measured GB/s (~2) sits
		// well under the weight ops' ~10.6: attn is latency/overhead-bound, not BW-bound.
		p.rec(opAttn, int64(2*nH*nPos*hd), int64(2*nH*nPos*hd*4), time.Since(t).Nanoseconds())

		t = time.Now()
		o := parMatRows(m.tensor(lp("self_attn.o_proj.weight")), attnOut, H, nH*hd)
		for i := 0; i < H; i++ {
			x[i] += o[i]
		}
		p.rec(opOProj, int64(H*nH*hd), int64(H*nH*hd*4), time.Since(t).Nanoseconds())

		t = time.Now()
		xn2 := rmsnorm(x, m.tensor(lp("post_attention_layernorm.weight")), eps)
		p.rec(opNorm, int64(H), int64(H*4), time.Since(t).Nanoseconds())

		t = time.Now()
		I := cfg.IntermediateSize
		g := parMatRows(m.tensor(lp("mlp.gate_proj.weight")), xn2, I, H)
		u := parMatRows(m.tensor(lp("mlp.up_proj.weight")), xn2, I, H)
		for i := 0; i < I; i++ {
			g[i] = silu(g[i]) * u[i]
		}
		down := parMatRows(m.tensor(lp("mlp.down_proj.weight")), g, H, I)
		for i := 0; i < H; i++ {
			x[i] += down[i]
		}
		p.rec(opMLP, int64(3*I*H), int64(3*I*H*4), time.Since(t).Nanoseconds())
	}

	s.Cache.pos = append(s.Cache.pos, pos)
	t := time.Now()
	xf := rmsnorm(x, m.tensor("model.norm.weight"), eps)
	p.rec(opNorm, int64(H), int64(H*4), time.Since(t).Nanoseconds())
	if !withHead {
		return xf
	}
	t = time.Now()
	logits := parMatRows(m.lmHead(), xf, cfg.VocabSize, H)
	p.rec(opHead, int64(cfg.VocabSize*H), int64(cfg.VocabSize*H*4), time.Since(t).Nanoseconds())
	return logits
}

// ProfileDecode prefills a prompt (untimed warm-up of the cache) then profiles
// `steps` real decode steps — the batch=1 autoregressive regime the agent loop
// actually runs in. The head fires every decode step (its logits are consumed).
//
// profToken now drives the SAME parMatRows kernel Session.Step uses, so the per-token
// total here agrees with MeasureCleanDecode (and hence modelbench/q8bench) to within the
// small per-op timing overhead — TestProfileDecodeAgreesWithCleanDecode pins that. The
// only residual gap is the time.Now()/time.Since() bracketing each op class, which is why
// the modelprof header still labels this figure "instrumented".
func (m *Model) ProfileDecode(promptLen, steps int) *Profile {
	vocab := m.Cfg.VocabSize
	s := m.NewSession()
	// warm the cache with a prefill (not profiled): faults weights in + fills KV.
	for i := 0; i < promptLen; i++ {
		s.tokenHidden((i*97+13)%vocab, s.Cache.Len())
	}
	p := newProfiler()
	id := 7
	t0 := time.Now()
	for i := 0; i < steps; i++ {
		s.profToken(p, id, s.Cache.Len(), true)
		id = (id*48271 + 1) % vocab
	}
	total := time.Since(t0).Nanoseconds()
	return m.finishProfile(p, "decode", promptLen, steps, total)
}

// MeasureCleanDecode prefills a prompt, then times the proven uninstrumented
// Session.Step decode loop over the same deterministic token stream ProfileDecode
// uses. It deliberately records no per-op attribution.
func (m *Model) MeasureCleanDecode(promptLen, steps int) *CleanDecode {
	vocab := m.Cfg.VocabSize
	s := m.NewSession()
	prompt := make([]int, promptLen)
	for i := range prompt {
		prompt[i] = (i*97 + 13) % vocab
	}
	s.Prefill(prompt)

	id := 7
	t0 := time.Now()
	for i := 0; i < steps; i++ {
		s.Step(id)
		id = (id*48271 + 1) % vocab
	}
	total := time.Since(t0).Nanoseconds()
	perTok := 0.0
	if steps > 0 {
		perTok = float64(total) / 1e6 / float64(steps)
	}
	return &CleanDecode{
		Mode:         "clean_decode",
		Model:        "SmolLM2-135M (f32)",
		PromptTokens: promptLen,
		Steps:        steps,
		TotalNanos:   total,
		PerTokenMS:   perTok,
		Summary:      "uninstrumented Session.Step decode latency; compare with decode.per_token_ms, which is instrumented roofline attribution",
	}
}

// ProfilePrefill profiles a fresh P-token prefill (head only on the last position,
// matching Session.Prefill) — the compute-bound-in-principle regime that fak runs
// as GEMV-per-token, so the tool shows why it stays memory-bound here.
func (m *Model) ProfilePrefill(promptLen int) *Profile {
	vocab := m.Cfg.VocabSize
	s := m.NewSession()
	p := newProfiler()
	t0 := time.Now()
	for i := 0; i < promptLen; i++ {
		s.profToken(p, (i*97+13)%vocab, s.Cache.Len(), i == promptLen-1)
	}
	total := time.Since(t0).Nanoseconds()
	return m.finishProfile(p, "prefill", promptLen, promptLen, total)
}

func (m *Model) finishProfile(p *profiler, mode string, promptLen, steps int, totalNanos int64) *Profile {
	memBW := MeasureMemBandwidthGBps(len(m.raw))
	// roofline ridge: peak compute / peak bandwidth. We don't have a measured peak
	// FLOP/s here; report intensity per class and classify against a conservative
	// CPU ridge (~ machine can do far more flops/byte than batch=1 GEMV's 0.5).
	pr := &Profile{
		Mode: mode, Model: "SmolLM2-135M (f32)", PromptTokens: promptLen, Steps: steps,
		TotalNanos: totalNanos, MemBWGBps: memBW,
	}
	for _, c := range p.order {
		st := p.stat[c]
		pr.TotalMACs += st.MACs
		pr.TotalBytes += st.Bytes
	}
	secs := float64(totalNanos) / 1e9
	for _, c := range p.order {
		st := p.stat[c]
		st.TimePct = 100 * float64(st.Nanos) / float64(totalNanos)
		osecs := float64(st.Nanos) / 1e9
		if osecs > 0 {
			st.GFLOPs = 2 * float64(st.MACs) / osecs / 1e9
			st.GBps = float64(st.Bytes) / osecs / 1e9
		}
		if st.Bytes > 0 {
			st.IntensFB = 2 * float64(st.MACs) / float64(st.Bytes)
		}
		// batch=1 projection/head ops are pure GEMV: ~0.5 flops/weight-byte -> the
		// memory system, not the ALUs, is the limit. attn reads cache at similar AI.
		if st.IntensFB < 4 || st.Class == opAttn || st.Class == opHead {
			st.Verdict = "memory-bound"
		} else {
			st.Verdict = "compute-bound"
		}
		pr.Stats = append(pr.Stats, *st)
	}
	sort.Slice(pr.Stats, func(i, j int) bool { return pr.Stats[i].Nanos > pr.Stats[j].Nanos })
	if len(pr.Stats) > 0 {
		pr.Bottleneck = pr.Stats[0].Class
	}
	pr.PerTokenMS = float64(totalNanos) / 1e6 / float64(steps)
	if secs > 0 {
		pr.AchievedGFLOP = 2 * float64(pr.TotalMACs) / secs / 1e9
		pr.AchievedGBps = float64(pr.TotalBytes) / secs / 1e9
	}
	if memBW > 0 {
		pr.BWUtilPct = 100 * pr.AchievedGBps / memBW
	}
	if pr.TotalBytes > 0 {
		pr.RidgeFB = 2 * float64(pr.TotalMACs) / float64(pr.TotalBytes)
	}
	return pr
}

// MeasureMemBandwidthGBps runs a STREAM-triad (a[i]=b[i]+scalar*c[i]) over buffers
// sized like the model's weights, single-threaded, to anchor the roofline at the
// SAME access pattern the single-threaded forward pass sees. Returns GB/s.
func MeasureMemBandwidthGBps(modelBytes int) float64 {
	n := modelBytes / 4 / 3 // three f32 buffers ~ one model's worth of traffic
	if n < 1<<20 {
		n = 1 << 20
	}
	a := make([]float32, n)
	b := make([]float32, n)
	c := make([]float32, n)
	for i := range b {
		b[i] = float32(i % 7)
		c[i] = float32(i % 5)
	}
	const scalar = 3.0
	// warm
	for i := 0; i < n; i++ {
		a[i] = b[i] + scalar*c[i]
	}
	best := 0.0
	for rep := 0; rep < 5; rep++ {
		t := time.Now()
		for i := 0; i < n; i++ {
			a[i] = b[i] + scalar*c[i]
		}
		secs := time.Since(t).Seconds()
		if secs <= 0 {
			continue // a zero-duration rep (coarse timer on a small buffer) would be +Inf and poison best
		}
		// triad touches 3 arrays (2 read, 1 write) of n f32 = 12n bytes
		gbps := float64(12*n) / secs / 1e9
		if gbps > best {
			best = gbps
		}
	}
	return best
}
