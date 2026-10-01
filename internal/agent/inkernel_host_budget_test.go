package agent

import (
	"context"
	"errors"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// Pressure relief must preserve the signed retention policy, including keep-none,
// rather than restore it from MaxTokens, which also reports zero for keep-all.
// fak-test:runtime fast est=1ms lane=default
func TestInKernelHostMemoryBudgetReliefPreservesRetention(t *testing.T) {
	for _, tc := range []struct {
		name string
		tree *radixkv.Tree
		want int
	}{
		{"keep-none", radixkv.NewWithRetention(0), 0},
		{"signed-bounded", radixkv.NewWithRetention(4), 4},
		{"signed-unbounded", radixkv.NewWithRetention(-1), 8},
		{"legacy-bounded", radixkv.New(4), 4},
		{"legacy-unbounded", radixkv.New(0), 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := &InKernelPlanner{tree: tc.tree}
			p.trimPrefixCacheForHostPressure(func() bool { return false })
			for _, ids := range [][]int{{1, 2, 3, 4}, {5, 6, 7, 8}} {
				boundary, matched := p.tree.Lookup(ids)
				leaf := p.tree.Insert(boundary, ids[matched:], nil)
				p.tree.Done(leaf)
			}
			// Insert protects its own leased leaf. Apply the retained budget after
			// both requests release their leases, without changing that budget.
			p.tree.PlanBoundedEviction(1)
			p.tree.ConfirmEvictions()
			if got := p.tree.Stats().Tokens; got != tc.want {
				t.Fatalf("retained tokens after pressure relief = %d, want %d", got, tc.want)
			}
		})
	}
}

// These tests pin the #13267 host-memory admission for the host-session (Metal) seam: a
// planner with backend == nil has no compute.Backend for refuseOversizeRequest to ask, so an
// armed host ceiling must price every turn BEFORE any session, clone, or prefill allocation.
// They are hardware-free: no Metal device is touched and no real model is loaded. Ladder byte
// figures are derived from p.hostRequestDemand (itself pinned against the spec'd pricing in
// TestInKernelHostMemoryBudgetPricesBaseContextPlan) rather than hard-coded, so the tests pin
// the admission ladder, not a particular sizing constant.

// hostUsageProbe is a scripted process-usage probe for SetHostMemoryBudget. at maps the 1-based
// call number to the reported (used, known) pair, which lets a test model reclaimed garbage or
// cache (high usage first, low afterwards).
type hostUsageProbe struct {
	mu    sync.Mutex
	calls int
	at    func(call int) (int64, bool)
}

func (h *hostUsageProbe) used() (int64, bool) {
	h.mu.Lock()
	h.calls++
	call := h.calls
	h.mu.Unlock()
	return h.at(call)
}

func (h *hostUsageProbe) callCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.calls
}

func constHostUsage(used int64, known bool) *hostUsageProbe {
	return &hostUsageProbe{at: func(int) (int64, bool) { return used, known }}
}

// hostReserved reads the in-flight reservation of the armed budget (0 when unarmed).
func hostReserved(p *InKernelPlanner) int64 {
	p.hostBudgetMu.Lock()
	defer p.hostBudgetMu.Unlock()
	if p.hostBudget == nil {
		return 0
	}
	return p.hostBudget.reserved
}

// hostSessionReusePlanner builds the resident host-session planner the way `fak up` does
// (backend nil, so prefix reuse is on and the radix tree exists) over a tiny synthetic model.
// metal stays false at construction so PrepareMetalResidency never runs; the admission arm
// keys on backend == nil, which is the host-session seam either way. The env is pinned so the
// radix tree exists and its token budget is the model's declared context window.
func hostSessionReusePlanner(t *testing.T) *InKernelPlanner {
	t.Helper()
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	t.Setenv("FAK_INKERNEL_RADIX_BUDGET", "")
	cfg := tinyConcurrencyConfig()
	cfg.MaxPositionEmbeddings = 4096
	p := NewInKernelPlanner(model.NewSynthetic(cfg), loadProbeTok(t), "host-budget-reuse", false, nil, false)
	if p.tree == nil || !inKernelPlannerPrefixReuseSupported(p.m, p.backend) {
		t.Fatalf("host-session planner must have prefix reuse on (tree=%v supported=%v)",
			p.tree != nil, inKernelPlannerPrefixReuseSupported(p.m, p.backend))
	}
	return p
}

// bareHostPlanner is a host-session planner with prefix reuse off (no tree), so a turn's
// demand is its session alone.
func bareHostPlanner() *InKernelPlanner {
	return &InKernelPlanner{m: model.NewSynthetic(tinyConcurrencyConfig()), modelID: "host-budget-bare"}
}

// qwenQ4KHostPlanner is the served darwin `fak up` shape: a Q4_K Qwen3.5-hybrid host-session
// planner (backend nil, so it is on the chunked Qwen Q4_K prefill route) with the DEFAULT
// prefill-panel chunk. The model has no weights; nothing here runs a forward.
func qwenQ4KHostPlanner(t *testing.T, modelID string) *InKernelPlanner {
	t.Helper()
	cfg := qwen35RuntimeExtraConfig()
	cfg.MaxPositionEmbeddings = 4096
	p := NewInKernelPlanner(&model.Model{Cfg: cfg}, loadProbeTok(t), modelID, true, nil, false)
	// The resident darwin `fak up` seam: no compute.Backend, Metal host session. Set directly
	// so PrepareMetalResidency (a device touch) never runs.
	p.metal = true
	if p.backend != nil {
		t.Fatal("host-session planner must have a nil backend")
	}
	if !p.qwenQ4KPrefillChunkTarget() {
		t.Fatal("precondition: the Q4_K hybrid host-session planner must be on the chunked Qwen Q4_K prefill route")
	}
	return p
}

// hostDemand prices one turn and fails the test on a non-positive session.
func hostDemand(t *testing.T, p *InKernelPlanner, promptTokens, maxNew int) (session, retained int64) {
	t.Helper()
	session, retained, _ = p.hostRequestDemand(promptTokens, maxNew)
	if session <= 0 {
		t.Fatalf("hostRequestDemand(%d,%d) session = %d, want positive", promptTokens, maxNew, session)
	}
	return session, retained
}

// hostBaseContextPlan is the spec'd session basis: the single context auto-sizer's
// per-context plan over prompt+maxNew positions (KV + fixed session state + HAL transient),
// with no resident weights and no runtime extras.
func hostBaseContextPlan(p *InKernelPlanner, promptTokens, maxNew int) compute.MemoryPlan {
	cs := p.m.Cfg.ContextSizeConfigWithPrecision(computeKVPrecisionFor(p.kvPrecision))
	_, plan := compute.AutoSizeContextPlan(cs, nil, compute.FreeUnknown, promptTokens+maxNew)
	return plan
}

// hostPanelRowBytes is the f32 live bytes of one host prefill-panel token at the MLP:
// residual, normed and down-projected rows (3 x hidden) plus gate and up rows (2 x intermediate).
func hostPanelRowBytes(p *InKernelPlanner) int64 {
	return int64(3*p.m.Cfg.HiddenSize+2*p.m.Cfg.IntermediateSize) * 4
}

// hostRetainedPerCopy is one retained prefix-cache copy: KV(prompt+maxNew) + fixed session state.
func hostRetainedPerCopy(p *InKernelPlanner, promptTokens, maxNew int) int64 {
	cs := p.m.Cfg.ContextSizeConfigWithPrecision(computeKVPrecisionFor(p.kvPrecision))
	return compute.EstimateKVStoreBytes(cs.KV, promptTokens+maxNew) + cs.SessionState.Total()
}

// radixTokens reads the tree's cached token count under the planner lock.
func radixTokens(p *InKernelPlanner) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.tree.Stats().Tokens
}

// seedRadixTokens inserts n disjoint pure-accounting chains of width tokens into the tree
// (kv = nil: the tokens are indexed, which is what the LRU token budget counts) and returns
// the resulting cached token count.
func seedRadixTokens(t *testing.T, p *InKernelPlanner, chains, width int) int {
	t.Helper()
	for c := 0; c < chains; c++ {
		ids := make([]int, width)
		for i := range ids {
			ids[i] = 3 + c*width + i // every chain starts on a distinct token: disjoint leaves
		}
		b, m := p.tree.Lookup(ids)
		leaf := p.tree.Insert(b, ids[m:], nil)
		p.tree.Done(leaf)
	}
	st := p.tree.Stats()
	if st.Tokens <= 0 {
		t.Fatalf("seeded radix tree holds %d tokens, want positive", st.Tokens)
	}
	return st.Tokens
}

// TestInKernelHostMemoryBudgetDeclinesBeforePrefill is the symptom test: the served entry
// point on the host-session (Metal) seam must decline a request that cannot fit the armed
// process ceiling BEFORE any allocation. The model has NO weights, so any session build /
// prefill would panic; receiving the typed host-scope capacity error is the witness that the
// decline ran first. It runs on the served shape with the DEFAULT prefill-panel chunk, so the
// short chat prompt is priced by the host arm, not refused as an unknown runtime extra.
func TestInKernelHostMemoryBudgetDeclinesBeforePrefill(t *testing.T) {
	p := qwenQ4KHostPlanner(t, "host-budget-reach")
	p.batchDecode = false
	// refuseOversizeRequest is the device arm; on this seam it must not be what refuses.
	if err := p.refuseOversizeRequest(1<<20, 8); err != nil {
		t.Fatalf("device precheck must be a no-op on the host-session seam, got %v", err)
	}

	// The prompt is at least one token, and the session demand grows with the prompt, so a
	// headroom one byte below the one-token session is guaranteed too small for this turn.
	floorSession, _ := hostDemand(t, p, 1, 8)
	const ceiling = int64(1 << 30)
	avail := floorSession - 1
	probe := constHostUsage(ceiling-avail, true)
	p.SetHostMemoryBudget(ceiling, probe.used)

	var (
		comp     *Completion
		err      error
		panicked any
	)
	func() {
		defer func() { panicked = recover() }()
		comp, err = p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, WithMaxTokens(8))
	}()
	if panicked != nil {
		t.Fatalf("Complete panicked (model work ran: the host-memory decline did not fire before allocation): %v", panicked)
	}
	if comp != nil {
		t.Fatalf("declined request returned a completion: %+v", comp)
	}
	var capErr *InKernelCapacityError
	if !errors.As(err, &capErr) {
		t.Fatalf("Complete error = %T (%v), want *InKernelCapacityError from the host-memory precheck", err, err)
	}
	if capErr.Scope != compute.MemoryScopeHost {
		t.Fatalf("capacity error scope = %s, want %s", capErr.Scope, compute.MemoryScopeHost)
	}
	if capErr.Site != "host-memory-precheck" {
		t.Fatalf("capacity error site = %q, want host-memory-precheck", capErr.Site)
	}
	if capErr.Want <= capErr.Avail {
		t.Fatalf("capacity sizing want %d <= avail %d, want a refused budget", capErr.Want, capErr.Avail)
	}
	if capErr.Want < floorSession {
		t.Fatalf("declined want %d is below the one-token session floor %d", capErr.Want, floorSession)
	}
	if capErr.Avail != avail {
		t.Fatalf("capacity avail = %d, want ceiling-used = %d", capErr.Avail, avail)
	}
	if msg := err.Error(); !strings.Contains(msg, "host-memory capacity precheck refused request") || strings.Contains(msg, "GPU") {
		t.Fatalf("served host refusal message = %q, want the host-memory wording (no GPU)", msg)
	}
	if probe.callCount() == 0 {
		t.Fatal("the armed usage probe was never consulted")
	}
	if got := hostReserved(p); got != 0 {
		t.Fatalf("host reservation after a declined request = %d, want 0", got)
	}
}

// TestInKernelHostMemoryBudgetAdmitsShortQwenQ4KHostRequest pins ladder rung a on the served
// darwin `fak up` shape itself: a Q4_K Qwen3.5-hybrid host-session planner with the DEFAULT
// prefill-panel chunk. A short turn (prompt+maxNew below the 512-token panel) that fits an
// ample ceiling must be admitted with caching, exactly as a long one is. The host arm prices
// the base context plan plus its own panel, never the Vulkan runtime extras, so a short prompt
// is not an unknown capacity bound and arming --max-rss must not turn every short request
// into a 503.
func TestInKernelHostMemoryBudgetAdmitsShortQwenQ4KHostRequest(t *testing.T) {
	p := qwenQ4KHostPlanner(t, "host-budget-short")
	const ceiling = int64(1 << 40) // ample: every rung-a turn below fits
	probe := constHostUsage(0, true)
	p.SetHostMemoryBudget(ceiling, probe.used)

	for _, tc := range []struct {
		name                 string
		promptTokens, maxNew int
	}{
		{"long-prompt", 1024, 64},
		{"short-prompt", 32, 16},
		{"short-prompt-default-max-tokens", 64, 256},
		{"one-token", 1, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			session, retained := hostDemand(t, p, tc.promptTokens, tc.maxNew)
			before := probe.callCount()
			ctx, release, err := p.admitHostMemory(context.Background(), tc.promptTokens, tc.maxNew)
			if err != nil {
				t.Fatalf("prompt %d + maxNew %d under a %d-byte ceiling with zero usage: admitHostMemory = %v, want admitted",
					tc.promptTokens, tc.maxNew, ceiling, err)
			}
			defer release()
			if inKernelSkipPrefixAdmission(ctx) {
				t.Fatal("ample headroom must admit with prefix caching")
			}
			if got := hostReserved(p); got != session+retained {
				t.Fatalf("admitted turn reserved %d, want session+retained %d", got, session+retained)
			}
			if calls := probe.callCount() - before; calls != 1 {
				t.Fatalf("rung-a admission probed usage %d times, want exactly 1 (no reclaim, no trim)", calls)
			}
		})
	}
	if got := hostReserved(p); got != 0 {
		t.Fatalf("reservation after every short turn released = %d, want 0", got)
	}
}

// TestInKernelHostMemoryBudgetPricesBaseContextPlan pins hostRequestDemand: session is the
// base context plan total (compute.AutoSizeContextPlan over prompt+maxNew: KV + fixed session
// state + HAL transient, NOT the Vulkan runtime extras) plus the host prefill panel; retained
// is 3 x (KV(prompt+maxNew) + session state) when prefix reuse is on, else 0; and the
// returned plan is that base plan.
func TestInKernelHostMemoryBudgetPricesBaseContextPlan(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		p                    func(t *testing.T) *InKernelPlanner
		promptTokens, maxNew int
		panelTokens          int64
		reuse                bool
	}{
		{"reuse-off", func(*testing.T) *InKernelPlanner { return bareHostPlanner() }, 64, 16, 64, false},
		{"reuse-on", hostSessionReusePlanner, 64, 16, 64, true},
		{"reuse-on-long-unchunked", hostSessionReusePlanner, 1024, 32, 1024, true},
		{"qwen-q4k-short", func(t *testing.T) *InKernelPlanner { return qwenQ4KHostPlanner(t, "host-budget-price") }, 32, 16, 32, false},
		{"qwen-q4k-long-chunked", func(t *testing.T) *InKernelPlanner { return qwenQ4KHostPlanner(t, "host-budget-price") }, 1024, 64, inKernelQwenQ4KPrefillChunkTokens, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := tc.p(t)
			session, retained, plan := p.hostRequestDemand(tc.promptTokens, tc.maxNew)
			base := hostBaseContextPlan(p, tc.promptTokens, tc.maxNew)
			if base.Total() <= 0 {
				t.Fatalf("base context plan total = %d, want positive", base.Total())
			}
			if !reflect.DeepEqual(plan, base) {
				t.Fatalf("hostRequestDemand plan = %+v, want the base context plan %+v (no runtime extras)", plan, base)
			}
			wantPanel := tc.panelTokens * hostPanelRowBytes(p)
			if got := p.hostPrefillPanelBytes(tc.promptTokens); got != wantPanel {
				t.Fatalf("host prefill panel = %d, want %d tokens x %d row bytes = %d", got, tc.panelTokens, hostPanelRowBytes(p), wantPanel)
			}
			if session != base.Total()+wantPanel {
				t.Fatalf("session = %d, want base plan %d + panel %d = %d", session, base.Total(), wantPanel, base.Total()+wantPanel)
			}
			reuseOn := p.tree != nil && inKernelPlannerPrefixReuseSupported(p.m, p.backend)
			if tc.reuse && !reuseOn {
				t.Fatal("precondition: this planner must have prefix reuse on")
			}
			if !reuseOn {
				if retained != 0 {
					t.Fatalf("reuse-off retained = %d, want 0", retained)
				}
				return
			}
			perCopy := hostRetainedPerCopy(p, tc.promptTokens, tc.maxNew)
			if perCopy <= 0 {
				t.Fatalf("per-copy retained estimate = %d, want positive", perCopy)
			}
			if retained != 3*perCopy {
				t.Fatalf("retained = %d, want 3 copies x %d = %d", retained, perCopy, 3*perCopy)
			}
		})
	}
}

// TestInKernelHostMemoryBudgetPrefillPanelClampsOnlyOnQwenQ4K pins hostPrefillPanelBytes: the
// panel is clamped to the Qwen Q4_K prefill chunk ONLY on the chunked Qwen Q4_K route; every
// other host route prefills the whole divergent suffix in one pass, so it prices the whole
// prompt (prompt x (3 x hidden + 2 x intermediate) x 4).
func TestInKernelHostMemoryBudgetPrefillPanelClampsOnlyOnQwenQ4K(t *testing.T) {
	t.Run("non-qwen-route-prices-whole-prompt", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			p    *InKernelPlanner
		}{
			{"dense-q8", bareHostPlanner()},
			{"dense-q4k", &InKernelPlanner{m: model.NewSynthetic(tinyConcurrencyConfig()), q4k: true}},
			{"hybrid-not-q4k", &InKernelPlanner{m: model.NewSynthetic(qwen35RuntimeExtraConfig())}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				p := tc.p
				// A configured chunk must not clamp a route that never walks the prompt in panels.
				p.qwenQ4KPrefillChunkTokens = 8
				if p.qwenQ4KPrefillChunkTarget() {
					t.Fatal("precondition: planner must be off the chunked Qwen Q4_K route")
				}
				for _, prompt := range []int{1, 100, inKernelQwenQ4KPrefillChunkTokens, 4096} {
					want := int64(prompt) * int64(3*p.m.Cfg.HiddenSize+2*p.m.Cfg.IntermediateSize) * 4
					if got := p.hostPrefillPanelBytes(prompt); got != want {
						t.Fatalf("prompt %d: panel = %d, want the whole prompt priced %d", prompt, got, want)
					}
				}
			})
		}
	})

	t.Run("qwen-q4k-default-chunk", func(t *testing.T) {
		p := qwenQ4KHostPlanner(t, "host-budget-panel")
		row := hostPanelRowBytes(p)
		for _, tc := range []struct{ prompt, panel int }{
			{1, 1},
			{32, 32},
			{inKernelQwenQ4KPrefillChunkTokens - 1, inKernelQwenQ4KPrefillChunkTokens - 1},
			{inKernelQwenQ4KPrefillChunkTokens, inKernelQwenQ4KPrefillChunkTokens},
			{1024, inKernelQwenQ4KPrefillChunkTokens},
			{4096, inKernelQwenQ4KPrefillChunkTokens},
		} {
			if got, want := p.hostPrefillPanelBytes(tc.prompt), int64(tc.panel)*row; got != want {
				t.Fatalf("prompt %d: panel = %d, want min(prompt, chunk)=%d x %d = %d", tc.prompt, got, tc.panel, row, want)
			}
		}
	})

	t.Run("qwen-q4k-configured-chunk", func(t *testing.T) {
		p := qwenQ4KHostPlanner(t, "host-budget-panel")
		p.qwenQ4KPrefillChunkTokens = 8
		row := hostPanelRowBytes(p)
		if got, want := p.hostPrefillPanelBytes(100), 8*row; got != want {
			t.Fatalf("prompt 100 with chunk 8: panel = %d, want %d", got, want)
		}
		if got, want := p.hostPrefillPanelBytes(5), 5*row; got != want {
			t.Fatalf("prompt 5 with chunk 8: panel = %d, want %d", got, want)
		}
	})

	t.Run("degenerate-inputs-price-zero", func(t *testing.T) {
		p := bareHostPlanner()
		for _, prompt := range []int{0, -1} {
			if got := p.hostPrefillPanelBytes(prompt); got != 0 {
				t.Fatalf("prompt %d: panel = %d, want 0", prompt, got)
			}
		}
		if got := (&InKernelPlanner{}).hostPrefillPanelBytes(64); got != 0 {
			t.Fatalf("nil-model panel = %d, want 0", got)
		}
		var nilPlanner *InKernelPlanner
		if got := nilPlanner.hostPrefillPanelBytes(64); got != 0 {
			t.Fatalf("nil-planner panel = %d, want 0", got)
		}
	})
}

// TestInKernelHostMemoryBudgetUnarmedFailsOpen pins every fail-open path: an unarmed or
// disarmed planner, a device-backed planner, a planner with no model, and an unknown usage
// probe all admit unchanged (same ctx, nil error, no skip flag, a callable release).
func TestInKernelHostMemoryBudgetUnarmedFailsOpen(t *testing.T) {
	type parentKey struct{}
	parent := context.WithValue(context.Background(), parentKey{}, "parent")
	const ceiling = int64(1 << 30)

	assertOpen := func(t *testing.T, p *InKernelPlanner, label string) {
		t.Helper()
		ctx, release, err := p.admitHostMemory(parent, 64, 16)
		if err != nil {
			t.Fatalf("%s: admitHostMemory error = %v, want nil (fail open)", label, err)
		}
		if release == nil {
			t.Fatalf("%s: release func is nil; Complete defers it unconditionally", label)
		}
		release()
		release()
		if ctx != parent {
			t.Fatalf("%s: fail-open admission must return the caller's ctx unchanged", label)
		}
		if inKernelSkipPrefixAdmission(ctx) {
			t.Fatalf("%s: fail-open admission must not set the skip-prefix flag", label)
		}
		if got := hostReserved(p); got != 0 {
			t.Fatalf("%s: fail-open admission reserved %d bytes, want 0", label, got)
		}
	}

	// Positive control: this probe (no headroom at all) makes an ARMED host planner refuse,
	// so each fail-open case below is meaningful.
	full := constHostUsage(ceiling, true)
	armed := bareHostPlanner()
	armed.SetHostMemoryBudget(ceiling, full.used)
	if _, _, err := armed.admitHostMemory(parent, 64, 16); err == nil {
		t.Fatal("control: an armed host planner with zero headroom must refuse")
	}

	t.Run("never-armed", func(t *testing.T) {
		assertOpen(t, bareHostPlanner(), "never armed")
	})
	t.Run("zero-ceiling-disarms", func(t *testing.T) {
		p := bareHostPlanner()
		p.SetHostMemoryBudget(ceiling, full.used)
		p.SetHostMemoryBudget(0, full.used)
		assertOpen(t, p, "SetHostMemoryBudget(0, fn)")
	})
	t.Run("negative-ceiling-disarms", func(t *testing.T) {
		p := bareHostPlanner()
		p.SetHostMemoryBudget(ceiling, full.used)
		p.SetHostMemoryBudget(-1, full.used)
		assertOpen(t, p, "SetHostMemoryBudget(-1, fn)")
	})
	t.Run("nil-probe-disarms", func(t *testing.T) {
		p := bareHostPlanner()
		p.SetHostMemoryBudget(ceiling, full.used)
		p.SetHostMemoryBudget(ceiling, nil)
		assertOpen(t, p, "SetHostMemoryBudget(x, nil)")
	})
	t.Run("device-backed-planner", func(t *testing.T) {
		probe := constHostUsage(ceiling, true)
		p := bareHostPlanner()
		p.backend = compute.Default()
		p.SetHostMemoryBudget(ceiling, probe.used)
		assertOpen(t, p, "device-backed planner")
		if probe.callCount() != 0 {
			t.Fatalf("device-backed planner consulted the host probe %d times, want 0 (the device arm owns it)", probe.callCount())
		}
	})
	t.Run("nil-model", func(t *testing.T) {
		probe := constHostUsage(ceiling, true)
		p := &InKernelPlanner{modelID: "host-budget-nil-model"}
		p.SetHostMemoryBudget(ceiling, probe.used)
		assertOpen(t, p, "nil model")
	})
	t.Run("unknown-usage-probe", func(t *testing.T) {
		probe := constHostUsage(ceiling, false)
		p := hostSessionReusePlanner(t)
		p.SetHostMemoryBudget(ceiling, probe.used)
		assertOpen(t, p, "used probe known=false")
		if probe.callCount() == 0 {
			t.Fatal("armed host planner never consulted the usage probe")
		}
	})
}

// TestInKernelHostMemoryBudgetUnknownProbeMidLadderFailsOpen pins that an unknown usage probe
// at ANY probe point of the ladder (the garbage-reclaim re-probe, or a re-probe inside the
// prefix-cache trim) fails open: nil error, the caller's ctx unchanged, and no reservation.
func TestInKernelHostMemoryBudgetUnknownProbeMidLadderFailsOpen(t *testing.T) {
	const promptTokens, maxNew = 64, 16
	const ceiling = int64(1 << 30)
	type parentKey struct{}
	parent := context.WithValue(context.Background(), parentKey{}, "parent")

	for _, tc := range []struct {
		name      string
		unknownAt int  // the 1-based probe call that reports known=false
		trimmed   bool // whether the trim loop ran before the unknown probe
	}{
		{"unknown-after-garbage-reclaim", 2, false},
		{"unknown-during-prefix-cache-trim", 3, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := hostSessionReusePlanner(t)
			cached := seedRadixTokens(t, p, 4, 32)
			budget := p.tree.Stats().MaxTokens
			probe := &hostUsageProbe{at: func(call int) (int64, bool) {
				if call >= tc.unknownAt {
					return 0, false
				}
				return ceiling, true // no headroom until the probe goes unknown
			}}
			p.SetHostMemoryBudget(ceiling, probe.used)
			ctx, release, err := p.admitHostMemory(parent, promptTokens, maxNew)
			if err != nil {
				t.Fatalf("unknown probe at call %d: error = %v, want nil (fail open)", tc.unknownAt, err)
			}
			if release == nil {
				t.Fatal("fail-open admission must return a callable release")
			}
			release()
			if ctx != parent {
				t.Fatal("fail-open admission must return the caller's ctx unchanged")
			}
			if inKernelSkipPrefixAdmission(ctx) {
				t.Fatal("fail-open admission must not set the skip-prefix flag")
			}
			if got := hostReserved(p); got != 0 {
				t.Fatalf("fail-open admission reserved %d bytes, want 0", got)
			}
			if got := probe.callCount(); got != tc.unknownAt {
				t.Fatalf("usage probe called %d times, want %d (stop at the first unknown probe)", got, tc.unknownAt)
			}
			if got := p.tree.Stats().MaxTokens; got != budget {
				t.Fatalf("prefix-cache budget = %d, want the configured %d restored", got, budget)
			}
			if !tc.trimmed {
				if got := radixTokens(p); got != cached {
					t.Fatalf("prefix cache trimmed to %d tokens before the trim rung (had %d)", got, cached)
				}
			}
		})
	}
}

// TestInKernelHostMemoryBudgetSkipsPrefixAdmissionWhenOnlySessionFits pins ladder rung c: a
// turn whose working set fits but whose retained prefix-cache copies do not is served
// uncached (skip flag set, only the session reserved) WITHOUT trimming the prefix cache (the
// optional retained copies never trigger eviction), while ample headroom admits with caching.
// It also pins the retained-copy pricing the rung depends on.
func TestInKernelHostMemoryBudgetSkipsPrefixAdmissionWhenOnlySessionFits(t *testing.T) {
	p := hostSessionReusePlanner(t)
	const promptTokens, maxNew = 64, 16
	session, retained := hostDemand(t, p, promptTokens, maxNew)

	// Retained = 3 copies of (KV(prompt+maxNew) + fixed session state) when reuse is on.
	perCopy := hostRetainedPerCopy(p, promptTokens, maxNew)
	if perCopy <= 0 {
		t.Fatalf("per-copy retained estimate = %d, want positive", perCopy)
	}
	if retained != 3*perCopy {
		t.Fatalf("retained = %d, want 3 copies x %d = %d", retained, perCopy, 3*perCopy)
	}
	// Session = base context plan + the host prefill-panel live peak (the whole prompt: this
	// dense planner is off the chunked Qwen Q4_K route).
	plan := hostBaseContextPlan(p, promptTokens, maxNew)
	panel := p.hostPrefillPanelBytes(promptTokens)
	wantPanel := int64(promptTokens) * int64(3*p.m.Cfg.HiddenSize+2*p.m.Cfg.IntermediateSize) * 4
	if panel != wantPanel {
		t.Fatalf("host prefill panel bytes = %d, want %d (prompt x (3H+2I) x f32)", panel, wantPanel)
	}
	if session != plan.Total()+panel {
		t.Fatalf("session = %d, want plan %d + panel %d", session, plan.Total(), panel)
	}
	// With reuse off the same turn retains nothing.
	if _, r := hostDemand(t, bareHostPlanner(), promptTokens, maxNew); r != 0 {
		t.Fatalf("reuse-off retained = %d, want 0", r)
	}

	// Seed the prefix cache so any eviction by the admission is observable.
	cached := seedRadixTokens(t, p, 4, 32)

	type parentKey struct{}
	parent := context.WithValue(context.Background(), parentKey{}, "parent")
	const ceiling = int64(1 << 30)
	maxTokensBefore := p.tree.Stats().MaxTokens

	var probe *hostUsageProbe
	admitWith := func(t *testing.T, avail int64) (context.Context, func(), error) {
		t.Helper()
		probe = constHostUsage(ceiling-avail, true)
		p.SetHostMemoryBudget(ceiling, probe.used)
		return p.admitHostMemory(parent, promptTokens, maxNew)
	}

	cases := []struct {
		name  string
		avail int64
	}{
		{"midway", session + retained/2},
		{"exactly-session", session},
		{"one-byte-short-of-caching", session + retained - 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, release, err := admitWith(t, tc.avail)
			if err != nil {
				t.Fatalf("avail %d (session %d retained %d): error = %v, want admitted uncached", tc.avail, session, retained, err)
			}
			if !inKernelSkipPrefixAdmission(ctx) {
				t.Fatalf("avail %d < session+retained %d: skip-prefix flag not set", tc.avail, session+retained)
			}
			if ctx.Value(parentKey{}) != "parent" {
				t.Fatal("skip-flag ctx must derive from the caller's ctx")
			}
			if got := hostReserved(p); got != session {
				t.Fatalf("uncached admission reserved %d, want the session alone %d", got, session)
			}
			if got := probe.callCount(); got != 2 {
				t.Fatalf("rung-c admission probed usage %d times, want 2 (initial + one garbage-reclaim re-probe, no trim)", got)
			}
			if got := radixTokens(p); got != cached {
				t.Fatalf("rung-c admission trimmed the prefix cache to %d tokens, want it unchanged at %d (optional copies never evict)", got, cached)
			}
			release()
			if got := hostReserved(p); got != 0 {
				t.Fatalf("reservation after release = %d, want 0", got)
			}
			if got := p.tree.Stats().MaxTokens; got != maxTokensBefore {
				t.Fatalf("prefix-cache budget changed by admission: %d, want %d", got, maxTokensBefore)
			}
		})
	}

	for _, avail := range []int64{session + retained, ceiling} {
		ctx, release, err := admitWith(t, avail)
		if err != nil {
			t.Fatalf("avail %d >= session+retained %d: error = %v", avail, session+retained, err)
		}
		if inKernelSkipPrefixAdmission(ctx) {
			t.Fatalf("avail %d >= session+retained %d: skip-prefix flag set, want cached admission", avail, session+retained)
		}
		if got := hostReserved(p); got != session+retained {
			t.Fatalf("cached admission reserved %d, want session+retained %d", got, session+retained)
		}
		if got := probe.callCount(); got != 1 {
			t.Fatalf("rung-a admission probed usage %d times, want exactly 1", got)
		}
		release()
		if got := hostReserved(p); got != 0 {
			t.Fatalf("reservation after release = %d, want 0", got)
		}
		if got := radixTokens(p); got != cached {
			t.Fatalf("cached admission trimmed the prefix cache to %d tokens, want %d", got, cached)
		}
	}

	// One byte below the session with persistent pressure: nothing fits, typed host refusal.
	_, _, err := admitWith(t, session-1)
	var capErr *InKernelCapacityError
	if !errors.As(err, &capErr) || capErr.Scope != compute.MemoryScopeHost || capErr.Site != "host-memory-precheck" {
		t.Fatalf("avail session-1: error = %T (%v), want host-memory-precheck capacity error", err, err)
	}
	if capErr.Want != session || capErr.Avail != session-1 {
		t.Fatalf("refusal sizing want %d avail %d, want %d/%d", capErr.Want, capErr.Avail, session, session-1)
	}
	if got := hostReserved(p); got != 0 {
		t.Fatalf("declined turn reserved %d bytes, want 0", got)
	}
	if got := p.tree.Stats().MaxTokens; got != maxTokensBefore {
		t.Fatalf("prefix-cache budget after a refused relief = %d, want the configured %d restored", got, maxTokensBefore)
	}
}

// TestInKernelHostMemoryBudgetReservesInFlight pins that admitted turns reserve their bytes
// against the ceiling until released, that release returns them exactly once, and that a
// refusal reports the headroom left after the in-flight reservations.
func TestInKernelHostMemoryBudgetReservesInFlight(t *testing.T) {
	const promptTokens, maxNew = 32, 8
	const ceiling = int64(1 << 30)
	ctx := context.Background()

	t.Run("second-refused-until-first-released", func(t *testing.T) {
		p := bareHostPlanner()
		session, retained := hostDemand(t, p, promptTokens, maxNew)
		if retained != 0 {
			t.Fatalf("reuse-off planner retained = %d, want 0", retained)
		}
		free := session + session/2 // one turn fits, two do not
		p.SetHostMemoryBudget(ceiling, constHostUsage(ceiling-free, true).used)

		_, release1, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		if err != nil {
			t.Fatalf("first admission: %v", err)
		}
		if got := hostReserved(p); got != session {
			t.Fatalf("reservation after first admission = %d, want %d", got, session)
		}
		_, release2, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		var capErr *InKernelCapacityError
		if !errors.As(err, &capErr) {
			t.Fatalf("second admission error = %T (%v), want *InKernelCapacityError (first turn's bytes are in flight)", err, err)
		}
		if capErr.Scope != compute.MemoryScopeHost || capErr.Site != "host-memory-precheck" {
			t.Fatalf("second refusal scope/site = %s/%q, want host/host-memory-precheck", capErr.Scope, capErr.Site)
		}
		if capErr.Want != session || capErr.Avail != free-session {
			t.Fatalf("second refusal want %d avail %d, want %d/%d (headroom net of the first reservation)",
				capErr.Want, capErr.Avail, session, free-session)
		}
		if release2 == nil {
			t.Fatal("refused admission must still return a callable release")
		}
		release2() // a refused turn reserved nothing; releasing it must not free the first turn's bytes
		if got := hostReserved(p); got != session {
			t.Fatalf("reservation after releasing the refused turn = %d, want %d", got, session)
		}

		release1()
		if got := hostReserved(p); got != 0 {
			t.Fatalf("reservation after first release = %d, want 0", got)
		}
		_, release3, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		if err != nil {
			t.Fatalf("third admission after release: %v", err)
		}
		release1() // idempotent: must not return the third turn's bytes
		if got := hostReserved(p); got != session {
			t.Fatalf("a second release of the first turn freed another turn's reservation: reserved %d, want %d", got, session)
		}
		release3()
		release3()
		if got := hostReserved(p); got != 0 {
			t.Fatalf("reservation after all releases = %d, want 0", got)
		}
	})

	t.Run("double-release-with-peer-in-flight", func(t *testing.T) {
		p := bareHostPlanner()
		session, _ := hostDemand(t, p, promptTokens, maxNew)
		free := 2*session + session/2 // two turns fit, three do not
		p.SetHostMemoryBudget(ceiling, constHostUsage(ceiling-free, true).used)

		_, releaseA, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		if err != nil {
			t.Fatalf("admission A: %v", err)
		}
		_, releaseB, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		if err != nil {
			t.Fatalf("admission B: %v", err)
		}
		if got := hostReserved(p); got != 2*session {
			t.Fatalf("reservation with A and B in flight = %d, want %d", got, 2*session)
		}
		if _, _, err := p.admitHostMemory(ctx, promptTokens, maxNew); err == nil {
			t.Fatal("admission C must be refused while A and B hold their reservations")
		}
		releaseA()
		releaseA()
		if got := hostReserved(p); got != session {
			t.Fatalf("double release of A left reservation %d, want B's %d", got, session)
		}
		_, releaseD, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		if err != nil {
			t.Fatalf("admission D after A released: %v", err)
		}
		releaseB()
		releaseD()
		if got := hostReserved(p); got != 0 {
			t.Fatalf("reservation after all releases = %d, want 0", got)
		}
	})

	t.Run("cached-reservation-shrinks-peer-headroom", func(t *testing.T) {
		p := hostSessionReusePlanner(t)
		session, retained := hostDemand(t, p, promptTokens, maxNew)
		if retained <= 0 {
			t.Fatalf("reuse-on planner retained = %d, want positive", retained)
		}
		free := session + retained + session // A fits cached; then only B's session fits
		p.SetHostMemoryBudget(ceiling, constHostUsage(ceiling-free, true).used)

		ctxA, releaseA, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		if err != nil {
			t.Fatalf("admission A: %v", err)
		}
		if inKernelSkipPrefixAdmission(ctxA) {
			t.Fatal("A fits session+retained: must admit with caching")
		}
		if got := hostReserved(p); got != session+retained {
			t.Fatalf("reservation after cached A = %d, want %d", got, session+retained)
		}
		ctxB, releaseB, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		if err != nil {
			t.Fatalf("admission B (session fits net of A's reservation): %v", err)
		}
		if !inKernelSkipPrefixAdmission(ctxB) {
			t.Fatal("B's retained copies do not fit net of A's reservation: skip flag must be set")
		}
		if got := hostReserved(p); got != 2*session+retained {
			t.Fatalf("reservation with A cached and B uncached = %d, want %d", got, 2*session+retained)
		}
		_, _, err = p.admitHostMemory(ctx, promptTokens, maxNew)
		var capErr *InKernelCapacityError
		if !errors.As(err, &capErr) || capErr.Want != session || capErr.Avail != 0 {
			t.Fatalf("admission C with no headroom left = %T (%v), want host refusal want %d avail 0", err, err, session)
		}
		releaseA()
		releaseB()
		if got := hostReserved(p); got != 0 {
			t.Fatalf("reservation after all releases = %d, want 0", got)
		}
	})
}

// TestInKernelHostMemoryBudgetRelievesBeforeDeclining pins ladder rungs b and d: before
// declining, the arm collects garbage and re-probes once, then (only when the session itself
// does not fit) trims the prefix cache step by step, re-probing after each step, admits the
// session WITHOUT caching once it fits, and restores the prefix cache's configured token budget.
func TestInKernelHostMemoryBudgetRelievesBeforeDeclining(t *testing.T) {
	const promptTokens, maxNew = 64, 16
	const ceiling = int64(1 << 30)
	ctx := context.Background()

	t.Run("garbage-reclaim-suffices", func(t *testing.T) {
		p := hostSessionReusePlanner(t)
		session, retained := hostDemand(t, p, promptTokens, maxNew)
		cached := seedRadixTokens(t, p, 4, 32)
		budget := p.tree.Stats().MaxTokens
		// High usage on the first probe only: the reclaim re-probe sees the space freed.
		probe := &hostUsageProbe{at: func(call int) (int64, bool) {
			if call == 1 {
				return ceiling, true
			}
			return 0, true
		}}
		p.SetHostMemoryBudget(ceiling, probe.used)
		rctx, release, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		if err != nil {
			t.Fatalf("admission after relief: %v (relief must re-probe before declining)", err)
		}
		defer release()
		if got := probe.callCount(); got != 2 {
			t.Fatalf("usage probe called %d times, want 2 (initial + exactly one garbage-reclaim re-probe)", got)
		}
		if inKernelSkipPrefixAdmission(rctx) {
			t.Fatal("garbage reclaim freed room for the retained copies; the turn must be admitted with caching")
		}
		if got := hostReserved(p); got != session+retained {
			t.Fatalf("reservation = %d, want session+retained %d", got, session+retained)
		}
		st := p.tree.Stats()
		if st.MaxTokens != budget {
			t.Fatalf("prefix-cache budget after relief = %d, want the configured %d", st.MaxTokens, budget)
		}
		if st.Tokens != cached {
			t.Fatalf("prefix cache trimmed to %d tokens although garbage reclaim alone made room (had %d)", st.Tokens, cached)
		}
	})

	t.Run("garbage-reclaim-fits-session-only-skips-without-trim", func(t *testing.T) {
		p := hostSessionReusePlanner(t)
		session, retained := hostDemand(t, p, promptTokens, maxNew)
		cached := seedRadixTokens(t, p, 4, 32)
		budget := p.tree.Stats().MaxTokens
		// Before reclaim not even the session fits; after it the session fits but the copies do not.
		probe := &hostUsageProbe{at: func(call int) (int64, bool) {
			if call == 1 {
				return ceiling - (session - 1), true
			}
			return ceiling - (session + retained/2), true
		}}
		p.SetHostMemoryBudget(ceiling, probe.used)
		rctx, release, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		if err != nil {
			t.Fatalf("session fits after garbage reclaim: error = %v, want admitted uncached", err)
		}
		defer release()
		if !inKernelSkipPrefixAdmission(rctx) {
			t.Fatal("only the session fits after reclaim: skip-prefix flag must be set")
		}
		if got := hostReserved(p); got != session {
			t.Fatalf("reservation = %d, want the session alone %d", got, session)
		}
		if got := probe.callCount(); got != 2 {
			t.Fatalf("usage probe called %d times, want 2 (no trim once the session fits)", got)
		}
		if got := radixTokens(p); got != cached {
			t.Fatalf("prefix cache trimmed to %d tokens although the session fit after reclaim (had %d)", got, cached)
		}
		if got := p.tree.Stats().MaxTokens; got != budget {
			t.Fatalf("prefix-cache budget = %d, want the configured %d", got, budget)
		}
	})

	t.Run("prefix-cache-trim-then-restore-budget", func(t *testing.T) {
		p := hostSessionReusePlanner(t)
		session, retained := hostDemand(t, p, promptTokens, maxNew)
		cached := seedRadixTokens(t, p, 4, 32)
		budget := p.tree.Stats().MaxTokens
		if budget <= 0 {
			t.Fatalf("configured prefix-cache budget = %d, want the positive context-window bound", budget)
		}
		// Pressure persists through the garbage reclaim and lifts (to session-only room) once
		// the cache sheds its first step.
		probe := &hostUsageProbe{at: func(call int) (int64, bool) {
			if call <= 2 {
				return ceiling, true
			}
			return ceiling - (session + retained/2), true
		}}
		p.SetHostMemoryBudget(ceiling, probe.used)
		rctx, release, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		if err != nil {
			t.Fatalf("admission after prefix-cache relief: %v", err)
		}
		defer release()
		if !inKernelSkipPrefixAdmission(rctx) {
			t.Fatal("the session fit only after trimming the prefix cache: skip-prefix flag must be set")
		}
		if got := hostReserved(p); got != session {
			t.Fatalf("reservation = %d, want the session alone %d", got, session)
		}
		if got := probe.callCount(); got != 3 {
			t.Fatalf("usage probe called %d times, want 3 (initial, reclaim, and the first trim step that fit)", got)
		}
		st := p.tree.Stats()
		if st.Tokens > cached/2 {
			t.Fatalf("prefix cache holds %d tokens after one halving step, want at most %d (half of %d)", st.Tokens, cached/2, cached)
		}
		if st.Tokens <= 0 {
			t.Fatal("trim must stop at the first step that fits the session, not empty the cache")
		}
		if st.MaxTokens != budget {
			t.Fatalf("prefix-cache budget after relief = %d, want the configured %d restored", st.MaxTokens, budget)
		}
	})

	t.Run("prefix-cache-trim-with-ample-room-still-skips-admission", func(t *testing.T) {
		p := hostSessionReusePlanner(t)
		session, _ := hostDemand(t, p, promptTokens, maxNew)
		seedRadixTokens(t, p, 4, 32)
		budget := p.tree.Stats().MaxTokens
		// The first trim step frees everything: the session (and more) now fits. Rung d still
		// admits WITHOUT caching: the turn only fit by evicting the cache.
		probe := &hostUsageProbe{at: func(call int) (int64, bool) {
			if call <= 2 {
				return ceiling, true
			}
			return 0, true
		}}
		p.SetHostMemoryBudget(ceiling, probe.used)
		rctx, release, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		if err != nil {
			t.Fatalf("admission after prefix-cache relief: %v", err)
		}
		defer release()
		if !inKernelSkipPrefixAdmission(rctx) {
			t.Fatal("rung d (session fit only after trimming the prefix cache) must admit with the skip-prefix flag")
		}
		if got := hostReserved(p); got != session {
			t.Fatalf("rung-d reservation = %d, want the session alone %d", got, session)
		}
		if got := p.tree.Stats().MaxTokens; got != budget {
			t.Fatalf("prefix-cache budget after relief = %d, want the configured %d restored", got, budget)
		}
	})

	t.Run("no-relief-helps-declines-and-restores-budget", func(t *testing.T) {
		p := hostSessionReusePlanner(t)
		session, _ := hostDemand(t, p, promptTokens, maxNew)
		cached := seedRadixTokens(t, p, 4, 32)
		budget := p.tree.Stats().MaxTokens
		// Record the cached token count each probe observes (the trim re-probes outside p.mu).
		var seen []int
		probe := &hostUsageProbe{}
		probe.at = func(int) (int64, bool) {
			seen = append(seen, radixTokens(p))
			return ceiling, true
		}
		p.SetHostMemoryBudget(ceiling, probe.used)
		_, _, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		var capErr *InKernelCapacityError
		if !errors.As(err, &capErr) || capErr.Scope != compute.MemoryScopeHost || capErr.Site != "host-memory-precheck" {
			t.Fatalf("persistent pressure: error = %T (%v), want host-memory-precheck capacity error", err, err)
		}
		if capErr.Want != session || capErr.Avail != 0 {
			t.Fatalf("refusal sizing want %d avail %d, want %d/0", capErr.Want, capErr.Avail, session)
		}
		// initial + reclaim + three halvings + the final empty step.
		if got := probe.callCount(); got != 2+inKernelHostReliefHalvings+1 {
			t.Fatalf("usage probe called %d times (tokens seen %v), want %d: initial, reclaim, %d halvings, then empty",
				got, seen, 2+inKernelHostReliefHalvings+1, inKernelHostReliefHalvings)
		}
		if seen[0] != cached || seen[1] != cached {
			t.Fatalf("tokens seen %v: the cache must not be trimmed before the session is known not to fit after reclaim", seen)
		}
		if seen[2] > cached/2 {
			t.Fatalf("tokens seen %v: the first trim step must halve the %d cached tokens", seen, cached)
		}
		for i := 3; i < len(seen); i++ {
			if seen[i] > seen[i-1] {
				t.Fatalf("tokens seen %v: the trim must never grow the cache", seen)
			}
		}
		if last := seen[len(seen)-1]; last != 0 {
			t.Fatalf("tokens seen %v: the final relief step must empty the prefix cache", seen)
		}
		if got := p.tree.Stats().MaxTokens; got != budget {
			t.Fatalf("prefix-cache budget after a failed relief = %d, want the configured %d restored", got, budget)
		}
		if got := hostReserved(p); got != 0 {
			t.Fatalf("declined turn reserved %d bytes, want 0", got)
		}
	})

	t.Run("over-committed-probe-reports-zero-avail", func(t *testing.T) {
		p := bareHostPlanner()
		session, _ := hostDemand(t, p, promptTokens, maxNew)
		p.SetHostMemoryBudget(ceiling, constHostUsage(ceiling+(1<<20), true).used)
		_, _, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		var capErr *InKernelCapacityError
		if !errors.As(err, &capErr) {
			t.Fatalf("usage above ceiling: error = %T (%v), want *InKernelCapacityError", err, err)
		}
		if capErr.Want != session || capErr.Avail != 0 {
			t.Fatalf("usage above ceiling: want %d avail %d, want %d/0 (avail clamps at zero)", capErr.Want, capErr.Avail, session)
		}
	})
}

// TestInKernelCapacityErrorMessageNamesScope pins the refusal wording: a host-scope refusal
// names host memory (the host-session seam has no GPU to blame), everything else names the GPU.
func TestInKernelCapacityErrorMessageNamesScope(t *testing.T) {
	for _, tc := range []struct {
		name   string
		scope  compute.MemoryScope
		prefix string
	}{
		{"host", compute.MemoryScopeHost, "in-kernel host-memory capacity precheck refused request ("},
		{"device", compute.MemoryScopeDevice, "in-kernel GPU capacity precheck refused request ("},
		{"unset-defaults-to-device", "", "in-kernel GPU capacity precheck refused request ("},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := &InKernelCapacityError{Want: 123456, Avail: 789, Class: compute.MemoryKVCache, Scope: tc.scope, Site: "x"}
			msg := e.Error()
			if !strings.HasPrefix(msg, tc.prefix) {
				t.Fatalf("Error() = %q, want prefix %q", msg, tc.prefix)
			}
			if !strings.Contains(msg, strconv.FormatInt(e.Want, 10)) || !strings.Contains(msg, strconv.FormatInt(e.Avail, 10)) {
				t.Fatalf("Error() = %q, want the want/avail byte figures", msg)
			}
			if tc.scope == compute.MemoryScopeHost && strings.Contains(msg, "GPU") {
				t.Fatalf("host-scope Error() = %q must not name the GPU", msg)
			}
		})
	}
}

// TestAdmitDeviceOOMRetryIsBounded pins every branch of the bounded device OOM retry gate.
func TestAdmitDeviceOOMRetryIsBounded(t *testing.T) {
	oom := &InKernelOOMError{Bytes: 1 << 20, Class: compute.MemoryKVCache, Site: "kv-grow"}
	roomy := func() *pressureTrimBackend {
		return &pressureTrimBackend{Backend: compute.Default(), total: 1 << 30, free: 1 << 30, trimFree: 1 << 30}
	}
	cramped := func() *pressureTrimBackend {
		return &pressureTrimBackend{Backend: compute.Default(), total: 1 << 20, free: 1 << 20}
	}
	planner := func(be compute.Backend) *InKernelPlanner {
		return &InKernelPlanner{m: model.NewSynthetic(tinyConcurrencyConfig()), backend: be, modelID: "oom-retry-gate"}
	}
	noRetryBooked := func(t *testing.T, p *InKernelPlanner) {
		t.Helper()
		if rows := p.InKernelOOMRetryStats().Rows; len(rows) != 0 {
			t.Fatalf("retry gate booked retry stats %+v on a branch that never attempted a retry", rows)
		}
	}

	t.Run("cancelled-context", func(t *testing.T) {
		be := roomy()
		p := planner(be)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		retry, out := p.admitDeviceOOMRetry(ctx, oom, 10, 5)
		if retry {
			t.Fatal("a cancelled request must not retry")
		}
		if !errors.Is(out, context.Canceled) {
			t.Fatalf("cancelled request surfaced %v, want context.Canceled", out)
		}
		if be.recycle != 0 || be.trim != 0 || len(be.trimLarge) != 0 {
			t.Fatalf("cancelled request trimmed pools: recycle %d trim %d trimLarge %v", be.recycle, be.trim, be.trimLarge)
		}
		noRetryBooked(t, p)
	})

	t.Run("coalesced-past-decode-pass", func(t *testing.T) {
		be := roomy()
		p := planner(be)
		req := &inKernelCoalesceRequest{}
		req.decodePass.Add(1)
		ctx := context.WithValue(context.Background(), inKernelCoalesceContextKey{}, req)
		retry, out := p.admitDeviceOOMRetry(ctx, oom, 10, 5)
		if retry {
			t.Fatal("a coalesced lane that crossed the shared decode pass must not retry")
		}
		if out != error(oom) {
			t.Fatalf("coalesced no-retry surfaced %v, want the original error", out)
		}
		if be.recycle != 0 || be.trim != 0 || len(be.trimLarge) != 0 {
			t.Fatalf("coalesced no-retry trimmed pools: recycle %d trim %d trimLarge %v", be.recycle, be.trim, be.trimLarge)
		}
		noRetryBooked(t, p)
	})

	t.Run("coalesced-before-decode-pass-may-retry", func(t *testing.T) {
		be := roomy()
		p := planner(be)
		req := &inKernelCoalesceRequest{}
		ctx := context.WithValue(context.Background(), inKernelCoalesceContextKey{}, req)
		retry, out := p.admitDeviceOOMRetry(ctx, oom, 10, 5)
		if !retry || out != nil {
			t.Fatalf("coalesced lane before any decode pass = (%v, %v), want (true, nil)", retry, out)
		}
	})

	t.Run("non-oom-error", func(t *testing.T) {
		be := roomy()
		p := planner(be)
		plain := errors.New("ordinary upstream error")
		retry, out := p.admitDeviceOOMRetry(context.Background(), plain, 10, 5)
		if retry {
			t.Fatal("a non-OOM failure must not retry")
		}
		if out != plain {
			t.Fatalf("non-OOM no-retry surfaced %v, want the original error", out)
		}
		noRetryBooked(t, p)
	})

	t.Run("backend-without-trim-hooks", func(t *testing.T) {
		p := planner(compute.Default())
		retry, out := p.admitDeviceOOMRetry(context.Background(), oom, 10, 5)
		if retry {
			t.Fatal("a backend with no trim hooks must not retry")
		}
		if out != error(oom) {
			t.Fatalf("no-hooks no-retry surfaced %v, want the original OOM error", out)
		}
		noRetryBooked(t, p)
	})

	t.Run("host-session-planner", func(t *testing.T) {
		p := planner(nil)
		retry, out := p.admitDeviceOOMRetry(context.Background(), oom, 10, 5)
		if retry || out != error(oom) {
			t.Fatalf("host-session planner = (%v, %v), want (false, original OOM)", retry, out)
		}
		noRetryBooked(t, p)
	})

	t.Run("trimmed-but-still-oversize", func(t *testing.T) {
		be := cramped()
		p := planner(be)
		if err := p.refuseOversizeRequest(100_000, 256); err == nil {
			t.Fatal("control: cramped backend must refuse the oversize request")
		}
		be.recycle, be.trim, be.trimLarge = 0, 0, nil
		retry, out := p.admitDeviceOOMRetry(context.Background(), oom, 100_000, 256)
		if retry {
			t.Fatal("a retry that re-prices over the recovered capacity must not run")
		}
		var capErr *InKernelCapacityError
		if !errors.As(out, &capErr) {
			t.Fatalf("oversize retry surfaced %T (%v), want *InKernelCapacityError", out, out)
		}
		if capErr.Site != "capacity-precheck" || capErr.Scope != compute.MemoryScopeDevice {
			t.Fatalf("oversize retry refusal site/scope = %q/%s, want capacity-precheck/device", capErr.Site, capErr.Scope)
		}
		if be.trim == 0 {
			t.Fatal("the gate must trim idle pools before re-pricing the retry")
		}
		// No retry ran, so the retry metric (attempted retries only) books nothing.
		noRetryBooked(t, p)
	})

	t.Run("trimmed-and-fits", func(t *testing.T) {
		be := roomy()
		p := planner(be)
		retry, out := p.admitDeviceOOMRetry(context.Background(), oom, 10, 5)
		if !retry || out != nil {
			t.Fatalf("trimmed OOM that re-prices under capacity = (%v, %v), want (true, nil)", retry, out)
		}
		if be.trim == 0 {
			t.Fatal("an admitted retry must have trimmed idle pools first")
		}
		// The caller books the retry's outcome after it runs; the gate must not pre-book it.
		noRetryBooked(t, p)
	})
}
