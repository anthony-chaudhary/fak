package model

import (
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// expert_ring_shared_test.go — the R7 witnesses for #5618 (epic #5606, and epic #5243's L2 lever).
//
// The rung's claim is that routed-expert residency belongs to the (model, device) pair rather than
// to a conversation, so B agents' activated sets land on ONE bounded object and a page-in charged
// once is served B times. Three properties carry it, and the issue names all three:
//
//	coalescing is MEASURED, not assumed — the ledger reports distinct agents served per page-in, and
//	  one shared ring pages in strictly less than B private rings of the same budget running the
//	  same routing;
//	bytes per agent-token FALL as B grows at a fixed budget, while every agent's output stays
//	  byte-identical to a solo run — coalescing changes who pays for a page-in, never what is
//	  computed;
//	nothing else is shared — weights are model-constant bytes; KV, conversation state and halW stay
//	  per-session. The issue calls this "the load-bearing safety property of the rung", so it gets
//	  its own test rather than a comment, and the enforcement is a refusal at Attach rather than an
//	  assertion in prose.
//
// Plus the concurrency witness the sharing implies: B agents decoding at once against one ring
// still bound the footprint and still produce the solo answer.

// sharedRingBackend is ONE device shared by every agent, which is what makes attaching legal at all
// (a device handle is valid only on the device that produced it). It serializes its own forwarding
// so a concurrency witness measures the RING's discipline rather than a test double's bookkeeping.
type sharedRingBackend struct {
	compute.Backend
	mu      sync.Mutex
	uploads int
	frees   int
}

func (b *sharedRingBackend) Name() string                     { return "cuda-test-shared" }
func (b *sharedRingBackend) SupportsRoutedExpertKQuant() bool { return true }
func (b *sharedRingBackend) Caps() compute.Caps {
	return compute.Caps{DeviceMemory: true, UploadDtype: true}
}

func (b *sharedRingBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.uploads++
	return b.Backend.Upload(t, as)
}

func (b *sharedRingBackend) Free(t compute.Tensor) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.frees++
	b.Backend.Free(t)
}

func (b *sharedRingBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.Backend.MatMul(w, x)
}

func (b *sharedRingBackend) SwiGLU(g, u compute.Tensor) compute.Tensor {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.Backend.SwiGLU(g, u)
}

func (b *sharedRingBackend) Read(t compute.Tensor) []float32 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.Backend.Read(t)
}

// sharedRingAgent is one agent's session: its own KV cache, its own halW, its own conversation —
// everything a session owns today — with no routed-expert residency of its own until it attaches.
func sharedRingAgent(m *Model, be compute.Backend) *Session {
	return &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}, Cache: NewKVCache(m.Cfg)}
}

// sharedRingSoloOutputs is the answer every arm below is checked against: each expert's output from
// an ordinary unbounded resident-HAL session, with no ring of any kind. Coalescing may change who
// pays for a page-in; if it changes a single float, the rung is wrong.
func sharedRingSoloOutputs(t *testing.T, m *Model, experts int, x []float32) [][]float32 {
	t.Helper()
	solo := expertRingSession(m, 0)
	out := make([][]float32, experts)
	for e := 0; e < experts; e++ {
		out[e] = expertSwiGLU(m, 0, e, x, sessionQ4KKernel{s: solo})
	}
	return out
}

func sharedRingCheckOutput(t *testing.T, agent string, step, expert int, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s step %d expert %d: len=%d, solo len=%d", agent, step, expert, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s step %d expert %d: out[%d]=%v, solo=%v — sharing residency must change who pays "+
				"for a page-in, never what is computed", agent, step, expert, i, got[i], want[i])
		}
	}
}

// TestSharedRingCoalescesPageInsAcrossAgents is the rung's central witness: coalescing is measured,
// not assumed. Three agents route a shared hot core plus one private expert each — the shape #5243's
// thesis assumes — and the ledger reports how many distinct agents each page-in went on to serve.
//
// The contrast arm is deliberately generous to the status quo: each private ring gets the WHOLE
// shared budget, not its 1/B share. Sharing still pages in strictly less, because the private arm
// pays for the hot core B times over and one ring pays for it once.
func TestSharedRingCoalescesPageInsAcrossAgents(t *testing.T) {
	const H, E, B = 256, 6, 3
	m := expertRingTestModel(t, H, E)
	x := expertRingTestInput(H)
	solo := sharedRingSoloOutputs(t, m, E, x)
	budget := expertRingWeightBytes(t, m) * 9 // three whole experts

	// Each agent: the same hot core {0,1,2}, one private expert, then the core again — enough reuse
	// to exercise hits, enough spread to exercise eviction.
	windows := [B][]int{
		{0, 1, 2, 3, 0, 1},
		{0, 1, 2, 4, 0, 1},
		{0, 1, 2, 5, 0, 1},
	}

	be := &sharedRingBackend{Backend: compute.Default()}
	sh, err := NewSharedExpertRing(SharedExpertRingConfig{Model: m, Backend: be, BudgetBytes: budget})
	if err != nil {
		t.Fatalf("NewSharedExpertRing: %v", err)
	}
	agents := make([]*Session, B)
	for a := range agents {
		agents[a] = sharedRingAgent(m, be)
		if err := sh.Attach(agents[a], agentName(a)); err != nil {
			t.Fatalf("attach agent %d: %v", a, err)
		}
	}

	// Interleave the agents step by step, which is what a batched decode does and what gives the
	// union a chance to coalesce inside one step.
	for step := 0; step < len(windows[0]); step++ {
		for a := 0; a < B; a++ {
			e := windows[a][step]
			got := expertSwiGLU(m, 0, e, x, sessionQ4KKernel{s: agents[a]})
			sharedRingCheckOutput(t, agentName(a), step, e, got, solo[e])
		}
	}

	st := sh.Stats()
	if !st.Enabled || st.Agents != B || st.PeakAgents != B {
		t.Fatalf("shared ring reports %+v, want an enabled ring with %d agents attached", st, B)
	}
	if st.Ring.PeakBytes > budget {
		t.Fatalf("peak resident %d exceeds the shared budget %d — sharing must not unbound the footprint",
			st.Ring.PeakBytes, budget)
	}
	if st.CrossAgentHits == 0 {
		t.Fatal("no demand was served by bytes another agent paid for; the ring coalesced nothing and " +
			"the L2 lever has no mechanism")
	}
	if st.SharedResidencies == 0 {
		t.Fatal("no residency span served two or more agents")
	}
	if got := st.AgentsPerPageIn(); got <= 1 {
		t.Fatalf("agents served per page-in = %.3f, want > 1: at 1.0 every page-in served exactly the "+
			"agent that paid for it, which is B private rings wearing one name", got)
	}
	if got := st.CoalescingRatio(); got <= 1 {
		t.Fatalf("coalescing ratio (demands per page-in) = %.3f, want > 1", got)
	}
	if st.Demands != int64(B*len(windows[0])*3) {
		t.Fatalf("ledger booked %d demands, want %d (%d agents x %d steps x 3 projections)",
			st.Demands, B*len(windows[0])*3, B, len(windows[0]))
	}

	// The contrast: B private rings, each given the WHOLE shared budget, running the same routing.
	private := 0
	for a := 0; a < B; a++ {
		s := expertRingSession(m, budget)
		for step := 0; step < len(windows[a]); step++ {
			e := windows[a][step]
			got := expertSwiGLU(m, 0, e, x, sessionQ4KKernel{s: s})
			sharedRingCheckOutput(t, "private", step, e, got, solo[e])
		}
		private += s.ExpertRing().PageIns
	}
	if st.Ring.PageIns >= private {
		t.Fatalf("one shared ring paged in %d weights, %d private rings of the SAME budget paged in %d; "+
			"sharing must cost strictly fewer page-ins or it buys nothing", st.Ring.PageIns, B, private)
	}

	for _, s := range agents {
		sh.Detach(s)
	}
	if err := sh.Close(); err != nil {
		t.Fatalf("Close after every agent detached: %v", err)
	}
}

// agentName keeps the ledger's identity for "a distinct agent" explicit in the tests, since a
// duplicate name would silently merge two agents and understate coalescing.
func agentName(i int) string { return string(rune('a'+i)) + "-agent" }

// TestSharedRingBytesPerAgentTokenFallAsAgentsGrow is #5243's arithmetic made a witness: at a FIXED
// ring budget, the expert bytes each agent-token costs must fall as B grows, because the union a
// step activates is paid for once no matter how many agents want it.
//
// The budget is deliberately too small to hold the schedule's working set, so the ring really does
// page in every step and the effect cannot come from a warm cache that stopped paying. And every
// agent's output at every B is checked against the solo answer, because a "cheaper" token that
// computed something else would be no result at all.
func TestSharedRingBytesPerAgentTokenFallAsAgentsGrow(t *testing.T) {
	const H, E, steps = 256, 6, 8
	m := expertRingTestModel(t, H, E)
	x := expertRingTestInput(H)
	solo := sharedRingSoloOutputs(t, m, E, x)
	budget := expertRingWeightBytes(t, m) * 6 // two whole experts; the schedule wants more

	var prev float64
	for _, B := range []int{1, 2, 4} {
		be := &sharedRingBackend{Backend: compute.Default()}
		sh, err := NewSharedExpertRing(SharedExpertRingConfig{Model: m, Backend: be, BudgetBytes: budget})
		if err != nil {
			t.Fatalf("B=%d NewSharedExpertRing: %v", B, err)
		}
		agents := make([]*Session, B)
		for a := range agents {
			agents[a] = sharedRingAgent(m, be)
			if err := sh.Attach(agents[a], agentName(a)); err != nil {
				t.Fatalf("B=%d attach agent %d: %v", B, a, err)
			}
		}
		// Every agent activates the same top-2 at each step — maximal overlap, which is the regime
		// the lever is claimed in — and the set moves every step so the budget is always short.
		for step := 0; step < steps; step++ {
			for a := 0; a < B; a++ {
				for _, e := range []int{step % E, (step + 1) % E} {
					got := expertSwiGLU(m, 0, e, x, sessionQ4KKernel{s: agents[a]})
					sharedRingCheckOutput(t, agentName(a), step, e, got, solo[e])
				}
				sh.NoteAgentToken() // one token produced by one agent over this residency
			}
		}

		st := sh.Stats()
		if st.AgentTokens != int64(B*steps) {
			t.Fatalf("B=%d booked %d agent-tokens, want %d", B, st.AgentTokens, B*steps)
		}
		if st.PageInBytes <= 0 {
			t.Fatalf("B=%d paged in %d bytes; the budget was supposed to be short", B, st.PageInBytes)
		}
		if st.Ring.PeakBytes > budget {
			t.Fatalf("B=%d peak resident %d exceeds budget %d", B, st.Ring.PeakBytes, budget)
		}
		got := st.BytesPerAgentToken()
		if B == 1 {
			if r := st.AgentsPerPageIn(); r != 1 {
				t.Fatalf("a single agent measured %.3f agents per page-in, want exactly 1.0 — the meter "+
					"must have no cross-agent credit to give when there is no second agent", r)
			}
			if st.CrossAgentHits != 0 {
				t.Fatalf("a single agent recorded %d cross-agent hits", st.CrossAgentHits)
			}
		} else if got >= prev {
			t.Fatalf("B=%d costs %.1f expert bytes per agent-token, B=%d cost %.1f; at a fixed budget the "+
				"union is paid once and must amortize over more output as B grows", B, got, B/2, prev)
		}
		prev = got

		for _, s := range agents {
			sh.Detach(s)
		}
		if err := sh.Close(); err != nil {
			t.Fatalf("B=%d Close: %v", B, err)
		}
	}
}

// TestSharedRingSharesWeightBytesAndNothingElse is the safety property the issue calls load-bearing.
// Weights are model-constant bytes and may be shared; KV, conversation state and the permanent halW
// memoizer are per-agent and must not be. The enforcement is at Attach — a refusal, not a log —
// because a ring shared between two models would feed one model's expert bytes to the other's GEMM,
// which is the worst failure available to this rung and would surface as plausible garbage, not a
// crash.
func TestSharedRingSharesWeightBytesAndNothingElse(t *testing.T) {
	const H, E = 256, 4
	m := expertRingTestModel(t, H, E)
	x := expertRingTestInput(H)
	solo := sharedRingSoloOutputs(t, m, E, x)
	budget := expertRingWeightBytes(t, m) * 12 // every expert fits; residency is not the subject here

	be := &sharedRingBackend{Backend: compute.Default()}
	sh, err := NewSharedExpertRing(SharedExpertRingConfig{Model: m, Backend: be, BudgetBytes: budget})
	if err != nil {
		t.Fatalf("NewSharedExpertRing: %v", err)
	}

	// The refusals, one per way sharing could be made wrong.
	other := expertRingTestModel(t, H, E)
	if err := sh.Attach(sharedRingAgent(other, be), "wrong-model"); !errors.Is(err, ErrSharedRingModel) {
		t.Fatalf("attaching a session over a DIFFERENT model returned %v, want ErrSharedRingModel — the "+
			"ring would have served it another model's expert bytes", err)
	}
	otherBE := &sharedRingBackend{Backend: compute.Default()}
	if err := sh.Attach(sharedRingAgent(m, otherBE), "wrong-device"); !errors.Is(err, ErrSharedRingBackend) {
		t.Fatalf("attaching a session on a DIFFERENT backend returned %v, want ErrSharedRingBackend", err)
	}
	if err := sh.Attach(sharedRingAgent(m, be), ""); !errors.Is(err, ErrSharedRingAgent) {
		t.Fatalf("attaching an unnamed agent returned %v, want ErrSharedRingAgent", err)
	}
	withOwnRing := sharedRingAgent(m, be) // same model, same device — only the private ring disqualifies it
	withOwnRing.ExpertRingBytes = budget
	_ = expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: withOwnRing}) // build its private ring
	if err := sh.Attach(withOwnRing, "already-resident"); !errors.Is(err, ErrSharedRingResidency) {
		t.Fatalf("attaching a session that already owns routed-expert residency returned %v, want "+
			"ErrSharedRingResidency — its private handles would have been stranded", err)
	}

	a := sharedRingAgent(m, be)
	b := sharedRingAgent(m, be)
	if err := sh.Attach(a, "a"); err != nil {
		t.Fatalf("attach a: %v", err)
	}
	if err := sh.Attach(b, "a"); !errors.Is(err, ErrSharedRingAgent) {
		t.Fatalf("attaching a DUPLICATE agent name returned %v, want ErrSharedRingAgent — the ledger "+
			"would have merged two agents into one and understated coalescing", err)
	}
	if err := sh.Attach(b, "b"); err != nil {
		t.Fatalf("attach b: %v", err)
	}

	// Give agent a some conversation state, which must be invisible to b.
	a.Cache.K[0] = []float32{1, 2, 3}
	a.Cache.V[0] = []float32{4, 5, 6}

	// Agent a pays for expert 0; agent b is served off the same bytes.
	sharedRingCheckOutput(t, "a", 0, 0, expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: a}), solo[0])
	afterA := sh.Stats()
	sharedRingCheckOutput(t, "b", 0, 0, expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: b}), solo[0])
	afterB := sh.Stats()
	if afterB.Ring.PageIns != afterA.Ring.PageIns {
		t.Fatalf("agent b's demand paged in %d more weights over residency agent a had already paid for",
			afterB.Ring.PageIns-afterA.Ring.PageIns)
	}
	if afterB.CrossAgentHits != 3 {
		t.Fatalf("agent b recorded %d cross-agent hits over agent a's expert, want 3 (gate/up/down)",
			afterB.CrossAgentHits)
	}

	// WEIGHTS are shared: one residency entry per projection, in the ring, not in either halW.
	for _, proj := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
		key := "q4k:" + expertName(0, 0, proj)
		if _, live := sh.ring.resident[polymodel.ModelID(key)]; !live {
			t.Fatalf("%s is not resident in the shared ring; the agents did not share residency at all", key)
		}
		for name, s := range map[string]*Session{"a": a, "b": b} {
			if _, memo := s.halW[key]; memo {
				t.Fatalf("agent %s memoized routed expert %q in its own halW; the ring must own it", name, key)
			}
		}
	}

	// EVERYTHING ELSE is not. A dense projection staged by a alone stays in a's halW.
	dense := "model.layers.0.mlp.down_proj.weight"
	m.q4kw[dense] = &q4kTensor{out: H, in: H, nblk: H / qkK, raw: buildRawQ4K(t, H, H, 7)}
	a.weightHALQ4K(dense, m.q4kw[dense])
	if _, memo := a.halW["q4k:"+dense]; !memo {
		t.Fatal("a dense weight staged by agent a did not land in agent a's halW")
	}
	if _, leaked := b.halW["q4k:"+dense]; leaked {
		t.Fatal("a dense weight staged by agent a appeared in agent b's halW; only ROUTED EXPERT " +
			"residency is shared")
	}
	if _, leaked := sh.ring.resident[polymodel.ModelID("q4k:"+dense)]; leaked {
		t.Fatal("a dense weight entered the shared routed-expert ring")
	}

	// Conversation state stayed put: distinct caches, and b's forwards touched neither.
	if a.Cache == b.Cache {
		t.Fatal("the two agents share one KV cache")
	}
	if len(a.Cache.K[0]) != 3 || a.Cache.K[0][0] != 1 || a.Cache.V[0][1] != 5 {
		t.Fatalf("agent a's KV cache changed while agent b decoded: K=%v V=%v", a.Cache.K[0], a.Cache.V[0])
	}
	if len(b.Cache.K[0]) != 0 || len(b.Cache.V[0]) != 0 {
		t.Fatalf("agent b's KV cache holds agent a's positions: K=%v V=%v", b.Cache.K[0], b.Cache.V[0])
	}

	// Close refuses while agents are live: freeing a handle b is about to multiply is a use-after-free.
	if err := sh.Close(); !errors.Is(err, ErrSharedRingBusy) {
		t.Fatalf("Close with agents attached returned %v, want ErrSharedRingBusy", err)
	}

	// A conversation ending DETACHES: b's residency survives a's Close intact.
	before := sh.Stats()
	a.Close()
	if a.sharedRing != nil || a.expertRing != nil {
		t.Fatal("Session.Close left the closed session pointing at the shared ring")
	}
	if got := sh.Stats(); got.Agents != 1 || got.Ring.ResidentCount != before.Ring.ResidentCount {
		t.Fatalf("after agent a closed the ring holds %d agents and %d residents, want 1 and %d — a "+
			"conversation ending must not page out its peers' working set",
			got.Agents, got.Ring.ResidentCount, before.Ring.ResidentCount)
	}
	sharedRingCheckOutput(t, "b", 1, 0, expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: b}), solo[0])
	if got := sh.Stats(); got.Ring.PageIns != before.Ring.PageIns {
		t.Fatalf("agent b paged in %d weights after agent a closed; the residency should have survived",
			got.Ring.PageIns-before.Ring.PageIns)
	}

	b.Close()
	if err := sh.Close(); err != nil {
		t.Fatalf("Close after the last agent left: %v", err)
	}
	if err := sh.Attach(sharedRingAgent(m, be), "late"); !errors.Is(err, ErrSharedRingClosed) {
		t.Fatalf("attaching to a closed ring returned %v, want ErrSharedRingClosed", err)
	}
}

// TestSharedRingServesConcurrentAgentsBoundedAndUnchanged is the discipline witness. Sharing is only
// useful if agents decode at the same time, and the moment they do, the stage->hold window in
// expertSwiGLUHAL becomes reachable: a peer's staging landing between "here is your handle" and
// "this handle is pinned" would Free a tensor the first agent is about to multiply.
//
// The lost-update detector is the EXACT demand count rather than the race detector, deliberately:
// -race needs cgo and a C toolchain that a plain `go test` host may not have, whereas an unguarded
// `demands++` under four goroutines loses increments on its own and fails this assertion without
// any special build. Running it under -race where cgo is available adds the reads a counter cannot
// see, but nothing here depends on that.
//
// The budget is five experts for four agents, which is the sizing rule a shared ring introduces:
// each agent pins its current expert's three projections for the span of its GEMMs, so a budget
// below B x 3 projections has every resident pinned by a peer and refuses — correctly, but into the
// permanent residency the ring exists to replace. Five holds the four held experts with room to
// page, while the six-expert working set still forces eviction.
func TestSharedRingServesConcurrentAgentsBoundedAndUnchanged(t *testing.T) {
	const H, E, B, steps = 256, 6, 4, 24
	m := expertRingTestModel(t, H, E)
	x := expertRingTestInput(H)
	solo := sharedRingSoloOutputs(t, m, E, x)
	budget := expertRingWeightBytes(t, m) * 15 // five whole experts for four concurrently-holding agents

	be := &sharedRingBackend{Backend: compute.Default()}
	sh, err := NewSharedExpertRing(SharedExpertRingConfig{Model: m, Backend: be, BudgetBytes: budget})
	if err != nil {
		t.Fatalf("NewSharedExpertRing: %v", err)
	}
	agents := make([]*Session, B)
	for a := range agents {
		agents[a] = sharedRingAgent(m, be)
		if err := sh.Attach(agents[a], agentName(a)); err != nil {
			t.Fatalf("attach agent %d: %v", a, err)
		}
	}

	var wg sync.WaitGroup
	fail := make(chan string, B*steps)
	for a := 0; a < B; a++ {
		wg.Add(1)
		go func(a int) {
			defer wg.Done()
			for step := 0; step < steps; step++ {
				e := (step + a) % E // overlapping but offset, so hits and misses interleave across agents
				got := expertSwiGLU(m, 0, e, x, sessionQ4KKernel{s: agents[a]})
				want := solo[e]
				if len(got) != len(want) {
					fail <- "length mismatch under concurrent agents"
					return
				}
				for i := range want {
					if got[i] != want[i] {
						fail <- "a concurrently served expert returned a different value than the solo run"
						return
					}
				}
			}
		}(a)
	}
	wg.Wait()
	close(fail)
	for msg := range fail {
		t.Fatal(msg)
	}

	st := sh.Stats()
	if st.Ring.PeakBytes > budget {
		t.Fatalf("peak resident %d exceeds budget %d under %d concurrent agents — the bound is not held "+
			"across sessions", st.Ring.PeakBytes, budget, B)
	}
	if st.Ring.ResidentBytes > budget {
		t.Fatalf("resident %d exceeds budget %d", st.Ring.ResidentBytes, budget)
	}
	if st.Ring.Evictions == 0 {
		t.Fatalf("no evictions under a 5-expert budget over a %d-expert working set with %d agents: "+
			"the contended path was never taken", E, B)
	}
	if st.Refusals != 0 {
		t.Fatalf("%d stagings were refused: the budget cannot hold %d agents' concurrently held experts, "+
			"so they fell back to the permanent residency the ring replaces", st.Refusals, B)
	}
	if st.Demands != int64(B*steps*3) {
		t.Fatalf("ledger booked %d demands under concurrency, want %d — a lost update means ring state "+
			"was touched outside the span", st.Demands, B*steps*3)
	}
	for a, s := range agents {
		for key := range s.halW {
			if isRoutedExpertWeight(key) {
				t.Fatalf("agent %d promoted routed expert %q to permanent residency under contention; "+
					"the shared ring must own every routed expert it was sized for", a, key)
			}
		}
	}
	for _, s := range agents {
		sh.Detach(s)
	}
	if err := sh.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestSharedExpertRingPolicySwapIsAtomicWhileAttached(t *testing.T) {
	m := expertRingTestModel(t, 256, 6)
	be := &sharedRingBackend{Backend: compute.Default()}
	sh, err := NewSharedExpertRing(SharedExpertRingConfig{
		Model: m, Backend: be, BudgetBytes: expertRingWeightBytes(t, m) * 6,
	})
	if err != nil {
		t.Fatalf("NewSharedExpertRing: %v", err)
	}
	defer sh.Close()
	a, b := sharedRingAgent(m, be), sharedRingAgent(m, be)
	defer a.Close()
	defer b.Close()
	if err := sh.Attach(a, "a"); err != nil {
		t.Fatalf("attach a: %v", err)
	}
	if err := sh.Attach(b, "b"); err != nil {
		t.Fatalf("attach b: %v", err)
	}
	driveExpertWindow(a, m, []int{0})
	before := sh.Stats()

	// Hold an active shared-ring span. The swap must wait on the same mutex rather than changing
	// the plain policy field under an operation that already entered under generation one.
	doneSpan := a.ringEnter(a.expertRing)
	type result struct {
		receipt ExpertRingPolicySwapReceipt
		err     error
	}
	started := make(chan struct{})
	resultCh := make(chan result, 1)
	go func() {
		close(started)
		r, err := sh.SwapExpertRingEvictPolicy(ExpertRingEvictValueAware)
		resultCh <- result{r, err}
	}()
	<-started
	select {
	case got := <-resultCh:
		doneSpan()
		t.Fatalf("swap crossed an active ring span: %+v", got)
	case <-time.After(20 * time.Millisecond):
	}
	if a.expertRing.policyGeneration != 1 || a.expertRing.policy != ExpertRingEvictLRU {
		doneSpan()
		t.Fatalf("active span observed a premature swap: policy=%s generation=%d",
			a.expertRing.policy, a.expertRing.policyGeneration)
	}
	doneSpan()
	got := <-resultCh
	if got.err != nil || !got.receipt.Changed || got.receipt.PolicyGeneration != 2 {
		t.Fatalf("swap result=%+v err=%v", got.receipt, got.err)
	}
	after := sh.Stats()
	if after.Agents != 2 || after.Ring.ResidentCount != before.Ring.ResidentCount ||
		after.Ring.ResidentBytes != before.Ring.ResidentBytes || after.Ring.PageIns != before.Ring.PageIns {
		t.Fatalf("swap disturbed attachments/residency/counters: before=%+v after=%+v", before, after)
	}
	driveExpertWindow(b, m, []int{1})
	if st := b.ExpertRing(); st.Policy != ExpertRingEvictValueAware || st.PolicyGeneration != 2 {
		t.Fatalf("next attached operation did not see new epoch: %+v", st)
	}
}
