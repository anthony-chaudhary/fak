package comm

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// recordingKernel is a minimal abi.Kernel that records every Submit so a test can
// assert the per-member adjudication order and count.
type recordingKernel struct {
	verdict abi.Verdict
	mu      sync.Mutex
	calls   []*abi.ToolCall
	reaps   []abi.SubmissionHandle
	submits int64
}

func allowKernel() *recordingKernel {
	return &recordingKernel{verdict: abi.Verdict{Kind: abi.VerdictAllow, By: "test"}}
}

func denyKernel() *recordingKernel {
	return &recordingKernel{verdict: abi.Verdict{Kind: abi.VerdictDeny, By: "test", Reason: abi.ReasonNone}}
}

func (k *recordingKernel) Submit(ctx context.Context, c *abi.ToolCall) (abi.SubmissionHandle, abi.Verdict) {
	k.mu.Lock()
	k.calls = append(k.calls, c)
	k.mu.Unlock()
	seq := uint64(atomic.AddInt64(&k.submits, 1))
	return abi.SubmissionHandle{Seq: seq}, k.verdict
}

func (k *recordingKernel) Reap(ctx context.Context, h abi.SubmissionHandle) (*abi.Result, error) {
	k.mu.Lock()
	k.reaps = append(k.reaps, h)
	k.mu.Unlock()
	return &abi.Result{Status: abi.StatusOK}, nil
}

func (k *recordingKernel) Syscall(ctx context.Context, c *abi.ToolCall) (*abi.Result, abi.Verdict) {
	_, v := k.Submit(ctx, c)
	return &abi.Result{Call: c, Status: abi.StatusOK}, v
}

func (k *recordingKernel) Resolver() abi.Resolver                        { return nil }
func (k *recordingKernel) Negotiate(c []abi.Capability) []abi.Capability { return c }

func members(ids ...string) []Member {
	out := make([]Member, len(ids))
	for i, id := range ids {
		out[i] = Member{ID: id}
	}
	return out
}

// TestRankDeterministicOverPermutation is the DoD's "deterministic rank over a
// permuted member set": the same identity set must yield the same ranks regardless
// of the order the members are handed to New.
func TestRankDeterministicOverPermutation(t *testing.T) {
	perms := [][]string{
		{"alpha", "bravo", "charlie", "delta"},
		{"delta", "charlie", "bravo", "alpha"},
		{"charlie", "alpha", "delta", "bravo"},
		{"bravo", "delta", "alpha", "charlie"},
	}
	// Canonical (sorted) ranks the group must always assign.
	want := map[string]int{"alpha": 0, "bravo": 1, "charlie": 2, "delta": 3}

	for _, p := range perms {
		g, err := New("wave-1", "trace-1", members(p...))
		if err != nil {
			t.Fatalf("New(%v): %v", p, err)
		}
		if g.Size() != len(want) {
			t.Fatalf("Size()=%d, want %d", g.Size(), len(want))
		}
		for id, wantRank := range want {
			got, err := g.Rank(id)
			if err != nil {
				t.Fatalf("Rank(%q): %v", id, err)
			}
			if got != wantRank {
				t.Fatalf("permutation %v: Rank(%q)=%d, want %d", p, id, got, wantRank)
			}
		}
		// Member(rank) must invert Rank(id).
		for r := 0; r < g.Size(); r++ {
			m, err := g.Member(r)
			if err != nil {
				t.Fatalf("Member(%d): %v", r, err)
			}
			if want[m.ID] != r {
				t.Fatalf("Member(%d).ID=%q maps to rank %d", r, m.ID, want[m.ID])
			}
		}
	}
}

func TestNewRejectsEmptyAndDuplicate(t *testing.T) {
	if _, err := New("w", "", nil); err != ErrEmptyGroup {
		t.Fatalf("New(nil) err=%v, want ErrEmptyGroup", err)
	}
	if _, err := New("w", "", members("a", "b", "a")); err == nil {
		t.Fatal("New with duplicate ID should fail")
	}
	if _, err := New("w", "", members("a")); err != nil {
		t.Fatalf("New with one member: %v", err)
	}
}

func TestRankUnknownMember(t *testing.T) {
	g, _ := New("w", "", members("a", "b"))
	if _, err := g.Rank("zzz"); err == nil {
		t.Fatal("Rank of unknown member should fail")
	}
}

// TestSplitColorLaneEquivalence is the DoD's "split-color↔lease equivalence": a split
// by color partitions the group into exactly the color buckets, the lane a sub-group
// reports IS the color (so the split is the dos-arbitrate lease key), and within each
// split ranks are re-assigned deterministically from the sorted members.
func TestSplitColorLaneEquivalence(t *testing.T) {
	ms := []Member{
		{ID: "a", Lane: "kernel"},
		{ID: "b", Lane: "gateway"},
		{ID: "c", Lane: "kernel"},
		{ID: "d", Lane: "gateway"},
		{ID: "e", Lane: ""}, // joins no split
	}
	g, err := New("wave-x", "trace-x", ms)
	if err != nil {
		t.Fatal(err)
	}

	// Split by the member's Lane: color == lane.
	splits := g.SplitLane()
	if len(splits) != 2 {
		t.Fatalf("got %d colors, want 2 (kernel, gateway)", len(splits))
	}

	// The split keyed "kernel" must contain exactly {a, c}, re-ranked 0,1, and every
	// member's lane must equal the split color (color↔lease equivalence).
	kern := splits["kernel"]
	if kern == nil || kern.Size() != 2 {
		t.Fatalf("kernel split = %v", kern)
	}
	for _, id := range []string{"a", "c"} {
		if _, err := kern.Rank(id); err != nil {
			t.Fatalf("kernel split missing %q: %v", id, err)
		}
	}
	if got := kern.Lanes(); len(got) != 1 || got[0] != "kernel" {
		t.Fatalf("kernel split lanes=%v, want [kernel]", got)
	}
	if r, _ := kern.Rank("a"); r != 0 {
		t.Fatalf("kernel split Rank(a)=%d, want 0", r)
	}
	if r, _ := kern.Rank("c"); r != 1 {
		t.Fatalf("kernel split Rank(c)=%d, want 1", r)
	}

	gw := splits["gateway"]
	if gw == nil || gw.Size() != 2 {
		t.Fatalf("gateway split = %v", gw)
	}
	if got := gw.Lanes(); len(got) != 1 || got[0] != "gateway" {
		t.Fatalf("gateway split lanes=%v, want [gateway]", got)
	}

	// The empty-lane member 'e' joined no split.
	for _, sg := range splits {
		if _, err := sg.Rank("e"); err == nil {
			t.Fatal("empty-lane member should not appear in any split")
		}
	}

	// The whole group's lane partition is the sorted distinct lanes.
	want := []string{"gateway", "kernel"}
	got := g.Lanes()
	if len(got) != len(want) {
		t.Fatalf("group Lanes()=%v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("group Lanes()=%v, want %v", got, want)
		}
	}
}

// TestSplitIsOrderIndependent proves the split partition does not depend on the order
// members were passed to New (same property as rank determinism, at split scope).
func TestSplitIsOrderIndependent(t *testing.T) {
	a := []Member{{ID: "x", Lane: "L1"}, {ID: "y", Lane: "L2"}, {ID: "z", Lane: "L1"}}
	b := []Member{{ID: "z", Lane: "L1"}, {ID: "x", Lane: "L1"}, {ID: "y", Lane: "L2"}}
	ga, _ := New("w", "", a)
	gb, _ := New("w", "", b)
	sa, sb := ga.SplitLane(), gb.SplitLane()
	if sa["L1"].Size() != 2 || sb["L1"].Size() != 2 {
		t.Fatalf("L1 split sizes differ: %d vs %d", sa["L1"].Size(), sb["L1"].Size())
	}
	for _, id := range []string{"x", "z"} {
		ra, _ := sa["L1"].Rank(id)
		rb, _ := sb["L1"].Rank(id)
		if ra != rb {
			t.Fatalf("Rank(%q) differs across permutations: %d vs %d", id, ra, rb)
		}
	}
}

// TestSpawnMembership proves Spawn mints one rank-stamped Membership per member, in
// rank order, carrying wave/parent/size/lane.
func TestSpawnMembership(t *testing.T) {
	ms := []Member{{ID: "b", Lane: "L2"}, {ID: "a", Lane: "L1"}}
	g, _ := New("wave-9", "trace-9", ms)
	roster := g.Spawn()
	if len(roster) != 2 {
		t.Fatalf("Spawn() len=%d, want 2", len(roster))
	}
	// Rank 0 is "a" (sorted), rank 1 is "b".
	if roster[0].Rank != 0 || roster[0].Lane != "L1" {
		t.Fatalf("roster[0]=%+v, want rank 0 lane L1", roster[0])
	}
	if roster[1].Rank != 1 || roster[1].Lane != "L2" {
		t.Fatalf("roster[1]=%+v, want rank 1 lane L2", roster[1])
	}
	for _, m := range roster {
		if m.WaveID != "wave-9" || m.ParentTraceID != "trace-9" || m.Size != 2 {
			t.Fatalf("membership lost wave/parent/size: %+v", m)
		}
	}
}

func TestBroadcastRoutesThroughSubmitAndScopeBound(t *testing.T) {
	k := allowKernel()
	g, _ := New("w", "", members("c", "a", "b"))
	payload := abi.Ref{Kind: abi.RefInline, Inline: []byte("shared"), Len: 6, Scope: abi.ScopeFleet, Taint: abi.TaintTainted}

	req, err := g.Broadcast(context.Background(), k, payload)
	if err != nil {
		t.Fatal(err)
	}
	if req.Tool != ToolBroadcast || req.State != abi.StatusPending || len(req.Handles) != 3 {
		t.Fatalf("Broadcast request=%+v, want tool %q pending with 3 handles", req, ToolBroadcast)
	}
	if len(k.calls) != 3 {
		t.Fatalf("Submit count=%d, want 3", len(k.calls))
	}
	wantRanks := []string{"0", "1", "2"}
	for r, c := range k.calls {
		if c.Tool != ToolBroadcast {
			t.Fatalf("call %d tool=%q, want %q", r, c.Tool, ToolBroadcast)
		}
		if c.Meta["rank"] != wantRanks[r] {
			t.Fatalf("call %d rank meta=%q", r, c.Meta["rank"])
		}
		if c.Args.Scope != abi.ScopeFleet || c.Args.Taint != abi.TaintTainted || string(c.Args.Inline) != "shared" {
			t.Fatalf("call %d args=%+v, want unchanged fleet-scoped payload", r, c.Args)
		}
	}

	private := abi.Ref{Kind: abi.RefInline, Inline: []byte("private"), Len: 7, Scope: abi.ScopeAgent, Taint: abi.TaintTainted}
	k = allowKernel()
	if _, err := g.Broadcast(context.Background(), k, private); !errors.Is(err, ErrScopeWiden) {
		t.Fatalf("private Broadcast err=%v, want ErrScopeWiden", err)
	}
	if len(k.calls) != 0 {
		t.Fatalf("private Broadcast made %d submits, want 0", len(k.calls))
	}
}

func TestScatterRoutesOneGoalPerMember(t *testing.T) {
	k := allowKernel()
	g, _ := New("w", "", members("a", "b"))
	goals := []abi.Ref{
		{Kind: abi.RefInline, Inline: []byte("goal-a"), Len: 6, Scope: abi.ScopeAgent, Taint: abi.TaintTainted},
		{Kind: abi.RefInline, Inline: []byte("goal-b"), Len: 6, Scope: abi.ScopeAgent, Taint: abi.TaintTrusted},
	}

	req, err := g.Scatter(context.Background(), k, goals)
	if err != nil {
		t.Fatal(err)
	}
	if req.Tool != ToolScatter || req.State != abi.StatusPending || len(req.Handles) != 2 {
		t.Fatalf("Scatter request=%+v", req)
	}
	for r, c := range k.calls {
		if c.Tool != ToolScatter {
			t.Fatalf("call %d tool=%q, want %q", r, c.Tool, ToolScatter)
		}
		if string(c.Args.Inline) != string(goals[r].Inline) || c.Args.Scope != goals[r].Scope || c.Args.Taint != goals[r].Taint {
			t.Fatalf("call %d args=%+v, want rank goal %+v", r, c.Args, goals[r])
		}
	}
}

func TestBarrierRoutesThroughSubmitWithWitnessMeta(t *testing.T) {
	k := allowKernel()
	g, _ := New("w", "", members("a", "b", "c"))

	req, err := g.Barrier(context.Background(), k)
	if err != nil {
		t.Fatal(err)
	}
	if req.Tool != ToolBarrier || req.State != abi.StatusPending || len(req.Handles) != 3 {
		t.Fatalf("Barrier request=%+v", req)
	}
	for r, c := range k.calls {
		if c.Tool != ToolBarrier {
			t.Fatalf("call %d tool=%q, want %q", r, c.Tool, ToolBarrier)
		}
		if c.Meta["witness"] != "dos-witness-claim" {
			t.Fatalf("call %d witness meta=%q, want dos-witness-claim", r, c.Meta["witness"])
		}
		if c.Args.Scope != abi.ScopeAgent || c.Args.Taint != abi.TaintTainted {
			t.Fatalf("call %d args scope/taint=%v/%v, want ScopeAgent/Tainted", r, c.Args.Scope, c.Args.Taint)
		}
	}
}

func TestNonBlockingCollectiveReapUsesRankHandles(t *testing.T) {
	k := allowKernel()
	g, _ := New("w", "", members("a", "b"))
	payload := abi.Ref{Kind: abi.RefInline, Inline: []byte("shared"), Len: 6, Scope: abi.ScopeFleet, Taint: abi.TaintTainted}

	req, err := g.IBroadcast(context.Background(), k, payload)
	if err != nil {
		t.Fatal(err)
	}
	if req.State != abi.StatusPending {
		t.Fatalf("IBroadcast state=%v, want StatusPending", req.State)
	}
	if _, err := req.Reap(context.Background(), k); err != nil {
		t.Fatal(err)
	}
	if req.State != abi.StatusOK {
		t.Fatalf("Reap state=%v, want StatusOK", req.State)
	}
	if len(k.reaps) != len(req.Handles) {
		t.Fatalf("Reap count=%d, want %d", len(k.reaps), len(req.Handles))
	}
	for i, h := range req.Handles {
		if k.reaps[i] != h {
			t.Fatalf("reap %d handle=%+v, want %+v", i, k.reaps[i], h)
		}
	}
}

// TestGatherRoutesThroughSubmitInRankOrder is the wiring contract: a Gather expands to
// exactly N independently-adjudicated Submit calls (no floor bypass), in rank order,
// and the fold is rank-ordered into Combine.
func TestGatherRoutesThroughSubmitInRankOrder(t *testing.T) {
	k := allowKernel()
	g, _ := New("w", "", members("a", "b", "c"))
	// outputs indexed by rank: a=0, b=1, c=2.
	out, err := g.Gather(context.Background(), k, []string{"oa", "ob", "oc"}, modelroute.ReduceConcat)
	if err != nil {
		t.Fatal(err)
	}
	// Exactly one Submit per member.
	if len(k.calls) != 3 {
		t.Fatalf("Submit count=%d, want 3 (one per member)", len(k.calls))
	}
	// Every call went through the comm.gather tool (the adjudication floor).
	for i, c := range k.calls {
		if c.Tool != ToolGather {
			t.Fatalf("call %d tool=%q, want %q", i, c.Tool, ToolGather)
		}
		// Fail-closed default scope/taint on the gather descriptor.
		if c.Args.Scope != abi.ScopeAgent || c.Args.Taint != abi.TaintTainted {
			t.Fatalf("call %d args scope/taint=%v/%v, want ScopeAgent/Tainted", i, c.Args.Scope, c.Args.Taint)
		}
	}
	// ReduceConcat joins in RANK order (Combine separates members with a newline):
	// oa, ob, oc. The property under test is the ORDER, not the separator.
	if out.Output != "oa\nob\noc" {
		t.Fatalf("Gather concat=%q, want %q (rank order preserved into Combine)", out.Output, "oa\nob\noc")
	}
}

func TestIGatherCombinesAfterRankOrderedSubmit(t *testing.T) {
	k := allowKernel()
	g, _ := New("w", "", members("b", "a"))

	req, err := g.IGather(context.Background(), k, []string{"oa", "ob"}, modelroute.ReduceConcat)
	if err != nil {
		t.Fatal(err)
	}
	if req.Tool != ToolGather || req.State != abi.StatusPending || len(req.Handles) != 2 {
		t.Fatalf("IGather request=%+v", req.CollectiveRequest)
	}
	got, err := req.ReapCombine(context.Background(), k)
	if err != nil {
		t.Fatal(err)
	}
	if got.Output != "oa\nob" {
		t.Fatalf("IGather concat=%q, want rank-ordered output", got.Output)
	}
	if req.State != abi.StatusOK {
		t.Fatalf("ReapCombine state=%v, want StatusOK", req.State)
	}
}

// TestGatherFailsClosedOnDeny proves a refused Submit fails the whole Gather — no
// collective silently drops a refused call.
func TestGatherFailsClosedOnDeny(t *testing.T) {
	k := denyKernel()
	g, _ := New("w", "", members("a", "b"))
	_, err := g.Gather(context.Background(), k, []string{"x", "y"}, modelroute.ReduceConcat)
	if err == nil {
		t.Fatal("Gather over a denying kernel must fail closed")
	}
	// It must stop at the first refusal, not adjudicate the rest.
	if len(k.calls) != 1 {
		t.Fatalf("denied Gather made %d Submits, want 1 (stop at first refusal)", len(k.calls))
	}
}

func TestGatherArityAndNilKernel(t *testing.T) {
	g, _ := New("w", "", members("a", "b"))
	if _, err := g.Gather(context.Background(), allowKernel(), []string{"only-one"}, modelroute.ReduceConcat); !errors.Is(err, ErrArity) {
		t.Fatalf("arity mismatch err=%v, want ErrArity", err)
	}
	if _, err := g.Gather(context.Background(), nil, []string{"a", "b"}, modelroute.ReduceConcat); !errors.Is(err, ErrNoKernel) {
		t.Fatalf("nil kernel err=%v, want ErrNoKernel", err)
	}
	if _, err := g.Scatter(context.Background(), allowKernel(), []abi.Ref{{}}); !errors.Is(err, ErrArity) {
		t.Fatalf("scatter arity mismatch err=%v, want ErrArity", err)
	}
	if _, err := g.Barrier(context.Background(), nil); !errors.Is(err, ErrNoKernel) {
		t.Fatalf("barrier nil kernel err=%v, want ErrNoKernel", err)
	}
}
