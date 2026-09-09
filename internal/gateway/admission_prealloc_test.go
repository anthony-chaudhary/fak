package gateway

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"
	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestEstimateServedAdmissionTokensPreallocCeiling(t *testing.T) {
	TestEstimateServedAdmissionTokens_PreallocCeiling(t)
}

// Acceptance criterion 1: Unit test on estimateServedAdmissionTokens & estimateServedTokenUsage:
// max_tokens = 120000 charges ~PreallocCeiling (1024 + prompt), not 120000.
func TestEstimateServedAdmissionTokens_PreallocCeiling(t *testing.T) {
	msgs := []agent.Message{
		{Role: "user", Content: "Hello world!"}, // 4 + 12 = 16 chars -> 4 tokens
	}

	// 1. Uncapped request with max_tokens = 120000 charges 4 + DefaultPreallocCeiling (1024) = 1028
	got := estimateServedAdmissionTokens(msgs, nil, 120000)
	want := 4 + DefaultPreallocCeiling
	if got != want {
		t.Fatalf("estimateServedAdmissionTokens(120000) = %d, want %d (capped at PreallocCeiling %d, not 120000)",
			got, want, DefaultPreallocCeiling)
	}

	// 2. Request under ceiling (max_tokens = 500) charges full 4 + 500 = 504
	if gotUnder := estimateServedAdmissionTokens(msgs, nil, 500); gotUnder != 504 {
		t.Fatalf("estimateServedAdmissionTokens(500) = %d, want 504 (cap should not bite)", gotUnder)
	}

	// 3. Request with zero max_tokens charges prompt + 1
	if gotZero := estimateServedAdmissionTokens(msgs, nil, 0); gotZero != 5 {
		t.Fatalf("estimateServedAdmissionTokens(0) = %d, want 5", gotZero)
	}

	// 4. Custom prealloc ceiling via estimateServedAdmissionTokensWithCap
	gotCustom := estimateServedAdmissionTokensWithCap(msgs, nil, 120000, 256)
	if gotCustom != 4+256 {
		t.Fatalf("estimateServedAdmissionTokensWithCap(120000, 256) = %d, want %d", gotCustom, 4+256)
	}

	// 5. Non-positive ceiling defaults to DefaultPreallocCeiling
	gotDefault := estimateServedAdmissionTokensWithCap(msgs, nil, 120000, 0)
	if gotDefault != 4+DefaultPreallocCeiling {
		t.Fatalf("estimateServedAdmissionTokensWithCap(120000, 0) = %d, want %d", gotDefault, 4+DefaultPreallocCeiling)
	}

	// 6. Provider token usage estimate also caps output tokens at DefaultPreallocCeiling
	usage := estimateServedTokenUsage(msgs, nil, 120000)
	if usage.OutputTokens != int64(DefaultPreallocCeiling) {
		t.Fatalf("estimateServedTokenUsage(120000).OutputTokens = %d, want %d",
			usage.OutputTokens, DefaultPreallocCeiling)
	}
	if usage.InputTokens != 4 {
		t.Fatalf("estimateServedTokenUsage(120000).InputTokens = %d, want 4", usage.InputTokens)
	}

	// 7. Custom ceiling in estimateServedTokenUsageWithCap
	usageCustom := estimateServedTokenUsageWithCap(msgs, nil, 120000, 512)
	if usageCustom.OutputTokens != 512 {
		t.Fatalf("estimateServedTokenUsageWithCap(120000, 512).OutputTokens = %d, want 512", usageCustom.OutputTokens)
	}
}

// Acceptance criterion 2: Under a constrained TokenBudget, a single huge-max_tokens request
// no longer sheds a queue of small requests that would have fit (anti-hog witness).
func TestAdmissionAntiHog_DoesNotShedSmallRequests(t *testing.T) {
	// Budget is 2048 tokens.
	// Without preallocation cap, a 120,000-token request would require 120,000 tokens,
	// either getting shed as impossible or exhausting the whole 2048 pool and starving/shedding
	// all concurrent small requests.
	// With preallocation cap (1024), it reserves 1024 tokens, leaving 1024 tokens free for small requests.
	policy := AdmissionPolicy{
		TokenBudget:     2048,
		MaxNumSeqs:      10,
		MaxWaiting:      10,
		PreallocCeiling: 1024,
	}
	c := NewAdmissionController(policy)

	// Offer the huge request with max_tokens = 120000.
	msgs := []agent.Message{{Role: "user", Content: "query"}} // (4 + 5) / 4 = 2 tokens
	hugeTokens := estimateServedAdmissionTokensWithCap(msgs, nil, 120000, c.preallocCeiling())
	if hugeTokens != 2+1024 {
		t.Fatalf("huge request tokens = %d, want 1026", hugeTokens)
	}

	vHuge := c.Offer(SeqRequest{TraceID: "hog-request", Tokens: hugeTokens})
	if vHuge != VerdictAdmitted {
		t.Fatalf("huge request verdict = %s, want admitted (capped prealloc fits in 2048 budget)", vHuge)
	}

	// Verify controller is running the huge request with 1026 tokens in use
	st := c.Stats()
	if st.Running != 1 || st.TokensInUse != 1026 {
		t.Fatalf("after huge request: running=%d tokens=%d, want 1/1026", st.Running, st.TokensInUse)
	}

	// Now offer 4 small requests (each asking for 200 tokens). Total small requests = 800 tokens.
	// Remaining budget: 2048 - 1026 = 1022 tokens, so all 4 small requests fit and must be admitted!
	for i := 1; i <= 4; i++ {
		traceID := "small-" + string(rune('0'+i))
		smallTokens := 200
		v := c.Offer(SeqRequest{TraceID: traceID, Tokens: smallTokens})
		if v != VerdictAdmitted {
			t.Fatalf("small request %s shed or queued: verdict = %s, want admitted (anti-hog preallocation preserved budget headroom)",
				traceID, v)
		}
	}

	st = c.Stats()
	if st.Running != 5 || st.Shed != 0 {
		t.Fatalf("expected 5 running sequences and 0 shed, got running=%d shed=%d", st.Running, st.Shed)
	}
	if st.TokensInUse != 1026+800 {
		t.Fatalf("tokens in use = %d, want %d", st.TokensInUse, 1026+800)
	}
}

// Acceptance criterion 3: A long generation completes correctly by re-charging per chunk via
// lease.Grow(chunk) without premature stop.
func TestAdmissionLongGeneration_RechargesPerChunk(t *testing.T) {
	policy := AdmissionPolicy{
		TokenBudget:     4096,
		MaxNumSeqs:      4,
		MaxWaiting:      4,
		PreallocCeiling: 1024,
	}
	c := NewAdmissionController(policy)

	rateGate := NewTokenRateGate(TokenRatePolicy{
		Caps: ratelimit.TokenCaps{
			MaxOutputTokens: 4096,
		},
		Window: time.Minute,
	})

	srv := &Server{
		admissionCtl:  c,
		tokenRateGate: rateGate,
	}

	ctx := context.Background()
	turn := servedSessionTurn{traceID: "long-gen-req"}
	msgs := []agent.Message{{Role: "user", Content: "Write a long essay."}} // prompt tokens = 5

	// Requested max_tokens is 3500 (exceeds prealloc ceiling 1024)
	lease, err := srv.beginServedAdmission(ctx, turn, msgs, nil, 3500)
	if err != nil {
		t.Fatalf("beginServedAdmission failed: %v", err)
	}
	defer lease.Release()

	// Initial admission footprint: 5 prompt + 1024 cap = 1029 tokens
	initialTokens := c.Stats().TokensInUse
	if initialTokens != 1029 {
		t.Fatalf("initial tokens in use = %d, want 1029", initialTokens)
	}

	// Chunk 1: generation produces 1024 tokens and tops up for next chunk
	if err := lease.Grow(1024); err != nil {
		t.Fatalf("chunk 1 lease.Grow(1024) failed: %v", err)
	}
	if got := c.Stats().TokensInUse; got != 1029+1024 {
		t.Fatalf("after chunk 1: tokens in use = %d, want %d", got, 1029+1024)
	}

	// Chunk 2: generation produces another 1024 tokens and tops up for next chunk
	if err := lease.Grow(1024); err != nil {
		t.Fatalf("chunk 2 lease.Grow(1024) failed: %v", err)
	}
	if got := c.Stats().TokensInUse; got != 1029+2048 {
		t.Fatalf("after chunk 2: tokens in use = %d, want %d", got, 1029+2048)
	}

	// Chunk 3: attempt to grow beyond budget (1029 + 2048 + 2000 = 5077 > 4096)
	// Must return *AdmissionError with VerdictShed
	errShed := lease.Grow(2000)
	if errShed == nil {
		t.Fatal("expected lease.Grow to fail when exceeding TokenBudget, got nil")
	}
	var ae *AdmissionError
	if !errors.As(errShed, &ae) || ae.Verdict != VerdictShed {
		t.Fatalf("expected AdmissionError with VerdictShed on budget overflow, got: %v", errShed)
	}

	// Generation completes with actual usage: 5 prompt, 2800 completion = 2805 total
	lease.SettleUsage(agent.Usage{
		PromptTokens:     5,
		CompletionTokens: 2800,
		TotalTokens:      2805,
	})

	// Tokens in use in controller was reconciled to actual usage: 2805
	if got := c.Stats().TokensInUse; got != 2805 {
		t.Fatalf("after SettleUsage: tokens in use = %d, want 2805", got)
	}

	// Release frees the remaining budget cleanly
	lease.Release()
	if got := c.Stats().TokensInUse; got != 0 {
		t.Fatalf("after Release: tokens in use = %d, want 0", got)
	}
}

// Test that SettleUsage frees excess preallocated tokens back to the controller budget
// and promotes queued waiters.
func TestAdmissionSettleUsage_FreesExcessAndPromotesWaiters(t *testing.T) {
	// Budget = 1500 tokens.
	policy := AdmissionPolicy{
		TokenBudget:     1500,
		MaxNumSeqs:      4,
		MaxWaiting:      4,
		PreallocCeiling: 1024,
	}
	c := NewAdmissionController(policy)

	ctx := context.Background()

	// 1. Request 1 acquires lease: 1024 preallocated tokens
	lease1, err := c.Acquire(ctx, SeqRequest{
		TraceID: "req-1",
		Tokens:  1024,
	})
	if err != nil {
		t.Fatalf("req-1 Acquire failed: %v", err)
	}
	defer lease1.Release()

	if c.Stats().Running != 1 || c.Stats().TokensInUse != 1024 {
		t.Fatalf("req-1 running: running=%d tokens=%d, want 1/1024", c.Stats().Running, c.Stats().TokensInUse)
	}

	// 2. Request 2 arrives asking for 800 tokens.
	// 1024 + 800 = 1824 > 1500, so Request 2 cannot be admitted now and is queued.
	acquiredCh := make(chan *AdmissionLease, 1)
	errCh := make(chan error, 1)
	go func() {
		lease2, err := c.Acquire(ctx, SeqRequest{
			TraceID: "req-2",
			Tokens:  800,
		})
		if err != nil {
			errCh <- err
			return
		}
		acquiredCh <- lease2
	}()

	// Wait briefly for req-2 to be queued
	time.Sleep(20 * time.Millisecond)
	st := c.Stats()
	if st.Waiting != 1 || st.Running != 1 {
		t.Fatalf("expected req-2 queued: waiting=%d running=%d, want 1/1", st.Waiting, st.Running)
	}

	// 3. Request 1 actually finishes with only 150 tokens (over-estimated by 1024 - 150 = 874 tokens!)
	// SettleUsage reports actual usage = 150 tokens.
	lease1.SettleUsage(agent.Usage{
		PromptTokens:     50,
		CompletionTokens: 100,
		TotalTokens:      150,
	})

	// 4. SettleUsage reconciled c.tokens to 150 and called scheduleLocked().
	// Headroom is now 1500 - 150 = 1350 tokens, which easily accommodates req-2 (800 tokens).
	// req-2 MUST be promoted to running immediately!
	select {
	case lease2 := <-acquiredCh:
		if lease2 == nil {
			t.Fatal("req-2 acquired nil lease")
		}
		defer lease2.Release()
	case err := <-errCh:
		t.Fatalf("req-2 Acquire returned unexpected error: %v", err)
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for req-2 to be promoted after req-1 settled excess tokens")
	}

	st = c.Stats()
	if st.Waiting != 0 || st.Running != 2 {
		t.Fatalf("after promotion: waiting=%d running=%d, want 0/2", st.Waiting, st.Running)
	}
	// Total tokens in use: 150 (req-1 settled) + 800 (req-2) = 950
	if st.TokensInUse != 950 {
		t.Fatalf("tokens in use = %d, want 950 (150 settled + 800 req-2)", st.TokensInUse)
	}
}

// Test TokenRateGate and TokenReservation Grow and Rollback functionality.
func TestTokenRateGate_GrowAndRollback(t *testing.T) {
	gate := NewTokenRateGate(TokenRatePolicy{
		Caps: ratelimit.TokenCaps{
			MaxOutputTokens: 2000,
			MaxTotalTokens:  3000,
		},
		Window: time.Minute,
	})

	res, err := gate.Admit(ratelimit.NewTokenUsage(100, 0, 500))
	if err != nil {
		t.Fatalf("Admit failed: %v", err)
	}
	defer res.Release()

	// Grow by 500 output tokens: 500 + 500 = 1000 <= 2000
	if err := res.Grow(500); err != nil {
		t.Fatalf("res.Grow(500) failed: %v", err)
	}
	snap := gate.Snapshot()
	if snap.Reserved.OutputTokens != 1000 {
		t.Fatalf("reserved output tokens = %d, want 1000", snap.Reserved.OutputTokens)
	}

	// Grow by 1500 output tokens: 1000 + 1500 = 2500 > 2000 (exceeds cap) -> must shed
	errOverflow := res.Grow(1500)
	if errOverflow == nil {
		t.Fatal("expected res.Grow to fail on output token cap overflow, got nil")
	}
	// Reserved output tokens should still be 1000
	snap = gate.Snapshot()
	if snap.Reserved.OutputTokens != 1000 {
		t.Fatalf("reserved output tokens after failed grow = %d, want 1000", snap.Reserved.OutputTokens)
	}

	// Negative grow refunds output tokens
	if err := gate.Grow(res.id, -400); err != nil {
		t.Fatalf("gate.Grow(-400) failed: %v", err)
	}
	snap = gate.Snapshot()
	if snap.Reserved.OutputTokens != 600 {
		t.Fatalf("reserved output tokens after refund = %d, want 600", snap.Reserved.OutputTokens)
	}
}

func TestTokenReservation_LinkedRollback(t *testing.T) {
	gateOuter := NewTokenRateGate(TokenRatePolicy{
		Caps:   ratelimit.TokenCaps{MaxOutputTokens: 2000},
		Window: time.Minute,
	})
	gateInner := NewTokenRateGate(TokenRatePolicy{
		Caps:   ratelimit.TokenCaps{MaxOutputTokens: 600},
		Window: time.Minute,
	})

	resOuter, err := gateOuter.Admit(ratelimit.NewTokenUsage(0, 0, 500))
	if err != nil {
		t.Fatalf("gateOuter.Admit failed: %v", err)
	}
	defer resOuter.Release()

	resInner, err := gateInner.Admit(ratelimit.NewTokenUsage(0, 0, 500))
	if err != nil {
		t.Fatalf("gateInner.Admit failed: %v", err)
	}
	defer resInner.Release()

	resOuter.linked = resInner

	// Grow by 200: outer fits (500+200 = 700 <= 2000), but inner fails (500+200 = 700 > 600).
	// Because inner fails, outer MUST be rolled back to 500!
	err = resOuter.Grow(200)
	if err == nil {
		t.Fatal("expected Grow to fail due to inner linked gate, got nil")
	}

	snapOuter := gateOuter.Snapshot()
	if snapOuter.Reserved.OutputTokens != 500 {
		t.Fatalf("expected outer reserved output tokens rolled back to 500, got %d", snapOuter.Reserved.OutputTokens)
	}
	if resOuter.estimate.OutputTokens != 500 {
		t.Fatalf("expected resOuter estimate output tokens rolled back to 500, got %d", resOuter.estimate.OutputTokens)
	}
}

func TestAdmissionControllerGrow_SessionPool(t *testing.T) {
	pool := session.NewPool(2000)
	ctl := NewAdmissionController(AdmissionPolicy{
		TokenBudget: 5000,
		MaxNumSeqs:  4,
	})
	ctl.SetFleet(pool)

	v := ctl.Offer(SeqRequest{TraceID: "t1", Tokens: 1000})
	if v != VerdictAdmitted {
		t.Fatalf("offer verdict = %s, want admitted", v)
	}
	if rem := pool.Remaining(); rem != 1000 {
		t.Fatalf("pool remaining = %d, want 1000", rem)
	}

	// Grow by 500 fits in pool (rem becomes 500)
	if err := ctl.Grow("t1", 500); err != nil {
		t.Fatalf("ctl.Grow(500) failed: %v", err)
	}
	if rem := pool.Remaining(); rem != 500 {
		t.Fatalf("pool remaining = %d, want 500", rem)
	}
	if ctl.Stats().TokensInUse != 1500 {
		t.Fatalf("ctl tokens in use = %d, want 1500", ctl.Stats().TokensInUse)
	}

	// Grow by 600 exceeds remaining pool (500) -> fails with VerdictShed
	err := ctl.Grow("t1", 600)
	if err == nil {
		t.Fatal("expected ctl.Grow(600) to fail on pool exhaustion, got nil")
	}
	var ae *AdmissionError
	if !errors.As(err, &ae) || ae.Verdict != VerdictShed {
		t.Fatalf("expected AdmissionError with VerdictShed, got %v", err)
	}
	if rem := pool.Remaining(); rem != 500 {
		t.Fatalf("pool remaining after failed grow = %d, want 500", rem)
	}

	// Settle over-estimate: actual tokens used was only 1200 (1500 - 1200 = 300 returned to pool)
	ctl.Settle("t1", 1200)
	if rem := pool.Remaining(); rem != 800 {
		t.Fatalf("pool remaining after settle = %d, want 800", rem)
	}
	if ctl.Stats().TokensInUse != 1200 {
		t.Fatalf("ctl tokens in use after settle = %d, want 1200", ctl.Stats().TokensInUse)
	}

	// Complete frees request
	ctl.Complete("t1")
	if ctl.Stats().TokensInUse != 0 {
		t.Fatalf("ctl tokens in use after complete = %d, want 0", ctl.Stats().TokensInUse)
	}
}

func TestAdmissionPrealloc_NilSafety(t *testing.T) {
	var lease *AdmissionLease
	if err := lease.Grow(100); err != nil {
		t.Fatalf("nil lease.Grow failed: %v", err)
	}
	lease.SettleUsage(agent.Usage{})

	var res *TokenReservation
	if err := res.Grow(100); err != nil {
		t.Fatalf("nil res.Grow failed: %v", err)
	}

	var gate *TokenRateGate
	if err := gate.Grow(1, 100); err != nil {
		t.Fatalf("nil gate.Grow failed: %v", err)
	}

	var ctl *AdmissionController
	if err := ctl.Grow("foo", 100); err != nil {
		t.Fatalf("nil ctl.Grow failed: %v", err)
	}
	ctl.Settle("foo", 100)
	if ceil := ctl.preallocCeiling(); ceil != DefaultPreallocCeiling {
		t.Fatalf("nil ctl.preallocCeiling = %d, want %d", ceil, DefaultPreallocCeiling)
	}

	p := AdmissionPolicy{}
	if ceil := p.preallocCeiling(); ceil != DefaultPreallocCeiling {
		t.Fatalf("zero policy preallocCeiling = %d, want %d", ceil, DefaultPreallocCeiling)
	}
}
