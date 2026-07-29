package gateway

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"
)

// principalTestPrompt is a 200-char prompt (4 chars of role + 196 of content), which
// estimateServedTokenUsage charges as 50 uncached input tokens; with the maxTokens=1
// output floor every served turn below costs exactly 51 total tokens.
func principalTestPrompt() []agent.Message {
	return []agent.Message{{Role: "user", Content: strings.Repeat("x", 196)}}
}

// TestPrincipalAllotmentExhaustionSparesOtherTenants is the #5379 headline witness: one
// principal draining its whole allotment must not refuse a DIFFERENT principal's request.
// Before this change a single TokenRateGate held the whole server's budget, so tenant A's
// exhaustion was tenant B's 429 — the exact noisy-neighbour break the issue names.
func TestPrincipalAllotmentExhaustionSparesOtherTenants(t *testing.T) {
	book := NewPrincipalTokenRates(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 100}}, 0)

	// Tenant A drains its allotment to the cap and lets the truth settle there.
	resA, err := book.Admit("org-a", ratelimit.NewTokenUsage(100, 0, 0))
	if err != nil {
		t.Fatalf("org-a first Admit err = %v, want admit against an empty allotment", err)
	}
	resA.Settle(ratelimit.NewTokenUsage(100, 0, 0))

	// A's NEXT request is shed.
	_, err = book.Admit("org-a", ratelimit.NewTokenUsage(1, 0, 0))
	var ae *AdmissionError
	if !errors.As(err, &ae) || ae.Verdict != VerdictShed {
		t.Fatalf("org-a over-cap Admit err = %v, want a VerdictShed *AdmissionError", err)
	}

	// THE WITNESS, asserted before anything cosmetic so a regression reds HERE: tenant B
	// is untouched by tenant A's exhaustion.
	resB, err := book.Admit("org-b", ratelimit.NewTokenUsage(100, 0, 0))
	if err != nil || resB == nil {
		t.Fatalf("org-b Admit = (%v, %v), want admit — org-a's exhausted allotment must not refuse a different principal", resB, err)
	}

	// The refusal names A's own budget — not "provider", which would misdirect an operator
	// to the shared window.
	if !strings.Contains(ae.Reason, `principal "org-a"`) || !strings.Contains(ae.Reason, ratelimit.TokenCapTotalTokens) {
		t.Fatalf("shed reason = %q, want it to name principal \"org-a\" and the %q cap", ae.Reason, ratelimit.TokenCapTotalTokens)
	}

	// The two allotments are genuinely separate windows, not one window read twice.
	if got := book.SnapshotFor("org-a"); got.Settled.TotalTokens() != 100 || got.InFlight != 0 {
		t.Fatalf("org-a snapshot = %+v, want 100 settled and nothing in flight", got)
	}
	if got := book.SnapshotFor("org-b"); got.Settled.TotalTokens() != 0 || got.InFlight != 1 {
		t.Fatalf("org-b snapshot = %+v, want an empty settled window with 1 reservation in flight", got)
	}
	if named, total := book.AllotmentCounts(); named != 2 || total != 2 {
		t.Fatalf("allotment counts = (%d named, %d total), want exactly the two named tenants", named, total)
	}
}

// TestUnidentifiedCallersShareOneAllotment pins the decision the issue leaves open: a
// request carrying NO principal is charged to a single shared allotment, capped exactly
// like a tenant's. The alternative — a fresh allotment per unidentified caller — would
// make the whole limit bypassable by simply omitting identity, so the uncertainty is
// resolved toward metering rather than toward admitting.
func TestUnidentifiedCallersShareOneAllotment(t *testing.T) {
	book := NewPrincipalTokenRates(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 100}}, 0)

	res, err := book.Admit("", ratelimit.NewTokenUsage(100, 0, 0))
	if err != nil {
		t.Fatalf("unidentified Admit err = %v, want admit against an empty shared allotment", err)
	}
	res.Settle(ratelimit.NewTokenUsage(100, 0, 0))

	// A SECOND unidentified caller lands in the SAME allotment — omitting identity buys
	// no fresh budget.
	_, err = book.Admit("", ratelimit.NewTokenUsage(1, 0, 0))
	var ae *AdmissionError
	if !errors.As(err, &ae) || !strings.Contains(ae.Reason, "unidentified caller") {
		t.Fatalf("second unidentified Admit err = %v, want a shed naming the unidentified caller's shared budget", err)
	}

	// A whitespace-only principal is unidentified too, not a distinct tenant named " " —
	// otherwise the shared allotment is bypassed with a single space.
	if _, err = book.Admit("   ", ratelimit.NewTokenUsage(1, 0, 0)); err == nil {
		t.Fatal("whitespace-only principal Admit = admit, want the same shed as the empty principal")
	}

	// The shared allotment's saturation is confined to unidentified callers: an
	// authenticated tenant is unaffected.
	if _, err = book.Admit("org-a", ratelimit.NewTokenUsage(100, 0, 0)); err != nil {
		t.Fatalf("org-a Admit err = %v, want admit — the unidentified allotment must not bill named tenants", err)
	}
	if named, total := book.AllotmentCounts(); named != 1 || total != 2 {
		t.Fatalf("allotment counts = (%d named, %d total), want 1 named tenant plus the single shared unidentified allotment", named, total)
	}
}

// TestPrincipalAllotmentCeilingBoundsBookGrowth witnesses the map bound. With no keyset
// armed the principal is caller-supplied (principalFor reads X-Fak-Principal), so a book
// keyed by it would otherwise grow without limit — the rate limiter becoming the denial of
// service. Two mechanisms hold it: principals past the ceiling fold into ONE shared
// overflow allotment (still metered — never a fresh, unbounded-in-effect budget), and an
// allotment with no in-flight reservation whose window has already rolled is swept, since
// dropping it forgives no budget.
func TestPrincipalAllotmentCeilingBoundsBookGrowth(t *testing.T) {
	book := NewPrincipalTokenRates(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 100}}, 3)
	clock := time.Unix(1_700_000_000, 0)
	book.now = func() time.Time { return clock }

	// Three named tenants, each holding a live reservation so none is sweepable.
	held := make([]*TokenReservation, 0, 3)
	for i := 0; i < 3; i++ {
		res, err := book.Admit(fmt.Sprintf("tenant-%d", i), ratelimit.NewTokenUsage(10, 0, 0))
		if err != nil {
			t.Fatalf("tenant-%d Admit err = %v, want admit under the ceiling", i, err)
		}
		held = append(held, res)
	}
	if named, total := book.AllotmentCounts(); named != 3 || total != 3 {
		t.Fatalf("allotment counts = (%d, %d), want the 3 named tenants at the ceiling", named, total)
	}

	// A FOURTH distinct principal cannot get its own allotment, so it is metered in the
	// shared overflow budget — and that budget is a real cap, not a formality.
	if _, err := book.Admit("tenant-overflow-a", ratelimit.NewTokenUsage(100, 0, 0)); err != nil {
		t.Fatalf("overflow Admit err = %v, want the ceiling to fold a new principal into the shared overflow allotment", err)
	}
	var ae *AdmissionError
	_, err := book.Admit("tenant-overflow-b", ratelimit.NewTokenUsage(1, 0, 0))
	if !errors.As(err, &ae) || !strings.Contains(ae.Reason, "shared overflow principal") {
		t.Fatalf("saturated overflow Admit err = %v, want a shed naming the shared overflow budget", err)
	}

	// Minting principals is therefore not a growth lever: 500 fresh identities add no
	// gates at all.
	for i := 0; i < 500; i++ {
		_, _ = book.Admit(fmt.Sprintf("mint-%d", i), ratelimit.NewTokenUsage(1, 0, 0))
	}
	if named, total := book.AllotmentCounts(); named != 3 || total != 4 {
		t.Fatalf("allotment counts after 500 minted principals = (%d named, %d total), want (3, 4) — the ceiling plus one shared overflow allotment", named, total)
	}

	// Now the three tenants go quiet and their windows roll: the allotments forgive no
	// budget on eviction, so the sweep reclaims them and a genuinely new tenant gets its
	// own allotment back instead of being punished with overflow.
	for _, res := range held {
		res.Settle(ratelimit.NewTokenUsage(10, 0, 0))
	}
	clock = clock.Add(defaultTokenRateWindow)
	if _, err := book.Admit("tenant-late", ratelimit.NewTokenUsage(100, 0, 0)); err != nil {
		t.Fatalf("post-sweep Admit err = %v, want a reclaimed allotment for a new tenant", err)
	}
	if named, _ := book.AllotmentCounts(); named != 1 {
		t.Fatalf("named allotments after the idle sweep = %d, want 1 (the three idle ones reclaimed)", named)
	}
	if got := book.SnapshotFor("tenant-late"); got.Settled.TotalTokens() != 0 {
		t.Fatalf("tenant-late snapshot = %+v, want a fresh window — a reclaimed slot must not inherit settled usage", got)
	}
}

// TestServedAdmissionIsolatesTenantsEndToEnd is the served-path seam witness: the book
// composes with the shared provider gate through beginServedAdmission, a tenant over its
// own cap gets the typed 429, and — the property that matters — a DIFFERENT tenant is
// still admitted. It also pins the lockstep settlement: the provider's real usage lands in
// both the tenant's allotment and the shared provider window.
func TestServedAdmissionIsolatesTenantsEndToEnd(t *testing.T) {
	srv := newTestServer(t)
	shared := NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 100_000}})
	srv.SetTokenRateGate(shared)
	book := NewPrincipalTokenRates(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 60}}, 0)
	srv.SetPrincipalTokenRates(book)

	turn := servedSessionTurn{traceID: "per-principal", state: SessionState{Priority: 0}, maxTokens: 1}
	msgs := principalTestPrompt() // 51 estimated total tokens per turn
	ctxA := WithPrincipal(context.Background(), "org-a")
	ctxB := WithPrincipal(context.Background(), "org-b")

	leaseA, err := srv.beginServedAdmission(ctxA, turn, msgs, nil, 1)
	if err != nil || leaseA == nil {
		t.Fatalf("org-a beginServedAdmission = (%v, %v), want an admitted lease", leaseA, err)
	}
	leaseA.SettleUsage(agent.Usage{PromptTokens: 55, CompletionTokens: 5}) // 60 total: A's cap
	leaseA.Release()
	if got := book.SnapshotFor("org-a"); got.Settled.TotalTokens() != 60 {
		t.Fatalf("org-a allotment after settle = %+v, want the provider's real 60 tokens charged to the tenant", got)
	}
	if got := shared.Snapshot(); got.Settled.TotalTokens() != 60 {
		t.Fatalf("shared window after settle = %+v, want the same 60 tokens settled in lockstep", got)
	}

	// A is at its cap: the next A turn is a typed 429 naming A.
	_, err = srv.beginServedAdmission(ctxA, turn, msgs, nil, 1)
	status, code, _, ok := admissionErrorStatus(err)
	if !ok || status != http.StatusTooManyRequests || code != "scheduler_overloaded" {
		t.Fatalf("org-a shed mapping = (%d, %q, %v), want (429, scheduler_overloaded, true); err=%v", status, code, ok, err)
	}
	shedErr := err

	// THE WITNESS, end to end and asserted first: org-b is admitted while org-a is shed.
	leaseB, err := srv.beginServedAdmission(ctxB, turn, msgs, nil, 1)
	if err != nil || leaseB == nil {
		t.Fatalf("org-b beginServedAdmission = (%v, %v), want admit — org-a exhausting its allotment must not 429 org-b", leaseB, err)
	}
	var ae *AdmissionError
	if !errors.As(shedErr, &ae) || !strings.Contains(ae.Reason, `principal "org-a"`) {
		t.Fatalf("org-a shed reason = %v, want the refusal to name the principal whose cap fired", shedErr)
	}
	if got := book.SnapshotFor("org-b"); got.InFlight != 1 || got.Settled.TotalTokens() != 0 {
		t.Fatalf("org-b allotment = %+v, want its own empty window with 1 reservation in flight", got)
	}
	leaseB.Release()

	// With no book installed the path is byte-for-byte historical: a fresh server with
	// only the shared gate still admits, so this is additive rather than a semantic swap.
	plain := newTestServer(t)
	plain.SetTokenRateGate(NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 100_000}}))
	if lease, err := plain.beginServedAdmission(ctxA, turn, msgs, nil, 1); err != nil || lease == nil {
		t.Fatalf("unwired-book beginServedAdmission = (%v, %v), want the historical single-budget admit", lease, err)
	}
}

// TestServedAdmissionOrdersPrincipalBeforeProviderWindow pins the two ordering
// consequences. (1) The tenant's own allotment is consulted FIRST, so a tenant already
// over its cap is shed without pushing the shared window any closer to saturation — the
// refusal names the principal, not the provider. (2) When the SHARED window is the one
// that refuses, the allotment the tenant held for an instant is cancelled rather than
// charged: a busy neighbour saturating the provider must not also spend this tenant's
// budget, which would starve it twice over.
func TestServedAdmissionOrdersPrincipalBeforeProviderWindow(t *testing.T) {
	turn := servedSessionTurn{traceID: "ordering", state: SessionState{Priority: 0}, maxTokens: 1}
	msgs := principalTestPrompt() // 51 estimated total tokens
	ctxA := WithPrincipal(context.Background(), "org-a")

	// Both budgets are too small for the turn, so either could refuse it; the reason names
	// whichever was consulted first.
	both := newTestServer(t)
	both.SetTokenRateGate(NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 10}}))
	both.SetPrincipalTokenRates(NewPrincipalTokenRates(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 10}}, 0))
	_, err := both.beginServedAdmission(ctxA, turn, msgs, nil, 1)
	var ae *AdmissionError
	if !errors.As(err, &ae) {
		t.Fatalf("doubly-over-budget beginServedAdmission err = %v, want a typed *AdmissionError", err)
	}
	if !strings.Contains(ae.Reason, `principal "org-a"`) || strings.Contains(ae.Reason, "provider token budget") {
		t.Fatalf("shed reason = %q, want the PER-PRINCIPAL allotment consulted before the shared provider window", ae.Reason)
	}

	// Roomy tenant allotment, saturated provider window: the shared gate refuses.
	starved := newTestServer(t)
	starved.SetTokenRateGate(NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 10}}))
	book := NewPrincipalTokenRates(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 10_000}}, 0)
	starved.SetPrincipalTokenRates(book)
	_, err = starved.beginServedAdmission(ctxA, turn, msgs, nil, 1)
	if !errors.As(err, &ae) || !strings.Contains(ae.Reason, "provider token budget") {
		t.Fatalf("provider-saturated beginServedAdmission err = %v, want the shared provider window to name itself", err)
	}
	if got := book.SnapshotFor("org-a"); got.InFlight != 0 || got.Settled != (ratelimit.TokenUsage{}) {
		t.Fatalf("org-a allotment after a PROVIDER shed = %+v, want nothing reserved and nothing charged — the request never reached the provider", got)
	}
}

// TestPrincipalTokenRatesConcurrentAdmitIsRaceFree hammers the book from many goroutines
// across more identities than the ceiling admits, so the admit path, the idle sweep and
// the settle path all run concurrently. Meaningful under -race; the bound assertion holds
// either way.
func TestPrincipalTokenRatesConcurrentAdmitIsRaceFree(t *testing.T) {
	const ceiling = 8
	book := NewPrincipalTokenRates(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 1_000_000}}, ceiling)

	var wg sync.WaitGroup
	for worker := 0; worker < 24; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := 0; i < 40; i++ {
				principal := fmt.Sprintf("tenant-%d", (worker*7+i)%50)
				if i%5 == 0 {
					principal = "" // the shared unidentified allotment, contended too
				}
				res, err := book.Admit(principal, ratelimit.NewTokenUsage(1, 0, 0))
				if err != nil {
					continue
				}
				if i%2 == 0 {
					res.Settle(ratelimit.NewTokenUsage(1, 0, 0))
					continue
				}
				res.Release()
			}
			_ = book.SnapshotFor(fmt.Sprintf("tenant-%d", worker))
		}(worker)
	}
	wg.Wait()

	named, total := book.AllotmentCounts()
	if named > ceiling || total > ceiling+2 {
		t.Fatalf("allotment counts under concurrency = (%d named, %d total), want at most (%d, %d)", named, total, ceiling, ceiling+2)
	}
}
