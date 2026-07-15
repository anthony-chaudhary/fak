// star_fan_race_test.go — the #1493 QA-box-3 RACE witness for M2 star-anchor
// first-natural-request warming: racing sibling senders fan onto the star
// anchor only after the first natural request's FIRST STREAMED CONTENT DELTA
// opens the send-one-then-fan barrier — never on the HTTP status or
// message_start — and every fanned sibling re-derives the byte-identical
// canonical anchor prefix (the applied layout equals RecommendLayout's
// recommendation), so the fan reads the anchor the first request wrote.
//
// External test package ON PURPOSE: it couples the M2 star plan (vcachestar)
// to the send-one-then-fan barrier (vcachewarm) in a test only, adding no
// production import edge between the two decision layers. HONEST SCOPE: this
// witnesses the barrier discipline and the anchor byte-identity at the
// decision layer under -race; it moves no live HTTP traffic (the live fan
// wiring is the serve-side work tracked by the epic's live-loop siblings).
package vcachestar_test

import (
	"bytes"
	"reflect"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/vcachestar"
	"github.com/anthony-chaudhary/fak/internal/vcachewarm"
)

func fanScope() vcachestar.Identity {
	return vcachestar.Identity{
		ModelID:          "claude-sonnet-4.6",
		TokenizerEpoch:   "tok-2026-06",
		BreakpointLayout: "system-last-block",
		TTL:              "5m",
		ProviderSurface:  "anthropic:first-party",
	}
}

// fanParts builds one star request: a per-request volatile id (DIFFERENT for
// every sibling — the hoist must keep it out of the shared anchor), the shared
// stable system charter and tool schema, and a small per-unit question.
func fanParts(requestID string) []vcachestar.Part {
	return []vcachestar.Part{
		{Section: vcachestar.SectionMessages, Kind: cachemeta.SegVolatile, Content: []byte("request_id=" + requestID + "\n"), Tokens: 1},
		{Section: vcachestar.SectionSystem, Content: []byte("system charter\n"), Tokens: 900},
		{Section: vcachestar.SectionTools, Name: "search", Content: []byte(`{"b":2,"a":1}`), Tokens: 200},
		{Section: vcachestar.SectionMessages, Content: []byte("unit question\n"), Tokens: 10},
	}
}

func TestFirstNaturalWarmFansSiblingsOnFirstStreamedDeltaNotHTTPStatus(t *testing.T) {
	const siblings = 8

	// The FIRST NATURAL REQUEST: preflight applies RecommendLayout (hoists the
	// volatile id to the tail) and yields the canonical anchor prefix bytes.
	first := vcachestar.Preflight(vcachestar.PreflightRequest{
		Scope:           fanScope(),
		MinAnchorTokens: 512,
		Parts:           fanParts("r-first"),
	})
	if first.Action != vcachestar.ActionRewrite {
		t.Fatalf("first preflight action = %q (%s), want rewrite", first.Action, first.Reason)
	}
	// The APPLIED layout IS the recommendation — applied, not just reported.
	if !reflect.DeepEqual(first.Applied, first.Recommendation.Reordered) {
		t.Fatalf("applied layout diverges from RecommendLayout recommendation:\napplied  %+v\nwant     %+v", first.Applied, first.Recommendation.Reordered)
	}

	// M2 plans the anchor with FIRST-NATURAL warming: no dedicated warm is
	// spent; the first real request writes the anchor and siblings read it.
	plan := vcachestar.Plan(vcachestar.StarRequest{
		Key:                  first.Key,
		MinAnchorTokens:      512,
		UnitTokens:           10,
		ExpectedSiblingReads: siblings,
	})
	if plan.Strategy != vcachestar.StrategyFirstNaturalWarm || !plan.FirstNaturalRequestWarms {
		t.Fatalf("plan = %+v, want first-natural-request warming", plan)
	}
	if plan.DedicatedWarm {
		t.Fatalf("M2 star anchor spent a dedicated warm: %+v", plan)
	}

	// SEND-ONE-THEN-FAN: siblings race the first request's stream reader on the
	// shared barrier (concurrent Observe/Released — meaningful under -race) and
	// hold until it opens.
	gate := new(vcachewarm.FanoutGate)
	results := make([]vcachestar.PreflightResult, siblings)
	var fanned atomic.Int32
	var wg sync.WaitGroup
	deadline := time.Now().Add(5 * time.Second)
	for i := 0; i < siblings; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for time.Now().Before(deadline) {
				if gate.Released() {
					fanned.Add(1)
					// Each sibling carries its OWN volatile id but proposes the
					// first request's canonical prefix as its warm candidate:
					// preflight must accept it as byte-identical, or the sibling
					// would re-write instead of reading the shared anchor.
					results[i] = vcachestar.Preflight(vcachestar.PreflightRequest{
						Scope:              fanScope(),
						MinAnchorTokens:    512,
						Parts:              fanParts("r-sib-" + strconv.Itoa(i)),
						WarmCandidateBytes: first.PrefixBytes,
					})
					return
				}
				runtime.Gosched()
			}
		}(i)
	}

	// HTTP 200 and message_start arrive first; the barrier must HOLD — a 200
	// proves nothing about the provider having begun to cache the prefix.
	for _, event := range []vcachewarm.StreamEventKind{vcachewarm.StreamEventHTTPStatus, vcachewarm.StreamEventMessageStart} {
		if gate.Observe(event) {
			t.Fatalf("barrier released on %q before the first streamed content delta", event)
		}
	}
	time.Sleep(20 * time.Millisecond) // give a broken barrier the chance to leak a sibling
	if n := fanned.Load(); n != 0 {
		t.Fatalf("%d sibling(s) fanned before the first streamed content delta", n)
	}

	// The first streamed content delta opens the barrier: the whole fan goes.
	if !gate.Observe(vcachewarm.StreamEventContentDelta) {
		t.Fatal("barrier did not release on the first streamed content delta")
	}
	wg.Wait()
	if n := fanned.Load(); n != siblings {
		t.Fatalf("fanned = %d, want all %d siblings released after the first streamed content delta", n, siblings)
	}

	// Every fanned sibling landed on the SAME anchor the first request warmed:
	// byte-identical canonical prefix, matching manifest key, no refusal.
	for i, sib := range results {
		if sib.Action == vcachestar.ActionRefuse {
			t.Fatalf("sibling %d refused after fan: %s", i, sib.Reason)
		}
		if !bytes.Equal(sib.PrefixBytes, first.PrefixBytes) {
			t.Fatalf("sibling %d anchor prefix diverged from the warmed anchor:\n got %q\nwant %q", i, sib.PrefixBytes, first.PrefixBytes)
		}
		if ok, reason := sib.Key.Match(first.Key); !ok {
			t.Fatalf("sibling %d manifest key missed the warmed anchor: %s", i, reason)
		}
	}
}
