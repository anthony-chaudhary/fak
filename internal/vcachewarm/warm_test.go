package vcachewarm

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func fp(id string) PrefixFingerprint {
	return PrefixFingerprint{
		SerializerID: "chatml-v1",
		Digest:       id,
		Bytes:        4096,
		Tokens:       1024,
	}
}

func baseAnthropic() Request {
	return Request{
		Provider:               ProviderAnthropic,
		ActiveCapability:       ActiveCacheCapabilitySupported,
		ExpectedReuseBeforeTTL: 2,
		ReadDiscount:           0.1,
		WarmPrefix:             fp("same"),
		RealPrefix:             fp("same"),
		SharedBlockCount:       3,
	}
}

func TestAnthropicMaxTokensZeroUsesLastSharedBlock(t *testing.T) {
	dec := Plan(baseAnthropic())
	if dec.Primitive != PrimitiveAnthropicMaxTokens0 {
		t.Fatalf("primitive = %q, want %q (%s)", dec.Primitive, PrimitiveAnthropicMaxTokens0, dec.Reason)
	}
	if !dec.Dedicated {
		t.Fatal("max_tokens:0 warm must be recorded as dedicated")
	}
	if dec.MaxTokens != 0 {
		t.Fatalf("MaxTokens = %d, want 0", dec.MaxTokens)
	}
	if dec.BreakpointBlockIndex != 2 {
		t.Fatalf("BreakpointBlockIndex = %d, want last shared block 2", dec.BreakpointBlockIndex)
	}
}

func TestAnthropicRejectedCombinationsFallbackToDecodeOne(t *testing.T) {
	cases := []struct {
		name string
		mut  func(*Request)
	}{
		{"stream", func(r *Request) { r.Stream = true }},
		{"thinking", func(r *Request) { r.ExtendedThinking = true }},
		{"structured output", func(r *Request) { r.StructuredOutput = true }},
		{"tool choice tool", func(r *Request) { r.ToolChoice = ToolChoiceTool }},
		{"tool choice any", func(r *Request) { r.ToolChoice = ToolChoiceAny }},
		{"batch", func(r *Request) { r.Batch = true }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := baseAnthropic()
			tc.mut(&req)
			dec := Plan(req)
			if dec.Primitive != PrimitiveDecode1 {
				t.Fatalf("primitive = %q, want decode-1 fallback (%s)", dec.Primitive, dec.Reason)
			}
			if dec.MaxTokens != 1 || !dec.FallbackFromExplicit {
				t.Fatalf("fallback evidence = max_tokens %d fallback %v", dec.MaxTokens, dec.FallbackFromExplicit)
			}
			if len(dec.RejectedExplicit) == 0 {
				t.Fatal("expected rejected explicit combination evidence")
			}
		})
	}
}

func TestDecodeOneRequiresByteIdenticalPrefix(t *testing.T) {
	req := Request{
		Provider:               "implicit-provider",
		ActiveCapability:       ActiveCacheCapabilitySupported,
		ExpectedReuseBeforeTTL: 3,
		ReadDiscount:           0.5,
		WarmPrefix:             fp("warm"),
		RealPrefix:             fp("real"),
	}
	dec := Plan(req)
	if dec.Primitive != PrimitiveNone {
		t.Fatalf("primitive = %q, want none", dec.Primitive)
	}
	if dec.Reason != ReasonPrefixFingerprintMismatch {
		t.Fatalf("reason = %q, want prefix mismatch", dec.Reason)
	}

	req.WarmPrefix = PrefixFingerprint{}
	req.RealPrefix = PrefixFingerprint{}
	dec = Plan(req)
	if dec.Primitive != PrimitiveNone || dec.Reason != ReasonPrefixFingerprintMismatch {
		t.Fatalf("empty fingerprint decision = %q/%q, want none/prefix mismatch", dec.Primitive, dec.Reason)
	}
}

func TestBreakEvenGateIsStrict(t *testing.T) {
	openAIRequired, ok := DedicatedDecode1ReuseFloor(0.5)
	if !ok || openAIRequired != 3 {
		t.Fatalf("r=0.5 floor = %d/%v, want 3/true", openAIRequired, ok)
	}
	anthropicRequired, ok := DedicatedDecode1ReuseFloor(0.1)
	if !ok || anthropicRequired != 2 {
		t.Fatalf("r=0.1 floor = %d/%v, want 2/true", anthropicRequired, ok)
	}

	req := Request{
		Provider:               "implicit-provider",
		ActiveCapability:       ActiveCacheCapabilitySupported,
		ExpectedReuseBeforeTTL: 2,
		ReadDiscount:           0.5,
		WarmPrefix:             fp("same"),
		RealPrefix:             fp("same"),
	}
	if dec := Plan(req); dec.Primitive != PrimitiveNone || dec.Reason != ReasonBelowBreakEven {
		t.Fatalf("k=2 r=0.5 decision = %q/%q, want none/below-break-even", dec.Primitive, dec.Reason)
	}
	req.ExpectedReuseBeforeTTL = 3
	if dec := Plan(req); dec.Primitive != PrimitiveDecode1 {
		t.Fatalf("k=3 r=0.5 primitive = %q, want decode-1", dec.Primitive)
	}
}

func TestAutoCacheProvidersNeverSpendDecodeOne(t *testing.T) {
	for _, provider := range []Provider{ProviderOpenAI, ProviderDeepSeek} {
		t.Run(string(provider), func(t *testing.T) {
			dec := Plan(Request{
				Provider:               provider,
				ActiveCapability:       ActiveCacheCapabilitySupported,
				ExpectedReuseBeforeTTL: 99,
				ReadDiscount:           0.5,
				WarmPrefix:             fp("same"),
				RealPrefix:             fp("same"),
			})
			if dec.Primitive != PrimitiveOrderFirstReal {
				t.Fatalf("primitive = %q, want order-first-real", dec.Primitive)
			}
			if dec.Dedicated || dec.MaxTokens == 1 {
				t.Fatalf("auto-cache provider spent a dedicated decode-1 warm: %+v", dec)
			}
		})
	}
}

func TestFanoutGateReleasesOnlyOnContentDelta(t *testing.T) {
	var gate FanoutGate
	for _, event := range []StreamEventKind{
		StreamEventHTTPStatus,
		StreamEventMessageStart,
		StreamEventMessageDelta,
	} {
		if gate.Observe(event) {
			t.Fatalf("gate released on %q before content delta", event)
		}
	}
	if !gate.Observe(StreamEventContentDelta) {
		t.Fatal("gate did not release on first content delta")
	}
	if !gate.Released() {
		t.Fatal("Released returned false after content delta")
	}
}

// TestFanoutGateHoldsRacingSiblingsUntilFirstContentDelta is the #1493 QA-box-3
// race witness at the barrier itself: sibling goroutines race the first
// request's stream reader — Observe and Released run concurrently, so the test
// is meaningful under -race — and nobody fans on the HTTP status or
// message_start; the whole fan releases only on the first streamed content
// delta.
func TestFanoutGateHoldsRacingSiblingsUntilFirstContentDelta(t *testing.T) {
	var gate FanoutGate
	const siblings = 8
	var fanned atomic.Int32
	var wg sync.WaitGroup
	deadline := time.Now().Add(5 * time.Second)
	for i := 0; i < siblings; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for time.Now().Before(deadline) {
				if gate.Released() {
					fanned.Add(1)
					return
				}
				runtime.Gosched()
			}
		}()
	}

	// The first request's stream: HTTP 200 and message_start arrive first. The
	// barrier must hold — a 200 proves nothing about the provider having begun
	// to process (and cache) the prefix.
	for _, event := range []StreamEventKind{StreamEventHTTPStatus, StreamEventMessageStart} {
		if gate.Observe(event) {
			t.Fatalf("gate released on %q before content delta", event)
		}
	}
	time.Sleep(20 * time.Millisecond) // give a broken barrier the chance to leak a sibling
	if n := fanned.Load(); n != 0 {
		t.Fatalf("%d sibling(s) fanned before the first streamed content delta", n)
	}

	if !gate.Observe(StreamEventContentDelta) {
		t.Fatal("gate did not release on first content delta")
	}
	wg.Wait()
	if n := fanned.Load(); n != siblings {
		t.Fatalf("fanned = %d, want all %d siblings released after first content delta", n, siblings)
	}
}

func TestUnsupportedActiveCacheCapabilityFailsClosed(t *testing.T) {
	for _, capability := range []ActiveCacheCapability{ActiveCacheCapabilityUnknown, ActiveCacheCapabilityUnsupported} {
		t.Run(string(capability), func(t *testing.T) {
			dec := Plan(Request{
				Provider:               ProviderAnthropic,
				ActiveCapability:       capability,
				ExpectedReuseBeforeTTL: 99,
				ReadDiscount:           0.1,
				WarmPrefix:             fp("same"),
				RealPrefix:             fp("same"),
				SharedBlockCount:       3,
			})
			if dec.Primitive != PrimitiveNone || dec.Dedicated {
				t.Fatalf("unsupported capability chose an active primitive: %+v", dec)
			}
			if dec.Reason != ReasonUnsupportedActiveCacheCapability {
				t.Fatalf("reason = %q, want %q", dec.Reason, ReasonUnsupportedActiveCacheCapability)
			}
		})
	}
}

func TestDedicatedWarmWithoutLaterCacheReadIsWasted(t *testing.T) {
	dec := Plan(Request{
		Provider:               "implicit-provider",
		ActiveCapability:       ActiveCacheCapabilitySupported,
		ExpectedReuseBeforeTTL: 3,
		ReadDiscount:           0.5,
		WarmPrefix:             fp("same"),
		RealPrefix:             fp("same"),
	})
	if dec.Primitive != PrimitiveDecode1 {
		t.Fatalf("setup primitive = %q, want decode-1", dec.Primitive)
	}
	pending := ReconcileWarm(dec, false, nil)
	if pending.Status != WarmPending || pending.Wasted {
		t.Fatalf("pending accounting = %+v, want pending/not wasted", pending)
	}
	wasted := ReconcileWarm(dec, true, []CacheReadback{{CacheReadTokens: 0}})
	if wasted.Status != WarmWasted || !wasted.Wasted {
		t.Fatalf("zero-read accounting = %+v, want wasted", wasted)
	}
	confirmed := ReconcileWarm(dec, true, []CacheReadback{{CacheReadTokens: 128}})
	if confirmed.Status != WarmConfirmed || confirmed.Wasted || confirmed.CacheReadTokens != 128 {
		t.Fatalf("readback accounting = %+v, want confirmed 128", confirmed)
	}
}

func TestNaturalAutoCacheWarmIsNotBookedAsDedicated(t *testing.T) {
	dec := Plan(Request{
		Provider:               ProviderOpenAI,
		ActiveCapability:       ActiveCacheCapabilitySupported,
		ExpectedReuseBeforeTTL: 3,
		WarmPrefix:             fp("same"),
		RealPrefix:             fp("same"),
	})
	acc := ReconcileWarm(dec, true, nil)
	if acc.Status != WarmNotDedicated || acc.Wasted {
		t.Fatalf("auto-cache accounting = %+v, want not dedicated", acc)
	}
}
