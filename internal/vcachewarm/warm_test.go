package vcachewarm

import "testing"

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

func TestDedicatedWarmWithoutLaterCacheReadIsWasted(t *testing.T) {
	dec := Plan(Request{
		Provider:               "implicit-provider",
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
		ExpectedReuseBeforeTTL: 3,
		WarmPrefix:             fp("same"),
		RealPrefix:             fp("same"),
	})
	acc := ReconcileWarm(dec, true, nil)
	if acc.Status != WarmNotDedicated || acc.Wasted {
		t.Fatalf("auto-cache accounting = %+v, want not dedicated", acc)
	}
}
