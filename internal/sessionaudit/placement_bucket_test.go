package sessionaudit

import "testing"

// TestPricedModelsAreNeverBucketedSelfHosted is the keystone invariant of the
// placement split, and it is TOTAL over the rate card: every key this package
// carries a published third-party price for must not be classified as self-hosted.
//
// The two claims are mutually exclusive by construction. A rate card exists because
// somebody BILLS you per token, which means the weights ran on their hardware. A
// self-hosted bucket asserts the opposite. Before this test, "deepseek" was priced
// from the published DeepSeek API card AND bucketed "local / self-hosted", so one
// turn was simultaneously reported as costing third-party dollars and as having been
// served on our own silicon — an error that inflates exactly the number epic #5416
// exists to report honestly.
func TestPricedModelsAreNeverBucketedSelfHosted(t *testing.T) {
	for _, key := range pricingOrder {
		if _, ok := PriceFor(key); !ok {
			t.Fatalf("pricingOrder key %q does not price itself — the price book is inconsistent", key)
		}
		bucket := ProviderBucket(key)
		selfHosted, known := BucketIsSelfHosted(bucket)
		if known && selfHosted {
			t.Fatalf("model id %q is priced from a published third-party rate card (%+v) "+
				"but ProviderBucket puts it in %q, which claims self-hosted placement. "+
				"A billed token was served on someone else's hardware — the two claims cannot both hold.",
				key, Pricing[key], bucket)
		}
	}
}

// TestPricedModelsNeverLandInTheUnpricedBucket pins the other direction of the same
// drift. "glm" and "kimi" carried published Z.AI and Moonshot rate cards while
// matching no bucket substring at all, so they fell through to the bucket literally
// named "UNKNOWN (unpriced bucket)" — a label contradicting a price in the same file.
// Both are middle-rung models for epic #5416, so silently losing them was expensive.
func TestPricedModelsNeverLandInTheUnpricedBucket(t *testing.T) {
	for _, key := range pricingOrder {
		if got := ProviderBucket(key); got == BucketUnknownPrices {
			t.Fatalf("model id %q is priced (%+v) but buckets as %q — a priced model must land in a named bucket",
				key, Pricing[key], got)
		}
	}
}

// TestOpenWeightsFamiliesAreNeverPricedAsVendor guards the seam #5115 has to cross.
// #5115 asks for "pricingOrder keys for gpt and gemini" from the vendors' published
// per-MTok cards, while its own Out-of-scope line requires self-hosted weights to stay
// UNPRICED. Those two instructions collide, and nothing in the package caught it:
// PriceFor matches by SUBSTRING, first hit wins, so the literal key "gpt" also matches
// gpt-oss — an open-weights family fak serves on its own hardware.
//
// The failure is silent and one-directional. Adding the key flips PriceFor from
// ok=false to ok=true, which REMOVES the id from Aggregate.UnpricedModels: instead of
// an honest UNMEASURED hold (#4490) the KPI emits a confident dollar figure computed
// from an OpenAI vendor rate card for tokens that were never billed by OpenAI.
//
// So this test is total over pricingOrder rather than a fixed expectation: any future
// key that reaches an open-weights id reds here, naming the id it captured.
func TestOpenWeightsFamiliesAreNeverPricedAsVendor(t *testing.T) {
	// Open-weights ids fak can serve itself. gpt-oss leads because it is the one that
	// embeds a vendor family name; the rest pin the families already in the bucket row.
	ids := []string{
		"gpt-oss-120b", "gpt-oss:20b", "gptoss-120b", "openai/gpt_oss-20b",
		"qwen2.5:14b", "llama3.2", "mistral-large", "gemma-3-27b",
	}
	for _, id := range ids {
		if r, ok := PriceFor(id); ok {
			t.Fatalf("PriceFor(%q) = %+v, but %q names OPEN WEIGHTS that fak can serve itself. "+
				"A pricingOrder key (%v) captured it by substring and attached a VENDOR rate card, "+
				"so the provider-cost KPI will report billed dollars for tokens no vendor billed. "+
				"Price the vendor's own tier ids instead of the bare family name (#5115).",
				id, r, id, pricingOrder)
		}
		if got := ModelTier(id); got != "unpriced" {
			t.Fatalf("ModelTier(%q) = %q, want \"unpriced\" — an open-weights id must not "+
				"inherit a vendor tier (#5115)", id, got)
		}
	}
}

// TestOpenWeightsBucketMakesNoPlacementClaim is the honesty witness: an open-weights
// family id must report placement as NOT KNOWN, rather than defaulting to either
// answer. Every one of these ids is servable both ways — from a machine the company
// operates and from a vendor API — so the id alone cannot settle it.
func TestOpenWeightsBucketMakesNoPlacementClaim(t *testing.T) {
	ids := []string{
		"qwen2.5:14b", "qwen/qwen3.6-27b", "llama3.2", "mistral-large",
		"mixtral-8x7b", "phi-4", "deepseek-v4-pro", "glm-5.2", "kimi-k3",
		// The gpt-oss family embeds a VENDOR family name ("gpt") in an
		// OPEN-WEIGHTS id, so it is the one case where substring order decides
		// the answer. fak serves these weights itself — MXFP4 dequant-on-load in
		// internal/ggufload, and a CI oracle in internal/covmatrix
		// ({Name: "gpt-oss-MoE", ResolverToken: "gptoss", OracleInCI: true}) — so
		// bucketing them "OpenAI" asserts a vendor placement fak's own tree
		// disproves (#5115).
		"gpt-oss-120b", "gpt-oss:20b", "gptoss-120b", "openai/gpt_oss-20b",
	}
	for _, id := range ids {
		bucket := ProviderBucket(id)
		if bucket != BucketOpenWeights {
			t.Fatalf("ProviderBucket(%q) = %q, want %q", id, bucket, BucketOpenWeights)
		}
		selfHosted, known := BucketIsSelfHosted(bucket)
		if known {
			t.Fatalf("bucket %q (from id %q) claims to know its placement; an open-weights id cannot", bucket, id)
		}
		if selfHosted {
			t.Fatalf("bucket %q must not report self-hosted without a placement signal", bucket)
		}
	}
}

// TestShippedRosterCounterexample is the concrete case that proves the old
// name-based rule wrong, using fak's OWN shipped configuration rather than a
// hypothetical: examples/model-accounts.example.json binds "qwen/qwen3.6-27b" to the
// Groq account — a VENDOR. The old classifier reported those tokens as self-hosted.
// With a real placement signal the same id now classifies correctly in either zone,
// which is the entire point of taking the zone as an input.
func TestShippedRosterCounterexample(t *testing.T) {
	const id = "qwen/qwen3.6-27b"

	viaGroq := BucketForPlacement(id, "vendor")
	if viaGroq != BucketVendorOpen {
		t.Fatalf("qwen served by Groq = %q, want %q", viaGroq, BucketVendorOpen)
	}
	if selfHosted, known := BucketIsSelfHosted(viaGroq); !known || selfHosted {
		t.Fatalf("qwen-via-vendor must be KNOWN and NOT self-hosted, got selfHosted=%v known=%v", selfHosted, known)
	}

	onCompanyGPUs := BucketForPlacement(id, "fleet")
	if onCompanyGPUs != BucketFleet {
		t.Fatalf("qwen on company hardware = %q, want %q", onCompanyGPUs, BucketFleet)
	}
	if selfHosted, known := BucketIsSelfHosted(onCompanyGPUs); !known || !selfHosted {
		t.Fatalf("qwen-on-fleet must be KNOWN and self-hosted, got selfHosted=%v known=%v", selfHosted, known)
	}

	// Same id, opposite answers — which is only possible because placement is an
	// input rather than a guess.
	if viaGroq == onCompanyGPUs {
		t.Fatal("the same model id must be able to classify differently in different zones")
	}
}

// TestBucketForPlacementCoversEveryZone walks the whole PlacementZone ladder
// (modelroute.Zones() in string form) plus the absent-signal case. The zone strings
// are duplicated here rather than imported because this package is an import-free
// architest pureRoot leaf; TestPlacementZoneStringsMatchModelroute in
// internal/engine pins them against the real typed values.
func TestBucketForPlacementCoversEveryZone(t *testing.T) {
	cases := []struct {
		zone       string
		model      string
		want       string
		selfHosted bool
		known      bool
	}{
		{"device", "qwen3.6-4b", BucketDevice, true, true},
		{"fleet", "glm-5.2", BucketFleet, true, true},
		{"fleet", "kimi-k3", BucketFleet, true, true},
		{"vendor", "deepseek-v4-pro", BucketVendorOpen, false, true},
		{"vendor", "claude-opus-4-6", BucketAnthropic, false, true},
		{"vendor", "gpt-5.5", BucketOpenAI, false, true},
		{"vendor", "gemini-3-pro", BucketGoogle, false, true},
		// Case and whitespace tolerated, like the residency floor's parser.
		{" FLEET ", "glm-5.2", BucketFleet, true, true},
		// No signal => no claim, in either direction.
		{"", "glm-5.2", BucketOpenWeights, false, false},
		{"nonsense", "glm-5.2", BucketOpenWeights, false, false},
		// Harness turns stay non-billed regardless of zone.
		{"vendor", "<synthetic>", BucketNonBilled, false, true},
	}
	for _, c := range cases {
		got := BucketForPlacement(c.model, c.zone)
		if got != c.want {
			t.Fatalf("BucketForPlacement(%q, %q) = %q, want %q", c.model, c.zone, got, c.want)
		}
		selfHosted, known := BucketIsSelfHosted(got)
		if selfHosted != c.selfHosted || known != c.known {
			t.Fatalf("BucketIsSelfHosted(%q) = (%v, %v), want (%v, %v)",
				got, selfHosted, known, c.selfHosted, c.known)
		}
	}
}

// TestSelfHostedFractionExcludesTheUnknownRemainder is the arithmetic the headline
// number depends on. Unattributable tokens must sit OUTSIDE both the numerator and
// the denominator of a self-hosted fraction; folding them into either one turns "we
// cannot tell" into a claim. This is why BucketIsSelfHosted returns two values.
func TestSelfHostedFractionExcludesTheUnknownRemainder(t *testing.T) {
	tokensByBucket := map[string]int64{
		BucketFleet:       600,
		BucketDevice:      200,
		BucketAnthropic:   100,
		BucketOpenWeights: 900, // unattributable — must not count as a saving
	}
	var selfHostedTokens, attributedTokens, unattributed int64
	for bucket, n := range tokensByBucket {
		selfHosted, known := BucketIsSelfHosted(bucket)
		if !known {
			unattributed += n
			continue
		}
		attributedTokens += n
		if selfHosted {
			selfHostedTokens += n
		}
	}
	if unattributed != 900 {
		t.Fatalf("unattributed = %d, want 900", unattributed)
	}
	if attributedTokens != 900 {
		t.Fatalf("attributed = %d, want 900", attributedTokens)
	}
	if got, want := float64(selfHostedTokens)/float64(attributedTokens), 800.0/900.0; got != want {
		t.Fatalf("self-hosted fraction = %v, want %v", got, want)
	}
	// The naive single-bool fold would have scored the unknown 900 as not-self-hosted
	// and reported 800/1800 = 44%, understating a measured 89% — the failure mode the
	// two-valued predicate exists to prevent.
	if naive := float64(selfHostedTokens) / float64(attributedTokens+unattributed); naive == 800.0/900.0 {
		t.Fatal("the unknown remainder must change the answer — otherwise this test proves nothing")
	}
}
