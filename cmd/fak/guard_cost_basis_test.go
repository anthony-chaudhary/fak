package main

// guard_cost_basis_test.go — the basis stamp on `fak guard`'s dollar estimate (#5483).
//
// The defect these tests pin is an honesty one, so they assert on the OPERATOR-FACING
// STRING, not on an internal float: a correct number nobody can attribute is exactly
// what the issue calls unfalsifiable. Each test therefore names the surface it guards.

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// costBasisTestSummary is a session whose four billable token axes are chosen so the
// Opus-class {5,25} card prices each one to an exact round dollar — $1.0000 uncached
// input, $0.5000 cache read (1M at 0.1x), $1.0000 cache write (160k at the 5m 1.25x
// tier), $1.0000 output — totalling exactly $3.5000. Round axes mean the assertion
// can name the rendered figure verbatim instead of chasing float formatting.
func costBasisTestSummary() gateway.AdjudicationSummary {
	return gateway.AdjudicationSummary{
		Total:               4,
		Allowed:             4,
		InputTokens:         200_000,
		CachedPromptTokens:  1_000_000,
		CacheCreationTokens: 160_000,
		OutputTokens:        40_000,
	}
}

// armCostBasisForTest arms the process-wide served-spend meter from a real
// provider/context pair (the same call guard boot makes) and restores the previous
// state on cleanup. It goes through armServedSpendPricing rather than poking the
// globals so the test exercises the actual resolution path whose (source, ok) return
// both guard call sites used to discard.
func armCostBasisForTest(t *testing.T, provider, context string) {
	t.Helper()
	// An operator env override outranks the built-in table inside resolveSpendPricing,
	// so a developer with FAK_SPEND_* exported must not silently change what this test
	// resolves. Clear both for the duration.
	t.Setenv(spendInputPriceEnv, "")
	t.Setenv(spendOutputPriceEnv, "")
	restoreCostBasisAfterTest(t)
	armServedSpendPricing(provider, context)
}

// restoreCostBasisAfterTest snapshots the process-wide meter field-by-field — never a
// struct copy, servedSpend embeds its own RWMutex — and puts it back on cleanup, so a
// test that arms pricing cannot leak its card into the rest of the package.
func restoreCostBasisAfterTest(t *testing.T) {
	t.Helper()
	prevP, prevSource, prevProvider, prevContext, prevArmed, prevOK := servedSpendPricingBasis()
	t.Cleanup(func() {
		servedSpend.mu.Lock()
		servedSpend.armed, servedSpend.ok = prevArmed, prevOK
		servedSpend.p, servedSpend.source = prevP, prevSource
		servedSpend.provider, servedSpend.context = prevProvider, prevContext
		servedSpend.mu.Unlock()
	})
}

// TestGuardExitSummaryPrintsBasisStampedSessionCost is the ticket's headline: the exit
// summary an operator actually reads must print a dollar figure for the session AND
// name the rate card that produced it — with no --budget-envelope configured. Before
// #5483 the summary rendered the same session purely in token-equivalents and the
// resolved card was thrown away, so the assertions here fail against the un-wired
// formatter even with every helper in place.
func TestGuardExitSummaryPrintsBasisStampedSessionCost(t *testing.T) {
	armCostBasisForTest(t, "anthropic", "claude")

	got := formatAuditSummary(costBasisTestSummary())

	for _, want := range []string{
		"estimated cost",     // the section exists at all
		"~$3.5000 ESTIMATED", // the figure, marked as an estimate
		"basis card",         // the stamp's label
		gateway.CachePricingSourceAnthropicClaudeOpus48, // WHICH card produced it
		"$5/MTok in, $25/MTok out",                      // the rates it applied
		"provider=anthropic context=claude",             // what the card was resolved FROM
		"modifiers applied",                             // which rate modifiers were folded in
		"modifiers NOT applied",                         // and which were not, stated not hidden
		"cards disagree",                                // the in-tree divergence, on the face of it
	} {
		if !strings.Contains(got, want) {
			t.Errorf("guard exit summary missing %q; #5483 wants a VISIBLE, basis-stamped estimate.\n--- summary ---\n%s", want, got)
		}
	}
}

// TestGuardSessionCostReportsDollarBlindRatherThanGuessing pins the other half of the
// honesty contract: when no card resolves, the surface says DOLLAR-BLIND and prints no
// dollar figure at all. Substituting a default that reads like a measurement is the
// failure mode the issue names, so a `$` here would be worse than silence.
func TestGuardSessionCostReportsDollarBlindRatherThanGuessing(t *testing.T) {
	armCostBasisForTest(t, "some-local-runtime", "qwen3-coder")

	got := formatGuardSessionCost(costBasisTestSummary(), guardServedCostBasis())

	if !strings.Contains(got, "DOLLAR-BLIND") {
		t.Errorf("unpriced session did not report DOLLAR-BLIND:\n%s", got)
	}
	if strings.Contains(got, "~$") {
		t.Errorf("unpriced session printed a dollar figure anyway:\n%s", got)
	}
	if !strings.Contains(got, spendInputPriceEnv) {
		t.Errorf("dollar-blind row did not name the env override that would price it:\n%s", got)
	}
}

// TestGuardSessionCostStampsAnOperatorSuppliedRateAsSuch covers the third basis the
// meter can resolve. An operator's FAK_SPEND_* rate outranks the built-in table, so
// printing it under the same unqualified "basis card" as a fak default would let a
// hand-set number read as a fak-authored one — and would make the in-tree divergence
// warning below it look like the source of the figure when it is not.
func TestGuardSessionCostStampsAnOperatorSuppliedRateAsSuch(t *testing.T) {
	t.Setenv(spendInputPriceEnv, "7")
	t.Setenv(spendOutputPriceEnv, "11")
	restoreCostBasisAfterTest(t)
	armServedSpendPricing("anthropic", "claude")

	basis := guardServedCostBasis()
	if basis.Card != guardCostEnvOverrideCard {
		t.Fatalf("env-priced basis card = %q, want %q", basis.Card, guardCostEnvOverrideCard)
	}
	got := formatGuardSessionCost(costBasisTestSummary(), basis)
	if !strings.Contains(got, "OPERATOR-SUPPLIED") {
		t.Errorf("an operator-set rate was not stamped as operator-supplied:\n%s", got)
	}
	if !strings.Contains(got, "$7/MTok in, $11/MTok out") {
		t.Errorf("the stamp did not report the operator's actual rates:\n%s", got)
	}
}

// TestGuardSessionCostStaysSilentOnASessionThatServedNothing keeps the clean-run
// silence every other exit-summary formatter holds: a session with no billable token
// on any axis has nothing to price, and "$0.0000" would be noise, not evidence.
func TestGuardSessionCostStaysSilentOnASessionThatServedNothing(t *testing.T) {
	armCostBasisForTest(t, "anthropic", "claude")
	if got := formatGuardSessionCost(gateway.AdjudicationSummary{Total: 3, Allowed: 3}, guardServedCostBasis()); got != "" {
		t.Errorf("token-less session rendered a cost section:\n%s", got)
	}
}

// TestGuardSessionCostSplitsCacheWriteTiers pins that the estimate prices the
// gateway-attributed 1h-upgraded cache-creation slice at the 2.0x write tier and the
// remainder at 5m's 1.25x — the same split MechanismSavings applies — so the two
// surfaces cannot disagree about what the same session's writes cost.
func TestGuardSessionCostSplitsCacheWriteTiers(t *testing.T) {
	p := gateway.CachePricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25}
	base := gateway.AdjudicationSummary{CacheCreationTokens: 160_000}

	all5m := guardSessionCostUSD(base, p)
	upgraded := base
	upgraded.CacheCreationTokensUpgraded = 160_000
	all1h := guardSessionCostUSD(upgraded, p)

	// 160k tokens at $5/MTok = $0.80 of base input: 1.25x = $1.00, 2.0x = $1.60.
	if diff := all5m - 1.0; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("all-5m cache writes = $%.6f, want $1.000000", all5m)
	}
	if diff := all1h - 1.6; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("all-1h cache writes = $%.6f, want $1.600000", all1h)
	}

	// An inconsistent upgraded > total pair must clamp, never inflate.
	overclaimed := base
	overclaimed.CacheCreationTokensUpgraded = 999_999
	if got := guardSessionCostUSD(overclaimed, p); got > all1h+1e-9 {
		t.Errorf("upgraded > total inflated the estimate: $%.6f > $%.6f", got, all1h)
	}
}

// TestGuardOpusClassRateCardsReadLiveConstants is the anti-rot pin on the divergence
// inventory: every row must equal the symbol it names, read live. A transcribed
// provenance table that drifts from its sources is worse than no table, because it
// makes a stale claim look audited.
func TestGuardOpusClassRateCardsReadLiveConstants(t *testing.T) {
	cards := guardOpusClassRateCards()
	if len(cards) != 4 {
		t.Fatalf("rate-card inventory has %d row(s), want the 4 #5483 enumerates: %+v", len(cards), cards)
	}
	for _, c := range cards {
		if c.Site == "" || c.Claims == "" {
			t.Errorf("rate card %+v is missing its site or its self-description", c)
		}
		if c.InPerMTokUSD <= 0 || c.OutPerMTokUSD <= 0 {
			t.Errorf("rate card %s carries a non-positive base %s", c.Site, c.Base())
		}
	}
	if got, want := cards[0].InPerMTokUSD, gateway.ClaudeOpus48InputPerMTokUSD; got != want {
		t.Errorf("gateway card in = %v, want the live constant %v", got, want)
	}
	if got, want := cards[3].InPerMTokUSD, float64(guardResumeFallbackInputPerMTokUSD); got != want {
		t.Errorf("resume-fallback card in = %v, want the live constant %v", got, want)
	}

	// The premise of the whole stamp: the cards do NOT agree today. If a future change
	// reconciles them, this expectation flips deliberately and the warning line goes
	// quiet on its own — which is the intended end state, not a regression.
	if bases := guardRateCardBases(); len(bases) < 2 {
		t.Fatalf("expected the in-tree Opus-class cards to disagree, got a single base %v", bases)
	}
}

// TestGuardRateCardDivergenceNamesEverySite makes the warning actionable: it is not
// enough to say "the cards disagree", the reader must be told where each one lives so
// they can go settle it.
func TestGuardRateCardDivergenceNamesEverySite(t *testing.T) {
	got := formatGuardRateCardDivergence()
	if got == "" {
		t.Fatal("divergence line is empty while the cards still disagree")
	}
	for _, c := range guardOpusClassRateCards() {
		if !strings.Contains(got, c.Site) {
			t.Errorf("divergence line does not name %s:\n%s", c.Site, got)
		}
	}
	if !strings.Contains(got, "does not reconcile") {
		t.Errorf("divergence line does not state that fak declines to pick a winner:\n%s", got)
	}
}

// TestGuardSessionCostFlagsTheUnpricedLongContextVariant covers the one modifier whose
// information is ALREADY in the process: guard detects the `[1m]` long-context marker
// for the auto-compact window, but the pricer has no published premium for it and
// resolves the standard-window rate. Rather than guess a multiplier, the stamp says the
// figure is a lower bound — a visible under-estimate beats a silent one.
func TestGuardSessionCostFlagsTheUnpricedLongContextVariant(t *testing.T) {
	armCostBasisForTest(t, "anthropic", "claude-opus-4-8[1m]")

	basis := guardServedCostBasis()
	if !basis.Priced {
		t.Fatalf("[1m] variant did not resolve a card at all: %+v", basis)
	}
	got := formatGuardSessionCost(costBasisTestSummary(), basis)
	if !strings.Contains(got, "long-context") || !strings.Contains(got, "LOWER BOUND") {
		t.Errorf("[1m] session was priced at the standard-window rate with no warning:\n%s", got)
	}
}

// TestGuardResumePlanStampsWhoPricedIt covers the other operator-facing `$` guard
// emits. The projection runs on a SUBSTITUTED {5,25} when no price is supplied, which
// turns an unpriced input into a priced number; the plan must say which of the two
// happened, in both the rendered text and the JSON a consumer binds.
func TestGuardResumePlanStampsWhoPricedIt(t *testing.T) {
	desc := session.Descriptor{ID: "sess-basis", Argv: []string{"claude"}}

	substituted := planGuardResume("sess-basis", desc, guardResumeInput{ResidentTokens: 120_000, IdleSeconds: 60})
	if substituted.PricingSource != guardResumePricingSourceSubstituted {
		t.Errorf("unpriced plan source = %q, want the substituted stamp %q", substituted.PricingSource, guardResumePricingSourceSubstituted)
	}
	if substituted.PricingInputPerMTokUSD != guardResumeFallbackInputPerMTokUSD {
		t.Errorf("unpriced plan input rate = %v, want %v", substituted.PricingInputPerMTokUSD, guardResumeFallbackInputPerMTokUSD)
	}

	priced := planGuardResume("sess-basis", desc, guardResumeInput{
		ResidentTokens: 120_000, IdleSeconds: 60,
		Pricing: resume.Pricing{InputPerMTokUSD: 3, OutputPerMTokUSD: 15},
	})
	if priced.PricingSource != guardResumePricingSourceOperator {
		t.Errorf("operator-priced plan source = %q, want %q", priced.PricingSource, guardResumePricingSourceOperator)
	}
	if priced.PricingInputPerMTokUSD != 3 {
		t.Errorf("operator-priced plan input rate = %v, want 3", priced.PricingInputPerMTokUSD)
	}

	var b strings.Builder
	renderGuardResumePlan(&b, substituted)
	if !strings.Contains(b.String(), "$ basis:") || !strings.Contains(b.String(), "SUBSTITUTED") {
		t.Errorf("rendered resume plan does not stamp its dollar basis:\n%s", b.String())
	}
}

// TestGuardResumeCLIDoesNotClaimAnOperatorPricedItByDefault closes the gap the pure
// planner test cannot see: --input-price / --output-price carry NON-ZERO defaults, so
// before the fs.Visit check every CLI plan handed the planner a fully-populated {5,25}
// and got stamped "operator:..." — a basis the operator never supplied. The stamp is
// only worth printing if it reports who actually priced the projection, so this drives
// all three labels through the real CLI entry point.
func TestGuardResumeCLIDoesNotClaimAnOperatorPricedItByDefault(t *testing.T) {
	regPath := filepath.Join(t.TempDir(), "session-registry.json")
	reg := session.NewRegistry(session.NewFileStore(regPath))
	if _, err := reg.RegisterWithMeta("guard-basis", "host",
		session.State{TraceID: "guard-basis", Run: session.Running},
		session.DefaultDescriptorTTL, time.Now(),
		session.DescriptorMeta{Argv: []string{"claude"}}); err != nil {
		t.Fatalf("seed registry: %v", err)
	}

	planOf := func(t *testing.T, extra ...string) guardResumePlan {
		t.Helper()
		var out, errb bytes.Buffer
		argv := append([]string{"--registry", regPath, "--json", "--idle-seconds", "60", "--resident-tokens", "120000"}, extra...)
		argv = append(argv, "guard-basis")
		if rc := runGuardResume(&out, &errb, argv); rc != 0 {
			t.Fatalf("runGuardResume%v: rc=%d (%s)", extra, rc, errb.String())
		}
		var p guardResumePlan
		if err := json.Unmarshal(out.Bytes(), &p); err != nil {
			t.Fatalf("plan JSON: %v\n%s", err, out.String())
		}
		return p
	}

	// No price flags at all: the dollars are still projected on the substituted {5,25},
	// and the plan must SAY so rather than crediting the operator.
	bare := planOf(t)
	if bare.PricingSource != guardResumePricingSourceSubstituted {
		t.Errorf("unflagged CLI plan source = %q, want %q", bare.PricingSource, guardResumePricingSourceSubstituted)
	}
	if bare.PricingInputPerMTokUSD != guardResumeFallbackInputPerMTokUSD || bare.PricingOutputPerMTokUSD != guardResumeFallbackOutputPerMTokUSD {
		t.Errorf("substitution changed the projected rates: got %v/%v, want %v/%v",
			bare.PricingInputPerMTokUSD, bare.PricingOutputPerMTokUSD,
			guardResumeFallbackInputPerMTokUSD, guardResumeFallbackOutputPerMTokUSD)
	}

	// Both supplied: a genuine operator basis.
	both := planOf(t, "--input-price", "3", "--output-price", "15")
	if both.PricingSource != guardResumePricingSourceOperator {
		t.Errorf("fully-priced CLI plan source = %q, want %q", both.PricingSource, guardResumePricingSourceOperator)
	}
	if both.PricingInputPerMTokUSD != 3 || both.PricingOutputPerMTokUSD != 15 {
		t.Errorf("operator rates not carried: got %v/%v, want 3/15", both.PricingInputPerMTokUSD, both.PricingOutputPerMTokUSD)
	}

	// Half supplied is neither: the other half is still fak's number.
	half := planOf(t, "--input-price", "3")
	if half.PricingSource != guardResumePricingSourceMixed {
		t.Errorf("half-priced CLI plan source = %q, want %q", half.PricingSource, guardResumePricingSourceMixed)
	}
	if half.PricingInputPerMTokUSD != 3 || half.PricingOutputPerMTokUSD != guardResumeFallbackOutputPerMTokUSD {
		t.Errorf("half-priced plan rates = %v/%v, want 3/%v",
			half.PricingInputPerMTokUSD, half.PricingOutputPerMTokUSD, guardResumeFallbackOutputPerMTokUSD)
	}
}
