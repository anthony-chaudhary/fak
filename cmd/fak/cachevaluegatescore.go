package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// cmdCacheValueGateScore is `fak score cachevalue-gate` (issue #3642, epic #3569): the FIRST
// production consumer of pkg/scorecard's D2 net-regression gate (ComposeD2 /
// CacheValueGateBlocks). The gate shipped fully-built and unit-tested but had ZERO real callers
// -- a defined-but-uninstalled reward-hack fence. This command installs it on a real path so a
// net regression (or a fresh gross-up-without-net) actually BLOCKS with a non-zero exit.
//
// It sources the CANDIDATE facts from the SAME live fleet-benefit fold the report/cache-health
// cards read (FoldFleetBenefit over the Track-1 kernel + Track-2 savings + gateway-usage
// ledgers), mapping the report's REAL observed gross/net fak_share (FakShareGrossPct /
// FakSharePct) into the D2 gate's 0..1 candidate shares. The BASELINE is the pinned
// "last-accepted honest" floor: a prior gate --json snapshot's candidate shares (via --baseline),
// or -- with no pin -- a zero floor, under which the net-nonregression fence can never
// false-positive but the divergence-ceiling / gross-up-without-net fences still red a fresh
// reward-hack (the fences #2783 designed to be baseline-independent).
//
// The valuation-basis-honesty fence is fed 0/0 (honesty 1.0, floor 0) here: the labelled-figure
// facts live on the D1 SCORE surface, not this fleet fold, so its floor is 0 (never reds) rather
// than a fabricated value -- an honest "not sourced from this fold yet" position, documented.
func cmdCacheValueGateScore(argv []string) {
	os.Exit(runCacheValueGateScore(os.Stdout, os.Stderr, argv))
}

func runCacheValueGateScore(stdout, stderr io.Writer, argv []string) int {
	const prefix = "fak score cachevalue-gate"
	fs := flag.NewFlagSet(prefix, flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevalueledger.DefaultLedgerRel, "Track-1 WITNESSED kernel ledger (docs/nightrun/cache-value.jsonl)")
	savingsLedger := fs.String("savings-ledger", cachevaluereport.DefaultSavingsLedgerRel, "Track-2 OBSERVED-$ savings ledger (.fak/nightrun/cache-savings.jsonl)")
	usageLedger := fs.String("usage-ledger", gatewayusageledger.DefaultLedgerRel, "gateway usage ledger (.fak/nightrun/gateway-usage.jsonl)")
	since := fs.String("since", "", "fold only rows on or after this date (YYYY-MM-DD)")
	baselinePath := fs.String("baseline", "", "pinned last-accepted-honest gate --json snapshot; the candidate's net/gross shares ratchet against it (default: a zero floor -- the divergence-ceiling and gross-up-without-net fences still red a fresh reward-hack)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit the committed snapshot body")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s: unexpected argument %q\n", prefix, fs.Arg(0))
		return 2
	}
	if *since != "" {
		if _, err := time.Parse("2006-01-02", *since); err != nil {
			fmt.Fprintf(stderr, "%s: --since must be YYYY-MM-DD: %v\n", prefix, err)
			return 2
		}
	}

	track1 := filterTrack1Since(cachevalueledger.ReadLedgerFile(*ledger), *since)
	track2 := filterTrack2Since(cachevaluereport.ReadSavingsLedgerFile(*savingsLedger), *since)
	usage := filterGatewayUsageSince(gatewayusageledger.ReadLedgerFile(*usageLedger), *since)
	rep := cachevaluereport.FoldFleetBenefit(track1, track2, usage, cachevaluereport.FleetBenefitOptions{})

	candidate := cacheValueGateFactsFromReport(rep)
	baseline, ok := loadCacheValueGateBaseline(stderr, prefix, *baselinePath)
	if !ok {
		return 2
	}

	// The production gate call this issue installs: CacheValueGateBlocks is the merge/loop verb
	// that refuses a gross-up-without-net change. In the default terminal render we surface its
	// BLOCK reason on stderr so an operator sees WHY the exit is non-zero; --json/--markdown/
	// --compare keep the machine payload clean.
	if !*asJSON && !*asMarkdown && *comparePath == "" {
		if blocked, reason := scorecard.CacheValueGateBlocks(baseline, candidate); blocked {
			fmt.Fprintf(stderr, "%s: BLOCK -- %s\n", prefix, reason)
		}
	}

	payload := scorecard.ComposeD2(baseline, candidate)
	return emitScorecard(stdout, stderr, prefix, scorecard.CacheValueGateDebtKey, payload,
		*comparePath, *asJSON, *asMarkdown, scorecard.MarkdownDoc{
			Title: "fak cache-value gate scorecard - the net-regression reward-hack fence",
			Description: "fak's D2 cache-value gate: it BLOCKS a change that regresses the honest fak_share_net below a " +
				"pinned floor, or grosses up the fak-authored share without a matching net gain, or lets gross diverge " +
				"from net past the reward-hack alarm. The candidate's gross/net shares are the live observed fleet-benefit " +
				"fold; the baseline is the last-accepted-honest snapshot.",
			Heading: "Cache-value net-regression gate",
			AutoGen: "Auto-generated by `fak score cachevalue-gate --markdown`. Do not hand-edit; re-run the tool.",
			Law: "The law: fak_share_net is the headline and gross a labelled upper bound; a candidate may not regress net, " +
				"nor gross up without a matching net gain, nor open the gross-net gap past the alarm. A red fence is retired " +
				"by making the candidate honest (raise net / converge gross->net / label the basis), never by weakening the fence.",
			DebtKey: scorecard.CacheValueGateDebtKey,
			HeaderExtra: fmt.Sprintf(" - candidate net %v gross %v vs pinned net floor %v",
				payload.Corpus["candidate_fak_share_net"], payload.Corpus["candidate_fak_share_gross"], payload.Corpus["baseline_fak_share_net"]),
		})
}

// cacheValueGateFactsFromReport maps the folded fleet-benefit report's REAL observed gross/net
// fak_share (percent, nil when the corpus is empty or upside-down) into the D2 gate's candidate
// facts (0..1 fractions). DollarFigures/WithBasis stay 0 -- the labelled-$ facts live on the D1
// SCORE surface, not this fleet fold, so ValuationBasisHonesty folds to the honest 1.0 (nothing
// unlabeled asserted here) rather than a fabricated value.
func cacheValueGateFactsFromReport(rep cachevaluereport.FleetBenefitReport) scorecard.CacheValueFacts {
	return scorecard.CacheValueFacts{
		FakShareGross: pctPtrToFraction(rep.FakShareGrossPct),
		FakShareNet:   pctPtrToFraction(rep.FakSharePct),
	}
}

// pctPtrToFraction converts a report's *percent share (nil when the corpus is empty or
// upside-down) into a 0..1 fraction, folding a nil/non-positive share to 0 -- the honest "no
// positive share to defend" floor, which never spuriously reds the net-nonregression fence.
func pctPtrToFraction(p *float64) float64 {
	if p == nil || *p <= 0 {
		return 0
	}
	return *p / 100
}

// loadCacheValueGateBaseline pins the D2 baseline. An empty path yields a zero floor (net floor 0
// can never false-positive; the divergence-ceiling and gross-up-without-net fences still red a
// fresh reward-hack independent of a pin). Otherwise it reads a prior gate --json snapshot and
// pins the floor to that snapshot's candidate shares (candidate_fak_share_gross /
// candidate_fak_share_net), so today's fold ratchets against the last accepted honest state.
func loadCacheValueGateBaseline(stderr io.Writer, prefix, path string) (scorecard.CacheValueBaseline, bool) {
	if path == "" {
		return scorecard.CacheValueBaseline{}, true
	}
	base, ok := readCompareBase(stderr, prefix, path)
	if !ok {
		return scorecard.CacheValueBaseline{}, false
	}
	corpus, _ := base["corpus"].(map[string]any)
	return scorecard.CacheValueBaseline{
		FakShareGross: corpusFloat(corpus, "candidate_fak_share_gross"),
		FakShareNet:   corpusFloat(corpus, "candidate_fak_share_net"),
		// ValuationBasisHonesty stays 0: the candidate basis honesty is not sourced from this
		// fleet fold (it lives on the D1 score surface), so its floor is 0 (never reds).
		// corpusFloat (shared with steering.go) tolerantly reads the numeric baseline shares.
	}, true
}
