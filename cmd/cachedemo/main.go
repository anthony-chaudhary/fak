// Command cachedemo renders the multi-turn shared-cache-savings demo: it reads the
// durable, WITNESSED gateway-usage and cache-savings ledgers and tells the per-turn
// story of how a single fak-guarded coding session earns cache value across turns —
// the provider prompt cache reading a stable prefix back, AND fak's own authored slice
// (per-fire compaction trim + tool prune) holding the re-sent window inside budget.
//
// HONESTY LAW (#3095, retraction 2026-07-06): compaction_shed_tokens is CUMULATIVE
// across fires (fak re-trims the full re-sent history every turn and keeps no compacted
// body), so the session sum re-counts the same aged middle and over-counts distinct
// tokens. This demo therefore cites shed PER FIRE as the honest headline, labels every
// cumulative sum as a re-counted over-count ceiling (not distinct tokens), never ratios
// shed against cache_read, and books the fak $ at the price-honest 0.1× cache-read
// marginal on warm fires (#2794/#2798). The distinct-token COUNT axis is open in #3095.
//
// It is a read-only fold over recorded evidence: it never hits a live model, never
// needs an API key or GPU, and prints only counts/token totals (never a prompt byte).
// The owner split is the point — provider prompt-cache dollars are labeled OBSERVED and
// attributed to the provider; fak's compaction dollars are labeled WITNESSED and
// attributed to fak — so a provider-heavy corpus can never read as a fak win, and fak's
// real, positive slice is never smuggled under a provider label.
//
//	cachedemo                          # human multi-turn narrative over the default ledgers
//	cachedemo --session-pid 56116      # pin the per-turn spine to one witnessed session
//	cachedemo --since 2026-07-04       # fold only rows on or after this date
//	cachedemo --json                   # emit the raw fleet-benefit report JSON
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

func main() {
	os.Exit(run(os.Stdout, os.Stderr, os.Args[1:]))
}

func run(stdout, stderr *os.File, argv []string) int {
	fs := flag.NewFlagSet("cachedemo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	valLedger := fs.String("ledger", cachevalueledger.DefaultLedgerRel, "Track-1 kernel cache-value ledger")
	savLedger := fs.String("savings-ledger", cachevaluereport.DefaultSavingsLedgerRel, "Track-2 OBSERVED-$ savings ledger")
	useLedger := fs.String("usage-ledger", gatewayusageledger.DefaultLedgerRel, "gateway-usage counter ledger (the per-turn spine)")
	since := fs.String("since", "2026-07-04", "fold only savings rows on or after this date (YYYY-MM-DD); empty = all")
	pinPID := fs.Int("session-pid", 0, "pin the per-turn spine to one witnessed guard session (0 = auto-pick the richest multi-turn session)")
	asJSON := fs.Bool("json", false, "emit the raw fleet-benefit report JSON instead of the narrative")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	track1 := filterTrack1Since(cachevalueledger.ReadLedgerFile(*valLedger), *since)
	track2 := filterTrack2Since(cachevaluereport.ReadSavingsLedgerFile(*savLedger), *since)
	usage := gatewayusageledger.ReadLedgerFile(*useLedger)

	report := cachevaluereport.FoldFleetBenefit(track1, track2, usage, cachevaluereport.FleetBenefitOptions{})

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "cachedemo: encode: %v\n", err)
			return 1
		}
		return 0
	}

	spine := pickSpine(usage, *pinPID)
	fmt.Fprint(stdout, renderDemo(report, spine, *since))
	return 0
}

// pickSpine selects the single guard session that best tells the multi-turn story: the
// richest session that BOTH read the provider cache back across turns AND fired fak's
// own compaction shed. A pinned PID wins when present and found.
func pickSpine(rows []gatewayusageledger.Row, pinPID int) *gatewayusageledger.Row {
	var best *gatewayusageledger.Row
	var bestScore uint64
	for i := range rows {
		r := &rows[i]
		if r.SessionType != "guard" {
			continue
		}
		c := r.Counters
		if pinPID != 0 {
			if r.PID == pinPID {
				return r
			}
			continue
		}
		if c.CachedTurns < 2 || c.CompactionShedTokens == 0 || c.CachedPromptTokens == 0 {
			continue
		}
		score := c.CachedTurns + c.CompactionFired
		if best == nil || score > bestScore {
			best, bestScore = r, score
		}
	}
	return best
}

func renderDemo(rep cachevaluereport.FleetBenefitReport, spine *gatewayusageledger.Row, since string) string {
	var b strings.Builder
	b.WriteString("fak — shared cache savings, multi-turn demo\n")
	b.WriteString("═══════════════════════════════════════════\n\n")
	b.WriteString("A read-only fold over WITNESSED, durable ledgers. No live model, no API key,\n")
	b.WriteString("no GPU. Counts and token totals only — never a prompt byte.\n\n")

	if spine == nil {
		b.WriteString("PER-TURN SPINE: no qualifying multi-turn guard session in the ledger yet.\n\n")
	} else {
		renderSpine(&b, spine)
	}

	renderFleet(&b, rep, since)
	return b.String()
}

// renderSpine tells the multi-turn story of ONE real session: across N turns the
// provider caches a stable prefix (read back cheaply), while fak fires compaction to
// trim the client's re-sent history so the window stays inside budget.
//
// HONESTY LAW (#3095, retraction 2026-07-06): compaction_shed_tokens is CUMULATIVE
// across fires — fak keeps no compacted body, it re-trims the FULL re-sent history
// every turn, so the same aged middle is dropped AND counted on every subsequent fire.
// The session sum therefore over-counts distinct tokens (~4.8× on the witnessed
// long session). So we headline shed PER FIRE (shed / fired), print the session sum
// only under an explicit "cumulative, re-counts the same middle each fire" label, and
// NEVER ratio it against cache_read (a different cumulative in a different currency).
func renderSpine(b *strings.Builder, s *gatewayusageledger.Row) {
	c := s.Counters
	fmt.Fprintf(b, "PER-TURN SPINE — one witnessed guard session (pid %d, %s, %.0f min)\n",
		s.PID, s.GeneratedAt, s.UptimeSecs/60)
	fmt.Fprintf(b, "  turns with cache activity ......... %d\n", c.CachedTurns)
	fmt.Fprintf(b, "  adjudicated tool calls ........... %d\n\n", c.Total)

	b.WriteString("  Provider prompt cache (OBSERVED — the provider's own cache, across turns):\n")
	fmt.Fprintf(b, "    cache_read  = %s tokens  (stable prefix read back, not re-billed at full rate)\n", commas(c.CachedPromptTokens))
	fmt.Fprintf(b, "    cache_write = %s tokens  (prefix first established — paid once)\n", commas(c.CacheCreationTokens))
	if c.CacheCreationTokens > 0 {
		fmt.Fprintf(b, "    read : write = %.1f : 1  (each written token was reused ~%.1f×)\n\n",
			ratio(c.CachedPromptTokens, c.CacheCreationTokens), ratio(c.CachedPromptTokens, c.CacheCreationTokens))
	} else {
		b.WriteString("\n")
	}

	b.WriteString("  fak-authored slice (WITNESSED — fak trims the re-sent history to hold budget):\n")
	fmt.Fprintf(b, "    compaction fired ......... %d turns  (bailed %d — bail is a positive: no profitable shed)\n", c.CompactionFired, c.CompactionBailed)
	fmt.Fprintf(b, "    turns dropped ............ %s across those fires (cumulative — re-trimmed each fire)\n", commas(c.CompactionDroppedTurns))
	// Per-fire is the honest headline: the marginal history fak trims each time it
	// fires. The session sum is cumulative and re-counts the same middle, so it is
	// shown only under an explicit label, never as distinct tokens.
	if c.CompactionFired > 0 {
		fmt.Fprintf(b, "    shed PER FIRE ............ ~%s tokens  (%s cumulative ÷ %d fires — the honest marginal)\n",
			commas(c.CompactionShedTokens/c.CompactionFired), commas(c.CompactionShedTokens), c.CompactionFired)
	} else {
		fmt.Fprintf(b, "    shed (cumulative) ........ %s tokens\n", commas(c.CompactionShedTokens))
	}
	fmt.Fprintf(b, "    tool results pruned ...... %s across %s turns\n", commas(c.ToolPruneCount), commas(c.ToolPruneTurns))
	if len(c.CompactionBailReasons) > 0 {
		fmt.Fprintf(b, "    bail reasons ............. %s\n", fmtReasons(c.CompactionBailReasons))
	}
	b.WriteString("\n")

	// The multi-turn crux, stated honestly: fak firing compaction repeatedly is what
	// holds a long session inside budget. We do NOT claim the cumulative sum is
	// distinct tokens saved (that is the retracted double-count); the load-bearing
	// fact is that WITHOUT the per-fire trim the re-sent window grows unbounded.
	if c.CompactionFired > 0 && c.CachedTurns > 1 {
		fmt.Fprintf(b, "  Multi-turn effect: fak fired compaction %d times over %d turns, trimming ~%s\n",
			c.CompactionFired, c.CachedTurns, commas(c.CompactionShedTokens/c.CompactionFired))
		fmt.Fprintf(b, "  tokens of aged history per fire. That repeated trim — not any one-time saving —\n")
		fmt.Fprintf(b, "  is what keeps the re-sent window inside budget as the session grows; the\n")
		fmt.Fprintf(b, "  provider cache reduces the PRICE of the prefix but cannot shrink its SIZE.\n")
		fmt.Fprintf(b, "  (Honest $ value of shed = per-fire count × 0.1× cache-read marginal on warm\n")
		fmt.Fprintf(b, "  fires, #2794/#2798; the distinct-token COUNT axis is open in epic #3095.)\n\n")
	}
}

// renderFleet prints the cumulative owner split: whose cache work earned the saved
// token-equivalent and the avoided dollars, provider vs fak, never blended.
func renderFleet(b *strings.Builder, r cachevaluereport.FleetBenefitReport, since string) {
	b.WriteString("CUMULATIVE OWNER SPLIT")
	if since != "" {
		fmt.Fprintf(b, " (savings rows since %s)", since)
	}
	b.WriteString("\n")
	fmt.Fprintf(b, "  sessions folded .......... %d usage rows, %d exit sessions\n", r.UsageRows, r.ExitSessions)
	// The fak $ figure is now price-honest (shed booked at 0.1× cache-read marginal
	// on warm fires, #2794/#2798). The fak COUNT/token-equiv still rides the
	// cumulative-re-counted shed (epic #3095 open), so it is labeled COUNT-AXIS OPEN
	// and the saved-token-equiv split is presented as an over-count ceiling, not a
	// settled distinct-token figure.
	fmt.Fprintf(b, "  API cost avoided ......... provider $%.2f (OBSERVED) + fak $%.2f (WITNESSED, price-honest) = $%.2f\n",
		r.ProviderAPICostAvoidedUSD, r.FakAPICostAvoidedUSD, r.ObservedAPICostAvoidedUSD)
	fmt.Fprintf(b, "  saved token-equiv ........ provider %s + fak %s = %s  [fak count = CUMULATIVE, re-counted per fire → over-count ceiling, #3095]\n",
		commasF(r.ProviderPromptCacheTokenEq), commasF(r.FakAuthoredTokenEq), commasF(r.TotalSavedTokenEq))
	if r.FakSharePct != nil {
		fmt.Fprintf(b, "  fak share (count-axis) ... %.1f%% — an UPPER BOUND off the over-counted shed; honest owner share is ~0.3–16%% via `fak cachevalue report`\n", *r.FakSharePct)
	}
	fmt.Fprintf(b, "  context shed ............. %s cumulative shed tokens (NOT distinct — re-trimmed each fire, #3095)\n", commas(r.ContextExtensionTokens))
	if r.SpanDays > 0 {
		prov := ""
		if r.RateProvisional {
			prov = " [PROVISIONAL — thin window]"
		}
		fmt.Fprintf(b, "  run-rate ................. $%.2f/week avoided (provider $%.2f/day + fak $%.2f/day over %.1fd)%s\n",
			r.USDAvoidedPerWeek, r.ProviderUSDAvoidedPerDay, r.FakUSDAvoidedPerDay, r.SpanDays, prov)
	}
	b.WriteString("\n")
	fmt.Fprintf(b, "  provenance: %s\n", r.Provenance)
	b.WriteString("  note: fak $ is price-honest (0.1× marginal, #2794/#2798); the shed token COUNT is\n")
	b.WriteString("        cumulative-re-counted pending epic #3095 — cite shed PER FIRE, never the session sum.\n")
}

// --- small local helpers (kept here so the demo has no cmd/fak dependency) ---

func filterTrack1Since(rows []cachevalueledger.Row, since string) []cachevalueledger.Row {
	if since == "" {
		return rows
	}
	out := rows[:0:0]
	for _, r := range rows {
		if r.Date >= since {
			out = append(out, r)
		}
	}
	return out
}

func filterTrack2Since(rows []cachevaluereport.SavingsRow, since string) []cachevaluereport.SavingsRow {
	if since == "" {
		return rows
	}
	out := rows[:0:0]
	for _, r := range rows {
		if r.Date >= since {
			out = append(out, r)
		}
	}
	return out
}

func ratio(a, b uint64) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}

func fmtReasons(m map[string]uint64) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, m[k]))
	}
	return strings.Join(parts, " ")
}

func commas(n uint64) string { return groupThousands(fmt.Sprintf("%d", n)) }

func commasF(f float64) string { return groupThousands(fmt.Sprintf("%.0f", f)) }

func groupThousands(s string) string {
	neg := strings.HasPrefix(s, "-")
	s = strings.TrimPrefix(s, "-")
	n := len(s)
	if n <= 3 {
		if neg {
			return "-" + s
		}
		return s
	}
	var b strings.Builder
	pre := n % 3
	if pre > 0 {
		b.WriteString(s[:pre])
	}
	for i := pre; i < n; i += 3 {
		if b.Len() > 0 {
			b.WriteByte(',')
		}
		b.WriteString(s[i : i+3])
	}
	if neg {
		return "-" + b.String()
	}
	return b.String()
}
