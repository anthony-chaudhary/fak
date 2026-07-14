// floorreloadcache.go is the mid-session-floor-reload prompt-cache cost bench
// (issue #3971). It puts a NUMBER on a contract the gateway already regulates but
// never priced: the gateway prunes provably-floor-denied tool DEFINITIONS from the
// outbound tools[] each turn (gateway.go ToolFloorDenies / CompactInboundTools),
// splicing on the original bytes so the provider prompt-cache prefix stays
// byte-identical — but only WHILE the floor predicate's answers are stable. A
// mid-session floor reload (POST /v1/fak/policy/reload -> SetPolicy, which swaps
// rt.Adjudicator.NeverAdmits, cmd/fak/main.go + guard.go) flips one tool's
// reachability; on the NEXT turn tools[] changes and the cache prefix busts from
// the tools block onward. #2849 (deferred invalidation vs --now) and #2915
// regulate that mutation as a contract, and #2916 will count live break events —
// but the contract has been trading against an unknown token cost. This file
// makes the design-time cost of one reload measurable.
//
// The mechanic modeled (faithful to the gateway splice, gateway.go:600-624):
//
//	prompt = [ stable system prefix | tools[] block | conversation turns... ]
//	                                 ^ cache_control breakpoint sits on the stable
//	                                   head that PRECEDES tools[]
//
// The prompt cache is a PREFIX cache. Across ordinary turns the whole resident
// prefix (system + tools + prior conversation) is re-read cheaply and only the new
// turn's growth is written. A floor reload rewrites the tools[] bytes, so the cache
// is valid ONLY up to the system prefix that precedes tools[]; the tools block and
// EVERY conversation turn resident behind it are re-sent fresh (cache-write price)
// on the reload turn. Those forfeited tokens are the cost. After the reload turn
// the new tools[] is itself cached, so the session returns to the ordinary regime —
// the reload is a one-time surcharge, not a recurring tax.
//
// It reports, for a sweep of session lengths (a single reload injected mid-session
// in each), the three numbers the issue names:
//
//   - tokens forfeited per reload: the cached tokens that flip reused -> fresh
//     because the reload busted the prefix from the tools block onward.
//   - fraction of session spend: the reload's extra billed tokens as a fraction of
//     the whole session's cache-aware bill (how much a single reload moves the
//     needle — smaller in a longer session that amortizes it over more turns).
//   - turns-to-amortize: the reload's extra cost expressed in units of ordinary
//     per-turn spend (the reload burned the equivalent of this many normal turns).
//
// PROVENANCE (net-true doctrine, docs/standards/net-true-value.md): this is a
// hermetic ANALYTIC model under named, policy-visible constants, NOT a live
// provider run. It witnesses the MEASUREMENT and the definitions; the promotion
// seam is #2916's live break-event metric, which can feed a MEASURED forfeited-
// token count into the same report shape and confirm or demote these numbers. The
// invalidating assumptions are listed in the report's `assumptions` block.
//
// Re-run: `go test ./internal/bench -run FloorReloadCache` (the report is also
// regenerable into testdata/floorreloadcache_report.json with UPDATE_GOLDEN=1).
package bench

import "encoding/json"

// FloorReloadModel is the hermetic, named constant set the reload-cost model runs
// over. Every field is policy-visible data (DOS lesson #3: no magic number buried
// in code) so a reviewer can see exactly what economics the numbers rest on, and a
// live #2916 run can override any of it. The cache-price multipliers mirror
// RVCModel's (Anthropic-style: a read is ~0.1x base, a write ~1.25x base) so the
// two benches price the cache the same way.
type FloorReloadModel struct {
	// SystemPrefixTokens is the stable warm head that PRECEDES tools[] (system
	// prompt + the cache_control breakpoint). A floor reload never rewrites it, so
	// it survives the bust — the floor of what stays cached on a reload turn.
	SystemPrefixTokens int `json:"system_prefix_tokens"`
	// ToolsBlockTokens is the outbound tools[] array the floor predicate shapes.
	// This is the block a reload mutates, and the first thing the bust forfeits.
	ToolsBlockTokens int `json:"tools_block_tokens"`
	// GrowthPerTurn is the tokens each turn appends to the conversation AFTER the
	// tools block (tool calls + results + reasoning) — resident context that a
	// reload also forfeits, because it sits behind the busted tools block.
	GrowthPerTurn int `json:"growth_per_turn_tokens"`
	// CacheReadMult / CacheWriteMult are the prompt-cache price multipliers: a
	// reused (cache-read) token bills at ReadMult, a fresh (cache-write) token at
	// WriteMult. The reload's surcharge is forfeited * (WriteMult - ReadMult).
	CacheReadMult  float64 `json:"cache_read_mult"`
	CacheWriteMult float64 `json:"cache_write_mult"`
	// ReloadAtFrac is where in the session the single reload fires, as a fraction
	// of its length (~0.5: mid-session, the issue's "turn k"). A later reload
	// forfeits more resident conversation; mid-session is the representative case.
	ReloadAtFrac float64 `json:"reload_at_frac"`
}

// DefaultFloorReloadModel is the representative constant set. The token sizes are
// chosen so every cache-billing product is an exact integer (no float truncation
// in the golden), and are of the same order as RVCModel's system prefix / growth.
func DefaultFloorReloadModel() FloorReloadModel {
	return FloorReloadModel{
		SystemPrefixTokens: 4_000,
		ToolsBlockTokens:   3_000,
		GrowthPerTurn:      6_000,
		CacheReadMult:      0.1,
		CacheWriteMult:     1.25,
		ReloadAtFrac:       0.5,
	}
}

// DefaultFloorReloadSessions is the sweep of session lengths, in turns. Three
// lengths (the acceptance criterion asks for "3 session lengths") spanning a 4x
// range so the fraction-of-session-spend trend — a single reload matters LESS the
// longer the session that amortizes it — is witnessed, not asserted at one point.
func DefaultFloorReloadSessions() []int { return []int{50, 100, 200} }

// FloorReloadArm is one simulated session's cache-aware token accounting: the
// reused/fresh input-token split the issue asks for, and the effective billed
// tokens under the cache-price multipliers.
type FloorReloadArm struct {
	// ReusedTokens is the total cache-READ (reused) input tokens across the
	// session — the prefix re-read cheaply each turn.
	ReusedTokens int `json:"reused_tokens"`
	// FreshTokens is the total cache-WRITE (fresh / uncached) input tokens across
	// the session — the cold prefix load plus each turn's new growth, plus (in the
	// reload arm) the tokens the reload forced back to write price.
	FreshTokens int `json:"fresh_tokens"`
	// BilledTokens is the cache-aware effective token spend:
	// ReusedTokens*CacheReadMult + FreshTokens*CacheWriteMult.
	BilledTokens int `json:"billed_tokens"`
}

// FloorReloadPoint is one session length: the baseline (no reload) and reload arms
// side by side, and the three headline numbers derived from their difference. Only
// the reload turn differs between the arms — every other turn is byte-identical —
// so the deltas below isolate exactly one reload's cost.
type FloorReloadPoint struct {
	SessionTurns int `json:"session_turns"`
	ReloadAtTurn int `json:"reload_at_turn"`
	// Baseline is the session run with NO reload; Reload injects one at ReloadAtTurn.
	Baseline FloorReloadArm `json:"baseline"`
	Reload   FloorReloadArm `json:"reload"`
	// ForfeitedCachedTokens is the headline: cached tokens that flip reused ->
	// fresh because the reload busted the prefix from the tools block onward. It
	// equals ToolsBlockTokens + (ReloadAtTurn-1)*GrowthPerTurn — the tools block
	// plus every conversation turn resident before the reload.
	ForfeitedCachedTokens int `json:"forfeited_cached_tokens"`
	// ReloadExtraBilled is the reload's surcharge in effective tokens:
	// Reload.BilledTokens - Baseline.BilledTokens, and equivalently
	// ForfeitedCachedTokens * (CacheWriteMult - CacheReadMult).
	ReloadExtraBilled int `json:"reload_extra_billed_tokens"`
	// FractionOfSessionSpend is ReloadExtraBilled / Reload.BilledTokens — how much
	// of the whole session's bill this one reload is responsible for.
	FractionOfSessionSpend float64 `json:"fraction_of_session_spend"`
	// AvgPerTurnBaselineBilled is the baseline arm's mean per-turn billed tokens —
	// the denominator of TurnsToAmortize, surfaced so the metric is transparent.
	AvgPerTurnBaselineBilled float64 `json:"avg_per_turn_baseline_billed"`
	// TurnsToAmortize is ReloadExtraBilled / AvgPerTurnBaselineBilled: the reload
	// burned the equivalent of this many ordinary turns of session spend.
	TurnsToAmortize float64 `json:"turns_to_amortize"`
}

// reloadNone is the sentinel reload turn meaning "no reload this session" — the
// baseline arm. A real reload turn is always > 0 (a reload at turn 0 would bust a
// cache that is still cold and cost nothing, so the model rejects it).
const reloadNone = -1

// simulateFloorSession replays an N-turn session under cache-aware billing. If
// reloadAt is a valid mid-session turn (0 < reloadAt < turns) it injects one floor
// reload there: the tools[] bytes change, so that turn's cache is valid ONLY up to
// the system prefix and everything from the tools block onward is re-sent fresh.
func simulateFloorSession(m FloorReloadModel, turns, reloadAt int) FloorReloadArm {
	prefix := m.SystemPrefixTokens + m.ToolsBlockTokens
	resident := prefix  // input re-sent at the start of a turn, before this turn's growth
	cacheValidUpTo := 0 // tokens of the resident prefix already cached from prior turns
	var reused, fresh int
	var billed float64
	for turn := 0; turn < turns; turn++ {
		cached := cacheValidUpTo
		if cached > resident {
			cached = resident
		}
		// A floor reload this turn flips a tool -> the tools[] block bytes change,
		// so the cache survives only up to the system prefix that PRECEDES tools[];
		// the tools block and all conversation resident behind it re-send fresh.
		if turn == reloadAt && cached > m.SystemPrefixTokens {
			cached = m.SystemPrefixTokens
		}
		uncached := resident - cached
		reused += cached
		fresh += uncached
		billed += float64(cached)*m.CacheReadMult + float64(uncached)*m.CacheWriteMult
		cacheValidUpTo = resident // everything sent this turn is cached going forward
		resident += m.GrowthPerTurn
	}
	return FloorReloadArm{ReusedTokens: reused, FreshTokens: fresh, BilledTokens: int(billed)}
}

// buildFloorReloadPoint folds the baseline and reload arms for one session length
// into the point, deriving the three headline numbers from their difference.
func buildFloorReloadPoint(m FloorReloadModel, turns int) FloorReloadPoint {
	reloadAt := int(m.ReloadAtFrac * float64(turns))
	if reloadAt <= 0 {
		reloadAt = 1 // never a turn-0 reload (nothing cached yet to forfeit)
	}
	if reloadAt >= turns {
		reloadAt = turns - 1
	}
	baseline := simulateFloorSession(m, turns, reloadNone)
	reload := simulateFloorSession(m, turns, reloadAt)

	forfeited := reload.FreshTokens - baseline.FreshTokens
	extra := reload.BilledTokens - baseline.BilledTokens

	var frac float64
	if reload.BilledTokens > 0 {
		frac = round4(float64(extra) / float64(reload.BilledTokens))
	}
	avgPerTurn := float64(baseline.BilledTokens) / float64(turns)
	var amortize float64
	if avgPerTurn > 0 {
		amortize = round4(float64(extra) / avgPerTurn)
	}
	return FloorReloadPoint{
		SessionTurns:             turns,
		ReloadAtTurn:             reloadAt,
		Baseline:                 baseline,
		Reload:                   reload,
		ForfeitedCachedTokens:    forfeited,
		ReloadExtraBilled:        extra,
		FractionOfSessionSpend:   frac,
		AvgPerTurnBaselineBilled: round4(avgPerTurn),
		TurnsToAmortize:          amortize,
	}
}

// FloorReloadCacheReport is the full floorreloadcache.v1 report: the model, the
// per-session-length sweep, a headline at the deepest horizon, and the net-true
// provenance/assumptions block.
type FloorReloadCacheReport struct {
	Schema     string             `json:"schema"`
	Provenance Provenance         `json:"provenance"`
	Model      FloorReloadModel   `json:"model"`
	Sweep      []FloorReloadPoint `json:"sweep"`
	// Headline is the reload cost at the DEEPEST swept session (the horizon that
	// forfeits the most resident conversation — the honest worst case in absolute
	// forfeited tokens, and the best case for fraction-of-spend since it amortizes
	// over the most turns). Empty sweep leaves it zero.
	Headline FloorReloadPoint `json:"headline"`
	// Finding is the one-line human read of the sweep.
	Finding             string   `json:"finding"`
	Assumptions         []string `json:"assumptions"`
	Promotion           string   `json:"promotion"`
	DemotionRetirement  string   `json:"demotion_or_retirement"`
	InvalidatingUnknown string   `json:"invalidating_assumption"`
}

// FloorReloadCacheSchema is the report's stable schema tag.
const FloorReloadCacheSchema = "floorreloadcache.v1"

// BuildFloorReloadCacheReport runs the default model over the default session sweep.
func BuildFloorReloadCacheReport() FloorReloadCacheReport {
	return BuildFloorReloadCacheReportFor(DefaultFloorReloadModel(), DefaultFloorReloadSessions())
}

// BuildFloorReloadCacheReportFor folds an arbitrary model + session sweep — the
// seam a live #2916 break-event run feeds observed constants into.
func BuildFloorReloadCacheReportFor(m FloorReloadModel, sessions []int) FloorReloadCacheReport {
	sweep := make([]FloorReloadPoint, 0, len(sessions))
	for _, n := range sessions {
		sweep = append(sweep, buildFloorReloadPoint(m, n))
	}
	var headline FloorReloadPoint
	for _, p := range sweep {
		if p.SessionTurns >= headline.SessionTurns {
			headline = p
		}
	}
	return FloorReloadCacheReport{
		Schema:     FloorReloadCacheSchema,
		Provenance: simulatedFloorReloadProvenance(),
		Model:      m,
		Sweep:      sweep,
		Headline:   headline,
		Finding: "One mid-session floor reload forfeits the tools block plus every " +
			"conversation turn resident behind it (ToolsBlockTokens + (k-1)*GrowthPerTurn) " +
			"back to cache-write price; its share of session spend shrinks as the session " +
			"length that amortizes it grows.",
		Assumptions: []string{
			"SIMULATED, not a live provider run: the token split is an analytic prefix-cache " +
				"model under the named constants in `model`, not a provider-billed total.",
			"The cache_control breakpoint sits on the stable head that PRECEDES tools[], so a " +
				"reload busts from the tools block onward and the system prefix survives " +
				"(gateway.go ToolFloorDenies / CompactInboundTools byte splice).",
			"Exactly ONE reload fires, mid-session at ReloadAtFrac, flipping one tool's " +
				"reachability so the tools[] bytes change on the next turn. Multiple reloads " +
				"in one session add linearly, each re-forfeiting the resident prefix at its turn.",
			"Context grows by a fixed GrowthPerTurn each turn with no compaction/relay rotation " +
				"in the window — so the only prefix bust modeled is the reload itself.",
			"Cache prices are Anthropic-style multipliers (read ~0.1x, write ~1.25x base); a " +
				"provider with different cache economics moves the surcharge proportionally.",
		},
		Promotion: "Promote by feeding #2916's live floor-reload break-event metric (measured " +
			"forfeited cached tokens per reload) into this same report shape; a MEASURED count " +
			"that matches the modeled ToolsBlockTokens + (k-1)*GrowthPerTurn confirms the model " +
			"and flips provenance to OBSERVED, mirroring #2915's contract and the " +
			"adjudication_latency_test projected->measured precedent.",
		DemotionRetirement: "Demote/retire if #2916 measures a materially different forfeit (e.g. " +
			"the provider caches tools[] independently of the conversation, or the breakpoint " +
			"placement changes so the bust does not extend from the tools block onward), or if " +
			"the floor-reload path is removed so tools[] no longer mutates mid-session.",
		InvalidatingUnknown: "The load-bearing unknown is the breakpoint's true position relative " +
			"to tools[]: if the provider prefix-cache boundary falls AFTER tools[] (tools cached " +
			"in a separate segment), a reload would forfeit far fewer tokens than modeled here.",
	}
}

// simulatedFloorReloadProvenance labels the hermetic analytic path.
func simulatedFloorReloadProvenance() Provenance {
	return Provenance{
		Kind:        ProvenanceSimulated,
		Command:     "go test ./internal/bench -run FloorReloadCache",
		GeneratedBy: "fak/internal/bench.BuildFloorReloadCacheReport",
		Note: "Hermetic ANALYTIC prefix-cache model of ONE mid-session floor reload busting " +
			"the tools[] cache prefix (#3971). It witnesses the MEASUREMENT and the token-split " +
			"definitions under the named constants in `model`; #2916's live break-event metric " +
			"is the promotion seam that can feed a MEASURED forfeited-token count into the same " +
			"report shape and confirm or demote these numbers.",
	}
}

// JSON renders the report as canonical indented JSON (the golden artifact form).
func (r FloorReloadCacheReport) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}
