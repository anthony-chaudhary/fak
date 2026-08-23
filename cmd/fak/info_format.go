package main

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
)

func guardInfoLegend() string {
	var b strings.Builder
	fmt.Fprintln(&b, "what this means:")
	fmt.Fprintln(&b, "  cache  = fak re-uses text it already sent so the model costs less. \"saving money\" = the re-use has paid off; \"reused %\" = how much was re-used; \"×N cheaper\" = how much cheaper; tokens = how much you've saved so far (can start below zero).")
	fmt.Fprintln(&b, "  safety = what fak did to keep you safe: blocked an unsafe action, fixed a risky one before it ran, or set a suspicious result aside.")
	fmt.Fprintln(&b, "  floor  = the capability floor those safety counts come from: every tool call the agent proposes is adjudicated against it BEFORE it runs, so \"nothing blocked\" means the floor admitted every call so far — never that it is off. The counts are session running totals from 0: blocked = refused outright, fixed = rewritten to a safe form and then allowed, set aside = the result quarantined instead of being fed back to the model.")
	fmt.Fprintln(&b, "  why    = the reason code(s) behind those blocks — the same breakdown fak prints when the session ends, now live — plus anything held for a witness or deferred.")
	fmt.Fprintln(&b, "  saved  = \"turns saved\": engine calls fak avoided for you (served from its own cache or handled in-kernel) so the agent never had to make them — shown only once at least one was avoided.")
	fmt.Fprintln(&b, "  replies = model answers completed this session, read from inference.turns — \"turns\" is this same count's name in the JSON and in the exit summary. A whole number counting up from 0 that never decreases while the gateway is up.")
	fmt.Fprintln(&b, "  busy with = requests in flight through the gateway at this instant, read from gateway.inflight_requests. It is a gauge, not a total: it rises when a call starts and drops back to 0 when the call returns, so 0 between turns is normal and one agent usually reads 0 or 1. Staying above 0 while \"replies\" stops moving is the stuck-call tell.")
	fmt.Fprintln(&b, "  running = how long THIS gateway process has been up, read from gateway.uptime_seconds and shown as whole seconds (Ns), minutes (NmNs) or hours (NhNm). It counts up from 0 at gateway start, so a value that suddenly drops means the gateway restarted — not that the pane stalled.")
	fmt.Fprintln(&b, "  assumptions = active facts the session is relying on, with source class, confidence, expiry, and origin reference from public session/debug state.")
	fmt.Fprintln(&b, "  agents = live sessions running through this fak — the main agent plus any sub-agents it spawned, with remaining budget and wall-clock.")
	fmt.Fprintln(&b, "  watchdog = health of the layer that resumes stranded agents and restarts a dead monitor: the rollup verdict (healthy / healing / down / gave up), how many times it has ticked (alive proof), how many resumes it has proven, and whether any monitor needs attention. Absent when no watchdog is running here.")
	fmt.Fprintln(&b, "  \"nothing yet\" = no re-use has happened.")
	return b.String()
}

func guardInfoAssumptionsText(rows []gateway.SessionAssumption) string {
	rows = cleanGuardInfoAssumptions(rows)
	if len(rows) == 0 {
		return ""
	}
	limit := len(rows)
	if limit > 3 {
		limit = 3
	}
	parts := make([]string, 0, limit+1)
	for _, row := range rows[:limit] {
		detail := fmt.Sprintf("%s %s (%s, expires %s",
			guardInfoAssumptionSource(row.Source),
			guardInfoAssumptionSubject(row),
			guardInfoAssumptionConfidence(row.Confidence),
			guardInfoAssumptionExpiry(row.Expiry))
		if ref := strings.TrimSpace(row.SourceRef); ref != "" {
			detail += ", from " + ref
		}
		detail += ")"
		parts = append(parts, detail)
	}
	if extra := len(rows) - limit; extra > 0 {
		parts = append(parts, fmt.Sprintf("+%d more", extra))
	}
	return fmt.Sprintf("assumptions: %d active — %s", len(rows), strings.Join(parts, "; "))
}

func cleanGuardInfoAssumptions(rows []gateway.SessionAssumption) []gateway.SessionAssumption {
	out := make([]gateway.SessionAssumption, 0, len(rows))
	for _, row := range rows {
		row.TraceID = strings.TrimSpace(row.TraceID)
		row.Key = strings.TrimSpace(row.Key)
		row.Statement = strings.TrimSpace(row.Statement)
		row.Source = strings.TrimSpace(row.Source)
		row.Expiry = strings.TrimSpace(row.Expiry)
		row.SourceRef = strings.TrimSpace(row.SourceRef)
		if row.Key == "" && row.Statement == "" {
			continue
		}
		out = append(out, row)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].TraceID != out[j].TraceID {
			return out[i].TraceID < out[j].TraceID
		}
		if out[i].Key != out[j].Key {
			return out[i].Key < out[j].Key
		}
		if out[i].Statement != out[j].Statement {
			return out[i].Statement < out[j].Statement
		}
		return out[i].Source < out[j].Source
	})
	return out
}

func guardInfoAssumptionSource(source string) string {
	switch strings.ToLower(strings.TrimSpace(source)) {
	case "user_stated", "user-stated", "user":
		return "user-stated"
	case "inferred", "infer":
		return "inferred"
	case "queried", "query", "asked":
		return "queried"
	case "witnessed", "witness":
		return "witnessed"
	case "stale":
		return "stale"
	case "unknown":
		return "unknown"
	default:
		return "unknown"
	}
}

func guardInfoAssumptionSubject(row gateway.SessionAssumption) string {
	key := strings.TrimSpace(row.Key)
	stmt := strings.TrimSpace(row.Statement)
	switch {
	case key != "" && stmt != "" && key != stmt:
		return key + "=" + stmt
	case key != "":
		return key
	case stmt != "":
		return stmt
	default:
		return "assumption"
	}
}

func guardInfoAssumptionConfidence(v float64) string {
	if v < 0 {
		v = 0
	}
	if v > 1 {
		v = 1
	}
	return fmt.Sprintf("%.0f%%", v*100)
}

func guardInfoAssumptionExpiry(expiry string) string {
	if expiry = strings.TrimSpace(expiry); expiry != "" {
		return expiry
	}
	return "none"
}

// guardInfoManagedContextText renders issue #1577's concise managed-context status
// (resident/budget tokens, cache state, reset count, query-needed count, stale-
// assumption count) via internal/scorecardpane.RenderContextStatusLine, or "" when
// the gateway has not reported the block at all (a nil pointer — the same "no field
// yet" contract guardInfoPrefixStabilityText already keeps for PrefixStability, so a
// build that has not wired a live managed-context tracker renders nothing extra
// rather than a fabricated all-zero line).
func guardInfoManagedContextText(m *guardInfoManagedContext) string {
	if m == nil {
		return ""
	}
	return scorecardpane.RenderContextStatusLine(scorecardpane.ContextStatusSignals{
		Severity:             scorecardpane.ContextHealthSeverity(m.Severity),
		ResidentTokens:       m.ResidentTokens,
		BudgetTokens:         m.BudgetTokens,
		CacheState:           m.CacheState,
		ResetCount:           m.ResetCount,
		QueryNeededCount:     m.QueryNeededCount,
		StaleAssumptionCount: m.StaleAssumptionCount,
	})
}

// guardInfoPrefixStabilityText renders issue #1602's managed-context prefix-stability
// score in plain words: "prefix: stable", "prefix: mutated (diverged at segment N,
// offset M tokens — kind)", or nothing when the gateway has not reported the block at
// all (a nil pointer — distinct from an explicit "unknown" state, which DOES render, so
// a first-turn session visibly says "no baseline yet" instead of going silent).
func guardInfoPrefixStabilityText(p *guardInfoPrefixStability) string {
	if p == nil {
		return ""
	}
	switch cachemeta.PrefixStabilityState(p.State) {
	case cachemeta.PrefixStable:
		return "prefix: stable"
	case cachemeta.PrefixMutated:
		detail := fmt.Sprintf("prefix: mutated (diverged at segment %d, offset %d tokens", p.FirstDivergentSegment, p.FirstDivergentTokenOffset)
		if p.FirstDivergentKind != "" {
			detail += ", " + p.FirstDivergentKind
		}
		detail += ")"
		if p.ProtectedSpanBroken {
			detail += " [sealed]"
		}
		return detail
	case cachemeta.PrefixUnknown:
		return "prefix: unknown (no baseline yet)"
	default:
		return ""
	}
}

// guardInfoPrefixStabilityFromScore lowers a live cachemeta.PrefixStabilityScore into
// the wire shape rendered above — the seam a gateway (or the offline --prefix-transcript
// path below) uses to populate guardInfoVars.PrefixStability.
func guardInfoPrefixStabilityFromScore(s cachemeta.PrefixStabilityScore) *guardInfoPrefixStability {
	return &guardInfoPrefixStability{
		State:                     string(s.State),
		FirstDivergentSegment:     s.FirstDivergentSegment,
		FirstDivergentTokenOffset: s.FirstDivergentTokenOffset,
		FirstDivergentKind:        string(s.FirstDivergentKind),
		ProtectedSpanBroken:       s.ProtectedSpanBroken,
		Reason:                    s.Reason,
	}
}

// guardCacheWord puts the re-use savings in plain words. The cache lets fak send the same text
// to the model once and re-use it, which costs less. "saving money" means the re-use has more
// than paid back the small extra cost of setting it up; "not saving yet" means it has not — the
// saved-tokens number is below zero until then, so it carries its own sign. reused% is how much
// of the text was served from the cache; ×N is how many times cheaper that made those tokens.
func guardCacheWord(status string, multiplier, savedTokens, hitRate float64) string {
	lead := "cache: saving money"
	if !strings.EqualFold(strings.TrimSpace(status), "PROVEN") {
		lead = "cache: not saving yet"
	}
	return fmt.Sprintf("%s — reused %.0f%% of text, ×%.2f cheaper, %s tokens",
		lead, hitRate*100, multiplier, signedTokens(savedTokens))
}

func guardInfoCacheAttributionText(v guardInfoVars) string {
	if v.CacheAttribution == nil {
		return ""
	}
	provider := v.CacheAttribution.ProviderTokenEquiv
	fak := v.CacheAttribution.FakTokenEquiv
	total := v.CacheAttribution.TotalTokenEquiv
	providerPct, fakPct := ownerSplitPct(provider, fak, total)
	return fmt.Sprintf("split default cache %.0f%% (~%s tok) + fak %.0f%% (~%s tok)",
		providerPct, gateway.HumanTokenEquiv(provider),
		fakPct, gateway.HumanTokenEquiv(fak))
}

// guardInfoManagedCacheText renders the managed-cache posture clause for the live line: the
// 1h TTL-upgrade lever state and, when ACTIVE, whether it is paying (upgrades) or inert
// (every head refused — the #2190 misconfiguration signal, named with its dominant reason so
// the operator can see WHY the lever bought nothing). Empty when the gateway omitted the
// block (a passive, cold session), so the common passthrough line stays quiet.
func guardInfoManagedCacheText(v guardInfoVars) string {
	mc := v.ManagedCache
	if mc == nil {
		return ""
	}
	if !mc.Active {
		return "managed cache off"
	}
	// Wire-aware: the OpenAI Responses (codex) wire has no 1h-TTL lever, so a zero-upgrade
	// ACTIVE session is NOT inert — fak's lever there is the pinned prompt_cache_key. Name that
	// lever instead of the Anthropic "ACTIVE but inert (0 upgrades)" false posture, mirroring
	// bannerLine's `provider == "openai-responses"` branch. Empty wire keeps Anthropic behavior.
	if mc.WireHasNo1hTTLLever() {
		return "managed cache ACTIVE (OpenAI Responses wire: no 1h-TTL lever; stable prompt_cache_key pinned)"
	}
	if mc.Inert {
		if reason := topTTLUpgradeReason(mc.Reasons); reason != "" {
			return fmt.Sprintf("managed cache ACTIVE but inert (0 upgrades, mostly %s)", reason)
		}
		return "managed cache ACTIVE but inert (0 upgrades)"
	}
	return fmt.Sprintf("managed cache ACTIVE (%d TTL-upgraded head(s))", mc.Upgraded)
}

// topTTLUpgradeReason returns the most-frequent refusal reason in the ttl-upgrade outcome
// map, ties broken by reason name so the line is deterministic despite Go's random map
// iteration order. Empty string when the map is empty.
func topTTLUpgradeReason(reasons map[string]uint64) string {
	best, bestN := "", uint64(0)
	for r, n := range reasons {
		if n > bestN || (n == bestN && (best == "" || r < best)) {
			best, bestN = r, n
		}
	}
	return best
}

func ownerSplitPct(provider, fak, total float64) (providerPct, fakPct float64) {
	if total > 0 {
		return provider / total * 100, fak / total * 100
	}
	denom := absFloat(provider) + absFloat(fak)
	if denom == 0 {
		return 0, 0
	}
	return absFloat(provider) / denom * 100, absFloat(fak) / denom * 100
}

func absFloat(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

// guardSafetyWord is the safety summary the live line + panels render. It prefers the
// operation-ledger ADJUDICATION tally — the honest source on the flagship `fak guard --
// claude` proxy, where the kernel adjudicates with Decide and so the raw kernel.Counters
// (the Kernel block) stay structurally ~0, which made the old "safety: blocked N" read a
// vacuous 0 while the floor was actually refusing calls. It falls back to the kernel
// counters for a fak serve gateway that has no adjudication block. Same formatting either
// way via guardFloorSafetyWord.
func guardSafetyWord(v guardInfoVars) string {
	if a := v.Adjudication; a != nil {
		return guardFloorSafetyWord(int64(a.Denied), int64(a.Transformed), int64(a.Quarantined), 0)
	}
	return guardFloorSafetyWord(v.Kernel.Denies, v.Kernel.Transforms, v.Kernel.Quarantines, v.Kernel.ResultDenies)
}

// guardFloorSafetyWord summarizes, in plain words, what fak did this session to keep you safe:
// blocked an unsafe tool call (Denies), fixed a risky one before it ran (Transforms), or set a
// suspicious result aside instead of feeding it to the model (Quarantines plus result-admission
// denials). A clean session reads "safety: nothing blocked" so the all-clear is visible, not a
// blank.
func guardFloorSafetyWord(denies, transforms, quarantines, resultDenies int64) string {
	setAside := quarantines + resultDenies
	if denies == 0 && transforms == 0 && setAside == 0 {
		return "safety: nothing blocked"
	}
	var parts []string
	if denies > 0 {
		parts = append(parts, fmt.Sprintf("blocked %d", denies))
	}
	if transforms > 0 {
		parts = append(parts, fmt.Sprintf("fixed %d", transforms))
	}
	if setAside > 0 {
		parts = append(parts, fmt.Sprintf("set aside %d", setAside))
	}
	return "safety: " + strings.Join(parts, ", ")
}

// guardInfoAdjudicationDetail promotes the guard EXIT summary's forensic "why" into the LIVE
// pane: the top deny/quarantine reason codes (the ByReason breakdown formatAuditSummary prints
// as "blocked: reason xN"), plus the held-for-witness (escalated) and deferred tallies. The
// safety word above answers "how many blocked"; this answers "blocked WHY" — so an operator
// watching the pane sees the reason a call was refused as it happens, not only in the closing
// summary. Empty when the gateway reported no adjudication block (a nil pointer — an older
// gateway or a fak serve gateway) or the session has refused nothing with a recorded reason and
// holds nothing pending, so a clean session stays silent rather than printing a vacuous "why".
func guardInfoAdjudicationDetail(a *gateway.AdjudicationSummary) string {
	if a == nil {
		return ""
	}
	var parts []string
	if reasons := guardInfoTopReasons(a.ByReason, 3); reasons != "" {
		parts = append(parts, "why "+reasons)
	}
	if a.Escalated > 0 {
		parts = append(parts, fmt.Sprintf("%d held for witness", a.Escalated))
	}
	if a.Deferred > 0 {
		parts = append(parts, fmt.Sprintf("%d deferred", a.Deferred))
	}
	return strings.Join(parts, " · ")
}

// guardInfoTopReasons renders the top-`limit` reason codes by count (count desc, then code asc
// for a stable order regardless of the map's iteration order), each as "code xN", with a
// "+K more" tail folding the reasons past the cap so a long tail can never scroll the pane.
// Empty for an empty/all-zero map.
func guardInfoTopReasons(byReason map[string]uint64, limit int) string {
	if len(byReason) == 0 {
		return ""
	}
	type reasonCount struct {
		code  string
		count uint64
	}
	rows := make([]reasonCount, 0, len(byReason))
	for code, n := range byReason {
		if n == 0 {
			continue
		}
		rows = append(rows, reasonCount{code, n})
	}
	if len(rows) == 0 {
		return ""
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].count != rows[j].count {
			return rows[i].count > rows[j].count
		}
		return rows[i].code < rows[j].code
	})
	shown := rows
	extra := 0
	if limit > 0 && len(rows) > limit {
		shown = rows[:limit]
		extra = len(rows) - limit
	}
	parts := make([]string, 0, len(shown)+1)
	for _, r := range shown {
		parts = append(parts, fmt.Sprintf("%s x%d", r.code, r.count))
	}
	if extra > 0 {
		parts = append(parts, fmt.Sprintf("+%d more", extra))
	}
	return strings.Join(parts, ", ")
}

// signedTokens renders a net saved-token-equiv with an explicit sign, because the value is
// NEGATIVE until cache reads repay the cache-creation premium — a "-1,234" reads correctly as
// "still in the red", where a bare "1234" would look like a saving.
func signedTokens(v float64) string {
	n := int64(v)
	if n < 0 {
		return "-" + groupThousands(-n)
	}
	return "+" + groupThousands(n)
}

// groupThousands formats a non-negative integer with comma separators (12345 -> "12,345").
func groupThousands(n int64) string {
	s := strconv.FormatInt(n, 10)
	if len(s) <= 3 {
		return s
	}
	var b strings.Builder
	pre := len(s) % 3
	if pre > 0 {
		b.WriteString(s[:pre])
		if len(s) > pre {
			b.WriteByte(',')
		}
	}
	for i := pre; i < len(s); i += 3 {
		b.WriteString(s[i : i+3])
		if i+3 < len(s) {
			b.WriteByte(',')
		}
	}
	return b.String()
}
