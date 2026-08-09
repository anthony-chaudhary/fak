package cachevaluereport

// inert.go — the CONFIGURED_BUT_INERT per-session diff loop (#3649), one scoped leaf
// under epic #3569 (independent trust-but-verify LOOPS for managed cache).
//
// The headline question this loop answers is "shed works in theory — is it working in
// PRACTICE?": a managed-cache lever can be CONFIGURED on (posture ACTIVE,
// --defer-cold-tools, 1h-TTL upgrade enabled) yet be INERT — fire nothing (0 upgrades,
// 0 cold-defers, a byte-identical body with no realized reuse). Every other cache-value
// fold in this package prices what the cache SAVED; none diffs a session's config
// INTENT against its observed EFFECT to name a lever that was armed and did nothing.
// This one does.
//
// The fold is PURE and deterministic (sessions in, report out; `now` only stamps
// GeneratedAt), mirroring FoldWeeklyDigest / FoldShapes. It is a diagnostic LOOP, not a
// CI gate: it reports CONFIGURED_BUT_INERT findings and keeps OK=true, exactly the
// "report, not a gate" posture FoldShapes takes — a surfaced inert lever is a cue to
// investigate its wiring, not a red build. Fixing the inert lever is out of scope.

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// InertSchema versions the CONFIGURED_BUT_INERT report envelope so a downstream reader
// can pin it independently of the other cache-value fold schemas.
const InertSchema = "fak-cache-configured-but-inert/1"

// The closed lever vocabulary this loop diffs. Each lever pairs a CONFIG intent (was it
// armed this session) with the single OBSERVED effect that proves it did something.
const (
	// LeverManagedCache — managed-cache posture was ACTIVE; its effect is realized cache
	// reuse (fak-authored KV-prefix reuse, the provider's own prompt-cache read, or both).
	// Nothing reused turn-over-turn (reuse ratio 0) under an active posture is the lever
	// doing nothing.
	LeverManagedCache = "managed_cache"
	// LeverDeferColdTools — --defer-cold-tools was configured on; its effect is cold
	// tool defs actually deferred. Zero cold-defers under the flag is inert.
	LeverDeferColdTools = "defer_cold_tools"
	// LeverCacheTTLUpgrade — the managed-cache 1h-TTL upgrade lever was enabled; its
	// effect is upgrades fired. Zero upgrades under the enabled lever is inert — the
	// "armed but every head refused" session.
	LeverCacheTTLUpgrade = "cache_ttl_upgrade"
)

// leverRank orders findings deterministically by lever after the session key.
var leverRank = map[string]int{
	LeverManagedCache:    0,
	LeverDeferColdTools:  1,
	LeverCacheTTLUpgrade: 2,
}

// The report verdicts. This is an observability loop, so CONFIGURED_BUT_INERT is a
// finding to read, not a failure — OK stays true for every verdict.
const (
	VerdictConfiguredButInert = "CONFIGURED_BUT_INERT"
	VerdictClean              = "CLEAN"
	VerdictInsufficient       = "INSUFFICIENT"
)

// SessionLevers is one session's CONFIG intent crossed with its OBSERVED effect — the
// diff loop's input pair. The intent flags say which levers were armed; the effect
// counters say what each actually did. ReuseRatio is the realized reuse share
// (reused/prompt, 0 = nothing was reused turn-over-turn), supplied already computed by
// the caller so this fold stays pure — see usageRowReuseRatio for what the durable-row
// caller counts as reuse.
type SessionLevers struct {
	Session string `json:"session"` // session id/label for the finding

	// CONFIG intent — which levers were configured ON for this session.
	ManagedCacheActive bool `json:"managed_cache_active"`
	DeferColdTools     bool `json:"defer_cold_tools"`
	UpgradeEnabled     bool `json:"upgrade_enabled"`

	// OBSERVED effect — what each armed lever actually did this session.
	UpgradesFired uint64  `json:"upgrades_fired"`
	ColdDefers    uint64  `json:"cold_defers"`
	ReuseRatio    float64 `json:"reuse_ratio"`
}

// InertLever is one CONFIGURED_BUT_INERT finding: a lever that was armed for a session
// and produced its zero effect. Intent names what was configured on; Effect names the
// observed zero that makes it inert.
type InertLever struct {
	Session string `json:"session"`
	Lever   string `json:"lever"`
	Intent  string `json:"intent"`
	Effect  string `json:"effect"`
	Reason  string `json:"reason"`
}

// InertReport is the whole per-session diff: every armed-but-inert lever across the
// supplied sessions, plus a verdict. It is a diagnostic LOOP report, not a CI gate — OK
// stays true even when findings exist (mirroring FoldShapes).
type InertReport struct {
	Schema      string `json:"schema"`
	GeneratedAt string `json:"generated_at"`

	Sessions    int `json:"sessions"`     // sessions diffed
	LeversOn    int `json:"levers_on"`    // total armed levers across all sessions
	InertLevers int `json:"inert_levers"` // len(Findings)

	Findings []InertLever `json:"findings,omitempty"`

	Verdict    string `json:"verdict"` // CONFIGURED_BUT_INERT | CLEAN | INSUFFICIENT
	Finding    string `json:"finding"`
	NextAction string `json:"next_action,omitempty"`
	OK         bool   `json:"ok"`
}

// FoldConfiguredButInert diffs each session's config intent against its observed effect
// and emits a CONFIGURED_BUT_INERT finding for every lever that was armed yet did
// nothing. Pure and deterministic: findings are ordered by session then lever rank, and
// `now` only stamps GeneratedAt. A fully-effective session (every armed lever fired)
// yields no finding; a session with no armed lever contributes nothing to verify.
func FoldConfiguredButInert(sessions []SessionLevers, now time.Time) InertReport {
	rep := InertReport{
		Schema:      InertSchema,
		GeneratedAt: now.UTC().Format(time.RFC3339),
		Sessions:    len(sessions),
		OK:          true,
	}
	for _, s := range sessions {
		label := strings.TrimSpace(s.Session)
		if label == "" {
			label = "unknown"
		}
		// managed-cache posture ACTIVE → expects realized KV-prefix reuse.
		if s.ManagedCacheActive {
			rep.LeversOn++
			if s.ReuseRatio <= 0 {
				rep.Findings = append(rep.Findings, InertLever{
					Session: label,
					Lever:   LeverManagedCache,
					Intent:  "managed-cache posture ACTIVE",
					Effect:  "0 realized KV-prefix reuse (byte-identical body)",
					Reason:  "managed cache was active but the body reused nothing turn-over-turn",
				})
			}
		}
		// --defer-cold-tools → expects cold tool defs to be deferred.
		if s.DeferColdTools {
			rep.LeversOn++
			if s.ColdDefers == 0 {
				rep.Findings = append(rep.Findings, InertLever{
					Session: label,
					Lever:   LeverDeferColdTools,
					Intent:  "--defer-cold-tools configured on",
					Effect:  "0 cold tool defs deferred",
					Reason:  "cold-tool deferral was armed but no cold def was ever deferred",
				})
			}
		}
		// 1h-TTL upgrade enabled → expects upgrades to fire.
		if s.UpgradeEnabled {
			rep.LeversOn++
			if s.UpgradesFired == 0 {
				rep.Findings = append(rep.Findings, InertLever{
					Session: label,
					Lever:   LeverCacheTTLUpgrade,
					Intent:  "managed-cache 1h-TTL upgrade enabled",
					Effect:  "0 upgrades fired",
					Reason:  "the 1h-TTL upgrade lever was enabled but every head was refused (0 upgrades)",
				})
			}
		}
	}
	sort.SliceStable(rep.Findings, func(i, j int) bool {
		if rep.Findings[i].Session != rep.Findings[j].Session {
			return rep.Findings[i].Session < rep.Findings[j].Session
		}
		return leverRank[rep.Findings[i].Lever] < leverRank[rep.Findings[j].Lever]
	})
	rep.InertLevers = len(rep.Findings)
	rep.finalize()
	return rep
}

// finalize sets the report-contract fields from the folded totals.
func (rep *InertReport) finalize() {
	switch {
	case rep.LeversOn == 0:
		rep.Verdict = VerdictInsufficient
		if rep.Sessions == 0 {
			rep.Finding = "no sessions to diff"
			rep.NextAction = "supply per-session config intent + observed effect pairs (e.g. from the gateway usage ledger), then re-fold"
		} else {
			rep.Finding = fmt.Sprintf("%d session(s), no configured managed-cache lever to verify", rep.Sessions)
			rep.NextAction = "arm a managed-cache lever (--managed-cache / --defer-cold-tools / 1h-TTL upgrade) to give this loop something to verify"
		}
	case rep.InertLevers == 0:
		rep.Verdict = VerdictClean
		rep.Finding = fmt.Sprintf("%d armed lever(s) across %d session(s); every one fired (no inert lever)", rep.LeversOn, rep.Sessions)
	default:
		rep.Verdict = VerdictConfiguredButInert
		rep.Finding = fmt.Sprintf("%d CONFIGURED_BUT_INERT lever(s) across %d session(s) (%d armed): %s",
			rep.InertLevers, rep.Sessions, rep.LeversOn, rep.inertSummary())
		rep.NextAction = "each named lever was configured on and did nothing this session — verify its wiring is reached in practice (this loop names the inert lever; fixing it is out of scope)"
	}
}

// inertSummary is a compact, deterministic lever×count roll-up for the finding line.
func (rep *InertReport) inertSummary() string {
	byLever := map[string]int{}
	for _, f := range rep.Findings {
		byLever[f.Lever]++
	}
	levers := make([]string, 0, len(byLever))
	for l := range byLever {
		levers = append(levers, l)
	}
	sort.Slice(levers, func(i, j int) bool { return leverRank[levers[i]] < leverRank[levers[j]] })
	parts := make([]string, 0, len(levers))
	for _, l := range levers {
		parts = append(parts, fmt.Sprintf("%s×%d", l, byLever[l]))
	}
	return strings.Join(parts, ", ")
}

// LeversFromUsageRow derives a SessionLevers diff-input from a durable gateway-usage exit
// row, arming all THREE managed-cache levers off that row's own intent fields (#4349).
//
// Each lever's intent comes from the narrowest durable witness the row carries:
//   - cache_ttl_upgrade is "enabled" when the lever left durable evidence this session —
//     an actual upgrade or a refusal-reason row — the same active-lever witness
//     FoldWeeklyDigest trusts (a zero upgraded count WITH refusal reasons is the "armed
//     but every head refused" session, i.e. CONFIGURED_BUT_INERT).
//   - managed_cache and defer_cold_tools read their explicit armed flags
//     (Counters.ManagedCacheActive / Counters.DeferColdToolsArmed). Those fields exist
//     precisely so intent is never INFERRED here: the previous version of this adapter
//     left both levers unarmed because deriving managed-cache intent from KVPrefix reuse
//     alone false-positives on a provider-prompt-cache-only session (legitimately zero
//     fak-authored KV reuse), and because the --defer-cold-tools intent and its
//     cold-defer effect were not in the durable row at all.
//
// HONESTY FENCE, still: an armed flag is `omitempty` on the wire, so a row that predates
// those fields is indistinguishable from a measured lever-OFF row. Both leave the lever
// unarmed here, which is the safe direction — this loop can under-report an inert lever on
// an old row, never invent one.
func LeversFromUsageRow(r gatewayusageledger.Row) SessionLevers {
	c := r.Counters
	upgradeArmed := c.CacheTTLUpgradesUpgraded > 0 || len(c.CacheTTLUpgradeReasons) > 0
	return SessionLevers{
		Session:            usageRowLabel(r),
		ManagedCacheActive: c.ManagedCacheActive,
		DeferColdTools:     c.DeferColdToolsArmed,
		UpgradeEnabled:     upgradeArmed,
		UpgradesFired:      c.CacheTTLUpgradesUpgraded,
		ColdDefers:         c.DeferColdCount,
		ReuseRatio:         usageRowReuseRatio(c),
	}
}

// usageRowReuseRatio is the row's REALIZED reuse share — the effect half of the
// managed_cache lever. It deliberately counts BOTH reuse mechanisms an active managed-cache
// session can pay off through: fak-authored in-kernel KV-prefix reuse (WITNESSED,
// KVPrefixReusedTokens) and the provider's own prompt-cache read (OBSERVED,
// CachedPromptTokens, which is what the 1h-TTL splice and a pinned prompt_cache_key protect).
// Counting only the first is the exact false "not working" this bridge must not emit: a
// passthrough session whose whole payoff is provider-side shows zero KV reuse.
//
// The denominator sums the prompt tokens each mechanism saw, so on a row that somehow
// booked both the share is a blend rather than a per-mechanism ratio — FoldWeeklyDigest is
// the reader for a mechanism-split number. That imprecision cannot move this fold's answer:
// FoldConfiguredButInert only reads the zero/nonzero boundary, and the numerator is zero
// exactly when neither mechanism reused a token.
func usageRowReuseRatio(c gatewayusageledger.Counters) float64 {
	reused := c.KVPrefixReusedTokens + c.CachedPromptTokens
	if reused == 0 {
		return 0
	}
	seen := c.KVPrefixPromptTokens + c.CachedPromptTokens + c.InputTokens + c.CacheCreationTokens
	if seen == 0 {
		return 0
	}
	return float64(reused) / float64(seen)
}

// usageRowLabel names a session for a finding, preferring the session id and falling
// back to pid@unix_millis (always enough to distinguish rows, matching the ledger).
func usageRowLabel(r gatewayusageledger.Row) string {
	if id := strings.TrimSpace(r.SessionID); id != "" {
		return id
	}
	return fmt.Sprintf("pid%d@%d", r.PID, r.UnixMillis)
}

// FoldUsageRowsConfiguredButInert runs the CONFIGURED_BUT_INERT loop over durable
// gateway-usage rows: it diffs every lever those rows arm (all three since #4349) for
// every exit row, skipping periodic/carryforward rows exactly as FoldWeeklyDigest does
// (a periodic row double-counts a live session; a carryforward is a synthetic pre-cut
// sum, not a session). It is the file-fed sibling of FoldConfiguredButInert.
func FoldUsageRowsConfiguredButInert(rows []gatewayusageledger.Row, now time.Time) InertReport {
	sessions := make([]SessionLevers, 0, len(rows))
	for _, r := range rows {
		if r.Kind != "exit" {
			continue
		}
		sessions = append(sessions, LeversFromUsageRow(r))
	}
	return FoldConfiguredButInert(sessions, now)
}

// RenderConfiguredButInert renders the diff as a compact, deterministic terminal block.
func RenderConfiguredButInert(r InertReport) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "managed-cache configured-but-inert diff (per session) — %s\n", r.Verdict)
	fmt.Fprintf(&sb, "  %s\n", r.Finding)
	if r.NextAction != "" {
		fmt.Fprintf(&sb, "  next: %s\n", r.NextAction)
	}
	if len(r.Findings) == 0 {
		return sb.String()
	}
	fmt.Fprintf(&sb, "\n  %-20s  %-18s  %-34s  %s\n", "session", "lever", "intent", "observed (inert)")
	for _, f := range r.Findings {
		fmt.Fprintf(&sb, "  %-20s  %-18s  %-34s  %s\n", f.Session, f.Lever, f.Intent, f.Effect)
	}
	return sb.String()
}
