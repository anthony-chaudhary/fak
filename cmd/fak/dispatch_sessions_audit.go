package main

// dispatch_sessions_audit.go — `fak dispatch sessions --file-issues`, the
// systemic-waste lens over the per-session view (#3336).
//
// `fak dispatch sessions` reports WHAT ran; this lens folds that same snapshot into
// systemic-waste findings — a retry/quota-storm rate concentrated on one
// backend/lane, a cache-read-share COLLAPSE across a lane's substantial sessions, or
// a cluster of token-heavy NO_OP sessions burning spend for nothing — and emits each
// as a FINGERPRINTED, DEDUPED improvement-ticket candidate.
//
//	fak dispatch sessions --file-issues            # dry-run: print candidates only
//	fak dispatch sessions --file-issues --confirm  # actually open a gh issue per NEW finding
//	fak dispatch sessions --file-issues --confirm --max-issues 5   # worst-first, capped
//
// The detector (foldDispatchSessionsWaste) is PURE — snapshot in, deterministic
// findings out, no I/O — so a test drives it hermetically. The filing half reuses the
// exact detect->fingerprint->dedup->file->mark substrate `fak dispatch audit
// --file-issues` already ships (dispatchaudit.SelectFindingsToFile + AlreadyFiled +
// the shared fileAuditFindings gh shell), so the two lenses never drift on how a
// candidate becomes an issue. The load-bearing safety choice: --file-issues DEFAULTS
// to a dry run — it prints candidates and touches nothing — and only --confirm opens
// gh issues, so a routine sweep can never storm the tracker.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
)

// dispatchSessionsWasteThresholds bounds the three systemic-waste detectors. Every
// knob has a conservative default (defaultDispatchSessionsWasteThresholds) tuned so a
// single cold/short session never trips a finding — waste must be a PATTERN across a
// backend/lane before it earns a ticket.
type dispatchSessionsWasteThresholds struct {
	// MinGroup is how many finished sessions a (backend, lane) group needs before a
	// rate is judged systemic rather than anecdotal.
	MinGroup int
	// WasteRate is the fraction of a group's finished sessions that must classify as a
	// waste outcome (retry storm / quota wall / wasted spawn) to flag the group.
	WasteRate float64
	// CacheReadFloor is the cache-read share below which a SUBSTANTIAL session (>=
	// CacheMinTokens) counts as "cache-collapsed".
	CacheReadFloor float64
	// CacheMinTokens is the token volume a session must move before its cache-read
	// share is meaningful (a tiny session legitimately reads no cache).
	CacheMinTokens uint64
	// CacheCollapseMin is how many cache-collapsed sessions a group needs to flag it.
	CacheCollapseMin int
	// NoOpMinTokens is the token spend that makes a NO_OP / WASTED_SPAWN session
	// "token-heavy" — spend burned for no shipped result.
	NoOpMinTokens uint64
	// NoOpMinCount is how many token-heavy no-ops a group needs to flag it.
	NoOpMinCount int
}

func defaultDispatchSessionsWasteThresholds() dispatchSessionsWasteThresholds {
	return dispatchSessionsWasteThresholds{
		MinGroup:         3,
		WasteRate:        0.5,
		CacheReadFloor:   0.2,
		CacheMinTokens:   20000,
		CacheCollapseMin: 2,
		NoOpMinTokens:    20000,
		NoOpMinCount:     2,
	}
}

// dispatchSessionsWasteFingerprint is the stable 16-hex hash over a code-site,
// namespaced away from dispatchaudit's own outcome fingerprints so the two feeders
// can never collide on a marker. Same (class, backend, lane) → same fingerprint
// across runs and log timestamps, which is what makes the dedup durable.
func dispatchSessionsWasteFingerprint(codeSite string) string {
	sum := sha256.Sum256([]byte("dispatch-sessions-waste/" + codeSite))
	return hex.EncodeToString(sum[:])[:16]
}

// dispatchSessionsWasteOutcomes is the set of finished-session outcomes that count as
// wasted work for the storm-rate detector.
var dispatchSessionsWasteOutcomes = map[string]bool{
	string(dispatchaudit.OutcomeRetryStorm):  true,
	string(dispatchaudit.OutcomeQuotaWalled): true,
	string(dispatchaudit.OutcomeWastedSpawn): true,
	string(dispatchaudit.OutcomeErrored):     true,
}

// sessionGroup accumulates the per-(backend, lane) stats the three detectors read.
type sessionGroup struct {
	backend   string
	lane      string
	finished  int // non-live sessions (a running worker is not yet waste)
	wasted    int // finished sessions with a waste outcome
	collapsed int // substantial sessions whose cache-read share fell below the floor
	worstCach float64
	noops     int    // token-heavy NO_OP / WASTED_SPAWN sessions
	noopToks  uint64 // tokens burned by those no-ops
}

// finding builds one waste finding for this (backend, lane) group. The three detectors differ
// only in their class prefix, headline, and evidence sentence — the code-site spelling, the
// namespaced fingerprint derived from it, and the backend binding are identical every time, so
// they are derived here once instead of being re-spelled (and re-diverged) per detector.
func (g *sessionGroup) finding(class, headline, detail string) dispatchaudit.Finding {
	site := class + "/" + g.backend + "/" + g.lane
	return dispatchaudit.Finding{
		Fingerprint: dispatchSessionsWasteFingerprint(site),
		Backend:     dispatchaudit.Backend(g.backend),
		CodeSite:    site,
		Title:       "dispatch waste: " + headline + " on " + g.backend + " (lane " + g.lane + ")",
		Detail:      detail,
	}
}

// foldDispatchSessionsWaste is the PURE systemic-waste detector: it folds the session
// snapshot into deterministic, fingerprinted findings. It groups by (backend, lane),
// then evaluates each group against the three detectors, walking the group keys in
// sorted order so the returned slice is stable for the same input.
func foldDispatchSessionsWaste(snap dispatchSessionsSnapshot, th dispatchSessionsWasteThresholds) []dispatchaudit.Finding {
	groups := map[string]*sessionGroup{}
	order := []string{}
	for _, s := range snap.Sessions {
		backend := s.Backend
		if backend == "" {
			backend = "unknown"
		}
		lane := s.Lane
		if lane == "" {
			lane = "unknown-lane"
		}
		key := backend + "\x00" + lane
		g, ok := groups[key]
		if !ok {
			g = &sessionGroup{backend: backend, lane: lane, worstCach: 1}
			groups[key] = g
			order = append(order, key)
		}
		if s.Live {
			continue // a still-running worker has not yet wasted anything
		}
		g.finished++
		if dispatchSessionsWasteOutcomes[s.Outcome] {
			g.wasted++
		}
		if s.Tokens >= th.CacheMinTokens && s.CacheReadShare < th.CacheReadFloor {
			g.collapsed++
			if s.CacheReadShare < g.worstCach {
				g.worstCach = s.CacheReadShare
			}
		}
		if (s.Outcome == string(dispatchaudit.OutcomeNoOp) || s.Outcome == string(dispatchaudit.OutcomeWastedSpawn)) && s.Tokens >= th.NoOpMinTokens {
			g.noops++
			g.noopToks += s.Tokens
		}
	}
	sort.Strings(order)

	var findings []dispatchaudit.Finding
	for _, key := range order {
		g := groups[key]

		// 1. Storm / waste rate concentrated on a backend+lane.
		if g.finished >= th.MinGroup {
			rate := float64(g.wasted) / float64(g.finished)
			if rate >= th.WasteRate {
				findings = append(findings, g.finding("waste-rate", "high retry/quota-storm rate",
					fmt.Sprintf("%d of %d finished %s/%s sessions wasted work (retry storm / quota wall / wasted spawn / errored) — a %.0f%% waste rate. Investigate the backend/lane routing or the account cooldown before spending more here.",
						g.wasted, g.finished, g.backend, g.lane, rate*100)))
			}
		}

		// 2. Cache-read-share collapse across substantial sessions.
		if g.collapsed >= th.CacheCollapseMin {
			findings = append(findings, g.finding("cache-read-collapse", "cache-read-share collapse",
				fmt.Sprintf("%d substantial %s/%s sessions read <%.0f%% of their prompt tokens from cache (worst %.0f%%). A cold prefix on every spawn means the stable head is not being reused — check the prompt-prefix stability / managed-cache lever for this lane.",
					g.collapsed, g.backend, g.lane, th.CacheReadFloor*100, g.worstCach*100)))
		}

		// 3. Token-heavy no-op cluster (spend with no shipped result).
		if g.noops >= th.NoOpMinCount {
			findings = append(findings, g.finding("token-heavy-noop", "token-heavy no-op sessions",
				fmt.Sprintf("%d %s/%s sessions burned ~%d tokens total and shipped nothing (NO_OP / WASTED_SPAWN). That spend is pure waste — tighten the pre-dispatch gate or the worker prompt for this lane.",
					g.noops, g.backend, g.lane, g.noopToks)))
		}
	}
	return findings
}

// runDispatchSessionsAudit runs the waste lens over an already-folded snapshot. It
// DEFAULTS to a dry run (live=false): it prints the deduped candidates and writes
// nothing. Only live=true (--confirm) reaches the shared fileAuditFindings gh shell —
// the same detect->dedup->file->mark path `fak dispatch audit --file-issues` uses —
// which opens an issue per NEW fingerprint and drops the marker so it is never
// re-filed.
func runDispatchSessionsAudit(stdout, stderr io.Writer, runsDir string, snap dispatchSessionsSnapshot, live bool, maxIssues int) int {
	findings := foldDispatchSessionsWaste(snap, defaultDispatchSessionsWasteThresholds())
	fmt.Fprintf(stdout, "dispatch sessions audit — %d session(s), %d systemic-waste finding(s)\n", snap.SessionCount, len(findings))
	if len(findings) == 0 {
		fmt.Fprintln(stdout, "no systemic waste detected.")
		return 0
	}

	if live {
		// Reuse the exact filing substrate `fak dispatch audit --file-issues` ships:
		// marker + open-title dedup, worst-first cap, gh issue create, marker write.
		return fileAuditFindings(stdout, stderr, runsDir, dispatchaudit.Report{Findings: findings}, maxIssues)
	}

	// Dry run: dedup against on-disk markers ONLY (no gh call, so a sweep is hermetic
	// and side-effect free), then print the candidates that WOULD be filed.
	filed := map[string]bool{}
	for _, f := range findings {
		if dispatchaudit.AlreadyFiled(runsDir, f.Fingerprint) {
			filed[f.Fingerprint] = true
		}
	}
	fresh, withheld := dispatchaudit.SelectFindingsToFile(findings, filed, map[string]bool{}, maxIssues)
	if len(fresh) == 0 {
		fmt.Fprintln(stdout, "all findings already filed (deduped by marker); nothing new. Pass --confirm to (re)file.")
		return 0
	}
	fmt.Fprintln(stdout, "\ncandidate improvement tickets (dry-run — pass --confirm to file):")
	for _, f := range fresh {
		fmt.Fprintf(stdout, "  %s  %s\n      %s\n", f.Fingerprint, f.Title, strings.TrimSpace(f.Detail))
	}
	if withheld > 0 {
		fmt.Fprintf(stdout, "(%d more withheld by --max-issues=%d)\n", withheld, maxIssues)
	}
	return 0
}
