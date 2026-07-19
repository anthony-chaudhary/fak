package main

// resume_backlog_page.go — the shell half of the post-reset backlog SLO gate (#3582).
//
// BOTTLENECK-MAP-2026-07-09.md §7 encodes a decision a human currently eyeballs at every
// throttle reset: "if auto_resume is still >= N with 0 throttled accounts, then the cap is
// the real limiter." At fleet scale nobody watches every reset, so the condition goes unseen
// until throughput visibly stalls. The pure gate (resume.FoldWatchdogStatus) decides; this
// file supplies the one fact the fold cannot read for itself — the roster's throttled-seat
// count — and routes a tripped page to the operator EXACTLY ONCE per signature.
//
// Dedup discipline (the agentic-volume-aware-ticketing rule): a gate that stays tripped keeps
// firing every tick. Filing one issue/toast per tick would bury the operator in the very noise
// the gate exists to cut through, so the page is keyed by its stable signature in _paged.json
// and refreshed with an occurrence count instead of re-notified.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// rwBacklogPageStore is the durable dedup store for fired SLO pages, next to the registry's
// _notified.json (the same once-per-blocker discipline rwAuthAlerts uses for auth walls).
const rwBacklogPageStore = "_paged.json"

// rwBacklogPageRecord is one deduped page's occurrence ledger. Count is what makes a
// refreshed page informative ("still tripped, 47 ticks now") rather than a repeated alarm.
type rwBacklogPageRecord struct {
	Reason     string `json:"reason"`
	Count      int    `json:"count"`
	FirstSeen  string `json:"first_seen"`
	LastSeen   string `json:"last_seen"`
	LastDepth  int    `json:"last_depth"`
	Threshold  int    `json:"threshold"`
	TicksHeld  int    `json:"ticks_held"`
	LastRedRun string `json:"last_red_run,omitempty"`
}

// rwThrottledAccounts counts the roster seats currently held closed by a usage throttle.
// ok=false means the roster could not be read at all — the gate must then FAIL CLOSED and
// never page, because an unreadable roster and a genuinely clear roster both look like
// "0 throttled", and paging on the former is crying wolf at the operator.
func rwThrottledAccounts(regDir string) (count int, ok bool) {
	raw, err := os.ReadFile(filepath.Join(regDir, "sessions.json"))
	if err != nil {
		return 0, false
	}
	var reg struct {
		Accounts []struct {
			Account   string `json:"account"`
			Throttled bool   `json:"throttled"`
			BlockKind string `json:"block_kind"`
		} `json:"accounts"`
	}
	if json.Unmarshal(raw, &reg) != nil {
		return 0, false
	}
	// A roster with no accounts at all is not evidence of an unthrottled fleet — it is a
	// roster we failed to populate. Fail closed there too.
	if len(reg.Accounts) == 0 {
		return 0, false
	}
	for _, a := range reg.Accounts {
		// "throttled" is the roster's own usage-cap flag; block_kind=="usage" is the same
		// fact spelled by the probe path (fleetaccounts.RuntimeStatus sets both together).
		if a.Throttled || strings.EqualFold(strings.TrimSpace(a.BlockKind), "usage") {
			count++
		}
	}
	return count, true
}

// rwEmitBacklogPage routes a tripped SLO page to the operator exactly once per signature and
// records the occurrence count durably. Returns true when it raised a NEW page (the first
// occurrence of this signature), false when it refreshed an already-open one or had nothing
// to do. Best-effort throughout: a notification or store failure must never kill a tick.
func rwEmitBacklogPage(regDir, logDir string, page *resume.WatchdogPage, note func(string, ...any)) bool {
	if page == nil || strings.TrimSpace(page.Signature) == "" {
		return false
	}
	storePath := filepath.Join(regDir, rwBacklogPageStore)
	store := map[string]rwBacklogPageRecord{}
	if b, err := os.ReadFile(storePath); err == nil {
		_ = json.Unmarshal(b, &store)
	}
	now := rwNowISO()
	rec, seen := store[page.Signature]
	rec.Reason = page.Reason
	rec.Count++
	rec.LastSeen = now
	rec.LastDepth = page.Depth
	rec.Threshold = page.Threshold
	rec.TicksHeld = page.Ticks
	if !seen {
		rec.FirstSeen = now
	}
	store[page.Signature] = rec
	if b, err := json.Marshal(store); err == nil {
		_ = os.WriteFile(storePath, b, 0o644)
	}

	if seen {
		// Already open: refresh the occurrence count, do NOT re-notify. This is the whole
		// point of the signature — one standing page, not one per tick.
		note("  PAGE (refreshed x%d, no new notification) %s — depth=%d still > %d",
			rec.Count, page.Reason, page.Depth, page.Threshold)
		return false
	}
	rwToast(logDir, "Resume backlog persists after throttle reset", page.Detail, "warn")
	note("  PAGE %s — %s", page.Reason, page.Detail)
	return true
}
