package main

import (
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

// accounts_headroom.go — the bridge that makes the config-plane ROTATION (fak accounts
// next / launch --rotate) headroom-aware. The config registry (internal/accounts) is pure
// over disk-derived identity and carries no usage/quota telemetry, so on its own it can only
// rotate stable-by-name — which happily hands back an account the live fleet layer already
// knows is capped or unservable. This derives the missing signal from the RUNTIME layer
// (internal/fleetaccounts: the same usage-throttle / offerability fold `fak fleet-accounts`
// prints) and folds it into an accounts.RotationHeadroom the pure planner consumes.
//
// The score is a BANDED, HONEST tier: the runtime layer tells us "offerable now", "capped /
// can't serve now", or "unknown", and within the two known tiers a coarse signal already on
// the row breaks what used to be an arbitrary name-order tie. The bands never overlap, so the
// pure planner's plain most-headroom-first sort orders tiers correctly AND orders within a
// tier without any planner change:
//
//	(1,2]  OFFERABLE (available, not throttled, creds present). Base +1, plus a load bonus
//	       1/(1+live_sessions) so the LEAST-loaded offerable bucket sorts first. Always > 1,
//	       so every offerable bucket still outranks unknown and walled.
//	 0     UNKNOWN — no runtime row for the bucket. Neither preferred nor penalised.
//	[-1,0) WALLED (usage-throttled or blocked). Base -1, plus a reset-soonness bonus in [0,1)
//	       so among walled buckets the SOONEST-to-reset (recovers first) sorts highest. Always
//	       < 0, so a walled bucket never outranks an unknown or offerable one.
//
// This is the "finer most-room-first" ordering the header used to defer: it is a TIE-BREAK
// within a tier from signals the roster already carries (live_sessions, the throttle reset
// string), NOT a continuous remaining-quota number — a real quota API remains a follow-on.

// rotationHeadroom builds the config-plane rotation headroom signal from the live runtime
// roster. homeDir is the same home the rotation registry was discovered under (the --home
// flag), so a test running against an isolated TempDir home discovers no real accounts and
// gets an empty signal — which makes RotationPlan fall back to the historical stable-by-name
// order, keeping unit tests hermetic. A nil/empty result means "no signal" by design.
func rotationHeadroom(homeDir string) accounts.RotationHeadroom {
	cwd, _ := os.Getwd()
	toolsDir := filepath.Join(findRepoRoot(cwd), "tools")
	paths := fleetaccounts.ResolvePaths(toolsDir)
	home := homeDir
	if home == "" {
		home = paths.Home
	}
	pol := fleetaccounts.LoadPolicy(paths)
	reg := fleetaccounts.LoadRegistry(paths.RegistryPath)
	rows := fleetaccounts.AnnotatedRoster(home, paths.ConfigHome, pol, reg)
	return headroomFromRoster(rows, time.Now().UTC())
}

// headroomFromRoster folds an annotated runtime roster into a per-bucket headroom score keyed
// by the SAME account-bucket key the config registry dedups on ("uuid:<AccountUUID>"), so the
// scores line up with the pool seats. It scores only Claude worker rows (the config-plane
// rotation is over Claude CLAUDE_CONFIG_DIR seats) that carry a resolved AccountUUID. When
// several dirs map to one bucket the BEST score wins: a bucket has room if ANY of its dirs can
// be offered (one rate-limit window, served by whichever dir holds live creds). now anchors the
// walled tier's reset-soonness tie-break (injected for deterministic tests). It is pure — no
// I/O — so the banded tiering is unit-tested directly.
func headroomFromRoster(rows []fleetaccounts.Account, now time.Time) accounts.RotationHeadroom {
	hr := accounts.RotationHeadroom{}
	for _, r := range rows {
		if r.Product != "claude" || r.AccountUUID == nil || *r.AccountUUID == "" {
			continue
		}
		key := "uuid:" + *r.AccountUUID
		score := bucketScore(r, now)
		if cur, ok := hr[key]; !ok || score > cur {
			hr[key] = score
		}
	}
	if len(hr) == 0 {
		return nil
	}
	return hr
}

// bucketScore maps one annotated Claude worker row to its banded headroom score (see the
// package-level table): OFFERABLE -> (1,2] least-loaded-first, WALLED -> [-1,0)
// soonest-reset-first, UNKNOWN -> 0. The tier is decided first (walled beats offerable when
// both flags are set — a throttled-but-still-marked-available row is walled, matching the old
// switch's throttle precedence), then a within-tier bonus that never leaves the tier's band.
func bucketScore(r fleetaccounts.Account, now time.Time) float64 {
	avail := r.Available != nil && *r.Available
	blocked := r.Blocked != nil && *r.Blocked
	throttled := r.Throttled != nil && *r.Throttled
	switch {
	case throttled || blocked:
		// Walled: base -1, plus a reset-soonness bonus in [0,1) so the account that recovers
		// soonest sorts highest among walled buckets while all stay strictly below 0.
		bonus := 0.0
		if r.Reset != nil {
			if s, ok := fleetaccounts.ResetSoonness(*r.Reset, now); ok {
				bonus = s
			}
		}
		return -1 + bonus
	case avail:
		// Offerable: base +1, plus a load bonus 1/(1+live) in (0,1] so the least-loaded bucket
		// sorts highest while all offerable buckets stay strictly above 1.
		live := liveLoad(r)
		return 1 + 1.0/float64(1+live)
	default:
		return 0 // unknown — no runtime availability signal
	}
}

// liveLoad reads a row's live session count for the offerable load tie-break, preferring
// LiveSessions (sessions seen live now), falling back to ActiveSessions, then 0. A negative
// count is clamped to 0 so the bonus denominator stays >= 1.
func liveLoad(r fleetaccounts.Account) int {
	n := 0
	switch {
	case r.LiveSessions != nil:
		n = *r.LiveSessions
	case r.ActiveSessions != nil:
		n = *r.ActiveSessions
	}
	if n < 0 {
		n = 0
	}
	return n
}
