package main

import (
	"os"
	"path/filepath"

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
// The score is a coarse, HONEST tier — the runtime layer tells us "offerable now", "capped /
// can't serve now", or "unknown", not a continuous remaining-quota number:
//
//	+1  the bucket is offerable right now (available, not usage-throttled, creds present)
//	-1  the bucket is usage-throttled or blocked (walled) — sort it LAST so a rotate never
//	    re-hands a capped account ahead of one with room
//	 0  no runtime row for the bucket (unknown) — neither preferred nor penalised
//
// A finer most-room-first ordering AMONG offerable buckets (real remaining-quota telemetry)
// is the documented follow-on; today equal-tier buckets keep the deterministic name order.

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
	return headroomFromRoster(rows)
}

// headroomFromRoster folds an annotated runtime roster into a per-bucket headroom score keyed
// by the SAME account-bucket key the config registry dedups on ("uuid:<AccountUUID>"), so the
// scores line up with the pool seats. It scores only Claude worker rows (the config-plane
// rotation is over Claude CLAUDE_CONFIG_DIR seats) that carry a resolved AccountUUID. When
// several dirs map to one bucket the BEST score wins: a bucket has room if ANY of its dirs can
// be offered (one rate-limit window, served by whichever dir holds live creds). It is pure —
// no I/O — so the tiering is unit-tested directly.
func headroomFromRoster(rows []fleetaccounts.Account) accounts.RotationHeadroom {
	hr := accounts.RotationHeadroom{}
	for _, r := range rows {
		if r.Product != "claude" || r.AccountUUID == nil || *r.AccountUUID == "" {
			continue
		}
		key := "uuid:" + *r.AccountUUID
		avail := r.Available != nil && *r.Available
		blocked := r.Blocked != nil && *r.Blocked
		throttled := r.Throttled != nil && *r.Throttled
		var score float64
		switch {
		case throttled || blocked:
			score = -1 // walled / can't be offered now -> sort last
		case avail:
			score = 1 // offerable with room now -> prefer
		default:
			score = 0 // unknown
		}
		if cur, ok := hr[key]; !ok || score > cur {
			hr[key] = score
		}
	}
	if len(hr) == 0 {
		return nil
	}
	return hr
}
