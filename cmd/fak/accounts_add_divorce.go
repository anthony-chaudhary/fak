package main

// accounts_add_divorce.go — the enroll path's operator-facing half of the OAuth token-family
// divorce (engine + witness: internal/accounts/credfamily.go, credfamily_divorce.go).
//
// An adopt copies a credential, so the new seat and its source briefly hold ONE refresh token.
// Two dirs cannot share one login: whichever refreshes first rotates the family and the other is
// 401'd with no warning, no matter how far its own expiresAt still is in the future. This file
// performs that unavoidable split deliberately at enroll time and — the load-bearing part —
// PRINTS the consequence, because the failure mode being fixed is not the logout itself but the
// logout nobody was told about.

import (
	"context"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// divorceRefreshTimeout bounds the throwaway refresh turn. It is generous relative to a haiku
// round-trip because a slow answer that still rotates the token is a SUCCESS, and a divorce that
// times out leaves the hazard armed — the expensive outcome we are paying this wait to avoid.
const divorceRefreshTimeout = 90 * time.Second

// divorceRefreshSpawn overrides the refresh mechanism the divorce uses for the whole package.
// Production leaves it nil, which resolves to accounts.DefaultRefreshSpawn (a real `claude -p`).
// The package's TestMain sets it so no test can exec a real binary or reach the network — the
// divorce runs on the DEFAULT enroll path now, so every adopt test would otherwise spawn.
var divorceRefreshSpawn accounts.RefreshSpawn

// resolveDivorceSpawn picks the refresh mechanism: an explicit per-call seam wins, then the
// package override, then nil (accounts.DivorceRefreshFamily's production default).
func resolveDivorceSpawn(p addParams) accounts.RefreshSpawn {
	if p.divorceSpawn != nil {
		return p.divorceSpawn
	}
	return divorceRefreshSpawn
}

// divorceAdoptedFamily moves a freshly adopted seat off the token family it shares with src and
// reports the outcome, including the source dir that a successful divorce logs out. It never fails
// the enroll: the seat and registry row are already correct, and every outcome here is either good
// news or a warning with an exact next command.
func divorceAdoptedFamily(stdout, stderr io.Writer, src, dir string, p addParams) {
	if p.noDivorce {
		// Explicitly declined. Say what state that leaves behind — an unmentioned shared family is
		// precisely the silent landmine this whole path exists to remove.
		if share := accounts.DetectSharedRefreshFamily(src, dir); share.Shared {
			fmt.Fprintf(stderr, "fak accounts: warning: --no-divorce: seat %q and %s share OAuth token family %s\n", p.name, src, share.FamilyID)
			fmt.Fprintln(stderr, "  the FIRST of the two to refresh will silently invalidate the other (a 401 long before its expiresAt)")
			fmt.Fprintf(stderr, "  resolve it deliberately with: fak accounts refresh --name %s\n", p.name)
		}
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), divorceRefreshTimeout)
	defer cancel()
	rep := accounts.DivorceRefreshFamily(ctx, src, dir, resolveDivorceSpawn(p), nil)

	switch rep.Outcome {
	case accounts.DivorceNotNeeded:
		// The source had no credential to share, or the two were already on separate families.
		// Nothing was spawned and nothing was invalidated.
		return

	case accounts.DivorceDone:
		fmt.Fprintf(stdout, "token family: seat moved onto its own OAuth family (%s -> %s); it can refresh independently\n", rep.Before, rep.After)
		// The cost, stated plainly and with the fix, because it has already happened by now.
		fmt.Fprintf(stdout, "NOTE: %s shared that old family, so its credential is now INVALID (a copied login is a MOVE, not a copy).\n", rep.SourceDir)
		fmt.Fprintf(stdout, "      re-login that dir when convenient:  CLAUDE_CONFIG_DIR=%s claude /login\n", rep.SourceDir)
		if !rep.Refreshed {
			// Family changed but the on-disk expiry did not advance: unexpected enough to surface
			// rather than smooth over.
			fmt.Fprintln(stderr, "fak accounts: note: the family rotated but the recorded expiry did not advance; check `fak accounts status` for this seat")
		}

	case accounts.DivorceFailed:
		fmt.Fprintf(stderr, "fak accounts: warning: seat %q could NOT move off the OAuth token family it shares with %s\n", p.name, src)
		if rep.Err != nil {
			fmt.Fprintf(stderr, "  refresh spawn: %v\n", rep.Err)
		}
		if rep.After == "" {
			// The credential is gone/hollow — the API-key failure mode, or a torn write.
			fmt.Fprintf(stderr, "  the seat now holds NO usable refresh token; restore it with `fak accounts restore-credential --name %s`\n", p.name)
			return
		}
		fmt.Fprintln(stderr, "  both dirs still work RIGHT NOW, but the first one to refresh will silently 401 the other")
		fmt.Fprintln(stderr, "  most likely cause: this credential cannot refresh, so the seat will need a human /login when its token lapses")
		fmt.Fprintf(stderr, "  retry the split with: fak accounts refresh --name %s\n", p.name)
	}
}
