package accounts

// credfamily_divorce.go — the ACTION half of the token-family hazard credfamily.go documents:
// move a freshly adopted seat off the token family it shares with its source, and grade the move
// from the file rather than from the spawn.
//
// It is deliberately a thin composition of two already-audited pieces — DetectSharedRefreshFamily
// (pure) and TriggerRefresh (which causes Claude Code to rotate its own credential and witnesses
// the rotation on disk) — so this file adds policy, not a second refresh mechanism. In particular
// it does NOT post the OAuth refresh grant itself, for the same reason TriggerRefresh does not.

import (
	"context"
	"time"
)

// DivorceRefreshFamily moves dstDir onto its own OAuth token family when it is sharing one with
// srcDir, and reports what actually happened.
//
// The move is performed by refreshing the TARGET, because the target is the seat the fleet will
// dispatch onto — a seat that cannot refresh independently is worthless, whereas the source is
// typically an interactive dir a human can re-login. The cost is real and unavoidable: a
// successful divorce INVALIDATES the source's copy of the old family, so DivorceDone means the
// source now needs a human /login. Callers must say so out loud; that stated, scheduled logout is
// the entire point, because the alternative is the same logout happening later by surprise.
//
// Grading is on the family fingerprint, not the spawn: the outcome is DivorceDone only when the
// target's refresh token actually CHANGED. A spawn that exits 0 while leaving the shared token in
// place is DivorceFailed — which usually means the credential cannot refresh at all, exactly the
// seat a fleet must not enroll silently.
//
// spawn and now are the injection seams TriggerRefresh already defines (nil = production
// defaults), so a test drives the whole policy without executing a binary or touching a network.
func DivorceRefreshFamily(ctx context.Context, srcDir, dstDir string, spawn RefreshSpawn, now func() time.Time) DivorceReport {
	share := DetectSharedRefreshFamily(srcDir, dstDir)
	rep := DivorceReport{Before: share.TargetID, After: share.TargetID, SourceDir: srcDir}
	if !share.Shared {
		rep.Outcome = DivorceNotNeeded
		return rep
	}

	// A COPIED credential is typically nowhere near expiry, and Claude Code only rotates when its
	// access token is due — so a plain spawn here would legitimately do nothing and leave the two
	// dirs sharing one family for hours (dogfooded 2026-08-06). Backdate ONLY the recorded expiry so
	// the refresh must happen, and undo that if it doesn't: a credential whose refresh failed must
	// be left exactly as it was, not left looking expired when it is still valid.
	nowFn := now
	if nowFn == nil {
		nowFn = time.Now
	}
	var restoreExpiry func() error
	if !CredentialDueForRefresh(dstDir, nowFn()) {
		if r, nerr := NudgeExpiryForRefresh(dstDir, nowFn()); nerr == nil {
			restoreExpiry = r
		} else {
			// Unparseable/hollow: don't rewrite it. The family check below reports the real state.
			rep.Err = nerr
		}
	}

	refreshed, err := TriggerRefresh(ctx, dstDir, spawn, now)
	rep.Refreshed = refreshed
	if err != nil {
		rep.Err = err
	}
	rep.After = RefreshFamilyID(dstDir)
	if rep.After == rep.Before && restoreExpiry != nil {
		// The nudge bought nothing — put the original expiry back so we leave no damage behind.
		if rerr := restoreExpiry(); rerr == nil {
			rep.After = RefreshFamilyID(dstDir)
		}
	}

	// The file is the witness. A changed (and still present) refresh token proves the target now
	// holds a family of its own; anything else leaves the hazard armed.
	if rep.After != "" && rep.After != rep.Before {
		rep.Outcome = DivorceDone
		return rep
	}
	rep.Outcome = DivorceFailed
	return rep
}
