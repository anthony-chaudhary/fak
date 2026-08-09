package main

// accounts_refresh_selfguard.go — `fak accounts refresh`'s refusal to log the operator out of the
// session they are typing in, and, when they choose it anyway, the dir that died plus the one
// command that revives it.
//
// WHY THIS EXISTS (witnessed 2026-08-08, #5954). An operator enrolled the seat they were signed in
// as with --no-divorce, then ran the retry that enroll points at:
//
//	fak accounts refresh --name aug8-netra --force
//	aug8-netra    ok       rotated onto family a190cbe5 (was 442fd283)
//
// The rotation was exactly right. It also killed the operator's own interactive session: ~/.claude
// held the other half of that ONE token family, so it went hollow (accessToken "", refreshToken "",
// expiresAt 0) and the next four turns each came back `Login expired · Please run /login`. Nothing
// was lost — a manual /login repaired it — but the operator had to diagnose a hard logout from a
// message that named neither the cause nor the fix. Worse, the UNforced run had actively invited
// the destructive path: "pass --force to prove it can rotate".
//
// The hazard itself is not removable: two dirs holding one refresh token cannot both survive a
// rotation, and there is no ordering that saves both (internal/accounts/credfamily.go states why).
// What was missing is the half the enroll path already gets right (accounts_add_divorce.go) — say
// it BEFORE it happens, make the operator choose it, and print the recovery afterwards. This file
// is that half for `refresh`, kept out of accounts_refresh.go because it is a policy about the
// CALLER's dir, not about grading a seat.
//
// Everything here is a fingerprint compare over on-disk bytes — no network, no spawn — so the
// refusal is decided before a snapshot is taken or a `claude -p` turn is burned.

import (
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// refreshAckLogoutFlag is the acknowledgement that converts the refusal into a deliberate,
// witnessed logout. It is spelled as the CONSEQUENCE rather than as another --force/--yes so it
// cannot be typed by reflex: the operator who passes it has read what it costs.
const refreshAckLogoutFlag = "--yes-log-me-out"

// refreshSelfHazard is the verdict on whether rotating one seat would invalidate the config dir
// the caller's OWN session is running out of.
type refreshSelfHazard struct {
	// Hit is true exactly when rotating the seat ends the caller's login.
	Hit bool
	// CallerDir is the session dir at risk ($CLAUDE_CONFIG_DIR, else ~/.claude).
	CallerDir string
	// FamilyID is the non-secret fingerprint of the OAuth token family the two dirs share.
	FamilyID string
}

// detectRefreshSelfHazard reports whether refreshing seatDir would log the session running out of
// callerDir out.
//
// The same dir is deliberately NOT a hazard: refreshing the caller's own dir IN PLACE is the
// healthy path — the session keeps reading the same file and simply finds a newer token in it.
// Only a SECOND dir holding a byte-identical refresh token turns a rotation into a logout, which is
// exactly what DetectSharedRefreshFamily decides.
func detectRefreshSelfHazard(callerDir, seatDir string) refreshSelfHazard {
	if callerDir == "" || seatDir == "" || sameDir(callerDir, seatDir) {
		return refreshSelfHazard{}
	}
	share := accounts.DetectSharedRefreshFamily(callerDir, seatDir)
	if !share.Shared {
		return refreshSelfHazard{}
	}
	return refreshSelfHazard{Hit: true, CallerDir: callerDir, FamilyID: share.FamilyID}
}

// refreshWouldRotate reports whether this seat is actually headed for a rotation: --force makes one
// happen, and a credential already DUE for refresh rotates on its own. A seat that would merely be
// graded `fresh` touches nothing, so it is not something to refuse — it only earns the amended hint
// below.
func refreshWouldRotate(dir string, force bool, now time.Time) bool {
	return force || accounts.CredentialDueForRefresh(dir, now)
}

// refreshSelfBlockDetail is the refused row's reason. It names the session it would have ended, the
// family that makes the two dirs one login, and both ways forward — accept the logout, or move the
// work to a session that is not the victim.
func refreshSelfBlockDetail(hz refreshSelfHazard) string {
	return fmt.Sprintf("REFUSED: shares OAuth token family %s with the config dir THIS session is running out of (%s) — rotating this seat invalidates that login (one family is one login; whichever side refreshes first wins). Re-run with %s to accept the logout, or run the refresh from a session on another seat",
		hz.FamilyID, hz.CallerDir, refreshAckLogoutFlag)
}

// refreshSelfForceHint replaces the `fresh` row's bare "pass --force to prove it can rotate"
// invitation when --force would log the caller out. The unqualified invitation is defect #1 of
// #5954: it described the destructive path as a proof, and the operator took it at its word.
func refreshSelfForceHint(hz refreshSelfHazard) string {
	return fmt.Sprintf("do NOT --force it blind: this seat shares OAuth family %s with the config dir THIS session is running out of (%s), so forcing the rotation logs this session out (refused without %s)",
		hz.FamilyID, hz.CallerDir, refreshAckLogoutFlag)
}

// printRefreshSelfLogout states what an acknowledged rotation just cost and how to undo it. It is
// deliberately the same shape as the divorce path's notice (accounts_add_divorce.go): the failure
// being fixed is not the logout, it is the logout nobody was told about.
func printRefreshSelfLogout(stdout io.Writer, r refreshRow) {
	if r.LoggedOutDir == "" {
		return
	}
	fmt.Fprintf(stdout, "NOTE: %s shared seat %s's old OAuth family %s, so its credential is now INVALID (a shared family is one login, not two).\n",
		r.LoggedOutDir, r.Seat, r.FamilyBefore)
	if accounts.CredentialHollow(r.LoggedOutDir) {
		fmt.Fprintf(stdout, "      that dir's credential is now HOLLOW on disk (both OAuth tokens empty): a session running there will fail EVERY turn with `Login expired · Please run /login` until it is repaired.\n")
	}
	fmt.Fprintf(stdout, "      re-login that dir when convenient:  CLAUDE_CONFIG_DIR=%s claude /login\n", r.LoggedOutDir)
}
