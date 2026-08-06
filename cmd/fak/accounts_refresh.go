package main

// `fak accounts refresh` — proactively rotate seats' OAuth credentials so a seat never decays into
// a state only a human /login can fix, and prove from disk that each one CAN still refresh.
//
// This is the operator + scheduled-task surface over accounts.TriggerRefresh, which until now was
// reachable only from inside a running guarded session (cmd/fak/guard_child.go). That left two
// gaps this verb closes:
//
//  1. An IDLE seat is never exercised. Its refresh capability is unknown until dispatch needs it,
//     which is the worst possible moment to discover a credential is dead. A periodic refresh turns
//     that unknown into a dated, per-seat fact.
//  2. A seat sharing an OAuth token FAMILY with another dir (what copying a credential produces —
//     see internal/accounts/credfamily.go) has no way to be split apart on demand. `--name <seat>`
//     here is the retry the enroll path points at when its automatic divorce could not run.
//
// The verdict is always the FILE, never the spawn's exit code: refreshed=true means the on-disk
// expiry actually advanced, and the reported family fingerprint means the refresh token actually
// rotated. A seat whose spawn exits 0 while rotating nothing is reported as STALE, because that is
// the signature of a credential that will demand a human login the moment its access token lapses.

import (
	"context"
	"fmt"
	"io"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

type refreshParams struct {
	name         string // limit to this seat; empty = every live seat
	timeout      time.Duration
	registryPath string
	homeDir      string
	asJSON       bool
	// force rotates even a credential that is not yet due, by backdating only its recorded expiry
	// so Claude Code must refresh. That is what turns "probably fine" into a witnessed
	// can-still-refresh fact; without it a fresh seat is reported `fresh` and left alone.
	force bool

	// spawn overrides the refresh mechanism (nil = the real `claude -p`). Test seam.
	spawn accounts.RefreshSpawn
}

// refreshRow is one seat's refresh outcome, shaped for both the table and the --json payload a
// scheduled task can alert on.
type refreshRow struct {
	Seat string `json:"seat"`
	Dir  string `json:"dir"`
	// Refreshed is TriggerRefresh's on-disk verdict: did the recorded expiry advance.
	Refreshed bool `json:"refreshed"`
	// FamilyBefore/FamilyAfter are the non-secret OAuth token-family fingerprints. A changed value
	// proves the refresh token itself rotated (the property that makes a seat independently
	// refreshable); an EMPTY After means the credential was cleared and the seat is now hollow.
	FamilyBefore string `json:"family_before,omitempty"`
	FamilyAfter  string `json:"family_after,omitempty"`
	// Status is the closed verdict: ok (rotated / expiry advanced) | fresh (nothing was due, so
	// nothing was spawned — healthy, NOT a failure) | stale (was due, ran, moved nothing) | hollow
	// (no usable credential left) | skipped (nothing refreshable in the first place).
	Status string `json:"status"`
	Detail string `json:"detail,omitempty"`
	Err    string `json:"error,omitempty"`
}

const (
	refreshStatusOK      = "ok"
	refreshStatusFresh   = "fresh"
	refreshStatusStale   = "stale"
	refreshStatusHollow  = "hollow"
	refreshStatusSkipped = "skipped"
)

// defaultRefreshTimeout bounds each seat's throwaway refresh turn. Seats are refreshed serially so
// a roster-wide sweep never fans out N concurrent `claude` processes across the box — the spawn
// churn that poisons dispatch preflight.
const defaultRefreshTimeout = 90 * time.Second

// refreshBackupRoot is where a pre-refresh snapshot lands (the shared #3987 store). Split out so
// the safety net is named at the call site.
func refreshBackupRoot(homeDir string) string { return accounts.BackupRoot(homeDir) }

func runAccountsRefresh(stdout, stderr io.Writer, p refreshParams) int {
	if p.homeDir == "" {
		fmt.Fprintln(stderr, "fak accounts: cannot resolve home dir")
		return 1
	}
	reg, err := loadOrDiscover(p.registryPath, p.homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	dirs, err := seatDirs(reg, p.name)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	timeout := p.timeout
	if timeout <= 0 {
		timeout = defaultRefreshTimeout
	}

	names := make([]string, 0, len(dirs))
	for n := range dirs {
		names = append(names, n)
	}
	sort.Strings(names)

	rows := make([]refreshRow, 0, len(names))
	backupRoot := refreshBackupRoot(p.homeDir)
	for _, seat := range names {
		// Snapshot BEFORE the refresh. A refresh is a credential-overwrite path like any other: when
		// the refresh grant fails, Claude Code can leave the file HOLLOW (both tokens blanked). This
		// verb hit that on its first roster-wide run — july16-netra's refresh token had already
		// expired, the spawn failed, and the credential came back empty. The pre-image had only been
		// captured because a separate `fak accounts backup` happened to run earlier that session;
		// relying on that is luck, so take it here (#3987's backup-on-write rule, applied to the one
		// overwrite path that was missing it). A backup miss warns, never blocks.
		if snaps, berr := accounts.SnapshotBeforeOverwrite(backupRoot, seat, dirs[seat], time.Now()); berr != nil {
			fmt.Fprintf(stderr, "fak accounts: warning: %s: pre-refresh credential backup failed: %v\n", seat, berr)
		} else if len(snaps) > 0 && !p.asJSON {
			fmt.Fprintf(stdout, "%-20s %-8s snapshotted %d blob(s) before refresh\n", seat, "backup", len(snaps))
		}
		rows = append(rows, refreshSeat(seat, dirs[seat], timeout, p.force, p.spawn))
	}

	if p.asJSON {
		stdout.Write(mustJSON(map[string]any{"seats": rows, "summary": refreshSummary(rows)}))
		fmt.Fprintln(stdout)
		return accountsRefreshExit(rows)
	}
	for _, r := range rows {
		fmt.Fprintf(stdout, "%-20s %-8s %s\n", r.Seat, r.Status, r.Detail)
		if r.Err != "" {
			fmt.Fprintf(stderr, "  %s: refresh spawn: %s\n", r.Seat, r.Err)
		}
	}
	sum := refreshSummary(rows)
	fmt.Fprintf(stdout, "summary: %d seat(s): ok=%d fresh=%d stale=%d hollow=%d skipped=%d\n",
		len(rows), sum["ok"], sum["fresh"], sum["stale"], sum["hollow"], sum["skipped"])
	if sum["hollow"] > 0 {
		fmt.Fprintln(stdout, "a hollow seat has no usable credential left: `fak accounts restore-credential --name <seat>` reverses the blanking (the pre-refresh snapshot above), but a credential whose REFRESH token has itself expired can only be revived by a human /login")
	}
	if sum["stale"] > 0 {
		fmt.Fprintln(stdout, "a stale seat was DUE for a refresh, ran, and rotated nothing: its refresh token is likely dead, so plan a human /login before dispatch relies on it")
	}
	return accountsRefreshExit(rows)
}

// refreshSeat refreshes one seat and grades it from disk. It records the token-family fingerprint
// on both sides of the spawn, so the report distinguishes "kept alive" (expiry advanced AND family
// rotated) from "ran but changed nothing" — a distinction the expiry alone cannot make.
//
// force decides what happens to a credential that is NOT yet due: by default it is left alone and
// graded `fresh`, because Claude Code only rotates a token that is near expiry and grading that
// no-op as a failure reports healthy seats as dead (the false alarm this verb shipped with).
// --force backdates the recorded expiry so the rotation must happen, which is how a sweep PROVES a
// seat can still refresh instead of assuming it.
func refreshSeat(seat, dir string, timeout time.Duration, force bool, spawn accounts.RefreshSpawn) refreshRow {
	row := refreshRow{Seat: seat, Dir: dir, FamilyBefore: accounts.RefreshFamilyID(dir)}
	if row.FamilyBefore == "" {
		// No refresh token to rotate: an api-key seat, a token-only seat, or an already-hollow
		// credential. Spawning `claude -p` could not refresh anything, so don't burn the turn.
		row.Status = refreshStatusSkipped
		row.Detail = "no OAuth refresh token on disk (api-key/token-only seat, or credential already hollow)"
		return row
	}

	due := accounts.CredentialDueForRefresh(dir, time.Now())
	if !due && !force {
		row.Status = refreshStatusFresh
		row.FamilyAfter = row.FamilyBefore
		if exp, ok := accounts.CredentialExpiry(dir); ok {
			row.Detail = fmt.Sprintf("not due yet (expires %s, in %s); pass --force to prove it can rotate",
				exp.UTC().Format("2006-01-02 15:04Z"), time.Until(exp).Round(time.Minute))
		} else {
			row.Detail = "not due yet; pass --force to prove it can rotate"
		}
		return row
	}

	// Forced and not yet due: backdate ONLY the expiry so the spawn must refresh, and put the
	// original bytes back if it turns out nothing rotated (never leave a valid token looking dead).
	var restoreExpiry func() error
	if !due && force {
		if r, err := accounts.NudgeExpiryForRefresh(dir, time.Now()); err == nil {
			restoreExpiry = r
		} else {
			row.Err = err.Error()
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	refreshed, err := accounts.TriggerRefresh(ctx, dir, spawn, nil)
	if restoreExpiry != nil && accounts.RefreshFamilyID(dir) == row.FamilyBefore {
		_ = restoreExpiry()
	}
	row.Refreshed = refreshed
	row.FamilyAfter = accounts.RefreshFamilyID(dir)
	if err != nil {
		row.Err = err.Error()
	}

	switch {
	case row.FamilyAfter == "":
		row.Status = refreshStatusHollow
		row.Detail = fmt.Sprintf("credential lost its refresh token (was family %s)", row.FamilyBefore)
	case row.FamilyAfter != row.FamilyBefore:
		row.Status = refreshStatusOK
		row.Detail = fmt.Sprintf("rotated onto family %s (was %s)", row.FamilyAfter, row.FamilyBefore)
	case refreshed:
		// Expiry advanced without a family rotation: the access token was renewed and the refresh
		// token reused. Still a working, independently-refreshable seat.
		row.Status = refreshStatusOK
		row.Detail = fmt.Sprintf("expiry advanced on family %s", row.FamilyAfter)
	default:
		row.Status = refreshStatusStale
		row.Detail = fmt.Sprintf("ran but neither expiry nor family moved (family %s)", row.FamilyAfter)
	}
	return row
}

func refreshSummary(rows []refreshRow) map[string]int {
	sum := map[string]int{
		refreshStatusOK:      0,
		refreshStatusFresh:   0,
		refreshStatusStale:   0,
		refreshStatusHollow:  0,
		refreshStatusSkipped: 0,
	}
	for _, r := range rows {
		sum[r.Status]++
	}
	return sum
}

// accountsRefreshExit is nonzero exactly when a seat needs attention a refresh could not provide
// (hollow or stale), so a scheduled `fak accounts refresh` can alert on the exit code alone.
func accountsRefreshExit(rows []refreshRow) int {
	for _, r := range rows {
		if r.Status == refreshStatusHollow || r.Status == refreshStatusStale {
			return 1
		}
	}
	return 0
}
