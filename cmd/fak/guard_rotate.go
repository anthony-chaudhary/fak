package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/guardrotate"
)

// guardRotateOffCooldown is the raw-`fak guard` launch guard against burning a turn on a
// rate-limited seat. A bare `fak guard -- claude` resolves its account purely from the
// environment (guardClaudeConfigDir: $CLAUDE_CONFIG_DIR else ~/.claude) and, unlike `fak
// accounts launch --rotate`, never consults the fleet-shared cooldown store. So it will
// happily launch against an account the launcher just watched bounce off its own
// weekly/usage cap. This closes that gap: it loads the registry + cooldown store + the live
// headroom signal and hands them to guardrotate.Plan; if the currently-resolved config dir
// belongs to an account inside an ACTIVE cooldown window with a live alternate, it returns
// that alternate seat's dir. The caller sets $CLAUDE_CONFIG_DIR to it before any of guard's
// five re-resolving config-dir consumers fire, so fak's own in-process OAuth read, the
// failover seed, the cap-recovery transcript path, and the child's inherited env all follow
// to the live seat.
//
// FAIL-OPEN is the invariant: cooldown bookkeeping must never block or error a launch.
// Every failure branch (unreadable registry/store, the dir is not an enrolled seat, no live
// alternate) returns the originally-resolved dir unchanged and rotated=false — byte-for-byte
// today's behavior. Only a proven cooled account with a proven live alternate returns a
// different dir.
func guardRotateOffCooldown(homeDir, registryPath string, now time.Time, warn io.Writer) (newDir string, rotated bool) {
	cur := guardClaudeConfigDir()

	reg, err := loadOrDiscover(registryPath, homeDir)
	if err != nil {
		return cur, false
	}
	// Serve/rotation read disk-derived identity (a seat that can't serve falls forward),
	// so refresh before matching or deciding.
	reg = reg.Refresh()

	store, err := accounts.LoadCooldownStore(defaultCooldownStorePath())
	if err != nil {
		return cur, false
	}

	dir, note, ok := guardrotate.Plan(reg, store, rotationHeadroom(homeDir), cur, now)
	if !ok {
		return cur, false
	}
	stampGuardRotationAccount(note)
	if warn != nil {
		// The one-line message is rendered by the pure guardrotate.Note.Explain (unit-tested there)
		// so this wrapper only does I/O.
		fmt.Fprintln(warn, note.Explain())
	}
	return dir, true
}

// stampGuardRotationAccount re-points $DISPATCH_ACCOUNT at the seat a resolved rotation
// actually landed on, and reports whether it did.
//
// This is the moment the serving account stops being the one the DISPATCHER chose. The
// #4805 long-Retry-After goal park is written from $DISPATCH_ACCOUNT (guard.go's park
// template) and read back from it (guardGoalParked), while the dispatcher stamps that
// variable at SPAWN time — before this rotation exists. Measured on the live fleet: 36 of
// 59 resolve units in one day carried a rotation line, so on a clear majority the
// spawn-time identity is NOT the account that served, and a park written under it would
// file the wall of the seat we rotated ONTO against the seat we rotated OFF: it would wall
// a healthy account and leave the walled one free. goalpark.SameAccount is a plain string
// compare and note.To is the same registry seat name the dispatcher's account tag carries,
// so both sides agree by construction once this fires.
//
// A blank target is a no-op rather than a delete: every fail-open branch in
// guardRotateOffCooldown returns before this, so the only way here with no name is a
// degenerate note, and clearing the dispatcher's stamp on one would trade a correct
// identity for an unattributed park — which blocks nobody at all.
//
// Scope, so the next reader does not over-trust this: it covers the STARTUP rotation
// only, which is the one that lands before guard.go builds parkTemplate (that struct
// reads DISPATCH_ACCOUNT once, so only a stamp written earlier can reach it). The
// mid-run guardRotationRuntime.rotate path — a rotation after a child failure — moves
// the serving seat AFTER that capture, so a park written by a relaunched child still
// names the pre-rotation seat. Closing that one needs parkTemplate.Account to be read
// at park time rather than at capture time; it is not done here.
func stampGuardRotationAccount(note guardrotate.Note) bool {
	to := strings.TrimSpace(note.To)
	if to == "" {
		return false
	}
	_ = os.Setenv("DISPATCH_ACCOUNT", to)
	return true
}

// guardRotateWarnWriter returns the writer the rotation note is emitted to, honoring
// --quiet by returning nil (guardRotateOffCooldown skips the note on a nil writer). The
// rotation itself still happens under --quiet; only the one-line explanation is silenced,
// matching how the rest of guard's startup chatter respects --quiet.
func guardRotateWarnWriter(w io.Writer, quiet bool) io.Writer {
	if quiet {
		return nil
	}
	return w
}

// guardDefaultAccountsRegistryPath mirrors the accounts subcommand's registry default
// ($FAK_ACCOUNTS_REGISTRY, else ~/.claude-accounts/registry.json) so the guard rotation
// step reads the same registry `fak accounts` writes. An unresolvable home yields "",
// which loadOrDiscover treats as "discover ~/.claude* fresh".
func guardDefaultAccountsRegistryPath(homeDir string) string {
	if v := strings.TrimSpace(os.Getenv("FAK_ACCOUNTS_REGISTRY")); v != "" {
		return v
	}
	if homeDir != "" {
		return filepath.Join(homeDir, ".claude-accounts", "registry.json")
	}
	return ""
}
