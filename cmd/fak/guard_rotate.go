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
	if warn != nil {
		msg := fmt.Sprintf("fak guard: account %q is cooling down — rotating to %q", note.From, note.To)
		if !note.ResetAt.IsZero() {
			msg += fmt.Sprintf(" (resets %s)", note.ResetAt.UTC().Format(time.RFC3339))
		}
		if note.Headroom != nil {
			msg += fmt.Sprintf(" (headroom=%s)", headroomLabel(*note.Headroom))
		}
		fmt.Fprintln(warn, msg)
	}
	return dir, true
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
