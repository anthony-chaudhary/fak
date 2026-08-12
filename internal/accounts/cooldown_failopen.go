package accounts

import (
	"fmt"
	"io"
	"strings"
)

// cooldown_failopen.go is the LOUD fail-open loader every cooldown-aware admission path
// (seat resolve, `fak accounts launch`, the dispatch preflight roster) uses to read the
// fleet-shared store. Fail-open itself is deliberate — bad cooldown state must never block
// a resolve, launch or preflight (#4675) — but folding a corrupt store to a nil (cooldown-
// blind) store SILENTLY downgrades a fleet safety gate to a no-op with no signal at all:
// every seat resolution made while the file is unreadable re-offers capped accounts, and
// nobody learns of it until someone happens to run `fak accounts cooldown` (#6027). This
// file keeps the fold and adds the missing signal.

// cooldownGateBlindWarning is the operator-visible warning template emitted when the
// cooldown store cannot be read. It is a single line so it survives a busy launch log,
// and it names three things a repair needs: the store path, the underlying error, and
// the consequence (the gate is off; the run continues anyway).
const cooldownGateBlindWarning = "%s: cooldown store unreadable (%s): %v — " +
	"the account-cooldown gate is OFF for this run and capped accounts may be re-offered; " +
	"repair or remove the file, then `fak accounts cooldown` to confirm\n"

// LoadCooldownStoreFailOpen loads the fleet-shared cooldown store for a cooldown-aware
// admission decision and FAILS OPEN: an unreadable or corrupt store yields nil — the
// cooldown-blind fold callers already handle — so no resolve, launch or dispatch preflight
// is ever blocked by bad cooldown state (#4675). Unlike the silent fold it replaces, it
// emits exactly one warning naming the store to warn (#6027), so the operator learns the
// gate is off on the path that turned it off rather than only from the `cooldown`/`status`
// verbs. surface labels the calling command ("fak accounts launch"); a nil warn or an
// empty surface is tolerated so a pure caller can opt out of the message without losing
// the fold. An ABSENT store is not an error (LoadCooldownStore yields an empty store) and
// stays silent: a fleet that has never cooled an account is healthy, not blind.
//
// Why the corrupt file is left in place rather than quarantined aside: this is a READ on a
// file every checkout and watchdog shares, and the read errors that reach here are not all
// permanent — a Windows sharing violation or a transient permission error against a live
// fleet store would, under a rename-aside policy, let any process destroy the fleet's real
// cooldown state (the observed corruption still held 11 valid entries) at the moment the
// gate is most needed. The write paths already refuse to overwrite state they could not
// read (recordLaunchCooldown / recordRehomeCooldown), so the evidence survives for the
// operator the warning just notified.
func LoadCooldownStoreFailOpen(path, surface string, warn io.Writer) *CooldownStore {
	store, err := LoadCooldownStore(path)
	if err != nil {
		if warn != nil {
			if surface = strings.TrimSpace(surface); surface == "" {
				surface = "fak accounts"
			}
			fmt.Fprintf(warn, cooldownGateBlindWarning, surface, path, err)
		}
		return nil
	}
	return store
}
