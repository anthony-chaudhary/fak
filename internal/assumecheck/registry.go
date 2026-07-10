package assumecheck

// registry.go — the DECLARATIVE assumption registry (#3820, epic #3818 C2): the
// canonical, ordered table of fleet assumptions, in the exact registry idiom of
// internal/antipattern (antipattern.go) and internal/brittleness (brittleness.go) —
// an ordered unexported slice of the EXISTING element type, a derived byID index, a
// copying Registry() accessor, and an ok-shaped Lookup. Rows are pure data; witness
// GATHERING stays in the impure shell (cmd/fak/assume.go), routed through the
// name-resolved driver registry (driver.go, #3821 C3). A declared-only row checks
// to UNVERIFIABLE by construction (Evidence{Witnessed:false}), never a fabricated
// HOLDS.

// WitnessStatus is the CLOSED per-row wiring marker: is the assumption's declared
// witness actually WIRED to an evidence gatherer, or is the row declared-only —
// registered and addressable, but with no driver behind it yet? It mirrors
// boundarylint's enforced/proposed Status axis (catalog.go): declared-only is the
// prioritized backlog, rendered as the EXPLANATION for an UNVERIFIABLE verdict
// rather than a silent one.
type WitnessStatus string

const (
	// WitnessWired — an evidence gatherer exists for this assumption; the shell can
	// witness it now.
	WitnessWired WitnessStatus = "wired"
	// WitnessDeclaredOnly — registered but no gatherer/driver yet: checking it
	// yields Evidence{Witnessed: false} -> OutcomeUnverifiable, never a guessed
	// decision.
	WitnessDeclaredOnly WitnessStatus = "declared-only"
)

// validWitnessStatuses is the membership set every registry row's status must
// belong to.
var validWitnessStatuses = map[WitnessStatus]bool{
	WitnessWired:        true,
	WitnessDeclaredOnly: true,
}

// ValidWitnessStatus reports whether s is a member of the closed vocabulary.
func ValidWitnessStatus(s WitnessStatus) bool { return validWitnessStatuses[s] }

func (s WitnessStatus) String() string {
	if ValidWitnessStatus(s) {
		return string(s)
	}
	if s == "" {
		return "(unset)"
	}
	return "unknown(" + string(s) + ")"
}

// registry is the canonical, ordered assumption table. Row 0 is the exported
// SeatLaunchable var (assumecheck.go) so the C1 shell reference and the registry
// share ONE source of truth — the same way antipattern/brittleness keep their
// exported class constants referenced inside the registry slice. Every other row is
// a REAL fleet assumption grounded in the subsystem that relies on it and a witness
// authority already in the tree. C3 (#3821) wired the config-flag and command-probe
// rows to name-resolved drivers (driver.go); seat-offerable-not-walled stays
// declared-only until a real ledger-read gatherer exists for it. RefusalReason
// tokens stay data-only SCREAMING_SNAKE — binding them into the closed DOS refusal
// vocabulary is C4.
var registry = []Assumption{
	SeatLaunchable,
	{
		// The rotation planner's banded headroom tiers (cmd/fak/accounts_headroom.go):
		// an OFFERABLE-scored seat (accounts.OfferableBase band) must not in fact sit
		// in the WALLED band (accounts.WalledBase) once the durable usage-cooldown
		// overlay folds in. The witness authority is the annotated runtime roster
		// (fleetaccounts.AnnotatedRoster) those bands are computed from.
		ID:              "seat-offerable-not-walled",
		Owner:           "accounts",
		Statement:       "a seat the headroom fold scores OFFERABLE is not runtime-walled once the usage-cooldown overlay applies, per the fleetaccounts.AnnotatedRoster runtime roster behind `fak accounts next`",
		Level:           LevelInfra,
		WitnessKind:     WitnessLedgerRead,
		RefusalReason:   "SEAT_RUNTIME_WALLED",
		ConfidenceClass: "declared",
		WitnessStatus:   WitnessDeclaredOnly,
	},
	{
		// The doctor's prune class (cmd/fak/accounts_doctor.go, doctorPrune "config
		// dir vanished; tombstone+rehome"): a registry seat whose config dir is gone
		// from disk is recovery material, not a launchable seat. The witness
		// authority is disk truth (the config dir) checked against the seat registry.
		ID:              "seat-config-dir-present",
		Owner:           "accounts",
		Statement:       "every in-rotation registry seat still has its config dir on disk — a vanished dir is `fak accounts doctor`'s prune (tombstone+rehome) class, not a launchable seat",
		Level:           LevelInfra,
		WitnessKind:     WitnessConfigFlag,
		RefusalReason:   "SEAT_CONFIG_DIR_MISSING",
		ConfidenceClass: "witnessed",
		WitnessStatus:   WitnessWired,
	},
	{
		// Dispatch admission (cmd/fak/dispatch_tick_preflight.go): the seat pool's
		// SeatCheck must not be Depleted — including depletion manufactured by
		// unattributed-live orphan workers (#3109) rather than real leases. The
		// witness authority is dispatchtick.EvaluatePreflight's folded verdict.
		ID:              "seat-pool-not-depleted",
		Owner:           "dispatch",
		Statement:       "the dispatch seat pool has free seats — preflight's SeatCheck is not Depleted, and not depleted merely by unattributed-live orphan workers, per dispatchtick.EvaluatePreflight",
		Level:           LevelLoop,
		WitnessKind:     WitnessCommandProbe,
		RefusalReason:   "SEAT_POOL_DEPLETED",
		ConfidenceClass: "witnessed",
		WitnessStatus:   WitnessWired,
	},
	{
		// Kernel preflight (cmd/fak/dispatch_tick_preflight.go,
		// dispatchPreflightKernel): dispatch assumes the DOS kernel loop it routes
		// work through is alive. The witness authority is the live `dos loop --json`
		// probe folded into KernelCheck{Alive, Target, Verdict}.
		ID:              "kernel-loop-alive",
		Owner:           "dispatch",
		Statement:       "the DOS kernel loop is alive and admitting work — the `dos loop --json` probe behind preflight's KernelCheck reports a live loop, not a dead or refusing one",
		Level:           LevelSubsystem,
		WitnessKind:     WitnessCommandProbe,
		RefusalReason:   "KERNEL_LOOP_DOWN",
		ConfidenceClass: "witnessed",
		WitnessStatus:   WitnessWired,
	},
}

// byID indexes the registry for O(1) lookup (the antipattern/brittleness
// specByClass idiom).
var byID = func() map[string]Assumption {
	m := make(map[string]Assumption, len(registry))
	for _, a := range registry {
		m[a.ID] = a
	}
	return m
}()

// Registry returns a copy of the ordered assumption table, so callers (the CLI
// list, a test, a doc generator) can read coverage without reaching into the
// unexported slice.
func Registry() []Assumption { return append([]Assumption(nil), registry...) }

// Lookup returns the registered assumption for an id, and ok=false for an
// unregistered one — the shell's fail-closed router: an unknown id is a usage
// error, never a guessed check.
func Lookup(id string) (Assumption, bool) {
	a, ok := byID[id]
	return a, ok
}
