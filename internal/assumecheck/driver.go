package assumecheck

// driver.go — the name-resolved witness-driver registry (#3821, epic #3818 C3):
// witness plurality as a CLOSED, kind-keyed driver table instead of per-assumption
// inline gatherers. A Driver PRODUCES Evidence for exactly one WitnessKind; the
// pure kernel (assumecheck.go) JUDGES it — drivers never call Check, and Check
// never gathers. The registry mirrors internal/abi's registration idiom (a
// mutex-guarded write side touched at init(), an immutable published snapshot on
// the read side) so the kernel file never imports a driver and the dispatch layer
// resolves by NAME (the assumption's declared WitnessKind), the same seam
// abi.RegisterWitnessResolver gives the require-witness gate.
//
// This is deliberately NOT the abi.WitnessResolver plane: that vocabulary
// (Abstain/Confirmed/Refuted) gates tool calls at the kernel boundary and its
// "dos_verify" id is reserved to internal/witness (architest
// TestSingleWitnessResolverRegistrant). Assumption drivers live entirely on the
// assumecheck plane and hand back assumecheck.Evidence.

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
)

// Target names WHAT a driver should probe. Exactly one field group is meaningful
// per WitnessKind; a driver reads only its own operand and treats a missing one
// as "cannot witness" (Evidence{Witnessed:false}), never a guess.
type Target struct {
	// Ref is the git ref a git-ancestry probe witnesses (a SHA, tag, or ref name).
	Ref string
	// Pattern is the literal token a worktree-grep probe searches the tracked
	// checkout for (git grep -F semantics — a fixed string, not a regexp).
	Pattern string
	// Argv is the command vector a command-probe spawns when no in-process Probe
	// is supplied.
	Argv []string
	// Path is the filesystem path a config-flag probe stats (a config dir/file
	// whose presence IS the declared flag's source of truth).
	Path string
	// Dir anchors relative operands and spawned commands ("" = the process
	// default: git's own discovery, the current working directory).
	Dir string
	// Probe is an OPTIONAL in-process probe for a command-probe target whose
	// signal lives in the caller's process (e.g. an already-loaded seat-pool
	// fold) rather than behind an exec. It returns the SAME exit-like tri-state
	// a spawned command would (0 holds / 1 does not hold / anything else or a
	// non-nil error cannot witness) plus an operator-readable detail; the driver
	// maps it exactly as it maps a real process exit. When set, Argv is ignored.
	Probe func(ctx context.Context) (detail string, code int, err error)
}

// Driver is one witness-evidence producer: it gathers Evidence for its OWN
// WitnessKind and no other. Gather must stamp Evidence.Kind with Kind() — the
// kernel's Check rule 1 UNVERIFIABLEs any evidence whose kind mismatches the
// assumption's declared witness, so a mis-stamped driver can never confirm.
// Drivers never judge: exit tri-states map onto (Witnessed, Holds) and the
// kernel produces the Outcome.
type Driver interface {
	Kind() WitnessKind
	Gather(ctx context.Context, t Target) Evidence
}

// driverReg is the mutex-guarded WRITE side, touched only by RegisterDriver at
// init() time (the abi registry shape). Readers never lock it — they read the
// published immutable snapshot below.
var driverReg = struct {
	mu     sync.Mutex
	byKind map[WitnessKind]Driver
}{byKind: map[WitnessKind]Driver{}}

// publishedDrivers is the immutable READ-side snapshot (map[WitnessKind]Driver).
// RegisterDriver rebuilds and atomically republishes it, so a resolve is a
// lock-free map read and a reader holding an old snapshot across a registration
// is race-free.
var publishedDrivers atomic.Value

// RegisterDriver adds a driver for its declared kind. Called from each driver's
// init() in this package; fails LOUD (panic) on a nil driver, a kind outside the
// closed WitnessKind vocabulary, or a duplicate registration — a broken driver
// table is a build defect, not a runtime condition to limp past.
func RegisterDriver(d Driver) {
	if d == nil {
		panic("assumecheck: RegisterDriver(nil)")
	}
	kind := d.Kind()
	if !ValidWitnessKind(kind) {
		panic(fmt.Sprintf("assumecheck: RegisterDriver for kind %q outside the closed WitnessKind vocabulary", string(kind)))
	}
	driverReg.mu.Lock()
	defer driverReg.mu.Unlock()
	if _, dup := driverReg.byKind[kind]; dup {
		panic(fmt.Sprintf("assumecheck: duplicate witness driver registration for kind %s", kind))
	}
	driverReg.byKind[kind] = d
	snap := make(map[WitnessKind]Driver, len(driverReg.byKind))
	for k, v := range driverReg.byKind {
		snap[k] = v
	}
	publishedDrivers.Store(snap)
}

// ResolveDriver is the name-resolved dispatch seam: the driver registered for a
// witness kind, ok=false when none is. A missing driver is the caller's
// fail-closed branch (hand the kernel unwitnessed evidence of the declared
// kind), never a cross-kind substitution.
func ResolveDriver(kind WitnessKind) (Driver, bool) {
	snap, _ := publishedDrivers.Load().(map[WitnessKind]Driver)
	d, ok := snap[kind]
	return d, ok
}

// Drivers returns the registered drivers in stable kind order — the coverage
// menu for the CLI, tests, and doc generators.
func Drivers() []Driver {
	snap, _ := publishedDrivers.Load().(map[WitnessKind]Driver)
	out := make([]Driver, 0, len(snap))
	for _, d := range snap {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Kind() < out[j].Kind() })
	return out
}
