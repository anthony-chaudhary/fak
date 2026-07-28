package accountprobe

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

// Registry dir file names. sessions.json is the roster the watchdog publishes;
// probe_ledger.jsonl is the append-only probe record account_probe.py writes. The two are
// SIBLINGS by convention, not by construction — a dir can carry one without the other,
// and that asymmetry is the whole of issue #5390.
const (
	sessionsFile = "sessions.json"
	ledgerFile   = "probe_ledger.jsonl"
)

// RegHealth grades what a registry directory can honestly say about seat BLOCKS.
//
// The grade exists because absence is not neutral here. A registry dir holding
// sessions.json with no probe_ledger.jsonl beside it can derive no probe verdict, so it
// can derive no block either — and the "zero blocked seats" it then reports means "cannot
// tell", not "nothing is blocked". A consumer that reads the second meaning routes work
// at a dead seat; because an entitlement failure answers 403 and a 403 burns no quota, a
// dead seat reports FULL headroom, so a headroom-weighted allocator ranks it among the
// emptiest in the fleet and sends it the most workers. Keeping unknown-health nameable is
// what stops that inversion, and it is deliberately NOT "treat unprobed as blocked":
// unknown is a third state, distinct from both OK and blocked, so nothing gets stranded.
type RegHealth string

const (
	// RegHealthBlocksKnown: a probe ledger is present, so a block verdict is derivable.
	// "Zero blocked" from such a dir is a finding.
	RegHealthBlocksKnown RegHealth = "blocks-known"
	// RegHealthBlocksUnknown: registry state (sessions.json) is present but NO ledger is
	// beside it. No probe verdict is derivable, so no block is. "Zero blocked" from such a
	// dir is not a finding and must not be reported as one.
	RegHealthBlocksUnknown RegHealth = "blocks-unknown"
	// RegHealthEmpty: no registry state at this path at all.
	RegHealthEmpty RegHealth = "empty"
)

// Rung names — the fixed, ordered set of places a fleet registry may live on one host.
// They are identifiers in a report, so an operator reading a fork note can tell WHICH
// writer produced each registry without guessing from the path.
const (
	RungEnv   = "env"   // $FLEET_REG_DIR, named outright by the operator or the fleet
	RungState = "state" // $FLEET_STATE_DIR/registry, named by the installed service
	RungUser  = "user"  // the per-user Fleet registry the prober writes under
	RungTemp  = "temp"  // the Windows temp-root Fleet registry (last machine-local resort)
	RungClone = "clone" // tools/_registry relative to the working directory
)

// RegSite is one candidate registry dir and what it actually holds on disk.
type RegSite struct {
	Dir         string    // the path, exactly as a consumer would use it
	Rung        string    // which rung proposed it (RungEnv … RungClone)
	DirExists   bool      // the directory itself is present
	HasSessions bool      // sessions.json is present
	HasLedger   bool      // probe_ledger.jsonl is present
	Health      RegHealth // what this dir can say about blocks
}

// RegChoice is the resolved registry dir plus the whole survey the choice was made from,
// so the decision is auditable and a FORK is reportable rather than silent.
type RegChoice struct {
	Dir    string    // the chosen registry dir
	Rung   string    // the rung that produced it
	Health RegHealth // the chosen dir's block-derivability
	Sites  []RegSite // every candidate, deduplicated, in fixed authority order
	Forked bool      // more than one DISTINCT dir on this host carries registry state
}

// BlocksDerivable reports whether the chosen registry can derive a block verdict at all.
//
// This is the explicit answer to "should a process that finds no ledger beside its
// sessions.json write blocks?". It must not: false here means the only honest statement
// about blocked seats is "unknown", and a caller that would otherwise publish "no seats
// blocked" is obliged to publish nothing instead. Reporting a derived zero from a dir that
// cannot derive anything is the precise shape of the #5390 failure.
func (c RegChoice) BlocksDerivable() bool { return c.Health == RegHealthBlocksKnown }

// ForkNote renders the one-line operator-visible fork report, or "" when this host runs a
// single registry. It is the observability half of the fix: converging on the
// block-bearing dir stops the wrong ANSWER, but only a note stops the fork itself from
// being invisible — the second registry is still on disk, still being written by whatever
// writes it, and still block-blind.
func (c RegChoice) ForkNote() string {
	if !c.Forked {
		return ""
	}
	var others []string
	for _, s := range c.Sites {
		if s.Dir == c.Dir || s.Health == RegHealthEmpty {
			continue
		}
		others = append(others, fmt.Sprintf("%s=%s (%s)", s.Rung, s.Dir, s.Health))
	}
	return fmt.Sprintf("accountprobe: fleet registry FORKED — reading %s=%s (%s); also carrying state: %s",
		c.Rung, c.Dir, c.Health, strings.Join(others, ", "))
}

// regSites enumerates every candidate registry dir on this host, in fixed AUTHORITY order,
// and stats each. The order is a constant of the program, never a function of the clock:
// ordering by modification time would let the choice FLAP between ticks, because two
// writers run on their own schedules and each is the freshest for part of every minute.
//
// Duplicates are collapsed to the highest-authority rung that names them, so the ordinary
// production wiring — FLEET_REG_DIR pointing AT the per-user Fleet registry — is one site,
// not a phantom fork.
func regSites() []RegSite {
	var out []RegSite
	seen := map[string]bool{}
	add := func(rung, dir string) {
		dir = strings.TrimSpace(dir)
		if dir == "" {
			return
		}
		key := regDirKey(dir)
		if seen[key] {
			return
		}
		seen[key] = true
		out = append(out, statRegSite(rung, dir))
	}
	add(RungEnv, os.Getenv("FLEET_REG_DIR"))
	if v := strings.TrimSpace(os.Getenv("FLEET_STATE_DIR")); v != "" {
		add(RungState, filepath.Join(v, "registry"))
	}
	if v := strings.TrimSpace(os.Getenv("LOCALAPPDATA")); v != "" {
		add(RungUser, filepath.Join(v, "Fleet", "registry"))
	}
	if runtime.GOOS == "windows" {
		add(RungTemp, filepath.Join(os.TempDir(), "Fleet", "registry"))
	}
	add(RungClone, filepath.Join("tools", "_registry"))
	return out
}

// regDirKey normalizes a candidate path for duplicate detection: absolute where possible,
// cleaned, and case-folded on the one platform whose paths are case-insensitive.
func regDirKey(dir string) string {
	key := dir
	if abs, err := filepath.Abs(dir); err == nil {
		key = abs
	}
	key = filepath.Clean(key)
	if runtime.GOOS == "windows" {
		key = strings.ToLower(key)
	}
	return key
}

func statRegSite(rung, dir string) RegSite {
	s := RegSite{Dir: dir, Rung: rung}
	if fi, err := os.Stat(dir); err == nil && fi.IsDir() {
		s.DirExists = true
	}
	s.HasSessions = regFilePresent(filepath.Join(dir, sessionsFile))
	s.HasLedger = regFilePresent(filepath.Join(dir, ledgerFile))
	switch {
	case s.HasLedger:
		s.Health = RegHealthBlocksKnown
	case s.HasSessions:
		s.Health = RegHealthBlocksUnknown
	default:
		s.Health = RegHealthEmpty
	}
	return s
}

func regFilePresent(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && !fi.IsDir()
}

// ResolveRegDir picks the registry dir this process reads, and surveys the rest.
//
// The rule is AUTHORITY, then DERIVABILITY — never recency:
//
//  1. $FLEET_REG_DIR, when set, wins outright. An operator (or the fleet launcher) naming
//     the dir is the highest authority there is, and every existing wiring depends on it.
//  2. Else $FLEET_STATE_DIR/registry, when set — the installed service's declared home.
//     Same reasoning: a declaration outranks a discovery.
//  3. Else the first DISCOVERED site, in the fixed rung order, that can actually derive a
//     block (it carries a probe ledger). This is the fix: the per-user Fleet dir, which
//     has the ledger, now outranks the cwd-relative clone-root dir, which never does.
//  4. Else the first discovered site carrying any registry state, with the choice marked
//     RegHealthBlocksUnknown so a caller cannot mistake its silence for "nothing blocked".
//  5. Else the first site whose directory merely exists — matching what the sibling
//     resolver in cmd/fak settles on, so two fak processes on one host converge.
//  6. Else nothing exists anywhere and the clone-root path is returned UNCHANGED, exactly
//     the pre-existing default, so a fresh checkout and CI behave as they always have.
//
// No branch consults a modification time, so the answer is a pure function of which files
// exist. Repeated calls over an unchanged filesystem return the same dir, and making the
// wrong dir the freshest changes nothing.
func ResolveRegDir() RegChoice {
	sites := regSites()
	c := RegChoice{Sites: sites, Health: RegHealthEmpty}

	carrying := 0
	for _, s := range sites {
		if s.Health != RegHealthEmpty {
			carrying++
		}
	}
	c.Forked = carrying > 1

	take := func(match func(RegSite) bool) bool {
		for _, s := range sites {
			if match(s) {
				c.Dir, c.Rung, c.Health = s.Dir, s.Rung, s.Health
				return true
			}
		}
		return false
	}
	switch {
	case take(func(s RegSite) bool { return s.Rung == RungEnv }):
	case take(func(s RegSite) bool { return s.Rung == RungState }):
	case take(func(s RegSite) bool { return s.Health == RegHealthBlocksKnown }):
	case take(func(s RegSite) bool { return s.Health == RegHealthBlocksUnknown }):
	case take(func(s RegSite) bool { return s.DirExists }):
	default:
		// Nothing anywhere: the clone rung is always last and is the legacy default.
		last := sites[len(sites)-1]
		c.Dir, c.Rung, c.Health = last.Dir, last.Rung, last.Health
	}
	return c
}

// regForksNoted remembers which fork notes this process has already emitted, keyed by the
// note text, so a standing fork is reported once instead of on every ledger read. It is
// bounded by the number of distinct registry layouts a single process can observe.
var regForksNoted sync.Map

// RegDir resolves the fleet registry dir the probe ledger lives under: $FLEET_REG_DIR when
// set (the fleet sets it in production, so the Go reader and the Python writer agree),
// else $FLEET_STATE_DIR/registry, else the first registry that can actually derive a
// block, else — unchanged — tools/_registry relative to the working directory. See
// ResolveRegDir for the full rule and for why it never consults a modification time.
//
// Before this resolution existed the unset-env default was the cwd-relative clone-root dir
// unconditionally, so a fak process started from the clone root read a registry that holds
// sessions.json and NO ledger. That registry cannot derive a block, so it reported every
// seat unblocked while the per-user registry beside it correctly carried entitlement
// blocks — the same fleet, two answers, and the wrong one was the silent one (#5390).
//
// A fork — a second registry alive beside the chosen one — is reported once per process on
// stderr. Converging on the right dir fixes this reader; the note is what keeps the fork
// itself from being invisible to the operator who has to go delete one of them.
func RegDir() string {
	c := ResolveRegDir()
	if note := c.ForkNote(); note != "" {
		if _, dup := regForksNoted.LoadOrStore(note, struct{}{}); !dup {
			fmt.Fprintln(os.Stderr, note)
		}
	}
	return c.Dir
}
