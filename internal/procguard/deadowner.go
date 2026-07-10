package procguard

// deadowner.go adds the OWNERSHIP-liveness dimension the resource (CPU/thread)
// and name+age orphan heuristics structurally miss: a fak-owned loop/worker tree
// whose OWNER (its run-id lease / run-registry row) is DEAD but whose children
// are still BUSY. A busy subtree of a crashed owner trips neither the idle-shell
// rule (it has live children) nor the CPU-pin rule (its load is normal), so those
// stranded trees accumulate. This classifier keys the reap on the owner lease
// instead of on idleness, catching exactly that class. Issue #3596.
//
// It stays PURE like the rest of the package: the lease lookup and the process
// topology are supplied by the caller, so it is table-testable with a synthetic
// lease map and never touches a real registry. The impure shell (read the loop
// ledger, fold it into a run-id -> alive lookup) lives in cmd/fak, the same
// leaf/shell split ClassifyOrphanSprawl / CollectRelations already use.

import (
	"fmt"
	"sort"
	"strings"
)

// KindDeadOwnerOrphan tags a fak-owned process tree whose owning run-lease is
// dead/absent. Distinct from the name+age orphan kinds (orphan-helper /
// idle-shell / orphan-console-shell) because the reap KEY is ownership liveness,
// not idleness.
const KindDeadOwnerOrphan = "dead-owner-orphan"

// DefaultFakOwnerMarkers are the cmdline substrings that mark a process as the
// root of a fak-owned loop/worker tree. A process matching one of these AND
// carrying a run tag whose lease is dead is a dead-owner orphan candidate. The
// trailing space on "fak c " keeps it from matching sibling verbs like
// "fak cachevalue" / "fak commit".
var DefaultFakOwnerMarkers = []string{"fak guard", "fak c ", "dos loop", "loop drive", "superloop drive"}

// DefaultRunIDFlags are the cmdline flags a fak run carries its run id in, tried
// in order (matches `fak loop --run`). The first that resolves keys the lease
// lookup.
var DefaultRunIDFlags = []string{"--run", "--run-id"}

// DeadOwnerOptions bundles the lease-keyed reaper knobs. LeaseAlive is injected
// (a nil LeaseAlive disables the whole mode and yields no findings) so the
// classifier stays pure and table-testable — never reaching for a real registry.
type DeadOwnerOptions struct {
	// OwnerMarkers overrides DefaultFakOwnerMarkers (empty => the defaults).
	OwnerMarkers []string
	// RunIDFlags overrides DefaultRunIDFlags (empty => the defaults).
	RunIDFlags []string
	// LeaseAlive reports whether the run that owns a tagged tree is still alive
	// (lease renewed within TTL / a live registry row). A run id absent from the
	// registry reads as NOT alive (a dead owner). A nil LeaseAlive disables the mode.
	LeaseAlive func(runID string) bool
	// InteractiveParentNames overrides DefaultInteractiveParentNames — an attended
	// terminal parent is reported but never reaped.
	InteractiveParentNames map[string]bool
	ProtectedPIDs          []int
	AllowNames             []string
}

// ClassifyDeadOwnerOrphans flags fak-owned process trees whose owning run-lease
// is dead/absent. Pure: LeaseAlive and the topology are supplied by the caller.
//
// The no-false-reap contract is preserved end to end: a process with no
// recognizable fak run tag is never a candidate (it cannot be keyed to a lease),
// a live owner lease spares the whole tree, and a protected OS name or an
// attended terminal parent is REPORTED (so an operator still sees the stranded
// tree) but marked Protected so Build's --enact reaper skips it.
func ClassifyDeadOwnerOrphans(procs []Proc, top RelationTopology, opt DeadOwnerOptions) []Finding {
	if opt.LeaseAlive == nil {
		return nil
	}
	markers := nonEmpty(opt.OwnerMarkers)
	if len(markers) == 0 {
		markers = DefaultFakOwnerMarkers
	}
	runIDFlags := opt.RunIDFlags
	if len(runIDFlags) == 0 {
		runIDFlags = DefaultRunIDFlags
	}
	interactive := opt.InteractiveParentNames
	if interactive == nil {
		interactive = DefaultInteractiveParentNames
	}
	allow := lowerSet(opt.AllowNames)
	protSet := intSet(opt.ProtectedPIDs)

	flagged := []Finding{}
	for _, p := range procs {
		name := strings.TrimSpace(p.Name)
		stem := stemLower(name)
		if allow[stem] {
			continue
		}
		if !cmdlineMatchesAnyMarker(name, p.Cmdline, markers) {
			continue
		}
		runID, ok := extractRunID(p.Cmdline, runIDFlags)
		if !ok {
			// No recognizable run tag: the tree cannot be keyed to a lease, so it is
			// never a candidate — the no-false-reap contract for an untagged process.
			continue
		}
		if opt.LeaseAlive(runID) {
			continue // owner lease live -> spare the whole tree
		}
		attended := attendedParent(p, top, interactive)
		reason := fmt.Sprintf("dead-owner orphan: run %q lease not live (owner dead/absent)", runID)
		if attended {
			reason += "; attended terminal (reported, not reaped)"
		}
		flagged = append(flagged, Finding{
			PID: p.PID, Name: name, PPID: p.PPID, ParentName: parentName(p, top),
			Threads: p.Threads, WSMB: p.WSMB,
			Reasons:   []string{reason},
			Protected: protSet[p.PID] || ProtectedNames[stem] || attended,
			Kind:      KindDeadOwnerOrphan,
		})
	}
	sort.Slice(flagged, func(i, j int) bool { return flagged[i].PID < flagged[j].PID })
	return flagged
}

// cmdlineMatchesAnyMarker reports whether a process name+cmdline contains any
// marker (case-insensitive substring).
func cmdlineMatchesAnyMarker(name, cmdline string, markers []string) bool {
	hay := strings.ToLower(name + " " + cmdline)
	for _, m := range markers {
		if m = strings.ToLower(strings.TrimSpace(m)); m != "" && strings.Contains(hay, m) {
			return true
		}
	}
	return false
}

// extractRunID lifts the run id from a cmdline: the first of the given flags that
// resolves, in either "--flag value" or "--flag=value" form. A value that is
// itself a flag (the space form's value was omitted) does not resolve. Returns
// ("", false) when none is present.
func extractRunID(cmdline string, flags []string) (string, bool) {
	fields := strings.Fields(cmdline)
	for i, f := range fields {
		for _, name := range flags {
			if f == name && i+1 < len(fields) {
				if v := fields[i+1]; !strings.HasPrefix(v, "-") {
					if v = strings.TrimSpace(v); v != "" {
						return v, true
					}
				}
			}
			if strings.HasPrefix(f, name+"=") {
				if v := strings.TrimSpace(strings.TrimPrefix(f, name+"=")); v != "" {
					return v, true
				}
			}
		}
	}
	return "", false
}
