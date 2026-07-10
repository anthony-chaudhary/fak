package procguard

import (
	"encoding/json"
	"testing"
)

// leaseSet turns a set of live run ids into an injected LeaseAlive lookup: a run
// id in the set is alive, everything else (including an absent id) is dead.
func leaseSet(live ...string) func(string) bool {
	set := map[string]bool{}
	for _, id := range live {
		set[id] = true
	}
	return func(runID string) bool { return set[runID] }
}

// TestClassifyDeadOwnerOrphans is acceptance bullet 1: a fak-tagged tree whose
// run id maps to a dead/absent lease is flagged; a live lease spares it. Also
// asserts the no-false-reap edges: no run tag, non-fak, and allow-listed are
// never candidates.
func TestClassifyDeadOwnerOrphans(t *testing.T) {
	procs := []Proc{
		{PID: 10, Name: "fak", PPID: ip(999), Cmdline: "fak c --run r-dead --lane gateway"},   // dead lease -> flag
		{PID: 11, Name: "fak", PPID: ip(999), Cmdline: "fak c --run r-live --lane gateway"},   // live lease -> spare
		{PID: 12, Name: "fak", PPID: ip(999), Cmdline: "fak guard --run=r-dead2"},             // =form, dead -> flag
		{PID: 13, Name: "fak", PPID: ip(999), Cmdline: "fak c --lane gateway"},                // fak-owned but no run tag -> spare
		{PID: 14, Name: "python", PPID: ip(999), Cmdline: "python -m http.server"},            // not fak-owned -> spare
		{PID: 15, Name: "fak", PPID: ip(999), Cmdline: "fak c --run r-dead3"},                 // allow-listed by name -> spare
	}
	top := NewRelationTopology(procs) // pid 999 absent -> owners not alive anyway
	got := ClassifyDeadOwnerOrphans(procs, top, DeadOwnerOptions{LeaseAlive: leaseSet("r-live")})

	flagged := map[int]Finding{}
	for _, f := range got {
		flagged[f.PID] = f
	}
	if f, ok := flagged[10]; !ok || f.Kind != KindDeadOwnerOrphan {
		t.Fatalf("pid 10 (dead lease) must flag as %s: %+v", KindDeadOwnerOrphan, got)
	}
	if _, ok := flagged[11]; ok {
		t.Fatalf("pid 11 (live lease) must be spared: %+v", got)
	}
	if f, ok := flagged[12]; !ok || f.Kind != KindDeadOwnerOrphan {
		t.Fatalf("pid 12 (--run= form, dead) must flag: %+v", got)
	}
	if _, ok := flagged[13]; ok {
		t.Fatalf("pid 13 (fak-owned, no run tag) must be spared (cannot key a lease): %+v", got)
	}
	if _, ok := flagged[14]; ok {
		t.Fatalf("pid 14 (not fak-owned) must be spared: %+v", got)
	}

	// Allow edge: naming "fak" exempts the fak-owned rows entirely.
	gotAllow := ClassifyDeadOwnerOrphans(procs, top, DeadOwnerOptions{
		LeaseAlive: leaseSet("r-live"), AllowNames: []string{"fak"},
	})
	for _, f := range gotAllow {
		if f.Name == "fak" {
			t.Fatalf("allow-listed name must be exempt: %+v", gotAllow)
		}
	}
}

// TestClassifyDeadOwnerNilLeaseDisabled: a nil LeaseAlive disables the mode (the
// classifier never guesses without a lease source).
func TestClassifyDeadOwnerNilLeaseDisabled(t *testing.T) {
	procs := []Proc{{PID: 10, Name: "fak", Cmdline: "fak c --run r-dead"}}
	if got := ClassifyDeadOwnerOrphans(procs, NewRelationTopology(procs), DeadOwnerOptions{}); got != nil {
		t.Fatalf("nil LeaseAlive must disable the mode, got %+v", got)
	}
}

// TestDeadOwnerProtectedAndAttendedReportedNotReaped is acceptance bullet 2: a
// protected OS name and an attended-terminal tree are REPORTED (still flagged as
// dead-owner) but marked Protected so --enact never reaps them, while a plain
// dead-owner tree IS reaped.
func TestDeadOwnerProtectedAndAttendedReportedNotReaped(t *testing.T) {
	procs := []Proc{
		// plain dead-owner tree, background parent -> reap candidate
		{PID: 20, Name: "fak", PPID: ip(900), Cmdline: "fak c --run r-dead"},
		// protected OS name carrying a fak marker -> reported, never reaped
		{PID: 21, Name: "System", PPID: ip(900), Cmdline: "System fak c --run r-dead"},
		// attended terminal parent (pwsh) -> reported, never reaped
		{PID: 22, Name: "fak", PPID: ip(500), Cmdline: "fak c --run r-dead"},
		{PID: 500, Name: "pwsh"}, // the attended terminal parent
	}
	top := NewRelationTopology(procs)
	rows := ClassifyDeadOwnerOrphans(procs, top, DeadOwnerOptions{LeaseAlive: leaseSet( /* none live */ )})

	byPID := map[int]Finding{}
	for _, r := range rows {
		byPID[r.PID] = r
	}
	if r, ok := byPID[21]; !ok || !r.Protected {
		t.Fatalf("protected OS name must be reported AND Protected: %+v", rows)
	}
	if r, ok := byPID[22]; !ok || !r.Protected {
		t.Fatalf("attended-terminal tree must be reported AND Protected: %+v", rows)
	}
	if r, ok := byPID[20]; !ok || r.Protected {
		t.Fatalf("plain background dead-owner tree must be a live (non-protected) candidate: %+v", rows)
	}

	// --enact contract: only the non-protected dead-owner tree dies.
	killed := map[int]bool{}
	killer := func(pid int) (bool, string) { killed[pid] = true; return true, "ok" }
	p := Build(procs, Options{
		Thresholds: DefaultThresholds(), DeadOwnerRows: rows,
		Enact: true, Killer: killer, Platform: "test",
	})
	if !killed[20] {
		t.Fatalf("non-protected dead-owner tree must be reaped under --enact")
	}
	if killed[21] || killed[22] {
		t.Fatalf("protected / attended dead-owner trees must NEVER be reaped: killed=%v", killed)
	}
	for _, r := range p.Flagged {
		if (r.PID == 21 || r.PID == 22) && r.Action != "protected-skip" {
			t.Fatalf("pid %d must be protected-skip, got %q", r.PID, r.Action)
		}
	}
}

// TestBuildDeadOwnerPayloadCarriesClassAndCount is acceptance bullet 3: the JSON
// payload carries the new candidate class (Kind) and its count so the control
// pane folds it, and a merged resource+dead-owner row keeps the dead-owner kind.
func TestBuildDeadOwnerPayloadCarriesClassAndCount(t *testing.T) {
	procs := []Proc{
		{PID: 30, Name: "fak", PPID: ip(900), Cmdline: "fak c --run r-dead"},
		{PID: 31, Name: "fak", PPID: ip(900), Cmdline: "fak guard --run r-dead2", Threads: ip(99999)}, // also a thread runaway
	}
	top := NewRelationTopology(procs)
	rows := ClassifyDeadOwnerOrphans(procs, top, DeadOwnerOptions{LeaseAlive: leaseSet()})
	p := Build(procs, Options{
		Thresholds: Thresholds{MaxThreads: 2000}, DeadOwnerRows: rows, Platform: "test",
	})
	if p.DeadOwnerOrphanCount != 2 {
		t.Fatalf("want dead_owner_orphan_count=2, got %d (flagged=%+v)", p.DeadOwnerOrphanCount, p.Flagged)
	}
	if p.OK {
		t.Fatalf("a live dead-owner orphan is actionable -> ok must be false")
	}

	// Round-trip the JSON contract: the class and count must be present.
	blob, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(blob, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got, _ := decoded["dead_owner_orphan_count"].(float64); int(got) != 2 {
		t.Fatalf("payload must carry dead_owner_orphan_count=2, got %v", decoded["dead_owner_orphan_count"])
	}
	kinds := map[string]bool{}
	for _, f := range p.Flagged {
		kinds[f.Kind] = true
	}
	if !kinds[KindDeadOwnerOrphan] {
		t.Fatalf("a flagged row must carry the %s kind: %+v", KindDeadOwnerOrphan, p.Flagged)
	}
}

// TestDeadOwnerCountOmittedWhenOff: with no dead-owner rows the count is zero and
// omitempty keeps it out of the JSON — the existing contract is unchanged when
// the mode is off.
func TestDeadOwnerCountOmittedWhenOff(t *testing.T) {
	p := Build([]Proc{{PID: 40, Name: "calm"}}, Options{Thresholds: DefaultThresholds(), Platform: "test"})
	if p.DeadOwnerOrphanCount != 0 {
		t.Fatalf("want 0 dead-owner orphans, got %d", p.DeadOwnerOrphanCount)
	}
	blob, _ := json.Marshal(p)
	var decoded map[string]any
	_ = json.Unmarshal(blob, &decoded)
	if _, present := decoded["dead_owner_orphan_count"]; present {
		t.Fatalf("dead_owner_orphan_count must be omitted when zero: %s", blob)
	}
}

func TestExtractRunID(t *testing.T) {
	cases := []struct {
		cmdline string
		want    string
		ok      bool
	}{
		{"fak c --run r-1 --lane gw", "r-1", true},
		{"fak guard --run=r-2", "r-2", true},
		{"fak c --lane gw", "", false},
		{"fak c --run --lane gw", "", false}, // value omitted (next token is a flag)
		{"fak c --run-id r-3", "r-3", true},
	}
	for _, c := range cases {
		got, ok := extractRunID(c.cmdline, DefaultRunIDFlags)
		if got != c.want || ok != c.ok {
			t.Fatalf("extractRunID(%q) = (%q,%v), want (%q,%v)", c.cmdline, got, ok, c.want, c.ok)
		}
	}
}
