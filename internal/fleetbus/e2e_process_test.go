package fleetbus

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

// This file is the e2e spine's load-bearing witness: everything else in this package
// proves the contract inside ONE process, where a mutex or a happens-before edge could
// be doing the work by accident. The claim fleetbus actually makes — one control point
// commands N independently scheduled OS processes, exactly once each, and every one of
// them answers — is only falsifiable across a real process boundary.
//
// So the test re-execs its own test binary as N drainers. They share nothing but the
// bus directory: no shared address space, no shared runtime, no shared file handles,
// separate schedulers. What survives that is the transport, not the language.
const (
	childIDEnv      = "FLEETBUS_E2E_CHILD"
	childDirEnv     = "FLEETBUS_E2E_DIR"
	childRoleEnv    = "FLEETBUS_E2E_ROLE"
	childMachineEnv = "FLEETBUS_E2E_MACHINE"

	// appliesDir holds one log per instance id, appended to ONLY by a real apply. It
	// is the double-apply detector: the acks alone cannot prove the applier ran once,
	// because a second apply followed by a suppressed ack would look identical.
	appliesDir = "e2e-applies"
)

// e2eOpUnknown is an op no applier in this test understands, so a refusal has to make
// the whole round trip on its own merits.
const e2eOpUnknown Op = "teleport"

type e2eChild struct {
	id      string
	machine string
	role    string
}

func TestFleetBusEndToEndAcrossProcesses(t *testing.T) {
	if testing.Short() {
		t.Skip("re-execs the test binary as several drainer processes")
	}
	self, err := os.Executable()
	if err != nil {
		t.Skipf("no test binary to re-exec: %v", err)
	}

	root := t.TempDir()
	b, err := OpenDir(root)
	if err != nil {
		t.Fatalf("OpenDir: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(root, appliesDir), 0o755); err != nil {
		t.Fatalf("mkdir applies: %v", err)
	}

	// One control point cuts four directives BEFORE any instance exists. That order is
	// deliberate: a directive that only works when the fleet is already listening is a
	// broadcast, not a control plane.
	now := time.Now()
	everyone := publish(t, b, "control-1", "steer", "go", Selector{All: true}, 5*time.Minute, now)
	servesOnly := publish(t, b, "control-1", "pause", "", Selector{Role: []string{"serve"}}, 5*time.Minute, now.Add(time.Millisecond))
	nonsense := publish(t, b, "control-2", e2eOpUnknown, "", Selector{All: true}, 5*time.Minute, now.Add(2*time.Millisecond))
	// Cut an hour ago with a one-second life: already dead when it is read. A second
	// control point issued it, which is the multi-issuer case in its cheapest form.
	stale := publish(t, b, "control-2", "steer", "too late", Selector{All: true}, time.Second, now.Add(-time.Hour))

	children := []e2eChild{
		{id: "serve-1", machine: "box-a", role: "serve"},
		{id: "serve-2", machine: "box-b", role: "serve"},
		{id: "worker-1", machine: "box-a", role: "worker"},
		{id: "worker-2", machine: "box-b", role: "worker"},
		// A fifth PROCESS wearing serve-1's identity — a restarted-but-not-yet-dead
		// instance, or a config that got copied. Two processes racing the same claim
		// across an OS boundary is exactly the case an in-process mutex cannot cover,
		// and serve-1's apply log must still show each directive exactly once.
		{id: "serve-1", machine: "box-a", role: "serve"},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	var wg sync.WaitGroup
	fail := make([]string, len(children))
	for i, c := range children {
		wg.Add(1)
		go func(i int, c e2eChild) {
			defer wg.Done()
			cmd := exec.CommandContext(ctx, self, "-test.run=^TestFleetBusDrainChild$", "-test.v")
			cmd.Env = append(os.Environ(),
				childIDEnv+"="+c.id,
				childDirEnv+"="+root,
				childRoleEnv+"="+c.role,
				childMachineEnv+"="+c.machine,
			)
			out, err := cmd.CombinedOutput()
			if err != nil {
				fail[i] = c.id + ": " + err.Error() + "\n" + string(out)
			}
		}(i, c)
	}
	wg.Wait()
	for _, msg := range fail {
		if msg != "" {
			t.Fatalf("drainer process failed:\n%s", msg)
		}
	}

	// --- the roster: five processes, four identities ----------------------------- //
	roster, err := b.Instances(time.Now(), DefaultInstanceTTL)
	if err != nil {
		t.Fatalf("Instances: %v", err)
	}
	if len(roster) != 4 {
		t.Fatalf("roster = %+v, want 4 distinct instances", roster)
	}
	// Prove the premise before trusting anything built on it: every announcement came
	// from a DIFFERENT live process, none of them this one. Without this the whole file
	// could pass vacuously while measuring nothing but a goroutine.
	pids := map[int]bool{}
	for _, inst := range roster {
		if inst.PID == 0 || inst.PID == os.Getpid() {
			t.Fatalf("%s announced pid %d from the parent process — this is not a cross-process witness", inst.ID, inst.PID)
		}
		pids[inst.PID] = true
	}
	if len(pids) != 4 {
		t.Fatalf("roster spans %d distinct pids, want 4: %+v", len(pids), roster)
	}

	// --- the fan-out: everyone applied, everyone answered ------------------------ //
	rep := foldOnBus(t, b, everyone, roster)
	if rep.Targeted != 4 || rep.Applied != 4 || rep.Outstanding != 0 || !rep.Complete {
		t.Fatalf("all-instances fold = %+v, want 4/4 applied and Complete", rep)
	}
	if rep.AffectedTotal != 4 {
		t.Fatalf("AffectedTotal = %d, want 4 — the number the operator came for", rep.AffectedTotal)
	}
	for _, row := range rep.Rows {
		if row.Witness == "" {
			t.Errorf("%s applied with no witness: %+v", row.Instance, row)
		}
		if !row.InRoster {
			t.Errorf("%s acked but is missing from the roster it just announced into", row.Instance)
		}
	}

	// --- targeting: a role filter really excludes the rest ----------------------- //
	rep = foldOnBus(t, b, servesOnly, roster)
	if rep.Targeted != 2 || rep.Applied != 2 || !rep.Complete {
		t.Fatalf("role fold = %+v, want the two serves and nobody else", rep)
	}
	for _, row := range rep.Rows {
		if strings.HasPrefix(row.Instance, "worker") {
			t.Errorf("a worker answered a serve-only directive: %+v", row)
		}
	}

	// --- the return path carries a refusal, not silence -------------------------- //
	rep = foldOnBus(t, b, nonsense, roster)
	if rep.Targeted != 4 || rep.Refused != 4 || rep.Applied != 0 || !rep.Complete {
		t.Fatalf("unknown-op fold = %+v, want four refusals and Complete", rep)
	}
	for _, row := range rep.Rows {
		if row.Reason != UnknownOp {
			t.Errorf("%s refused under %q, want %s", row.Instance, row.Reason, UnknownOp)
		}
	}
	// Complete means answered, never "succeeded". An op nobody understood must not
	// read the same as an op everybody ran.
	if rep.Applied == rep.Targeted {
		t.Error("a fleet-wide refusal folded as success")
	}

	// --- an expired directive is answered, and never applied --------------------- //
	rep = foldOnBus(t, b, stale, roster)
	if rep.Targeted != 4 || rep.Expired != 4 || rep.Applied != 0 || !rep.Complete {
		t.Fatalf("expired fold = %+v, want four expired acks", rep)
	}

	// --- exactly once, per identity, across processes ---------------------------- //
	want := map[string][]string{
		"serve-1":  {everyone.ID, servesOnly.ID},
		"serve-2":  {everyone.ID, servesOnly.ID},
		"worker-1": {everyone.ID},
		"worker-2": {everyone.ID},
	}
	for id, wantIDs := range want {
		got := readApplies(t, root, id)
		sort.Strings(got)
		sorted := append([]string(nil), wantIDs...)
		sort.Strings(sorted)
		if len(got) != len(sorted) {
			t.Fatalf("%s applied %v, want exactly %v — a repeat is a session steered twice", id, got, sorted)
		}
		for i := range got {
			if got[i] != sorted[i] {
				t.Fatalf("%s applied %v, want %v", id, got, sorted)
			}
		}
	}
}

// TestFleetBusDrainChild is one drainer process. It is inert unless the parent asked
// for it by environment, so it costs nothing in a normal run.
func TestFleetBusDrainChild(t *testing.T) {
	id := os.Getenv(childIDEnv)
	if id == "" {
		t.Skip("not a re-exec'd drainer; see TestFleetBusEndToEndAcrossProcesses")
	}
	root := os.Getenv(childDirEnv)
	b, err := OpenDir(root)
	if err != nil {
		t.Fatalf("child %s: OpenDir: %v", id, err)
	}
	self, refusal := NewInstance(id, os.Getenv(childMachineEnv), os.Getenv(childRoleEnv),
		os.Getpid(), "", []Op{"steer", "pause"}, time.Now())
	if refusal != nil {
		t.Fatalf("child %s: NewInstance: %v", id, refusal)
	}
	if err := b.Announce(self); err != nil {
		t.Fatalf("child %s: Announce: %v", id, err)
	}

	ap := ApplierFunc(func(d Directive) Outcome {
		if d.Op == e2eOpUnknown {
			return OutcomeRefused(UnknownOp, "op %q is outside this applier's vocabulary", d.Op)
		}
		recordApply(t, root, id, d.ID)
		return OutcomeApplied("op "+string(d.Op)+" landed on 1 session", 1)
	})

	// Drain twice. The second pass is the at-least-once redelivery every real
	// transport has: the log still holds every directive, and the claim — not the
	// reader — is what keeps the apply from happening again.
	for pass := 0; pass < 2; pass++ {
		rep, err := Drain(b, self, ap, time.Now())
		if err != nil {
			t.Fatalf("child %s: drain %d: %v", id, pass, err)
		}
		if len(rep.Errors) != 0 {
			t.Fatalf("child %s: drain %d errors: %v", id, pass, rep.Errors)
		}
		if pass == 1 && rep.Applied+rep.Refused+rep.Expired != 0 {
			// Its twin may legitimately have taken some claims on pass 0, so the
			// only universal rule is that a second pass answers NOTHING new.
			t.Fatalf("child %s: second drain answered again: %+v", id, rep)
		}
	}
}

func foldOnBus(t *testing.T, b Bus, d Directive, roster []Instance) Report {
	t.Helper()
	acks, err := b.Acks(d.ID)
	if err != nil {
		t.Fatalf("Acks(%s): %v", d.ID, err)
	}
	return Fold(d, roster, acks, time.Now())
}

// recordApply appends the directive to this identity's apply log. One short
// O_APPEND write per apply, so the twin processes sharing an id cannot interleave a
// line — which is the point: the log has to be trustworthy to accuse the bus.
func recordApply(t *testing.T, root, id, directiveID string) {
	t.Helper()
	f, err := os.OpenFile(filepath.Join(root, appliesDir, id+".log"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		t.Fatalf("record apply: %v", err)
	}
	defer f.Close()
	if _, err := f.Write([]byte(directiveID + "\n")); err != nil {
		t.Fatalf("record apply: %v", err)
	}
}

func readApplies(t *testing.T, root, id string) []string {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(root, appliesDir, id+".log"))
	if err != nil {
		t.Fatalf("read applies for %s: %v", id, err)
	}
	var out []string
	for _, line := range strings.Split(strings.TrimSpace(string(body)), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			out = append(out, line)
		}
	}
	return out
}
