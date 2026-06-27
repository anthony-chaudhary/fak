package loopmgr

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func regClock() time.Time { return time.Unix(1_700_000_000, 0).UTC() }

// TestRegistrySurvivesRestartDisabledStaysDisabled is the #764 acceptance:
// after a process restart (a save + a fresh load from disk), the registry
// re-lists every persisted job at the state it held, and a disabled job stays
// disabled — never silently re-armed.
func TestRegistrySurvivesRestartDisabledStaysDisabled(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.json")
	now := regClock()

	reg := Registry{Jobs: map[string]Job{}}
	mustPut(t, &reg, Job{
		Schedule: Schedule{JobID: "issue-dispatch/default", IntervalSeconds: 300, MissedRun: MissedSkip},
		State:    JobArmed,
	}, now)
	mustPut(t, &reg, Job{
		Schedule: Schedule{JobID: "nightrun/da33", IntervalSeconds: 3600, MissedRun: MissedCatchUp, JitterSeconds: 120},
		State:    JobDisabled,
	}, now)
	mustPut(t, &reg, Job{
		Schedule: Schedule{JobID: "retired/old", IntervalSeconds: 60, MissedRun: MissedSkip},
		State:    JobStopped,
	}, now)

	if err := SaveRegistry(path, reg); err != nil {
		t.Fatalf("SaveRegistry: %v", err)
	}

	// Simulate a restart: a fresh load from disk, no in-memory carryover.
	reloaded, err := LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	if len(reloaded.List()) != 3 {
		t.Fatalf("reloaded %d jobs, want 3", len(reloaded.List()))
	}

	// Every job re-lists at the exact state it held.
	want := map[string]JobState{
		"issue-dispatch/default": JobArmed,
		"nightrun/da33":          JobDisabled,
		"retired/old":            JobStopped,
	}
	for id, st := range want {
		job, ok := reloaded.Get(id)
		if !ok {
			t.Fatalf("job %q missing after restart", id)
		}
		if job.State != st {
			t.Fatalf("job %q reloaded state = %q, want %q", id, job.State, st)
		}
	}

	// The load-bearing rule: a disabled job is NOT re-armed at boot.
	armed := reloaded.ArmedJobs()
	if len(armed) != 1 || armed[0].JobID() != "issue-dispatch/default" {
		t.Fatalf("ArmedJobs after restart = %+v, want only issue-dispatch/default", jobIDs(armed))
	}
	for _, j := range armed {
		if j.JobID() == "nightrun/da33" {
			t.Fatalf("disabled job re-armed after restart — must reload disabled")
		}
	}

	// The schedule definition (cadence/policy/jitter) round-trips too.
	nr, _ := reloaded.Get("nightrun/da33")
	if nr.Schedule.IntervalSeconds != 3600 || nr.Schedule.MissedRun != MissedCatchUp || nr.Schedule.JitterSeconds != 120 {
		t.Fatalf("schedule definition did not survive restart: %+v", nr.Schedule)
	}
}

func TestRegistryMissingFileIsEmptyNotError(t *testing.T) {
	reg, err := LoadRegistry(filepath.Join(t.TempDir(), "absent.json"))
	if err != nil {
		t.Fatalf("LoadRegistry(missing): %v", err)
	}
	if len(reg.List()) != 0 {
		t.Fatalf("missing registry has %d jobs, want 0", len(reg.List()))
	}
	if len(reg.ArmedJobs()) != 0 {
		t.Fatalf("missing registry armed = %d, want 0", len(reg.ArmedJobs()))
	}
}

func TestRegistrySetStateAndPutBookkeeping(t *testing.T) {
	now := regClock()
	reg := Registry{Jobs: map[string]Job{}}
	mustPut(t, &reg, Job{
		Schedule: Schedule{JobID: "j", IntervalSeconds: 60, MissedRun: MissedSkip},
		State:    JobArmed,
	}, now)

	first, _ := reg.Get("j")
	if first.CreatedUnixNano != now.UnixNano() || first.UpdatedUnixNano != now.UnixNano() {
		t.Fatalf("initial bookkeeping = created %d updated %d", first.CreatedUnixNano, first.UpdatedUnixNano)
	}

	// Disable it later: created is preserved, updated advances, state flips.
	later := now.Add(time.Hour)
	if err := reg.SetState("j", JobDisabled, later); err != nil {
		t.Fatalf("SetState: %v", err)
	}
	got, _ := reg.Get("j")
	if got.State != JobDisabled {
		t.Fatalf("state after SetState = %q, want disabled", got.State)
	}
	if got.CreatedUnixNano != now.UnixNano() {
		t.Fatalf("CreatedUnixNano changed on SetState (%d != %d)", got.CreatedUnixNano, now.UnixNano())
	}
	if got.UpdatedUnixNano != later.UnixNano() {
		t.Fatalf("UpdatedUnixNano = %d, want %d", got.UpdatedUnixNano, later.UnixNano())
	}
	if len(reg.ArmedJobs()) != 0 {
		t.Fatalf("disabled job still armed")
	}

	// Re-put with a new schedule preserves CreatedUnixNano.
	if err := reg.Put(Job{Schedule: Schedule{JobID: "j", IntervalSeconds: 120, MissedRun: MissedCatchUp}, State: JobArmed}, later.Add(time.Hour)); err != nil {
		t.Fatalf("Put replace: %v", err)
	}
	rep, _ := reg.Get("j")
	if rep.CreatedUnixNano != now.UnixNano() {
		t.Fatalf("CreatedUnixNano not preserved on replace")
	}
	if rep.Schedule.IntervalSeconds != 120 {
		t.Fatalf("schedule not replaced")
	}

	if err := reg.SetState("does-not-exist", JobArmed, later); err == nil {
		t.Fatalf("SetState on unknown job did not error")
	}
}

func TestRegistryRejectsBadRowsOnLoadAndPut(t *testing.T) {
	// A row with an empty state must not load (state is never defaulted).
	bad := `{"schema":"fak.loop-registry.v1","jobs":{"j":{"schedule":{"job_id":"j","interval_seconds":60,"missed_run":"skip"},"state":""}}}`
	path := filepath.Join(t.TempDir(), "bad-state.json")
	if err := os.WriteFile(path, []byte(bad), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := LoadRegistry(path); err == nil {
		t.Fatalf("LoadRegistry accepted an empty job state")
	}

	// A key that disagrees with the job id must not load.
	mismatch := `{"schema":"fak.loop-registry.v1","jobs":{"k":{"schedule":{"job_id":"j","interval_seconds":60,"missed_run":"skip"},"state":"armed"}}}`
	path2 := filepath.Join(t.TempDir(), "mismatch.json")
	if err := os.WriteFile(path2, []byte(mismatch), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := LoadRegistry(path2); err == nil {
		t.Fatalf("LoadRegistry accepted a key != job id")
	}

	// Put rejects an unnamed missed-run policy (the schedule never defaults).
	reg := Registry{Jobs: map[string]Job{}}
	if err := reg.Put(Job{Schedule: Schedule{JobID: "x", IntervalSeconds: 60}, State: JobArmed}, regClock()); err == nil {
		t.Fatalf("Put accepted a schedule with no missed-run policy")
	}
	// Put rejects an unnamed job state.
	if err := reg.Put(Job{Schedule: Schedule{JobID: "x", IntervalSeconds: 60, MissedRun: MissedSkip}, State: ""}, regClock()); err == nil {
		t.Fatalf("Put accepted an empty job state")
	}
}

func mustPut(t *testing.T, r *Registry, job Job, at time.Time) {
	t.Helper()
	if err := r.Put(job, at); err != nil {
		t.Fatalf("Put(%s): %v", job.JobID(), err)
	}
}

func jobIDs(jobs []Job) []string {
	out := make([]string, 0, len(jobs))
	for _, j := range jobs {
		out = append(out, j.JobID())
	}
	return out
}
