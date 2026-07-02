package toolproc

import (
	"encoding/json"
	"testing"
)

func hookEnvNone(string) HookEnvelope { return HookEnvelope{} }

// TestHookEventsBridgesABackgroundLaunch: a launch post whose response
// announces a background id emits the launch's own exit PLUS a spawn for the
// job itself, tool-tagged "[bg]", with the envelope resolved for that tag.
func TestHookEventsBridgesABackgroundLaunch(t *testing.T) {
	existing := []Event{
		{Kind: EvSpawn, CallID: "tu-1", Tool: "Bash", Session: "s1", AtMS: 1_000},
	}
	p := HookPayload{SessionID: "s1", ToolName: "Bash", ToolUseID: "tu-1",
		ToolResponse: json.RawMessage(`"Command running in background with ID: bilsbrzwq. Output is being written to: /tmp/x"`)}
	envFor := func(tool string) HookEnvelope {
		if tool == "Bash[bg]" {
			return HookEnvelope{DeadlineMS: 600_000, HeartbeatEveryMS: 30_000}
		}
		return HookEnvelope{}
	}
	evs, err := HookEvents("post", p, envFor, 2_000, existing)
	if err != nil {
		t.Fatalf("HookEvents: %v", err)
	}
	if len(evs) != 2 {
		t.Fatalf("HookEvents = %+v, want [exit, bg spawn]", evs)
	}
	if evs[0].Kind != EvExit || evs[0].CallID != "tu-1" {
		t.Fatalf("primary = %+v, want the launch's exit", evs[0])
	}
	bg := evs[1]
	if bg.Kind != EvSpawn || bg.CallID != "bg:bilsbrzwq" || bg.Tool != "Bash[bg]" || bg.Session != "s1" {
		t.Fatalf("bridge = %+v, want spawn of bg:bilsbrzwq as Bash[bg] in s1", bg)
	}
	if bg.DeadlineMS != 600_000 || bg.HeartbeatEveryMS != 30_000 {
		t.Fatalf("bridge envelope = %+v, want the Bash[bg] grant", bg)
	}
	// A structured announcement (shellId key) bridges the same way.
	p.ToolResponse = json.RawMessage(`{"shellId": "sh-9", "stdout": ""}`)
	evs, err = HookEvents("post", p, hookEnvNone, 2_000, existing)
	if err != nil || len(evs) != 2 || evs[1].CallID != "bg:sh-9" {
		t.Fatalf("structured announce => %+v, %v; want a bg:sh-9 spawn", evs, err)
	}
	// Re-announcing an already-journaled id must not respawn it (dup spawn
	// refuses the fold).
	dup := append(existing, Event{Kind: EvSpawn, CallID: "bg:bilsbrzwq", Tool: "Bash[bg]", Session: "s1", AtMS: 1_500})
	p.ToolResponse = json.RawMessage(`"Command running in background with ID: bilsbrzwq"`)
	evs, err = HookEvents("post", p, hookEnvNone, 2_000, dup)
	if err != nil || len(evs) != 1 {
		t.Fatalf("re-announce => %+v, %v; want the primary exit only", evs, err)
	}
}

// TestHookEventsPulsesThePolledJob: a poll post whose input names a journaled
// background id pulses that job (Via = the poll's own call id); a poll
// reporting the job finished exits it instead — exactly once.
func TestHookEventsPulsesThePolledJob(t *testing.T) {
	existing := []Event{
		{Kind: EvSpawn, CallID: "bg:job7", Tool: "Bash[bg]", Session: "s1", AtMS: 1_000},
		{Kind: EvSpawn, CallID: "tu-poll", Tool: "BashOutput", Session: "s1", AtMS: 5_000},
	}
	p := HookPayload{SessionID: "s1", ToolName: "BashOutput", ToolUseID: "tu-poll",
		ToolInput:    json.RawMessage(`{"bash_id": "job7"}`),
		ToolResponse: json.RawMessage(`{"status": "running", "stdout": "chunk"}`)}
	evs, err := HookEvents("post", p, hookEnvNone, 6_000, existing)
	if err != nil {
		t.Fatalf("HookEvents: %v", err)
	}
	if len(evs) != 2 || evs[0].Kind != EvExit || evs[0].CallID != "tu-poll" {
		t.Fatalf("HookEvents = %+v, want [poll exit, pulse]", evs)
	}
	pulse := evs[1]
	if pulse.Kind != EvPulse || pulse.CallID != "bg:job7" || pulse.Via != "tu-poll" {
		t.Fatalf("bridge = %+v, want a pulse on bg:job7 via tu-poll", pulse)
	}

	// Completion poll: the job exits ok.
	done := append(existing, Event{Kind: EvExit, CallID: "tu-poll", AtMS: 6_000, Status: "ok"},
		Event{Kind: EvSpawn, CallID: "tu-poll2", Tool: "BashOutput", Session: "s1", AtMS: 7_000})
	p.ToolUseID = "tu-poll2"
	p.ToolResponse = json.RawMessage(`{"status": "completed", "stdout": ""}`)
	evs, err = HookEvents("post", p, hookEnvNone, 8_000, done)
	if err != nil || len(evs) != 2 || evs[1].Kind != EvExit || evs[1].CallID != "bg:job7" || evs[1].Status != "ok" {
		t.Fatalf("completion poll => %+v, %v; want the job's ok exit", evs, err)
	}
	// A second completion poll after the job exited emits no double exit.
	after := append(done, Event{Kind: EvExit, CallID: "bg:job7", AtMS: 8_000, Status: "ok"},
		Event{Kind: EvSpawn, CallID: "tu-poll3", Tool: "BashOutput", Session: "s1", AtMS: 9_000})
	p.ToolUseID = "tu-poll3"
	evs, err = HookEvents("post", p, hookEnvNone, 10_000, after)
	if err != nil || len(evs) != 1 {
		t.Fatalf("post-exit poll => %+v, %v; want the primary exit only", evs, err)
	}
	// A failed job exits error.
	p.ToolResponse = json.RawMessage(`{"status": "failed"}`)
	p.ToolUseID = "tu-poll2"
	evs, err = HookEvents("post", p, hookEnvNone, 8_000, done)
	if err != nil || len(evs) != 2 || evs[1].Status != "error" {
		t.Fatalf("failed poll => %+v, %v; want the job's error exit", evs, err)
	}
	// A poll for a job never journaled bridges nothing (a pulse for an unknown
	// call would refuse the whole fold).
	p.ToolInput = json.RawMessage(`{"bash_id": "ghost"}`)
	p.ToolResponse = json.RawMessage(`{"status": "running"}`)
	evs, err = HookEvents("post", p, hookEnvNone, 8_000, done)
	if err != nil || len(evs) != 1 {
		t.Fatalf("unknown-job poll => %+v, %v; want the primary exit only", evs, err)
	}
}

// TestHookEventsBridgedJournalFolds is the end-to-end honesty line: a hooked
// launch → polls → completion sequence folds into a table where the job ran
// LIVE (its polls were its pulses) and finished DONE — and a job whose polls
// stop arriving goes STALLED, which the plain mapping could never see.
func TestHookEventsBridgedJournalFolds(t *testing.T) {
	envFor := func(tool string) HookEnvelope {
		if tool == "Bash[bg]" {
			return HookEnvelope{HeartbeatEveryMS: 10_000}
		}
		return HookEnvelope{}
	}
	var journal []Event
	step := func(kind string, p HookPayload, atMS int64) {
		t.Helper()
		evs, err := HookEvents(kind, p, envFor, atMS, journal)
		if err != nil {
			t.Fatalf("HookEvents(%s @%d): %v", kind, atMS, err)
		}
		journal = append(journal, evs...)
	}
	launch := HookPayload{SessionID: "s1", ToolName: "Bash", ToolUseID: "tu-launch",
		ToolInput: json.RawMessage(`{"command": "make bench", "run_in_background": true}`)}
	step("pre", launch, 1_000)
	launch.ToolResponse = json.RawMessage(`"Command running in background with ID: j1"`)
	step("post", launch, 2_000)
	poll := HookPayload{SessionID: "s1", ToolName: "BashOutput", ToolUseID: "tu-p1",
		ToolInput: json.RawMessage(`{"bash_id": "j1"}`)}
	step("pre", poll, 8_000)
	poll.ToolResponse = json.RawMessage(`{"status": "running"}`)
	step("post", poll, 9_000)

	// Mid-flight: the job is LIVE — its poll pulsed it 1s ago on a 10s cadence.
	tab, err := Fold(journal, 10_000, Config{})
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}
	job := findProc(t, tab, "bg:j1")
	if job.State != StateRunning || job.Liveness != LivenessLive || job.Pulses != 1 {
		t.Fatalf("mid-flight job = state=%s live=%s pulses=%d, want RUNNING/LIVE/1", job.State, job.Liveness, job.Pulses)
	}

	// Silence: 40s past the last pulse on a 10s cadence => STALLED.
	tab, err = Fold(journal, 49_000, Config{})
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}
	if job = findProc(t, tab, "bg:j1"); job.Liveness != LivenessStalled {
		t.Fatalf("silent job liveness = %s, want STALLED", job.Liveness)
	}

	// Completion poll: the job goes DONE.
	poll.ToolUseID = "tu-p2"
	poll.ToolResponse = nil
	step("pre", poll, 50_000)
	poll.ToolResponse = json.RawMessage(`{"status": "completed"}`)
	step("post", poll, 51_000)
	tab, err = Fold(journal, 60_000, Config{})
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}
	if job = findProc(t, tab, "bg:j1"); job.State != StateDone || job.ExitStatus != "ok" {
		t.Fatalf("finished job = state=%s exit=%s, want DONE/ok", job.State, job.ExitStatus)
	}
}
