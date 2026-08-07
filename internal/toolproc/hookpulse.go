package toolproc

import (
	"encoding/json"
	"regexp"
	"strings"
)

// The pulse source for streamed/background output — the seam-4 gap the
// process-table note leaves open: a hooked harness reports pre/post/stop, so a
// background job (run_in_background, a task queue) is visible only as its
// launch call, which exits the instant the harness returns "running with ID
// x". The actual long-running work then streams output through poll calls
// (BashOutput, TaskOutput) that the plain mapping records as unrelated
// short-lived procs — the launch↔poll correlation gap.
//
// HookEvents closes it with two bridge derivations, both pure:
//
//   - a launch post whose response announces a background id spawns a SECOND
//     proc "bg:<session>:<id>" (tool "<tool>[bg]") — the job itself, now subject
//     to the deadline/stall/orphan machinery its launch call escaped by
//     returning early. The harness's background id is per-session and the
//     journal is workspace-shared, so the identity carries the owning session
//     (bgident.go, #5880);
//   - a poll post whose input names that id pulses the bg proc (Via = the
//     poll's own call id), and a poll response that reports the job finished
//     exits it — streamed output becomes liveness, so a healthy polled job
//     reads LIVE instead of QUIET.
//
// Everything here degrades to the plain single-event mapping when the shapes
// don't match: no announced id, no bridge; an id that was never spawned, no
// pulse (the fold refuses signals for unknown calls); a job already terminal,
// no second exit (double exit refuses the fold).

// HookEvents derives ALL journal events for one hook firing: the primary
// HookEvent mapping plus the background-job bridge events above. envFor
// resolves the runtime envelope per tool name; the bridged job resolves as
// "<tool>[bg]" so a manifest can grant the job a longer budget than its
// launch call. A nil envFor grants zero envelopes (fold defaults).
func HookEvents(kind string, p HookPayload, envFor func(tool string) HookEnvelope, nowMS int64, existing []Event) ([]Event, error) {
	if envFor == nil {
		envFor = func(string) HookEnvelope { return HookEnvelope{} }
	}
	var out []Event
	ev, emit, err := HookEvent(kind, p, envFor(p.ToolName), nowMS, existing)
	if err != nil {
		return nil, err
	}
	if emit {
		out = append(out, ev)
	}
	if kind != "post" {
		return withSessionResume(out, nowMS, existing), nil
	}

	// Launch bridge: the response announces a background id => spawn the job.
	if id := hookBackgroundID(p.ToolResponse); id != "" {
		callID := backgroundCallID(p.SessionID, id)
		if _, known := hookCallState(callID, existing); !known {
			env := envFor(p.ToolName + "[bg]")
			out = append(out, Event{Kind: EvSpawn, CallID: callID, Tool: p.ToolName + "[bg]",
				Session: p.SessionID, AtMS: nowMS,
				DeadlineMS: env.DeadlineMS, HeartbeatEveryMS: env.HeartbeatEveryMS})
		}
		return withSessionResume(out, nowMS, existing), nil
	}

	// Poll bridge: the input names a background id => pulse (or finish) the job.
	// Resolved through the same constructor as the launch, so the poll can only
	// reach a job in its OWN session (bgident.go).
	if id := hookPolledID(p.ToolInput); id != "" {
		callID := backgroundCallID(p.SessionID, id)
		state, known := hookCallState(callID, existing)
		if !known {
			return out, nil // job never journaled; a pulse would refuse the fold
		}
		via := p.ToolUseID
		switch hookPollStatus(p.ToolResponse) {
		case "completed":
			if state == "running" {
				out = append(out, Event{Kind: EvExit, CallID: callID, AtMS: nowMS, Status: "ok"})
			}
		case "failed", "killed":
			if state == "running" {
				out = append(out, Event{Kind: EvExit, CallID: callID, AtMS: nowMS, Status: "error"})
			}
		default:
			// Still streaming (or status unknown): the poll itself is the pulse.
			out = append(out, Event{Kind: EvPulse, CallID: callID, AtMS: nowMS, Via: via})
		}
	}
	return withSessionResume(out, nowMS, existing), nil
}

// withSessionResume prepends the session_resume retraction when this firing's
// own spawn refutes a session_end still standing in the journal (#3152). The
// retraction must lead: Fold consumes the journal in order, so a spawn placed
// ahead of it would still hit the armed boundary. Returns out unchanged when no
// boundary is standing, which is the overwhelmingly common case.
func withSessionResume(out []Event, nowMS int64, existing []Event) []Event {
	resume, ok := hookSessionResume(out, nowMS, existing)
	if !ok {
		return out
	}
	return append([]Event{resume}, out...)
}

// hookCallState reports whether callID was ever journaled and, if so, whether
// it is still running ("running") or terminal ("done").
func hookCallState(callID string, existing []Event) (state string, known bool) {
	for _, ev := range existing {
		if ev.CallID != callID {
			continue
		}
		switch ev.Kind {
		case EvSpawn:
			state, known = "running", true
		case EvExit, EvKill:
			state, known = "done", true
		}
	}
	return state, known
}

// hookBGAnnounceRe matches the human announcement line a background launch
// returns ("Command running in background with ID: abc123"). The id charset is
// the conservative token class harnesses use for task/shell ids.
var hookBGAnnounceRe = regexp.MustCompile(`background with ID:\s*([A-Za-z0-9_.-]+)`)

// hookBackgroundID extracts the background-task id a launch response
// announces: structured keys first (the spellings in the wild), then the
// announcement line anywhere in the raw response. Empty when the response
// announces no background job — the common, foreground case.
func hookBackgroundID(resp json.RawMessage) string {
	if len(resp) == 0 {
		return ""
	}
	var probe struct {
		ShellID  string `json:"shellId"`
		ShellID2 string `json:"shell_id"`
		BashID   string `json:"bash_id"`
		TaskID   string `json:"task_id"`
		TaskID2  string `json:"taskId"`
	}
	if err := json.Unmarshal(resp, &probe); err == nil {
		for _, id := range []string{probe.ShellID, probe.ShellID2, probe.BashID, probe.TaskID, probe.TaskID2} {
			if id != "" {
				return id
			}
		}
	}
	if m := hookBGAnnounceRe.FindSubmatch(resp); m != nil {
		// The id class includes '.', so a mid-sentence announcement drags its
		// trailing period along ("…with ID: abc. Output is…") — sentence
		// punctuation, not id. An interior dot stays.
		return strings.TrimRight(string(m[1]), ".")
	}
	return ""
}

// hookPolledID extracts the background id a poll-shaped call targets from its
// tool_input. Detection is by key shape, not tool name, so it survives harness
// renames; a bare generic "id" is deliberately NOT matched.
func hookPolledID(input json.RawMessage) string {
	if len(input) == 0 {
		return ""
	}
	var probe struct {
		BashID   string `json:"bash_id"`
		ShellID  string `json:"shell_id"`
		ShellID2 string `json:"shellId"`
		TaskID   string `json:"task_id"`
		TaskID2  string `json:"taskId"`
	}
	if err := json.Unmarshal(input, &probe); err != nil {
		return ""
	}
	for _, id := range []string{probe.BashID, probe.ShellID, probe.ShellID2, probe.TaskID, probe.TaskID2} {
		if id != "" {
			return id
		}
	}
	return ""
}

// hookPollStatus reads the job status a poll response reports, normalized to
// lower case. Empty when the response carries none — treated as "still
// streaming".
func hookPollStatus(resp json.RawMessage) string {
	if len(resp) == 0 {
		return ""
	}
	var probe struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(resp, &probe); err != nil {
		return ""
	}
	return strings.ToLower(strings.TrimSpace(probe.Status))
}
