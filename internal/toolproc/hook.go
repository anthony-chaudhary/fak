package toolproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

// HookPayload is the harness hook envelope this package understands — the
// fields Claude Code (and compatible harnesses) put on PreToolUse /
// PostToolUse / Stop hook stdin. Unknown fields are ignored (OPEN envelope);
// the derived journal events keep the CLOSED vocabulary.
type HookPayload struct {
	SessionID    string          `json:"session_id"`
	ToolName     string          `json:"tool_name"`
	ToolUseID    string          `json:"tool_use_id"`
	ToolInput    json.RawMessage `json:"tool_input"`
	ToolResponse json.RawMessage `json:"tool_response"`
}

// HookEnvelope is the runtime envelope the hook grants a spawned call (the
// CLI's flag defaults; per-tool envelopes belong to the policy seam).
type HookEnvelope struct {
	DeadlineMS       int64
	HeartbeatEveryMS int64
}

// HookEvent derives the journal event for one hook firing, given the events
// already in the journal (for identity correlation). kind is "pre" | "post" |
// "stop". It is a pure function: no clock, no IO.
//
// Identity: the harness's tool_use_id when present; otherwise a deterministic
// fallback of session + tool + a digest of tool_input, suffixed #2, #3, …
// when the same identity respawns after a terminal state (a repeated
// identical call is a new process, not a duplicate-spawn violation).
//
// emit=false, err=nil means "observed, nothing to record": a pre for an
// identity already running (the harness retried delivery), or a post whose
// spawn was never journaled (hook armed mid-session — an exit for a call the
// table never admitted would refuse at fold time).
func HookEvent(kind string, p HookPayload, env HookEnvelope, nowMS int64, existing []Event) (Event, bool, error) {
	switch kind {
	case "stop":
		if p.SessionID == "" {
			return Event{}, false, fmt.Errorf("toolproc: stop hook without session_id")
		}
		return Event{Kind: EvSessionEnd, Session: p.SessionID, AtMS: nowMS}, true, nil
	case "pre", "post":
		// fall through
	default:
		return Event{}, false, fmt.Errorf("toolproc: unknown hook kind %q", kind)
	}
	if p.ToolName == "" {
		return Event{}, false, fmt.Errorf("toolproc: %s hook without tool_name", kind)
	}
	base := p.ToolUseID
	if base == "" {
		sum := sha256.Sum256(p.ToolInput)
		base = fmt.Sprintf("hk:%s:%s:%s", p.SessionID, p.ToolName, hex.EncodeToString(sum[:])[:8])
	}
	id, running := hookResolveID(base, existing)

	switch kind {
	case "pre":
		if running {
			return Event{}, false, nil // already journaled as live; nothing to add
		}
		return Event{Kind: EvSpawn, CallID: id, Tool: p.ToolName, Session: p.SessionID,
			AtMS: nowMS, DeadlineMS: env.DeadlineMS, HeartbeatEveryMS: env.HeartbeatEveryMS}, true, nil
	default: // post
		if !running {
			return Event{}, false, nil // spawn never journaled (hook armed mid-flight)
		}
		return Event{Kind: EvExit, CallID: id, AtMS: nowMS, Status: hookExitStatus(p.ToolResponse)}, true, nil
	}
}

// hookResolveID walks the journal for the base identity and its #N respawn
// suffixes, returning the newest generation's id and whether that generation
// is still running (spawned without exit/kill).
func hookResolveID(base string, existing []Event) (id string, running bool) {
	gen := 0 // highest generation spawned; 0 = never spawned
	live := false
	for _, ev := range existing {
		g, ok := hookGeneration(base, ev.CallID)
		if !ok {
			continue
		}
		switch ev.Kind {
		case EvSpawn:
			if g >= gen {
				gen, live = g, true
			}
		case EvExit, EvKill:
			if g == gen {
				live = false
			}
		}
	}
	if gen == 0 {
		return base, false
	}
	if live {
		return hookIDForGeneration(base, gen), true
	}
	return hookIDForGeneration(base, gen+1), false
}

func hookIDForGeneration(base string, gen int) string {
	if gen <= 1 {
		return base
	}
	return fmt.Sprintf("%s#%d", base, gen)
}

// hookGeneration reports whether callID is base or one of its #N respawns,
// and which generation (base = 1).
func hookGeneration(base, callID string) (int, bool) {
	if callID == base {
		return 1, true
	}
	rest, ok := strings.CutPrefix(callID, base+"#")
	if !ok {
		return 0, false
	}
	var n int
	if _, err := fmt.Sscanf(rest, "%d", &n); err != nil || n < 2 {
		return 0, false
	}
	return n, true
}

// hookExitStatus maps a hook tool_response onto the closed exit vocabulary.
// The response shape varies by harness and tool; the only reliable error tell
// is an is_error/isError flag on an object response. Default is "ok" — the
// deadline/stall/orphan machinery does not depend on it.
func hookExitStatus(resp json.RawMessage) string {
	var obj struct {
		IsError  *bool `json:"is_error"`
		IsErrorC *bool `json:"isError"`
	}
	if err := json.Unmarshal(resp, &obj); err == nil {
		if (obj.IsError != nil && *obj.IsError) || (obj.IsErrorC != nil && *obj.IsErrorC) {
			return "error"
		}
	}
	return "ok"
}
