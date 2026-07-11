package gateway

import (
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// resultAdmissionNote names any inbound tool result the kernel PAGED OUT so a quarantine
// stub does not read as a broken tool — in ONE line. A quarantine is a routine safety
// precaution (most often a credential-shaped string or injection-shaped text in the
// tool's OWN output, which CAN be a false positive on placeholder/example or
// security-discussing content), not a sign the agent did anything wrong. The per-result
// verdicts (tool, reason, page-in id) still ride the machine-readable `fak` extension, so
// the prose only needs to carry the three things a reading model acts on: that something
// was held (not lost), that it is retrievable, and that it is not the agent's fault. The
// reason CODES (TRUST_VIOLATION / SECRET_EXFIL / OVERSIZE) are the closed-vocabulary
// labels — terse and self-describing — instead of a per-item paragraph. Returns "" when
// every result was a clean allow.
//
// This is the buffered/streaming Anthropic + Gemini prose; callers pass it through
// freshAdmissionNotes first so a held result is announced once — when the admission
// ledger screened it (fresh) — not on every replayed turn (#2417).
func resultAdmissionNote(adms []ResultAdmission) string {
	n := 0
	redacted := 0 // warn-first SECRET_REDACTED transforms: masked in place, NOT held out
	counts := map[string]int{}
	order := make([]string, 0, 4)
	livelocks := make([]string, 0, 1)
	for _, a := range adms {
		if a.Verdict.Kind == "TRANSFORM" && a.Verdict.Reason == reasonSecretRedacted {
			redacted++
			continue
		}
		if a.Verdict.Kind != "QUARANTINE" {
			continue
		}
		n++
		reason := a.Verdict.Reason
		if reason == "" {
			reason = "RESULT_FLOOR"
		}
		if _, seen := counts[reason]; !seen {
			order = append(order, reason)
		}
		counts[reason]++
		if a.Livelock != nil {
			livelocks = append(livelocks, resultLivelockInBandNote(a))
		}
	}
	if n == 0 {
		// No held-out result. If a credential span was MASKED in place (the warn-first
		// default), say so in one line — the rest of the result is in context, so this is
		// a WARN, not a "held out" banner, and never baits a re-read.
		return secretRedactedWarn(redacted)
	}
	noun, verb := "tool result", "was"
	if n > 1 {
		noun, verb = "tool results", "were"
	}
	parts := make([]string, 0, len(order))
	for _, reason := range order {
		if c := counts[reason]; c > 1 {
			parts = append(parts, reason+"×"+strconv.Itoa(c))
		} else {
			parts = append(parts, reason)
		}
	}
	// The retrievability clause must be HONEST per reason class. Most quarantines
	// (TRUST_VIOLATION / OVERSIZE / injection-shaped text) page back in via the kernel
	// page-in gate. Secret-class quarantines (SECRET_EXFIL / RESULT_SECRET_DISCOVERED)
	// do NOT: the page-in gate re-screens on release and refuses any bytes that still
	// match, so a credential-shaped result never returns to context — by design.
	// Telling a worker such bytes are "retrievable" is false AND actively harmful: it
	// baits a retrieval loop (worker #2704 re-read the same held result to ~125k tokens
	// chasing a promise the gate would never honor). So name the secret class honestly
	// and give it a concrete next step instead of a false promise.
	hasSecret := counts[reasonSecretExfil] > 0 || counts[reasonSecretDiscovered] > 0
	hasNonSecret := false
	for _, r := range order {
		if r != reasonSecretExfil && r != reasonSecretDiscovered {
			hasNonSecret = true
			break
		}
	}
	var retrievability string
	switch {
	case hasSecret && !hasNonSecret:
		retrievability = "held; credential-shaped bytes will NOT page back into context (secrets are absolute by design — the page-in gate re-screens and refuses release). If this was a placeholder/example or config value, do not re-read to retrieve it — proceed without the secret, or ask the operator to whitelist the value's shape."
	case hasSecret && hasNonSecret:
		retrievability = "paged out, not lost. Non-secret results are retrievable via the kernel page-in gate; credential-shaped results (SECRET_EXFIL) will NOT page back — do not re-read to retrieve them, proceed without them or ask the operator to whitelist the shape."
	default:
		retrievability = "paged out, not lost; retrievable via the kernel page-in gate."
	}
	note := "[fak] " + strconv.Itoa(n) + " " + noun + " " + verb + " held out of context (" +
		strings.Join(parts, ", ") + ") — " + retrievability + " " +
		"Routine guard behavior, not an error you caused; see the `fak` extension for per-result detail."
	if len(livelocks) > 0 {
		note += " " + strings.Join(livelocks, " ")
	}
	if w := secretRedactedWarn(redacted); w != "" {
		note += " " + w // a mixed turn: some held, some masked-in-place
	}
	return note
}

// freshAdmissionNotes selects the admissions the held-out banner should announce THIS
// turn, then hands them to resultAdmissionNote. Admission is now keyed to the ledger
// entry (#2417): a result is screened once, at first arrival (fresh), so its banner is
// emitted once — the client re-sends the full transcript every turn, but a replay is not
// fresh and is not re-announced. A result the livelock detector just annotated is always
// included even on a replay, so an escalating repeated-quarantine loop still surfaces.
// The machine-readable verdicts still ride the `fak` extension on every turn (the ledger
// record is consulted), so suppressing the repeated paragraph costs no signal. An empty
// trace never dedups upstream, so every admission is marked fresh — the un-deduped note.
func freshAdmissionNotes(adms []ResultAdmission) []ResultAdmission {
	out := make([]ResultAdmission, 0, len(adms))
	for _, a := range adms {
		if a.fresh || a.Livelock != nil {
			out = append(out, a)
		}
	}
	return out
}

func resultLivelockInBandNote(a ResultAdmission) string {
	if a.Livelock == nil {
		return ""
	}
	note := "LIVELOCK_DETECTED repeat=" + strconv.Itoa(a.Livelock.RepeatCount) +
		" repeated_result=" + livelockCallLabel(*a.Livelock) +
		" approach=" + a.Livelock.SuggestedChange
	if a.Livelock.Escalate {
		note += " ABORT=terminal (re-reading this same result keeps producing the same held/paged-out stub — it will NOT change; stop re-reading, proceed without it or report the blocker)"
	} else if a.Livelock.Fuse {
		note += " fuse=armed"
	}
	return note
}

// resultNoteKey is the stable per-result dedup key. The tool_call_id is replayed
// byte-identically across turns by the client, so it identifies the SAME held result over
// a session; an idless result (a nameless cross-boundary payload) falls back to
// tool|reason, which collapses repeats of the same shape without a stable id.
func resultNoteKey(a ResultAdmission) string {
	if a.ToolCallID != "" {
		return a.ToolCallID
	}
	return a.Tool + "|" + a.Verdict.Reason
}

// anyRepaired reports whether the kernel rewrote any admitted call's arguments.
func anyRepaired(adjs []ToolAdjudication) bool {
	for _, a := range adjs {
		if a.Admitted && a.Verdict.Kind == "TRANSFORM" {
			return true
		}
	}
	return false
}

func anyLivelock(adjs []ToolAdjudication) bool {
	for _, a := range adjs {
		if a.Livelock != nil {
			return true
		}
	}
	return false
}

// loopBreakThreshold is how long a degenerate tail of assistant turns must grow before
// the gateway short-circuits the loop. A capable model never repeats itself verbatim or
// echoes a kernel notice; a small local model fronted by the kernel often does exactly
// that — turn after turn, with no new tool call and no progress — until the harness
// turn-cap ends the session with an empty result. At threshold=2 the third such
// degenerate turn trips the break, reclaiming the rest of the turn budget.
const loopBreakThreshold = 2

// degenerateStreak counts the trailing run of NON-PROGRESSING assistant turns in the
// replayed history. A turn is degenerate when it is text-only (no tool call survived to
// drive work forward) AND it is either:
//
//   - a `[fak]` echo: text the KERNEL originates (adjudicationNote / resultAdmissionNote)
//     that a model should never produce — the model parroting the refusal note back; or
//   - a verbatim repeat of the previous assistant turn — the model emitting the SAME
//     prose every turn (e.g. the same graceful "I can't, policy blocks it" refusal).
//
// Counted from the END so a single stale line in a long healthy transcript never trips
// it; only an unbroken degenerate tail does. A turn that carries a tool_use, or differs
// from its predecessor and is not a `[fak]` echo, is real progress and breaks the run.
// Stateless: it reads exactly what Claude Code replayed.
func degenerateStreak(messages []agent.Message) int {
	// Collect the trailing run of text-only assistant turns (most recent first). A turn
	// carrying a tool_use is forward progress and ends the run.
	var tail []string
	for i := len(messages) - 1; i >= 0; i-- {
		m := messages[i]
		if m.Role != agent.RoleAssistant {
			continue // user / tool turns interleave; skip without breaking the run
		}
		if len(m.ToolCalls) > 0 {
			break
		}
		tail = append(tail, m.Content)
	}
	// Count, from the most recent backward, how many turns are degenerate: a `[fak]`
	// echo (kernel-originated text the model should never produce), or a verbatim repeat
	// of the adjacent more-recent assistant turn (the model emitting the same prose).
	n := 0
	for i, c := range tail {
		isEcho := strings.Contains(c, "[fak]")
		isRepeat := c != "" && ((i > 0 && c == tail[i-1]) || (i+1 < len(tail) && c == tail[i+1]))
		if isEcho || isRepeat {
			n++
			continue
		}
		break
	}
	return n
}

// pendingFreshUserInput reports whether the turns after the most recent assistant
// message — the input this completion must answer — carry fresh user prose: a plain
// RoleUser turn that is non-empty, not kernel-originated (`[fak]`), and not a verbatim
// repeat of a user turn the model already answered. Fresh input means the next turn is
// not predetermined to repeat, so the repetition-loop steer must stand down. The
// observed victim is an out-of-band evaluator call — Claude Code's prompt-based Stop
// hook replays the stuck session PLUS a novel judge prompt and must answer with
// structured JSON; steering it substituted prose for the verdict, and every stop
// errored with "JSON validation failed". Tool results decode as RoleTool and a
// re-injected identical nudge is a verbatim repeat, so the dead loops the steer exists
// for — mechanical continuations that add nothing new — still trip it.
func pendingFreshUserInput(messages []agent.Message) bool {
	last := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == agent.RoleAssistant {
			last = i
			break
		}
	}
	if last < 0 {
		return false
	}
	answered := make(map[string]struct{})
	for _, m := range messages[:last+1] {
		if m.Role == agent.RoleUser && m.Content != "" {
			answered[m.Content] = struct{}{}
		}
	}
	for _, m := range messages[last+1:] {
		if m.Role != agent.RoleUser || m.Content == "" || strings.Contains(m.Content, "[fak]") {
			continue
		}
		if _, ok := answered[m.Content]; !ok {
			return true
		}
	}
	return false
}

// repetitionLoopSteer returns a terminal corrective turn when the model is stuck in a
// degenerate tail (degenerateStreak ≥ loopBreakThreshold) — echoing the kernel's `[fak]`
// notes or repeating the same prose with no progress — or nil otherwise. The corrective
// turn is a single plain-text assistant message, deliberately NOT prefixed with `[fak]`
// (so it can't itself feed the echo detector) and distinct from any repeated line, that
// ends the turn (end_turn). Returned BEFORE the planner runs, it breaks the loop
// deterministically and cheaply (no model round-trip) and Claude Code reads a normal
// terminal assistant turn so its agent loop settles instead of grinding to the turn-cap.
// A pending fresh user turn vetoes the steer: new input (an operator question, a Stop
// hook evaluator's judge prompt) deserves a real model answer even behind a stuck tail.
func repetitionLoopSteer(messages []agent.Message, id, model string) *anthropicTurn {
	if degenerateStreak(messages) < loopBreakThreshold {
		return nil
	}
	if pendingFreshUserInput(messages) {
		return nil
	}
	const steer = "I was repeating myself without making progress: a tool I tried is " +
		"blocked by the security policy and cannot be used. Stopping that loop. If the " +
		"request needs the blocked tool I cannot complete it; otherwise tell me what to " +
		"answer and I will respond directly."
	if id == "" {
		id = "msg_fak_" + itoa(uint64(time.Now().UnixNano()))
	}
	return &anthropicTurn{
		ID:    id,
		Model: model,
		Blocks: []agent.AnthropicBlockOut{
			{Type: "text", Text: steer},
		},
		Stop: "end_turn",
	}
}

// prependTextBlock inserts an in-band [fak] note as the FIRST content block so a
// client that reads content top-to-bottom (Claude Code) sees the kernel's decision
// before the surviving tool_use blocks it is about to run. The note never replaces
// model prose — existing text/tool_use blocks follow it untouched.
func prependTextBlock(blocks []agent.AnthropicBlockOut, text string) []agent.AnthropicBlockOut {
	out := make([]agent.AnthropicBlockOut, 0, len(blocks)+1)
	out = append(out, agent.AnthropicBlockOut{Type: "text", Text: text})
	return append(out, blocks...)
}

// fakExtFrom builds the response extension from a turn's proposed-call
// adjudications and inbound-result admissions, or nil when there is nothing to
// report (so the `fak` key is omitted on a turn with no tool activity at all).
func fakExtFrom(adjs []ToolAdjudication, results []ResultAdmission) *FakExt {
	if len(adjs) == 0 && len(results) == 0 {
		return nil
	}
	return &FakExt{Adjudications: adjs, ResultAdmissions: results}
}
