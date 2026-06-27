// scan.go — the DETECT half of the resume-cache process. resume.go's Plan answers "given
// a dormant session, what should I do with the cache?"; Diagnose answers the question that
// comes BEFORE it on a real transcript store: "WHICH dormant sessions actually crashed on a
// rate limit and never resumed — the ones that need a managed restart at all?"
//
// # The gap it closes
//
// A rate-limited agent session does not end cleanly. Claude Code records the refusal as a
// synthetic assistant turn ("You've hit your session limit · resets 8pm", "hit your weekly
// limit", "Rate limited") and then stops — the transcript's last real model turn is followed
// by that error and nothing more. The operator wants to bring those sessions back, but a
// blind `claude --resume` re-prefills the WHOLE resident transcript from a cold cache (the
// single most expensive turn the session will ever run) only to often hit the same limit
// again. Diagnose is the deterministic verdict that turns a pile of `.jsonl` transcripts into
// the short list that needs a restart, each one paired with the managed-cache Plan (CUT /
// RESET vs RESUME_FULL) that makes the restart a "new session with cache managed" instead of
// a cold full re-prefill.
//
// # Content-blind by construction
//
// Diagnose reasons only over a closed set of EVENT FACTS — was this record a real model turn
// (and how big was its prompt), or a synthetic rate-limit refusal, or an unrelated error —
// never over transcript text. The I/O shell (cmd/fak/resume.go) parses the `.jsonl`, matches
// the provider's refusal strings against the closed limit vocabulary, and hands Diagnose only
// the typed events. So this leaf stays pure (same events + Input → same Diagnosis), stdlib-
// only, and incapable of leaking a byte of session content into a verdict.
package resume

// EventKind is the coarse class of one transcript record that the diagnosis cares about.
// Everything the classifier needs is captured by the kind plus the two payload fields on
// Event; the raw record (user turns, tool results, system bookkeeping) collapses to
// EventOther and is ignored.
type EventKind string

const (
	// EventRealAssistant is a genuine model turn: a non-synthetic assistant message that
	// carries prompt usage. Its PromptTokens is the resident context a resume would re-prefill;
	// the LAST one in the stream sets the session's resident size.
	EventRealAssistant EventKind = "real_assistant"
	// EventRateLimitError is a synthetic api-error turn whose text matched the closed
	// rate/usage/session-limit vocabulary — the refusal that ended a rate-limited session.
	// Its LimitReason is the closed token the shell classified it as.
	EventRateLimitError EventKind = "rate_limit_error"
	// EventOtherError is a synthetic api-error turn that is NOT a rate limit (an overloaded
	// server, a transport failure). It marks an unclean end, but not the one this verb restarts.
	EventOtherError EventKind = "other_error"
	// EventOther is any record the classifier ignores — user turns, tool results, system and
	// bookkeeping lines. Present so the shell can stream every record through without filtering.
	EventOther EventKind = "other"
)

// Event is the closed set of facts the diagnosis needs about ONE transcript record — never
// any transcript content. The shell extracts these; the leaf only reasons over them.
type Event struct {
	// Kind is the record's class (see EventKind).
	Kind EventKind `json:"kind"`
	// PromptTokens is meaningful only for EventRealAssistant: that turn's full prompt size
	// (input + cache-read + cache-creation), i.e. the context a resume of this session would
	// re-prefill. Zero for every other kind.
	PromptTokens int `json:"prompt_tokens,omitempty"`
	// LimitReason is meaningful only for EventRateLimitError: the closed token the shell
	// classified the refusal as (see the Limit* vocabulary). Empty for every other kind.
	LimitReason string `json:"limit_reason,omitempty"`
}

// The closed rate-limit reason vocabulary, mirroring the distinct refusals Claude Code
// records. The shell maps the provider's refusal text onto exactly one of these so an
// observability sink (and this leaf) never has to see the raw string. A refusal that matches
// the rate-limit family but none of the specific phrasings falls back to LimitRate.
const (
	// LimitSession: "You've hit your session limit · resets <time>" — the 5-hour rolling cap.
	LimitSession = "session_limit"
	// LimitWeekly: "You've hit your weekly limit · resets <date>" — the weekly cap.
	LimitWeekly = "weekly_limit"
	// LimitUsage: a generic "usage limit reached" refusal.
	LimitUsage = "usage_limit"
	// LimitRate: a server-side "Rate limited" / HTTP 429 throttle (not your usage cap).
	LimitRate = "rate_limited"
)

// CrashKind classifies how a dormant session's transcript ended.
type CrashKind string

const (
	// CrashNone: the transcript's last meaningful event is a real model turn (or any earlier
	// rate-limit was recovered by a later real turn) — the session ended cleanly enough that
	// it does not need this verb's managed restart.
	CrashNone CrashKind = "none"
	// CrashRateLimit: the last real model turn is followed by a rate-limit refusal and no
	// further real turn — the session crashed on a limit and never resumed. This is the target.
	CrashRateLimit CrashKind = "rate_limit"
	// CrashOther: the session ended on a non-rate error with no real turn after it — an unclean
	// end, but not a rate-limit one (restart remediation differs), so it is reported, not flagged.
	CrashOther CrashKind = "other_error"
)

// Diagnosis is the deterministic verdict for one dormant session: how it ended and, when it
// crashed on a rate limit and never resumed, the managed-cache restart Plan that makes the
// restore a "new session with cache managed" rather than a cold full re-prefill.
type Diagnosis struct {
	// Crash is why the transcript ended (see CrashKind).
	Crash CrashKind `json:"crash"`
	// LimitReason is the closed rate-limit token when Crash is CrashRateLimit; empty otherwise.
	LimitReason string `json:"limit_reason,omitempty"`
	// Unresumed reports that the terminal error had no real model turn after it — the session
	// never came back on its own. True for both CrashRateLimit and CrashOther.
	Unresumed bool `json:"unresumed"`
	// NeedsRestart is the one bit `fak resume scan` filters on: the session crashed on a rate
	// limit and never resumed (Crash == CrashRateLimit). These are the sessions to bring back.
	NeedsRestart bool `json:"needs_restart"`
	// ResidentTokens is the size of the last real model turn's prompt — the context a resume
	// would re-prefill. Derived from the events (the synthetic refusal turn is skipped), unless
	// the caller pinned Input.ResidentTokens. Zero when the transcript has no real model turn.
	ResidentTokens int `json:"resident_tokens"`
	// RealTurns is how many real model turns the transcript carried — a crashed session with
	// zero real turns has nothing to manage (the cache plan is trivially RESUME_FULL).
	RealTurns int `json:"real_turns"`
	// Plan is the managed-cache restart decision over the resident facts (the same Report the
	// `fak resume plan` verb emits). Always populated, so even a clean session is priced.
	Plan Report `json:"plan"`
}

// Diagnose is THE deterministic detect-and-plan decision: given a transcript's ordered event
// facts and the resume Input (idle, TTL, pricing, horizon), it classifies how the session
// ended and pairs the verdict with the managed-cache restart Plan. Pure — same events and
// Input yield the same Diagnosis, with no clock, I/O, or model — and total over any input
// (an empty event slice yields a clean, zero-resident diagnosis, never a panic).
//
// The resident size is taken from the LAST real model turn's prompt (the most recent context
// a resume would re-prefill), and the synthetic rate-limit turn — which carries an all-zero
// usage block — is deliberately skipped so a crashed session is never mis-sized to zero. If
// the caller pinned Input.ResidentTokens (> 0) that pin wins, mirroring the flag precedence
// the single-transcript path already uses.
func Diagnose(events []Event, in Input) Diagnosis {
	resident, realTurns := residentFromEvents(events)
	if in.ResidentTokens > 0 {
		resident = in.ResidentTokens // an explicit operator pin wins over the derived size
	}
	crash, reason := classifyTail(events)

	in.ResidentTokens = resident
	return Diagnosis{
		Crash:          crash,
		LimitReason:    reason,
		Unresumed:      crash != CrashNone,
		NeedsRestart:   crash == CrashRateLimit,
		ResidentTokens: resident,
		RealTurns:      realTurns,
		Plan:           Plan(in),
	}
}

// residentFromEvents returns the prompt size of the LAST real model turn (the resident
// context a resume re-prefills) and the count of real model turns seen. Synthetic refusal
// turns (EventRateLimitError / EventOtherError) and everything else are skipped, so the
// all-zero usage of a rate-limit turn never masquerades as a zero-token session.
func residentFromEvents(events []Event) (resident, realTurns int) {
	for _, e := range events {
		if e.Kind == EventRealAssistant {
			realTurns++
			resident = e.PromptTokens // keep overwriting: the LAST real turn is the resident size
		}
	}
	return resident, realTurns
}

// classifyTail decides how the transcript ended. It finds the last real model turn, then
// looks at the tail AFTER it: the error closest to the end (scanning backward) is the
// terminal failure. A rate-limit error there means the session crashed on a limit and never
// resumed; a non-rate error there is an unclean-but-not-rate end; no error there (the last
// meaningful event is a real turn, or any earlier rate limit was followed by a real turn)
// means a clean enough end that needs no managed restart.
func classifyTail(events []Event) (CrashKind, string) {
	lastReal := -1
	for i, e := range events {
		if e.Kind == EventRealAssistant {
			lastReal = i
		}
	}
	for i := len(events) - 1; i > lastReal; i-- {
		switch events[i].Kind {
		case EventRateLimitError:
			return CrashRateLimit, events[i].LimitReason
		case EventOtherError:
			return CrashOther, ""
		}
	}
	return CrashNone, ""
}
