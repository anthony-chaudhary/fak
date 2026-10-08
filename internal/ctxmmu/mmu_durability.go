package ctxmmu

import (
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/numfmt"
)

// ---------------------------------------------------------------------------
// Write-time durability classification (S7 rung 1 — issue #82).
//
// The write gate decides WHETHER a result may enter context; it never decided HOW
// LONG it should be believed. classifyDurability assigns a result a durability class
// from a cheap lexical/tense prior, and MMU.Admit stamps it on Verdict.Meta. The
// downstream durable boundary (recall's promotion gate) reads it to enforce the
// headline inversion: expire by default, promotion is the earned exception.
// ---------------------------------------------------------------------------

// Durability classes ride the OPEN Verdict.Meta map under DurabilityKey — orthogonal
// to the trust Kind, additive over the frozen ABI (TestABIGoldenFreeze does not move).
// v1 emits {turn, session, durable}; `bounded` is accepted as the explicit-expiry
// class even though the lexical prior does not infer it on its own. Readers degrade
// unknown values fail-closed to turn.
const (
	DurabilityKey     = "durability"
	DurabilityTurn    = "turn"    // true only this turn; the fail-closed default
	DurabilitySession = "session" // true for this session
	DurabilityBounded = "bounded" // true until a caller-supplied expiry
	DurabilityDurable = "durable" // true across sessions until revised — the only promotable class
)

const (
	ExpiryPolicyTurn     = "turn_end"
	ExpiryPolicySession  = "session_end"
	ExpiryPolicyRequired = "requires_expiry"
	ExpiryPolicyNone     = "none"
)

// DurabilityPolicy is the renderable context-ledger policy for a fact's truth duration.
type DurabilityPolicy struct {
	Class          string `json:"class"`
	ExpiryPolicy   string `json:"expiry_policy"`
	RequiresExpiry bool   `json:"requires_expiry,omitempty"`
}

// NormalizeDurabilityClass maps unknown durability tags to the fail-closed turn class.
func NormalizeDurabilityClass(class string) string {
	switch class {
	case DurabilityTurn, DurabilitySession, DurabilityBounded, DurabilityDurable:
		return class
	default:
		return DurabilityTurn
	}
}

// PolicyForDurability maps a durability class to its deterministic expiry policy.
func PolicyForDurability(class string) DurabilityPolicy {
	class = NormalizeDurabilityClass(class)
	switch class {
	case DurabilitySession:
		return DurabilityPolicy{Class: class, ExpiryPolicy: ExpiryPolicySession}
	case DurabilityBounded:
		return DurabilityPolicy{Class: class, ExpiryPolicy: ExpiryPolicyRequired, RequiresExpiry: true}
	case DurabilityDurable:
		return DurabilityPolicy{Class: class, ExpiryPolicy: ExpiryPolicyNone}
	default:
		return DurabilityPolicy{Class: DurabilityTurn, ExpiryPolicy: ExpiryPolicyTurn}
	}
}

// DurabilityLabel renders the compact ledger label a context fact can carry.
func DurabilityLabel(class string) string {
	p := PolicyForDurability(class)
	return "durability=" + p.Class + " expiry=" + p.ExpiryPolicy
}

var (
	// durableFrame: habitual/stative frames => durable (stated preferences, identity).
	// Deliberately NARROW — a false-positive promotion (a transient fact recalled as
	// current truth) is "strictly worse than absence" (CONTEXT-IS-NOT-MEMORY.md §4), so
	// weak copular/imperative alternations are excluded: `my <noun> is` is whitelisted
	// to identity/disposition nouns (NOT the generic `my \w+ is`, which fired on
	// `my build is failing right now`), and bare `i am a` / `call me` / `we work` are
	// dropped (they fire on `I am a bit busy`, `call me back later`, `we work until 5pm`).
	// RE2 has no negative lookahead, so the safe shape is a noun whitelist, not exclusion.
	durableFrame = regexp.MustCompile(`(?i)\b(prefers?|preferred|preference|i always|i usually|i normally|we (?:use|prefer)|my (?:name|role|title|pronouns?|timezone|tz|email|handle|username|nickname|birthday|address|favou?rite \w+) is)\b`)
	// sessionFrame: explicit session-scoped frames => session.
	sessionFrame = regexp.MustCompile(`(?i)(\bthis session\b|\bthis branch\b|\bworking on\b|\btoday'?s task\b|\bcurrent task\b|\bfor now\b)`)
	// turnFrame: punctual/progressive deictics + bare clock times => turn.
	turnFrame = regexp.MustCompile(`(?i)(\bright now\b|\bcurrently\b|\btoday\b|\bat the moment\b|\bas of now\b|\bit is now\b|\b\d{1,2}\s*(?:am|pm)\b|\b\d{1,2}:\d{2}\b|o'?clock)`)
)

// classifyDurability assigns a rung-1 write-time durability class to a produced result
// from a cheap lexical/tense prior over the bytes — NO model call, and explicitly NOT
// the Zhang-Choi fact-duration estimator (CONTEXT-IS-NOT-MEMORY.md §5), which has no
// callsite and is deferred. It leans on bytes (and may consult the tool) only; it does
// NOT take a turn index / session id / principal / as-of clock — threading those into
// the hot ResultAdmitter signature is a named follow-on, not this rung.
//
// Precedence is most-durable-first so a stated preference is durable even if it also
// mentions "today"; a clearly session-scoped frame ("today's task") beats the bare
// "today" deictic; everything unmatched fails closed to turn, because a false-positive
// promotion (a poltergeist fact recalled as current) is the expensive error direction.
func isReadFileTool(tool string) bool {
	t := strings.ToLower(strings.TrimSpace(tool))
	if idx := strings.LastIndex(t, "__"); idx >= 0 {
		t = t[idx+2:]
	}
	t = strings.TrimPrefix(t, "functions.")
	switch t {
	case "read", "fak_read", "read_file", "readfile", "get_file":
		return true
	default:
		return false
	}
}

func isReadFileCall(c *abi.ToolCall, r *abi.Result) bool {
	if r != nil && r.Meta != nil {
		switch r.Meta["engine"] {
		case "fakread", "codetools.read":
			return true
		}
	}
	if c != nil {
		switch c.Engine {
		case "fakread", "codetools.read":
			return true
		}
		if isReadFileTool(c.Tool) {
			return true
		}
	}
	return false
}

type outputToolClass uint8

const (
	outputToolOther outputToolClass = iota
	outputToolRead
	outputToolSearchExec
)

func isSearchExecTool(tool string) bool {
	t := strings.ToLower(strings.TrimSpace(tool))
	if idx := strings.LastIndex(t, "__"); idx >= 0 {
		t = t[idx+2:]
	}
	t = strings.TrimPrefix(t, "functions.")
	t = strings.TrimPrefix(t, "codetools.")
	switch t {
	case "bash", "powershell", "pwsh", "shell", "exec", "exec_command", "shell_command", "run_terminal_cmd", "rg", "ripgrep", "grep":
		return true
	default:
		return false
	}
}

func classifyOutputTool(c *abi.ToolCall, r *abi.Result) outputToolClass {
	if isReadFileCall(c, r) {
		return outputToolRead
	}
	if c != nil && (isSearchExecTool(c.Tool) || isSearchExecTool(c.Engine)) {
		return outputToolSearchExec
	}
	if r != nil && isSearchExecTool(r.Meta["engine"]) {
		return outputToolSearchExec
	}
	return outputToolOther
}

// oversizeThresholdForClass is the single tool-class to page-out limit mapping.
func oversizeThresholdForClass(class outputToolClass) int {
	switch class {
	case outputToolRead:
		return numfmt.EnvPositiveInt("FAK_READ_OVERSIZE_BYTES", ReadOversizeBytes)
	case outputToolSearchExec:
		return SearchExecOversizeBytes
	default:
		return OversizeBytes
	}
}

func (m *MMU) oversizeThreshold(c *abi.ToolCall, r *abi.Result) int {
	return oversizeThresholdForClass(classifyOutputTool(c, r))
}

func classifyDurability(c *abi.ToolCall, body []byte) string {
	_ = c // reserved: a future tool prior (a clock/now source is inherently turn-class)
	switch {
	case durableFrame.Match(body):
		return DurabilityDurable
	case sessionFrame.Match(body):
		return DurabilitySession
	case turnFrame.Match(body):
		return DurabilityTurn
	default:
		return DurabilityTurn
	}
}

// ClassifyText is the exported, chat-message-shaped entry to the SAME rung-1
// durability prior classifyDurability runs over tool-result bytes — it lets a caller
// outside the admit path (e.g. a budget-reset carryover builder that must decide which
// transcript lines a fresh session keeps) reuse the shipped tense/deixis classifier
// instead of reinventing it. It runs the identical durableFrame/sessionFrame/turnFrame
// priors over the message text and fails closed to turn, so "it's 3pm" => turn and "I
// prefer afternoons" => durable, exactly as the admit path classifies the same words.
//
// It takes the message text only (no tool call): a chat line has no producing tool,
// and classifyDurability's tool argument is reserved/unused at rung 1. role is accepted
// for forward compatibility (a future prior may weight assistant vs user vs system text)
// but does not change the rung-1 verdict.
func ClassifyText(role, content string) string {
	_ = role // reserved: a future prior may weight by author; rung 1 is text-only
	return classifyDurability(nil, []byte(content))
}
