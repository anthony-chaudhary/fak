package trajctl

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"sync"
)

// BlockedAlternativeMessage is the structured alternative returned when an
// apology-repeat loop is detected.
const BlockedAlternativeMessage = "Identical call blocked. Do not retry without modifying arguments or reading error diagnostics."

var defaultApologyPatterns = []string{
	"apolog",             // covers apologize, apologizes, apologized, apologizing, apology, apologies, apologetic, apologise
	"sorry",              // covers sorry, i'm sorry, i am sorry
	"my mistake",         // covers my mistake, that's my mistake
	"i was wrong",        // covers i was wrong
	"excuse the error",   // covers excuse the error
	"excuse my error",    // covers excuse my error
	"excuse this error",  // covers excuse this error
	"my fault",           // covers my fault
	"my bad",             // covers my bad
	"pardon the mistake", // covers pardon the mistake
	"pardon the error",   // covers pardon the error
	"pardon me",          // covers pardon me
	"made a mistake",     // covers made a mistake, i made a mistake
	"made an error",      // covers made an error, i made an error
	"stand corrected",    // covers stand corrected, i stand corrected
	"stood corrected",    // covers stood corrected
}

// HasApology reports whether text contains apologetic phrasing.
func HasApology(text string) bool {
	lower := strings.ToLower(text)
	for _, p := range defaultApologyPatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// CanonicalizeArgs normalizes arguments into a consistent canonical form.
// If args is valid JSON, it unmarshals and re-marshals the content to sort map keys
// and strip insignificant whitespace. If args is empty or whitespace, it returns "{}".
// Non-JSON arguments are returned with leading and trailing whitespace trimmed.
func CanonicalizeArgs(args string) string {
	trimmed := strings.TrimSpace(args)
	if trimmed == "" || trimmed == "{}" {
		return "{}"
	}
	var v any
	if err := json.Unmarshal([]byte(trimmed), &v); err == nil {
		if canon, err := json.Marshal(v); err == nil {
			return string(canon)
		}
	}
	return trimmed
}

// HashArgs returns the SHA-256 hex digest of the canonicalized arguments.
func HashArgs(args string) string {
	canon := CanonicalizeArgs(args)
	h := sha256.Sum256([]byte(canon))
	return hex.EncodeToString(h[:])
}

// identicalArgs reports whether two arguments representations have identical content or hash.
func identicalArgs(prevArgs, currArgs string) bool {
	return HashArgs(prevArgs) == HashArgs(currArgs)
}

// DetectApologyRepeat inspects the transition between a prior failing tool call
// and the current proposal. If the prior call failed, the current proposal targets
// the same tool with identical arguments, and the model's accompanying text contains
// apologetic remorse, it flags the repeat and returns a structured alternative message.
func DetectApologyRepeat(prevTool string, prevArgs string, prevFailed bool, currText string, currTool string, currArgs string) (isRepeat bool, alternativeMessage string) {
	if !prevFailed {
		return false, ""
	}
	if prevTool == "" || currTool == "" {
		return false, ""
	}
	if !strings.EqualFold(prevTool, currTool) {
		return false, ""
	}
	if !identicalArgs(prevArgs, currArgs) {
		return false, ""
	}
	if !HasApology(currText) {
		return false, ""
	}
	return true, BlockedAlternativeMessage
}

// ApologyRepeatTracker tracks consecutive tool calls and execution outcomes
// across turn boundaries to identify apology-repeat loops.
type ApologyRepeatTracker struct {
	mu         sync.Mutex
	lastTool   string
	lastArgs   string
	lastFailed bool
}

// NewApologyRepeatTracker creates an initialized ApologyRepeatTracker.
func NewApologyRepeatTracker() *ApologyRepeatTracker {
	return &ApologyRepeatTracker{}
}

// RecordOutcome records the outcome of a tool execution.
func (t *ApologyRepeatTracker) RecordOutcome(tool string, args string, failed bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.lastTool = tool
	t.lastArgs = args
	t.lastFailed = failed
}

// Check evaluates whether currTool and currArgs accompanied by currText
// constitute an apology-repeat of the previous failing call.
func (t *ApologyRepeatTracker) Check(currText string, currTool string, currArgs string) (bool, string) {
	t.mu.Lock()
	prevTool := t.lastTool
	prevArgs := t.lastArgs
	prevFailed := t.lastFailed
	t.mu.Unlock()
	return DetectApologyRepeat(prevTool, prevArgs, prevFailed, currText, currTool, currArgs)
}

// Reset clears the tracker's recorded state.
func (t *ApologyRepeatTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.lastTool = ""
	t.lastArgs = ""
	t.lastFailed = false
}

// LastCall returns the last recorded tool, arguments, and failure status.
func (t *ApologyRepeatTracker) LastCall() (tool string, args string, failed bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.lastTool, t.lastArgs, t.lastFailed
}
