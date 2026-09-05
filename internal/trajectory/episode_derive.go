package trajectory

import (
	"fmt"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// ClassifyToolIntent derives the semantic episode phase from a tool invocation, its input arguments,
// and whether the invocation resulted in an error.
// Mapping rules:
//   - Read, Grep, Glob, search -> EpisodeExplore
//   - Edit, Write, patch, modify -> EpisodeMutate
//   - Test, build, lint, vet -> EpisodeVerify
//   - Error, rollback, undo, revert -> EpisodeRecovery
func ClassifyToolIntent(tool string, input string, isError bool) ctxmmu.EpisodeType {
	if isError {
		return ctxmmu.EpisodeRecovery
	}

	normTool := normalizeToolName(tool)
	normInput := strings.ToLower(strings.TrimSpace(input))

	// 1. Explicit error / rollback / undo recovery tools
	if isRecoveryTool(normTool) {
		return ctxmmu.EpisodeRecovery
	}

	// 2. Command runners (bash, exec, terminal, sh): inspect command payload
	if isCommandRunner(normTool) {
		if ep, ok := classifyCommandPayload(normInput); ok {
			return ep
		}
	}

	// 3. Direct tool classifications
	if isExploreTool(normTool) {
		return ctxmmu.EpisodeExplore
	}
	if isMutateTool(normTool) {
		return ctxmmu.EpisodeMutate
	}
	if isVerifyTool(normTool) {
		return ctxmmu.EpisodeVerify
	}

	// 4. Heuristic inference from input if tool name is generic or unclassified
	if ep, ok := classifyCommandPayload(normInput); ok {
		return ep
	}

	// Default fall-through for unknown exploration
	return ctxmmu.EpisodeExplore
}

// DeriveEpisodeType is a convenience helper to classify intent from tool name and error status.
func DeriveEpisodeType(tool string, isError bool) ctxmmu.EpisodeType {
	return ClassifyToolIntent(tool, "", isError)
}

func normalizeToolName(tool string) string {
	t := strings.ToLower(strings.TrimSpace(tool))
	// Strip namespaces e.g. "default_api:read" or "mcp__read"
	if idx := strings.LastIndex(t, ":"); idx != -1 {
		t = t[idx+1:]
	}
	if idx := strings.LastIndex(t, "__"); idx != -1 {
		t = t[idx+2:]
	}
	return t
}

func isExploreTool(t string) bool {
	switch t {
	case "read", "read_file", "fak_read", "cat", "view", "head", "tail",
		"grep", "search", "content_search", "ripgrep", "rg",
		"glob", "file_search", "list_dir", "find_files", "find",
		"explore", "inspect", "fetch", "webfetch", "ask_question", "question":
		return true
	default:
		return strings.HasPrefix(t, "read") || strings.HasPrefix(t, "grep") || strings.HasPrefix(t, "glob") || strings.HasPrefix(t, "list")
	}
}

func isMutateTool(t string) bool {
	switch t {
	case "edit", "write", "write_file", "modify", "patch", "apply_patch", "apply_diff",
		"create_file", "delete_file", "replace", "sed", "rm", "mkdir", "touch", "append":
		return true
	default:
		return strings.HasPrefix(t, "edit") || strings.HasPrefix(t, "write") || strings.HasPrefix(t, "patch")
	}
}

func isVerifyTool(t string) bool {
	switch t {
	case "test", "go_test", "pytest", "run_tests", "unit_test",
		"build", "go_build", "make", "compile",
		"lint", "vet", "go_vet", "check", "validate", "audit", "verify", "eval":
		return true
	default:
		return strings.HasPrefix(t, "test") || strings.HasPrefix(t, "build") || strings.HasPrefix(t, "lint") || strings.HasPrefix(t, "vet")
	}
}

func isRecoveryTool(t string) bool {
	switch t {
	case "rollback", "undo", "revert", "recover", "restore", "git_reset", "git_checkout", "error":
		return true
	default:
		return strings.HasPrefix(t, "rollback") || strings.HasPrefix(t, "undo") || strings.HasPrefix(t, "revert") || strings.HasPrefix(t, "recover")
	}
}

func isCommandRunner(t string) bool {
	switch t {
	case "bash", "sh", "zsh", "powershell", "pwsh", "cmd", "exec", "terminal", "run", "process":
		return true
	default:
		return false
	}
}

func classifyCommandPayload(input string) (ctxmmu.EpisodeType, bool) {
	if input == "" {
		return "", false
	}

	// Recovery commands
	if strings.Contains(input, "git checkout") ||
		strings.Contains(input, "git restore") ||
		strings.Contains(input, "git reset") ||
		strings.Contains(input, "git revert") ||
		strings.Contains(input, "fak recover") ||
		strings.Contains(input, "rollback") ||
		strings.Contains(input, "undo") {
		return ctxmmu.EpisodeRecovery, true
	}

	// Verification commands
	if strings.Contains(input, "go test") ||
		strings.Contains(input, "pytest") ||
		strings.Contains(input, "cargo test") ||
		strings.Contains(input, "npm test") ||
		strings.Contains(input, "make test") ||
		strings.Contains(input, "go build") ||
		strings.Contains(input, "make build") ||
		strings.Contains(input, "cargo build") ||
		strings.Contains(input, "go vet") ||
		strings.Contains(input, "golangci-lint") ||
		strings.Contains(input, "fak validate") ||
		strings.Contains(input, "fak buildcheck") ||
		strings.Contains(input, "ci-preflight") ||
		strings.Contains(input, "lint") {
		return ctxmmu.EpisodeVerify, true
	}

	// Mutation commands
	if strings.Contains(input, "sed ") ||
		strings.Contains(input, "echo ") && strings.Contains(input, ">") ||
		strings.Contains(input, "cat <<") ||
		strings.Contains(input, "touch ") ||
		strings.Contains(input, "mkdir ") ||
		strings.Contains(input, "rm ") ||
		strings.Contains(input, "cp ") ||
		strings.Contains(input, "mv ") ||
		strings.Contains(input, "git commit") {
		return ctxmmu.EpisodeMutate, true
	}

	// Exploration commands
	if strings.Contains(input, "git status") ||
		strings.Contains(input, "git log") ||
		strings.Contains(input, "git diff") ||
		strings.Contains(input, "cat ") ||
		strings.Contains(input, "grep ") ||
		strings.Contains(input, "rg ") ||
		strings.Contains(input, "find ") ||
		strings.Contains(input, "ls ") ||
		strings.Contains(input, "pwd") {
		return ctxmmu.EpisodeExplore, true
	}

	return "", false
}

// EpisodeTransition captures a switch from one semantic episode to another.
type EpisodeTransition struct {
	FromEpisode ctxmmu.EpisodeType `json:"from_episode"`
	ToEpisode   ctxmmu.EpisodeType `json:"to_episode"`
	TurnSeq     int                `json:"turn_seq"`
	Tool        string             `json:"tool"`
	Reason      string             `json:"reason,omitempty"`
}

// EpisodeSpan represents a contiguous sequence of turns within the same semantic episode.
type EpisodeSpan struct {
	EpisodeType ctxmmu.EpisodeType   `json:"episode_type"`
	StartSeq    int                  `json:"start_seq"`
	EndSeq      int                  `json:"end_seq"`
	TurnCount   int                  `json:"turn_count"`
	Tools       []string             `json:"tools,omitempty"`
	Digest      ctxmmu.EpisodeDigest `json:"digest,omitempty"`
}

// EpisodeSequenceClassifier maintains sliding sequence state across turns and emits
// transitions whenever an agent crosses an episode boundary.
type EpisodeSequenceClassifier struct {
	mu             sync.Mutex
	currentEpisode ctxmmu.EpisodeType
	transitions    []EpisodeTransition
}

// NewEpisodeSequenceClassifier creates a classifier initialized to EpisodeExplore.
func NewEpisodeSequenceClassifier() *EpisodeSequenceClassifier {
	return &EpisodeSequenceClassifier{
		currentEpisode: ctxmmu.EpisodeExplore,
	}
}

// CurrentEpisode returns the current active episode.
func (c *EpisodeSequenceClassifier) CurrentEpisode() ctxmmu.EpisodeType {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.currentEpisode
}

// Step evaluates a tool call and returns the resulting episode type and an EpisodeTransition
// if a phase boundary was crossed.
func (c *EpisodeSequenceClassifier) Step(tool string, input string, isError bool, turnSeq int) (ctxmmu.EpisodeType, *EpisodeTransition) {
	c.mu.Lock()
	defer c.mu.Unlock()

	next := ClassifyToolIntent(tool, input, isError)
	if next != c.currentEpisode {
		trans := EpisodeTransition{
			FromEpisode: c.currentEpisode,
			ToEpisode:   next,
			TurnSeq:     turnSeq,
			Tool:        tool,
			Reason:      fmt.Sprintf("tool %s triggered %s -> %s transition", tool, c.currentEpisode, next),
		}
		c.transitions = append(c.transitions, trans)
		c.currentEpisode = next
		ret := trans
		return next, &ret
	}
	return next, nil
}

// Transitions returns a copy of all observed episode transitions.
func (c *EpisodeSequenceClassifier) Transitions() []EpisodeTransition {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.transitions) == 0 {
		return nil
	}
	return append([]EpisodeTransition(nil), c.transitions...)
}

// DeriveEpisodeTransitions walks a slice of trajectory Turn records and derives all episode transitions.
func DeriveEpisodeTransitions(turns []Turn) []EpisodeTransition {
	classifier := NewEpisodeSequenceClassifier()
	for _, t := range turns {
		isErr := t.Verdict == "DENY" || strings.Contains(strings.ToLower(t.Verdict), "error") || t.Reason != ""
		classifier.Step(t.Tool, t.Query, isErr, t.Seq)
	}
	return classifier.Transitions()
}

// DeriveEpisodeSpans groups a slice of Turn records into contiguous EpisodeSpan blocks.
func DeriveEpisodeSpans(turns []Turn) []EpisodeSpan {
	if len(turns) == 0 {
		return nil
	}

	var spans []EpisodeSpan
	var currentSpan *EpisodeSpan

	for _, t := range turns {
		isErr := t.Verdict == "DENY" || strings.Contains(strings.ToLower(t.Verdict), "error") || t.Reason != ""
		ep := ClassifyToolIntent(t.Tool, t.Query, isErr)

		if currentSpan == nil || currentSpan.EpisodeType != ep {
			if currentSpan != nil {
				spans = append(spans, *currentSpan)
			}
			currentSpan = &EpisodeSpan{
				EpisodeType: ep,
				StartSeq:    t.Seq,
				EndSeq:      t.Seq,
				TurnCount:   1,
				Tools:       []string{t.Tool},
			}
		} else {
			currentSpan.EndSeq = t.Seq
			currentSpan.TurnCount++
			currentSpan.Tools = append(currentSpan.Tools, t.Tool)
		}
	}

	if currentSpan != nil {
		spans = append(spans, *currentSpan)
	}

	return spans
}

// DeriveEpisodesFromEvents analyzes canonical Event records and produces both spans and transitions.
func DeriveEpisodesFromEvents(events []Event) ([]EpisodeSpan, []EpisodeTransition) {
	if len(events) == 0 {
		return nil, nil
	}

	var turns []Turn
	for i, e := range events {
		if e.Kind != EventTool && e.Kind != EventError {
			continue
		}
		tool := payloadString(e.Payload, "tool", "name")
		if tool == "" {
			tool = e.Action
		}
		verdict := e.Action
		if e.Kind == EventError || e.Action == "failed" || e.Action == "error" {
			verdict = "error"
		}
		turns = append(turns, Turn{
			Seq:     i + 1,
			Tool:    tool,
			Query:   payloadString(e.Payload, "query", "input", "command", "args"),
			Verdict: verdict,
			Reason:  payloadString(e.Payload, "reason", "error"),
			Bytes:   int64(len(e.Payload)),
		})
	}

	return DeriveEpisodeSpans(turns), DeriveEpisodeTransitions(turns)
}

// CompileEpisodesFromTurns feeds turns into an EpisodeTracker, executing transitions between episodes
// and compiling immutable digests.
func CompileEpisodesFromTurns(turns []Turn, tracker *ctxmmu.EpisodeTracker) ([]ctxmmu.EpisodeDigest, error) {
	if tracker == nil {
		tracker = ctxmmu.NewEpisodeTracker(nil)
	}

	for _, t := range turns {
		isErr := t.Verdict == "DENY" || strings.Contains(strings.ToLower(t.Verdict), "error") || t.Reason != ""
		next := ClassifyToolIntent(t.Tool, t.Query, isErr)

		if next != tracker.CurrentEpisode() {
			if _, err := tracker.Transition(next); err != nil {
				return nil, err
			}
		}

		toks := t.TokenEstimate
		if toks <= 0 && t.Bytes > 0 {
			toks = int(t.Bytes / 4)
		}
		if toks <= 0 {
			toks = 10
		}

		rec := ctxmmu.EpisodeTurnRecord{
			TurnIndex: t.Seq,
			ToolName:  t.Tool,
			Input:     t.Query,
			Tokens:    toks,
			Error:     t.Reason,
		}
		if _, err := tracker.RecordTurn(rec); err != nil {
			return nil, err
		}
	}

	return tracker.Digests(), nil
}
