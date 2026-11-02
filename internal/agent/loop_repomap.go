package agent

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/codetools"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// RepoMapPreload selects only explicitly named Go files for one fresh native arm.
// It never discovers files. Each selection still crosses the normal session,
// tool-budget, route, kernel policy and confined coding-Read boundary.
type RepoMapPreload struct {
	Files  []string
	Tokens int
}

// WithRepoMapPreload opts into bounded upfront Read calls and a derived map.
// The option is not inherited by spawned children and is refused for raw arms,
// existing conversations and resumed checkpoints. Unset preserves the old path.
func WithRepoMapPreload(req RepoMapPreload) (RunOption, error) {
	if req.Tokens == 0 {
		req.Tokens = ctxmmu.RepoMapDefaultTokens
	}
	if len(req.Files) == 0 || len(req.Files) > ctxmmu.RepoMapMaxFiles ||
		req.Tokens < 256 || req.Tokens > ctxmmu.RepoMapMaxTokens {
		return nil, errors.New("REPOMAP_OPTIONS")
	}
	req.Files = append([]string(nil), req.Files...)
	seen := make(map[string]bool, len(req.Files))
	for _, p := range req.Files {
		if err := ctxmmu.ValidateRepoMapPath(p); err != nil {
			return nil, err
		}
		if seen[p] {
			return nil, errors.New("REPOMAP_DUPLICATE")
		}
		seen[p] = true
	}
	return func(c *runConfig) {
		copyReq := req
		copyReq.Files = append([]string(nil), req.Files...)
		c.repoMap = &copyReq
	}, nil
}

type repoMapPreloadState struct {
	tools    *codetools.Toolset
	selected map[string]bool
	files    []ctxmmu.RepoMapFile
	bytes    int
}

// prepareRepoMap runs only after beginTurn has admitted the first turn. It
// reuses dispatchToolCalls, including native decisions, read observations, audit
// and tool budget debits; it never calls Toolset.Read or os.ReadFile directly.
func (r *armRunner) prepareRepoMap(ctx context.Context, turn int) error {
	if r.cfg == nil || r.cfg.repoMap == nil || r.repoMapPrepared {
		return nil
	}
	r.repoMapPrepared = true
	if !r.fak || r.kernel == nil || len(r.cfg.conversation) != 0 ||
		!r.cfg.sessionCheckpointCreatedAt.IsZero() || r.cfg.sessionCheckpointInitialTurn != 0 || !r.metrics.ResumedPendingTurn.IsZero() {
		return errors.New("REPOMAP_FRESH_NATIVE_ARM_REQUIRED")
	}
	tools := armedCodeTools.Load()
	if tools == nil {
		return errors.New("REPOMAP_CODE_TOOLS_REQUIRED")
	}
	// The registered engine bounds a single read BEFORE returning. Do not
	// silently accept an externally enlarged native read limit. Serial dispatch
	// and fail-fast result admission bound an oversized attempt to one 1MiB
	// result in addition to at most 512KiB of already accepted file bodies.
	if n := tools.Limits().MaxReadBytes; n <= 0 || n > 1<<20 {
		return errors.New("REPOMAP_NATIVE_READ_BOUND")
	}
	declared := false
	for _, d := range r.tools {
		if d.Function.Name == codetools.ToolRead {
			declared = true
			break
		}
	}
	if !declared {
		return errors.New("REPOMAP_READ_NOT_DECLARED")
	}
	preloadCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	ctx = preloadCtx // cooperative cancellation; not a hard filesystem I/O deadline
	req := r.cfg.repoMap
	state := &repoMapPreloadState{tools: tools, selected: make(map[string]bool, len(req.Files))}
	r.repoMapPreload = state
	defer func() { r.repoMapPreload = nil }()
	for _, p := range req.Files {
		state.selected[p] = true
	}
	for i, p := range req.Files {
		if err := ctx.Err(); err != nil {
			return err
		}
		if armedCodeTools.Load() != tools {
			return errors.New("REPOMAP_ARM_CHANGED")
		}
		b, _ := json.Marshal(codetools.ReadArgs{FilePath: p})
		var call ToolCall
		call.ID = fmt.Sprintf("fak-repomap-preload-%d", i+1)
		call.Type = "function"
		call.Function.Name = codetools.ToolRead
		call.Function.Arguments = string(b)
		// One complete assistant/result pair at a time: a refusal leaves no
		// dangling synthetic calls and cannot dispatch later selected files.
		asst := Message{Role: RoleAssistant, ToolCalls: []ToolCall{call}}
		r.messages = append(r.messages, asst)
		stopped, err := r.dispatchToolCalls(ctx, turn, asst)
		if err != nil {
			return err
		}
		if stopped {
			return errors.New("REPOMAP_PRELOAD_INCOMPLETE")
		}
	}
	if len(state.files) != len(req.Files) {
		return errors.New("REPOMAP_PRELOAD_INCOMPLETE")
	}
	mapped, err := ctxmmu.BuildRepoMap(ctx, state.files, ctxmmu.RepoMapOptions{
		Tokens: req.Tokens, Mentioned: repoMapMentions(r.task),
	})
	if err != nil {
		return err
	}
	// Deliberately ordinary user-role DATA, not a system prompt, instruction,
	// permission, persistent memory note or a hidden context-planner exemption.
	// The existing prompt assembler/compactor may retain or evict it normally.
	r.messages = append(r.messages, Message{Role: RoleUser, Content: mapped.Text})
	return nil
}

func repoMapMentions(task string) []string {
	if len(task) > 8192 {
		task = task[:8192]
		for !utf8.ValidString(task) {
			task = task[:len(task)-1]
		}
	}
	parts := strings.FieldsFunc(task, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_'
	})
	out := make([]string, 0, 64)
	seen := make(map[string]bool)
	for _, p := range parts {
		if len(p) > 128 || seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
		if len(out) == 64 {
			break
		}
	}
	return out
}

// compactRepoMapRead compacts ONLY the explicit preload's transcript result.
// The original body has already crossed native admission and remains the input
// to audit/trace and code-read version observations; no ordinary Read is changed.
func (r *armRunner) compactRepoMapRead(tool, rawArgs, body string, result *abi.Result, failed, denied bool) (string, error) {
	state := r.repoMapPreload
	if state == nil {
		return body, nil
	}
	code := "REPOMAP_READ_REFUSED"
	type receipt struct {
		Schema  string `json:"schema"`
		Status  string `json:"status"`
		Path    string `json:"file_path,omitempty"`
		Version string `json:"version,omitempty"`
		SHA256  string `json:"sha256,omitempty"`
		Bytes   int    `json:"bytes,omitempty"`
		Reason  string `json:"reason,omitempty"`
	}
	refused := func() (string, error) {
		b, _ := json.Marshal(receipt{Schema: "fak.repomap.read/1", Status: "error", Reason: code})
		return string(b), errors.New(code)
	}
	if failed || denied || tool != codetools.ToolRead || result == nil ||
		result.Status != abi.StatusOK || result.Meta["engine"] != codetools.EngineRead ||
		result.Meta["admit"] != "" {
		return refused()
	}
	if armedCodeTools.Load() != state.tools {
		code = "REPOMAP_ARM_CHANGED"
		return refused()
	}
	var args codetools.ReadArgs
	dec := json.NewDecoder(strings.NewReader(rawArgs))
	dec.DisallowUnknownFields()
	if dec.Decode(&args) != nil || args.Validate() != nil || args.Offset != 0 || args.Limit != 0 ||
		!state.selected[args.FilePath] {
		code = "REPOMAP_READ_IDENTITY"
		return refused()
	}
	var extra any
	if dec.Decode(&extra) != io.EOF {
		code = "REPOMAP_READ_IDENTITY"
		return refused()
	}
	// Native JSON escaping is at most six bytes per source byte. Refuse before
	// decoding if this receipt cannot fit a selected complete 64KiB file.
	if len(body) > 6*ctxmmu.RepoMapMaxFileBytes+4096 {
		code = "REPOMAP_INPUT_BOUND"
		return refused()
	}
	var payload struct {
		Path      string `json:"file_path"`
		Content   string `json:"content"`
		Version   string `json:"version"`
		Bytes     int    `json:"bytes"`
		Truncated bool   `json:"truncated"`
	}
	dec = json.NewDecoder(bytes.NewBufferString(body))
	dec.DisallowUnknownFields()
	if dec.Decode(&payload) != nil || dec.Decode(&extra) != io.EOF ||
		payload.Path != args.FilePath || payload.Truncated || !isFullFileVersion(payload.Version) ||
		payload.Bytes != len(payload.Content) || !utf8.ValidString(payload.Content) {
		code = "REPOMAP_READ_INCOMPLETE"
		return refused()
	}
	if len(payload.Content) > ctxmmu.RepoMapMaxFileBytes ||
		state.bytes+len(payload.Content) > ctxmmu.RepoMapMaxInputBytes {
		code = "REPOMAP_INPUT_BOUND"
		return refused()
	}
	delete(state.selected, args.FilePath)
	state.bytes += len(payload.Content)
	state.files = append(state.files, ctxmmu.RepoMapFile{
		Path: payload.Path, Content: payload.Content, Version: payload.Version,
	})
	h := sha256.Sum256([]byte(payload.Content))
	b, _ := json.Marshal(receipt{Schema: "fak.repomap.read/1", Status: "ok",
		Path: payload.Path, Version: payload.Version, SHA256: hex.EncodeToString(h[:]), Bytes: payload.Bytes})
	return string(b), nil
}
