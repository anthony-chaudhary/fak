package codetools

import (
	"context"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// engines.go — the filesystem read engine.
//
// Each engine re-validates. The rung already decoded, validated, canonicalized, and
// policy-checked the call, so on the loop's path these checks pass every time. They stay
// because an engine registered in the abi registry is reachable by ANY kernel — including
// one whose chain does not carry this package's rung — and an engine that trusted its
// caller to have confined the path would be a filesystem primitive with no confinement at
// all. The duplicated cost is a decode of a small argument object; the duplicated
// GUARANTEE is that confinement is a property of the engine, not of its wiring.
//
// Every engine returns a deny-as-VALUE on failure: Status=Error with a typed
// {"error":{"code",...}} payload, never a transport error, so the loop hands the model a
// tool_result it can act on instead of a dropped turn.

// readEngine serves Read: a bounded, confined file read with an optional line window.
type readEngine struct{ t *Toolset }

// Caps reports no optional capabilities.
func (readEngine) Caps() []abi.Capability { return nil }

// WeightBearing declares Read a deterministic classical tool engine, not a model.
func (readEngine) WeightBearing() bool { return false }

// Complete performs the read. This is the vDSO MISS path: on a hit the kernel served the
// bytes from the tier-2 cache and this never ran, and the per-path invalidator guarantees
// that hit was not stranded by a Write/Edit to the same path.
func (e readEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	in := bytesOf(ctx, c.Args)
	out, isErr := e.t.read(ctx, in)
	return result(ctx, c, in, out, isErr, EngineRead), nil
}

// applyPatchEngine serves apply_patch: atomic unified diff application with optimistic CAS.
type applyPatchEngine struct{ t *Toolset }

func (applyPatchEngine) Caps() []abi.Capability { return nil }
func (applyPatchEngine) WeightBearing() bool    { return false }
func (e applyPatchEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	in := bytesOf(ctx, c.Args)
	out, bad := e.t.applyPatch(ctx, in)
	return result(ctx, c, in, out, bad, EngineApplyPatch), nil
}

// read decodes, confines, and executes a Read, returning the JSON payload and whether it
// is an error.
func (t *Toolset) read(ctx context.Context, body []byte) ([]byte, bool) {
	RecordSubprocessAvoided()
	var a ReadArgs
	if r := decodeArgs(body, &a); r != nil {
		return r.JSON(), true
	}
	if r := a.Validate(); r != nil {
		return r.JSON(), true
	}
	if r := canceled(ctx); r != nil {
		return r.JSON(), true
	}
	res, r := t.resolve(a.FilePath)
	if r != nil {
		return r.JSON(), true
	}
	obs, r := observeFile(ctx, res.Abs, t.limits.MaxReadBytes)
	if r != nil {
		if r.Code == CodeNotFound || r.Code == CodeIsDir {
			r.Detail = res.Rel
		}
		return r.JSON(), true
	}
	content := string(obs.Content)
	truncated := obs.Truncated
	if a.Offset > 0 || a.Limit > 0 {
		content, truncated = window(content, a.Offset, a.Limit, truncated)
	}
	return okJSON(map[string]any{
		"file_path": res.Rel,
		"content":   content,
		"bytes":     len(content),
		"truncated": truncated,
		"version":   obs.Version,
	}), false
}

// window applies a 1-based line offset and a line limit to content. An offset past the
// end yields empty content rather than an error: "there is nothing at line 900" is a
// true answer to the question asked, not a fault.
func window(content string, offset, limit int, truncated bool) (string, bool) {
	lines := strings.Split(content, "\n")
	start := 0
	if offset > 0 {
		start = offset - 1
	}
	if start >= len(lines) {
		return "", truncated
	}
	end := len(lines)
	if limit > 0 && start+limit < end {
		end = start + limit
		truncated = true
	}
	return strings.Join(lines[start:end], "\n"), truncated
}

// canceled turns a dead context into the toolset's CANCELED refusal. Every engine calls
// it on entry, so a terminated session's queued tool call cannot still touch the disk
// after the loop that proposed it has stopped.
func canceled(ctx context.Context) *Refusal {
	if err := ctx.Err(); err != nil {
		return refuse(CodeCanceled, err.Error())
	}
	return nil
}
