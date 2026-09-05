package agent

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// readengine.go — the real filesystem-read engine that backs the `fak_read` MCP tool
// (#795 vToolcall, the live-harness serve seam). The demo `localtools` engine implements
// only the travel-domain toolset (no real file I/O), so a Read routed through the kernel
// had no engine to dispatch to on a cache miss. This engine is that miss path: a working-
// tree-confined os.ReadFile.
//
// Why it lives behind the kernel (not as a raw read): routing the read through
// k.Syscall means the vDSO fast path runs FIRST — on a cache hit the file is served from
// the tier-2 cache with NO disk read at all, and the #795 per-path invalidator
// (internal/vdso/pathscope.go: files:<path>) guarantees that hit is fresh (a Write/Edit to
// the path bumped its epoch). So `fak_read` is the live-harness expression of fak's
// closed-loop cache: the agent calls fak_read instead of the built-in Read, the kernel
// serves a fresh cached result without touching disk, and only a genuine miss reaches this
// engine. No Claude Code change is required — the model opts in via `claude mcp add fak`.
//
// Soundness boundary: this engine is READ-ONLY and path-confined. It never writes, and it
// refuses any path that escapes the configured read root (default: the working tree), so a
// model-supplied `file_path` cannot exfiltrate /etc/shadow or a path outside the project.
// A refused or failed read returns a Status=Error result (deny-as-value), never a panic.

// FakReadEngineID is the engine id `fak_read` binds on its abi.ToolCall so k.Syscall
// dispatches a cache MISS here (the vDSO fast path serves a hit before dispatch).
const FakReadEngineID = "fakread"

// readEngine performs a working-tree-confined filesystem read. root is the directory reads
// are confined to; a path resolving outside it is refused. An empty root means "the process
// cwd", resolved once at registration.
type readEngine struct {
	root string
}

// Caps reports no optional capabilities — a plain read engine advertises none.
func (readEngine) Caps() []abi.Capability { return nil }

// WeightBearing declares that fak_read is a deterministic classical tool engine.
func (readEngine) WeightBearing() bool { return false }

// Complete reads the file named by the call's `file_path` (or `path`) argument and returns
// its bytes, confined to the engine root. It is the Read tool's miss path; on a hit the
// vDSO served the result and this never runs.
func (e readEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	body, m := decodeCallArgs(ctx, c.Args)
	pathArg := ""
	for _, k := range []string{"file_path", "path", "filename", "filepath"} {
		if v, ok := m[k]; ok {
			if s, ok := v.(string); ok && s != "" {
				pathArg = s
				break
			}
		}
	}
	offset := parseOptInt(m["offset"])
	limit := parseOptInt(m["limit"])
	lineNumbers := parseOptBool(m["line_numbers"])
	out, isErr := e.readWithOptions(pathArg, offset, limit, lineNumbers)
	return engineResult(ctx, c, body, out, isErr, FakReadEngineID), nil
}

func parseOptInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case int64:
		return int(n)
	case string:
		if i, err := strconv.Atoi(strings.TrimSpace(n)); err == nil {
			return i
		}
	}
	return 0
}

func parseOptBool(v any) bool {
	switch b := v.(type) {
	case bool:
		return b
	case string:
		return strings.EqualFold(strings.TrimSpace(b), "true") || b == "1"
	}
	return false
}

// read resolves pathArg against the engine root, refuses an escape, and returns the file
// bytes (or a JSON error object on any failure). The result is always JSON so the MCP wire
// shape is stable; a successful read returns {"file_path":..., "content":...}.
func (e readEngine) read(pathArg string) (result []byte, isError bool) {
	return e.readWithOptions(pathArg, 0, 0, false)
}

func (e readEngine) readWithOptions(pathArg string, offset, limit int, lineNumbers bool) (result []byte, isError bool) {
	errResult := func(code, source, message string) ([]byte, bool) {
		b, _ := json.Marshal(map[string]any{
			"error":        "fak_read: " + message,
			"error_code":   code,
			"error_source": source,
		})
		return b, true
	}
	if pathArg == "" {
		return errResult("missing_path", "input", "missing required field: file_path")
	}
	abs := pathArg
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(e.root, abs)
	}
	abs = filepath.Clean(abs)
	if rel, err := filepath.Rel(e.root, abs); err != nil || rel == ".." || hasDotDotPrefix(rel) {
		return errResult("path_escape", "confinement", "path escapes the read root")
	}
	// Symlink confinement: resolve symlinks on abs and verify the resolved path remains inside realRoot (#11399).
	if realPath, err := filepath.EvalSymlinks(abs); err == nil {
		realRoot, rErr := filepath.EvalSymlinks(e.root)
		if rErr == nil {
			if rel, err := filepath.Rel(realRoot, realPath); err != nil || rel == ".." || hasDotDotPrefix(rel) {
				return errResult("path_escape", "confinement", "path escapes the read root via symlink")
			}
		}
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		if info, statErr := os.Stat(abs); statErr == nil && info.IsDir() {
			return errResult("is_directory", "filesystem", "path is a directory")
		}
		switch {
		case errors.Is(err, os.ErrNotExist):
			return errResult("not_found", "filesystem", "file not found")
		case errors.Is(err, os.ErrPermission):
			return errResult("permission_denied", "filesystem", "permission denied")
		default:
			return errResult("io_error", "filesystem", "filesystem read failed")
		}
	}
	body := map[string]any{"file_path": pathArg}
	if !utf8.Valid(data) {
		body["encoding"] = "base64"
		body["content_base64"] = base64.StdEncoding.EncodeToString(data)
		body["content"] = ""
	} else {
		content := string(data)
		if offset > 0 || limit > 0 || lineNumbers {
			lines := strings.Split(content, "\n")
			start := 0
			if offset > 0 {
				start = offset - 1
			}
			if start > len(lines) {
				start = len(lines)
			}
			end := len(lines)
			truncated := false
			if limit > 0 && start+limit < end {
				end = start + limit
				truncated = true
			}
			selected := lines[start:end]
			if lineNumbers {
				numbered := make([]string, len(selected))
				for i, l := range selected {
					numbered[i] = fmt.Sprintf("%d: %s", start+i+1, l)
				}
				selected = numbered
			}
			body["content"] = strings.Join(selected, "\n")
			if offset > 0 {
				body["offset"] = offset
			}
			if limit > 0 {
				body["limit"] = limit
			}
			if truncated {
				body["truncated"] = true
			}
			body["total_lines"] = len(lines)
		} else {
			body["content"] = content
		}
	}
	b, _ := json.Marshal(body)
	return b, false
}

// hasDotDotPrefix reports whether a filepath.Rel result begins with a parent-dir segment
// ("../" or "..\\"), i.e. the target escapes the base. A bare ".." is handled by the caller.
func hasDotDotPrefix(rel string) bool {
	return len(rel) >= 3 && rel[0] == '.' && rel[1] == '.' && (rel[2] == '/' || rel[2] == '\\')
}

// RegisterReadEngine registers the working-tree-confined read engine under FakReadEngineID,
// confined to root (empty => the process cwd). Idempotent-friendly: re-registering replaces
// the driver. Called from Configure so `fak guard` / `fak serve` arm the fak_read miss path.
func RegisterReadEngine(root string) {
	if root == "" {
		if cwd, err := os.Getwd(); err == nil {
			root = cwd
		}
	}
	root = filepath.Clean(root)
	abi.RegisterEngine(FakReadEngineID, readEngine{root: root})
}
