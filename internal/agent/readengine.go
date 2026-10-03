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
	"github.com/anthony-chaudhary/fak/internal/fakroot"
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
// refuses any path that escapes EVERY configured read root, so a model-supplied `file_path`
// cannot exfiltrate /etc/shadow or a path outside the project. The default root set is the
// working tree PLUS, when the canonical ladder proves one exists, the declared companion
// PUBLIC fak checkout — see RegisterReadEngine for why that widening is asymmetric and why
// every extra root is structurally proved to be the public checkout. A refused or failed
// read returns a Status=Error result (deny-as-value), never a panic.

// FakReadEngineID is the engine id `fak_read` binds on its abi.ToolCall so k.Syscall
// dispatches a cache MISS here (the vDSO fast path serves a hit before dispatch).
const FakReadEngineID = "fakread"

// readEngine performs a working-tree-confined filesystem read. roots are the directories
// reads are confined to (at least one); a path resolving outside EVERY root is refused.
// A RELATIVE path is resolved against roots[0] — the working tree — and may then land in
// any declared root, so the `../fak/README.md` spelling that motivated the widening works
// unchanged. Both halves of the confinement (lexical and symlink) test the UNION of the
// roots, so widening admits the union and nothing else. The zero engine has no roots and
// therefore refuses everything.
type readEngine struct {
	roots []string
}

// primaryRoot is the root a relative path argument resolves against.
func (e readEngine) primaryRoot() string {
	if len(e.roots) == 0 {
		return ""
	}
	return e.roots[0]
}

// lexicallyConfined reports whether abs is lexically inside at least one root. The
// per-root filepath.Rel + dot-dot test is the lexical half of the confinement; it runs
// once PER root, so widening the root set widens only the union of the admitted trees
// and never turns a "escapes every root" verdict into a pass.
func (e readEngine) lexicallyConfined(abs string) bool {
	for _, root := range e.roots {
		if rel, err := filepath.Rel(root, abs); err == nil && !escapes(rel) {
			return true
		}
	}
	return false
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
		abs = filepath.Join(e.primaryRoot(), abs)
	}
	abs = filepath.Clean(abs)
	if !e.lexicallyConfined(abs) {
		return errResult("path_escape", "confinement", "path escapes the read root")
	}
	if escaped := e.symlinkEscapes(abs); escaped {
		return errResult("path_escape", "confinement", "path escapes the read root via symlink")
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
// ("../" or "..\\"), i.e. the target escapes the base. A bare ".." is handled by escapes.
func hasDotDotPrefix(rel string) bool {
	return len(rel) >= 3 && rel[0] == '.' && rel[1] == '.' && (rel[2] == '/' || rel[2] == '\\')
}

// escapes reports whether a filepath.Rel result leaves its base.
func escapes(rel string) bool { return rel == ".." || hasDotDotPrefix(rel) }

// symlinkEscapes is the symlink half of the confinement (#11399): it resolves abs and
// every root and reports whether the resolved path lands outside ALL of them. The test
// runs once per root, so a widened root set cannot launder a symlink out of every root.
// A path that does not resolve (missing file) or a root that does not resolve is not a
// symlink escape — os.ReadFile reports the real reason, as before.
func (e readEngine) symlinkEscapes(abs string) bool {
	realPath, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return false
	}
	resolved := false
	for _, root := range e.roots {
		realRoot, rErr := filepath.EvalSymlinks(root)
		if rErr != nil {
			continue
		}
		resolved = true
		if rel, err := filepath.Rel(realRoot, realPath); err == nil && !escapes(rel) {
			return false
		}
	}
	return resolved
}

// RegisterReadEngine registers the working-tree-confined read engine under FakReadEngineID,
// confined to root (empty => the process cwd). Idempotent-friendly: re-registering replaces
// the driver. Called from Configure so `fak guard` / `fak serve` arm the fak_read miss path.
//
// An EMPTY root also admits the declared companion PUBLIC fak checkout, when the canonical
// discovery ladder (internal/fakroot) can prove one exists: a directory counts only when
// `cmd/fak/main.go` is on disk, and the private companion checkout has no cmd/fak tree — so
// this widening is asymmetric and can never admit fak-private from a public cwd. Without the
// widening, `fak serve --stdio` run from fak-private default-denied fak_read of the sibling
// public workspace it is literally joined to by go.work. A non-empty root is taken verbatim,
// with no discovery, so callers that pin a root keep exactly the confinement they had.
func RegisterReadEngine(root string) {
	defaulted := root == ""
	cwd, _ := os.Getwd()
	if defaulted {
		root = cwd
	}
	roots := []string{filepath.Clean(root)}
	if defaulted && cwd != "" {
		ladder := fakroot.Ladder{Cwd: cwd}
		if companion := ladder.Discover(); companion != "" {
			roots = appendUniqueRoots(roots, companion)
		}
	}
	abi.RegisterEngine(FakReadEngineID, readEngine{roots: roots})
}

// RegisterReadEngineRoots registers the read engine confined to EXACTLY the given roots —
// the additive-roots entry point. The first root resolves relative path arguments; every
// root admits absolute ones. No cwd defaulting and NO discovery run, so a caller can widen
// the set deliberately (add more roots) or narrow it back to one (opt out of the
// companion-root widening RegisterReadEngine performs).
func RegisterReadEngineRoots(roots ...string) {
	set := make([]string, 0, len(roots))
	for _, r := range roots {
		set = appendUniqueRoots(set, r)
	}
	abi.RegisterEngine(FakReadEngineID, readEngine{roots: set})
}

// appendUniqueRoots appends dir (cleaned) unless it is empty or already present, keeping
// registration order — roots[0] stays the primary working tree.
func appendUniqueRoots(roots []string, dir string) []string {
	if strings.TrimSpace(dir) == "" {
		return roots
	}
	dir = filepath.Clean(dir)
	for _, existing := range roots {
		if existing == dir {
			return roots
		}
	}
	return append(roots, dir)
}
