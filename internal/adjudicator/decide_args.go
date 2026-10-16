package adjudicator

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func decodeArgs(ctx context.Context, c *abi.ToolCall) map[string]any {
	b := refBytes(ctx, c.Args)
	if len(b) == 0 {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil
	}
	return m
}

func refBytes(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	res := abi.ActiveResolver()
	if res == nil {
		return nil
	}
	b, err := res.Resolve(ctx, r)
	if err != nil {
		return nil
	}
	return b
}

func putJSON(ctx context.Context, m map[string]any) (abi.Ref, bool) {
	b, err := json.Marshal(m)
	if err != nil {
		return abi.Ref{}, false
	}
	res := abi.ActiveResolver()
	if res == nil {
		return abi.Ref{}, false
	}
	ref, err := res.Put(ctx, b)
	if err != nil {
		return abi.Ref{}, false
	}
	return ref, true
}

func cloneArgs(args map[string]any) map[string]any {
	out := make(map[string]any, len(args))
	for k, v := range args {
		out[k] = v
	}
	return out
}
func selectArgs(preferred, fallback map[string]any) map[string]any {
	if preferred != nil {
		return preferred
	}
	return fallback
}

func redact(args map[string]any, fields []string) (map[string]any, bool) {
	changed := false
	out := make(map[string]any, len(args))
	for k, v := range args {
		out[k] = v
	}
	for _, f := range fields {
		if _, ok := out[f]; ok {
			out[f] = "[REDACTED]"
			changed = true
		}
	}
	return out, changed
}

// matchGlob returns the first glob fragment contained in path, or "".
//
// A fragment that names a DOTFILE (leading '.', e.g. ".env", ".aws/") must
// additionally occur at a token boundary: the byte before the match, if any,
// may not be a word byte. Bare substring matching turned the most ordinary
// env READ there is — `python -c "… os.environ …"` — into a SELF_MODIFY deny,
// because the inline-eval floor hands the whole opaque segment to this matcher
// as the write target and "os.environ" contains ".env" (witnessed live
// 2026-07-01: `python -c "import os; print(len(os.environ))"` denied with
// witness ".env"). A genuine dotfile target always follows a separator —
// `> .env`, `src/.env`, `--env-file=.env`, `~/.aws/credentials`, a leading
// `.env` — so every real deny is kept; only the mid-identifier '.' of a dotted
// name ("os.environ", "config.env", "repo.git/") stops counting as the dotfile.
func matchGlob(path string, globs []string) string {
	if path == "" {
		return ""
	}
	// Tool arguments use the client's path dialect, which may differ from the
	// gateway host. Normalize both separators explicitly so a Windows-style
	// filePath cannot evade a protected tree fragment on a Unix server.
	path = strings.ReplaceAll(path, `\`, "/")
	for _, g := range globs {
		if g == "" {
			continue
		}
		match := strings.ReplaceAll(g, `\`, "/")
		if match[0] != '.' {
			if strings.Contains(path, match) {
				return g
			}
			if strings.HasSuffix(match, "/") && treeDirFragmentIn(path, strings.TrimSuffix(match, "/")) {
				return g
			}
			continue
		}
		if dotfileFragmentIn(path, match) {
			return g
		}
	}
	return ""
}

// dotfileFragmentIn reports whether the dotfile fragment g occurs in path at a
// token boundary — the start of the string, or preceded by a non-word byte
// (a path separator, whitespace, a quote, '=', shell punctuation, …).
func dotfileFragmentIn(path, g string) bool {
	return fragmentAt(path, g, func(at, _ int) bool {
		return at == 0 || !wordByte(path[at-1])
	})
}

// treeDirFragmentIn reports whether a slash-bearing guarded-tree fragment occurs
// as the tree itself, not only as a parent prefix ending in '/'. This preserves
// the "internal/abi/" glob for descendants while also catching destructive calls
// that name the directory exactly, such as rmtree("internal/abi").
func treeDirFragmentIn(path, g string) bool {
	if g == "" || !strings.Contains(g, "/") {
		return false
	}
	return fragmentAt(path, g, func(at, end int) bool {
		leftOK := at == 0 || !wordByte(path[at-1])
		rightOK := end == len(path) || !wordByte(path[end])
		return leftOK && rightOK
	})
}

func fragmentAt(path, fragment string, accept func(at, end int) bool) bool {
	for from := 0; ; {
		i := strings.Index(path[from:], fragment)
		if i < 0 {
			return false
		}
		at := from + i
		if accept(at, at+len(fragment)) {
			return true
		}
		from = at + 1
	}
}

// wordByte reports whether b continues an identifier/word — a byte that, when
// it precedes a '.', makes that '.' the dot OF a dotted name rather than the
// start of a dotfile. Everything else is a boundary.
func wordByte(b byte) bool {
	return b == '_' || b == '-' ||
		(b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')
}
