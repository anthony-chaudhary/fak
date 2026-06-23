package codelint

// codelint is the kernel's language-server-pack surface: it lints code the AGENT
// produces (a write_file / edit tool call's content) the way boundarylint lints the
// repo's OWN trusted Go source. A Pack is one language's source of diagnostics,
// keyed by the file extensions it owns; DefaultRegistry wires the built-in set.
//
// Two deliberate differences from boundarylint, because the input is UNTRUSTED
// agent output rather than repo source:
//
//   - No in-content suppression. boundarylint honors a //boundarylint:ignore
//     comment because a human author owns that source. codelint does NOT honor any
//     ignore comment: the lint is a kernel judgment over code the model wrote, and
//     the model must not be able to switch the gate off by writing a magic comment
//     (the same "the model can't talk past the kernel" rule the adjudicator keeps).
//   - It may shell out. A real language server is an external process, so the packs
//     run OFF the hot path (architest's TestHotPathHasNoExec forbids os/exec on the
//     decide path; codelint is a foundation leaf, never on it) and degrade to "no
//     opinion" when a checker binary is absent or wedged past a timeout — linting is
//     a quality signal, never a security gate, so it fails OPEN.
//
// The simplest realization of a pack shells the language's own one-shot checker
// (python -m compile, nvcc); the Go and JSON packs are in-process because the
// stdlib already parses those. A long-lived LSP session (gopls, pyright,
// rust-analyzer) is a drop-in future Pack with the same Check shape.

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Severity ranks a finding. Error means the code does not parse/compile — a hard
// defect any toolchain would reject; Warning is a softer signal. The kernel
// integrations act only on Error findings, so the lint never nags about style and
// never false-blocks a write over a warning.
type Severity int

const (
	Warning Severity = iota
	Error
)

func (s Severity) String() string {
	if s == Error {
		return "error"
	}
	return "warning"
}

// MarshalJSON renders a severity as its name ("error"/"warning") so a findings
// dump reads for an operator instead of leaking the enum's integer.
func (s Severity) MarshalJSON() ([]byte, error) {
	return []byte(`"` + s.String() + `"`), nil
}

// Finding is one diagnostic at one source location. It mirrors the shape of
// boundarylint.Finding (Code/File/Line/Detail) so every lint surface the kernel
// emits reads the same, and adds Severity + Col because these come from external
// language servers rather than a single in-process Go AST.
type Finding struct {
	Pack     string   `json:"pack"`     // the language pack that produced it, e.g. "go"
	Code     string   `json:"code"`     // closed code, e.g. "GO_PARSE", "PYTHON_SYNTAX"
	File     string   `json:"file"`     // the path as handed to the linter
	Line     int      `json:"line"`     // 1-based; 0 when the checker did not report one
	Col      int      `json:"col"`      // 1-based; 0 when unknown
	Severity Severity `json:"severity"` // error | warning
	Detail   string   `json:"detail"`   // the human message
}

func (f Finding) String() string {
	loc := f.File
	if f.Line > 0 {
		loc = fmt.Sprintf("%s:%d", f.File, f.Line)
		if f.Col > 0 {
			loc = fmt.Sprintf("%s:%d", loc, f.Col)
		}
	}
	return fmt.Sprintf("%s: %s: %s (%s/%s)", loc, f.Severity, f.Detail, f.Pack, f.Code)
}

// Pack is one language-server pack: a self-contained source of diagnostics for one
// language, keyed by the file extensions it owns.
//
// Check inspects the file already on disk at path and returns its findings, or
// (nil, nil) when the pack has no opinion — the checker binary is absent, or the
// code is clean. Check MUST be hermetic in its findings (same file, same findings)
// and MUST NOT block indefinitely (an external pack is bounded by DefaultTimeout).
type Pack struct {
	Lang  string
	Exts  []string
	Check func(ctx context.Context, path string) ([]Finding, error)
}

// DefaultTimeout bounds one external checker run. Linting agent output must never
// hang the kernel, so a slow or wedged language server is treated as "no opinion".
const DefaultTimeout = 10 * time.Second

// Registry maps a file extension to the pack that owns it. Build one with
// DefaultRegistry (the built-in set) or NewRegistry (a custom set).
type Registry struct {
	byExt map[string]Pack
}

// NewRegistry indexes the given packs by (lower-cased) extension. A later pack that
// claims an extension already taken wins, so a caller can override a default pack by
// appending its own after DefaultPacks().
func NewRegistry(packs ...Pack) *Registry {
	r := &Registry{byExt: make(map[string]Pack, len(packs)*2)}
	for _, p := range packs {
		for _, e := range p.Exts {
			r.byExt[strings.ToLower(e)] = p
		}
	}
	return r
}

// DefaultRegistry is the kernel's built-in pack set — what `fak codelint` and the
// agent-code path use.
func DefaultRegistry() *Registry { return NewRegistry(DefaultPacks()...) }

// PackFor returns the pack that owns path's extension, if any.
func (r *Registry) PackFor(path string) (Pack, bool) {
	p, ok := r.byExt[strings.ToLower(filepath.Ext(path))]
	return p, ok
}

// Langs returns the distinct registered language ids, sorted — the menu of what the
// kernel can lint in this build.
func (r *Registry) Langs() []string {
	seen := map[string]bool{}
	var out []string
	for _, p := range r.byExt {
		if !seen[p.Lang] {
			seen[p.Lang] = true
			out = append(out, p.Lang)
		}
	}
	sort.Strings(out)
	return out
}

// LintFile lints a file already on disk. An unknown extension is not an error — the
// file is simply unlinted, and (nil, nil) is returned. A genuine I/O failure (the
// file vanished) is returned as the error.
func (r *Registry) LintFile(ctx context.Context, path string) ([]Finding, error) {
	p, ok := r.PackFor(path)
	if !ok {
		return nil, nil
	}
	return p.Check(ctx, path)
}

// LintBytes lints in-memory content that is not (or not yet) on disk by writing it
// to a temp file carrying name's extension, linting that, and re-labeling each
// finding's File back to name. This is how the kernel lints a proposed write before
// — or just after — it lands, attributing diagnostics to the agent's intended path.
func (r *Registry) LintBytes(ctx context.Context, name string, content []byte) ([]Finding, error) {
	if _, ok := r.PackFor(name); !ok {
		return nil, nil
	}
	dir, err := os.MkdirTemp("", "codelint-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	tmp := filepath.Join(dir, "input"+strings.ToLower(filepath.Ext(name)))
	if err := os.WriteFile(tmp, content, 0o600); err != nil {
		return nil, err
	}
	fs, err := r.LintFile(ctx, tmp)
	for i := range fs {
		fs[i].File = name
	}
	return fs, err
}

// HasError reports whether any finding is Error severity — the signal the kernel
// integrations (the `fak codelint` exit code, the fleet write path) act on.
func HasError(fs []Finding) bool {
	for _, f := range fs {
		if f.Severity == Error {
			return true
		}
	}
	return false
}

// Summary renders findings as a compact, model-facing block: one line per finding,
// errors first. Used to feed a coding agent its own diagnostics so it self-corrects.
func Summary(fs []Finding) string {
	if len(fs) == 0 {
		return ""
	}
	ordered := make([]Finding, len(fs))
	copy(ordered, fs)
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].Severity != ordered[j].Severity {
			return ordered[i].Severity > ordered[j].Severity // Error before Warning
		}
		return ordered[i].Line < ordered[j].Line
	})
	var b strings.Builder
	for _, f := range ordered {
		b.WriteString(f.String())
		b.WriteByte('\n')
	}
	return strings.TrimRight(b.String(), "\n")
}
