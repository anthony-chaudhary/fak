package agentsindex

// The effective-instructions resolver is deliberately separate from Load. Load is the
// byte-exact, root-only pager introduced by #3535; ResolveEffective is the target-scoped
// hierarchy contract introduced by #9391. Keeping the two entry points separate makes the
// legacy command's output and failure behaviour an explicit compatibility boundary.

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const DefaultEffectiveMaxBytes int64 = 128 << 10

// ResolutionStatus is the fail-closed capability outcome of one hierarchy walk.
type ResolutionStatus string

const (
	StatusComplete    ResolutionStatus = "complete"
	StatusPartial     ResolutionStatus = "partial"
	StatusUnknown     ResolutionStatus = "unknown"
	StatusTruncated   ResolutionStatus = "truncated"
	StatusUntrusted   ResolutionStatus = "untrusted"
	StatusOutsideRoot ResolutionStatus = "outside_root"
)

// ResolveOptions supplies the only policy inputs: fallbacks, one shared selected-source
// budget, and the caller's explicit trust decision.
type ResolveOptions struct {
	Fallbacks []string
	MaxBytes  int64
	Trusted   bool
}

// SourceSpan locates exact source bytes inside a complete Instructions value.
type SourceSpan struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

// EffectiveSource records every existing candidate in a directory. Lower-precedence
// files are retained as skipped diagnostics, so callers can explain why a file did not
// contribute without re-walking the tree.
type EffectiveSource struct {
	Path       string      `json:"path"`
	Directory  string      `json:"directory"`
	Name       string      `json:"name"`
	Precedence int         `json:"precedence"`
	Selected   bool        `json:"selected"`
	Included   bool        `json:"included"`
	Reason     string      `json:"reason"`
	Bytes      int         `json:"bytes,omitempty"`
	SHA256     string      `json:"sha256,omitempty"`
	Content    string      `json:"content,omitempty"`
	Trusted    bool        `json:"trusted"`
	Span       *SourceSpan `json:"span,omitempty"`
}

// ResolutionDiagnostic preserves a typed reason without making callers parse errors.
type ResolutionDiagnostic struct {
	Path   string `json:"path,omitempty"`
	Reason string `json:"reason"`
	Detail string `json:"detail,omitempty"`
}

// EffectiveResult is safe to inject only when Status is complete. Instructions is
// intentionally empty for every other status; Sources retain bounded diagnostics.
type EffectiveResult struct {
	Status          ResolutionStatus       `json:"status"`
	Target          string                 `json:"target"`
	TargetKind      string                 `json:"target_kind,omitempty"`
	Trusted         bool                   `json:"trusted"`
	MaxBytes        int64                  `json:"max_bytes"`
	Bytes           int                    `json:"bytes"`
	EffectiveSHA256 string                 `json:"effective_sha256,omitempty"`
	Instructions    string                 `json:"instructions,omitempty"`
	Sources         []EffectiveSource      `json:"sources"`
	Diagnostics     []ResolutionDiagnostic `json:"diagnostics,omitempty"`
}

// ResolveEffective resolves one instruction source per directory, root through the
// target directory inclusive. Relative targets are rooted at root, never at the process
// cwd. All paths in the result use root-relative forward slashes.
func ResolveEffective(root, target string, opts ResolveOptions) EffectiveResult {
	if opts.MaxBytes == 0 {
		opts.MaxBytes = DefaultEffectiveMaxBytes
	}
	r := EffectiveResult{Status: StatusUnknown, Trusted: opts.Trusted, MaxBytes: opts.MaxBytes, Sources: []EffectiveSource{}}
	if opts.MaxBytes < 0 {
		r.Diagnostics = append(r.Diagnostics, ResolutionDiagnostic{Reason: "invalid_max_bytes", Detail: "max bytes must be non-negative"})
		return r
	}

	rootReal, dir, ok := resolveEffectiveTarget(root, target, &r)
	if !ok {
		return r
	}
	names := effectiveSourceNames(opts.Fallbacks, &r)
	dirs, err := hierarchy(rootReal, dir)
	if err != nil {
		r.Diagnostics = append(r.Diagnostics, ResolutionDiagnostic{Reason: "hierarchy_failed", Detail: err.Error()})
		return r
	}
	state := resolveEffectiveSources(rootReal, dirs, names, opts, &r)
	finalizeEffectiveResult(&r, state)
	return r
}

func resolveEffectiveTarget(root, target string, r *EffectiveResult) (rootReal, dir string, ok bool) {
	rootAbs, err := filepath.Abs(root)
	if err != nil {
		r.Diagnostics = append(r.Diagnostics, ResolutionDiagnostic{Reason: "root_absolute_failed", Detail: err.Error()})
		return "", "", false
	}
	rootReal, err = filepath.EvalSymlinks(filepath.Clean(rootAbs))
	if err != nil {
		r.Diagnostics = append(r.Diagnostics, ResolutionDiagnostic{Reason: "root_canonicalization_failed", Detail: err.Error()})
		return "", "", false
	}
	rootReal, _ = filepath.Abs(rootReal)
	if target == "" {
		target = "."
	}
	targetPath := target
	if !filepath.IsAbs(targetPath) {
		targetPath = filepath.Join(rootReal, targetPath)
	}
	targetReal, err := filepath.EvalSymlinks(filepath.Clean(targetPath))
	if err != nil {
		r.Target = slashClean(target)
		r.Diagnostics = append(r.Diagnostics, ResolutionDiagnostic{Path: r.Target, Reason: "target_canonicalization_failed", Detail: err.Error()})
		return "", "", false
	}
	targetReal, _ = filepath.Abs(targetReal)
	relTarget, inside := relativeInside(rootReal, targetReal)
	if !inside {
		r.Status = StatusOutsideRoot
		r.Target = slashClean(target)
		r.Diagnostics = append(r.Diagnostics, ResolutionDiagnostic{Path: r.Target, Reason: "target_outside_root"})
		return "", "", false
	}
	info, err := os.Stat(targetReal)
	if err != nil {
		r.Target = filepath.ToSlash(relTarget)
		r.Diagnostics = append(r.Diagnostics, ResolutionDiagnostic{Path: r.Target, Reason: "target_stat_failed", Detail: err.Error()})
		return "", "", false
	}
	dir = targetReal
	if info.IsDir() {
		r.TargetKind = "directory"
	} else if info.Mode().IsRegular() {
		r.TargetKind = "file"
		dir = filepath.Dir(targetReal)
	} else {
		r.Target = filepath.ToSlash(relTarget)
		r.Diagnostics = append(r.Diagnostics, ResolutionDiagnostic{Path: r.Target, Reason: "unsupported_target_kind"})
		return "", "", false
	}
	r.Target = filepath.ToSlash(relTarget)
	if r.Target == "" {
		r.Target = "."
	}
	return rootReal, dir, true
}

func effectiveSourceNames(fallbacks []string, r *EffectiveResult) []string {
	names := []string{"AGENTS.override.md", FileName}
	seen := map[string]bool{"AGENTS.override.md": true, FileName: true}
	for _, fallback := range fallbacks {
		if filepath.Base(fallback) != fallback || fallback == "." || fallback == "" || filepath.IsAbs(fallback) {
			r.Diagnostics = append(r.Diagnostics, ResolutionDiagnostic{Path: slashClean(fallback), Reason: "invalid_fallback", Detail: "fallback must be a file name"})
			continue
		}
		if !seen[fallback] {
			seen[fallback] = true
			names = append(names, fallback)
		}
	}
	return names
}

type effectiveResolutionState struct {
	combined       []byte
	digestSources  []EffectiveSource
	selectedCount  int
	hadReadFailure bool
	truncated      bool
	escaped        bool
}

func resolveEffectiveSources(rootReal string, dirs, names []string, opts ResolveOptions, r *EffectiveResult) effectiveResolutionState {
	var state effectiveResolutionState
	var combined []byte
	var digestSources []EffectiveSource
	selectedCount := 0
	hadReadFailure := false
	truncated := false
	escaped := false
	for _, current := range dirs {
		selectedAtDir := false
		for precedence, name := range names {
			candidate := filepath.Join(current, name)
			li, statErr := os.Lstat(candidate)
			if os.IsNotExist(statErr) {
				continue
			}
			rel, _ := filepath.Rel(rootReal, candidate)
			s := EffectiveSource{Path: filepath.ToSlash(rel), Directory: filepath.ToSlash(filepath.Dir(rel)), Name: name, Precedence: precedence + 1, Trusted: opts.Trusted}
			if s.Directory == "." {
				s.Directory = "."
			}
			if statErr != nil {
				s.Reason = "stat_failed"
				r.Sources = append(r.Sources, s)
				r.Diagnostics = append(r.Diagnostics, ResolutionDiagnostic{Path: s.Path, Reason: s.Reason, Detail: statErr.Error()})
				hadReadFailure = true
				continue
			}
			if li.IsDir() {
				s.Reason = "not_regular_file"
				r.Sources = append(r.Sources, s)
				continue
			}
			candidateReal, evalErr := filepath.EvalSymlinks(candidate)
			if evalErr != nil {
				s.Reason = "canonicalization_failed"
				r.Sources = append(r.Sources, s)
				r.Diagnostics = append(r.Diagnostics, ResolutionDiagnostic{Path: s.Path, Reason: s.Reason, Detail: evalErr.Error()})
				hadReadFailure = true
				continue
			}
			if _, ok := relativeInside(rootReal, candidateReal); !ok {
				s.Selected = !selectedAtDir
				s.Reason = "outside_root"
				r.Sources = append(r.Sources, s)
				if s.Selected {
					escaped = true
					selectedAtDir = true
				}
				continue
			}
			body, readErr := os.ReadFile(candidateReal)
			if readErr != nil {
				s.Selected = !selectedAtDir
				s.Reason = "read_failed"
				r.Sources = append(r.Sources, s)
				r.Diagnostics = append(r.Diagnostics, ResolutionDiagnostic{Path: s.Path, Reason: s.Reason, Detail: readErr.Error()})
				hadReadFailure = true
				if s.Selected {
					selectedAtDir = true
				}
				continue
			}
			s.Bytes = len(body)
			h := sha256.Sum256(body)
			s.SHA256 = hex.EncodeToString(h[:])
			s.Content = string(body)
			if selectedAtDir {
				s.Reason = "higher_precedence_selected"
				r.Sources = append(r.Sources, s)
				continue
			}
			s.Selected = true
			selectedAtDir = true
			selectedCount++
			separator := 0
			if len(combined) > 0 && combined[len(combined)-1] != '\n' {
				separator = 1
			}
			if int64(len(combined)+separator+len(body)) > opts.MaxBytes {
				s.Reason = "byte_budget_exceeded"
				s.Content = "" // diagnostics keep identity and size, never the over-budget payload
				truncated = true
				r.Sources = append(r.Sources, s)
				continue
			}
			if separator != 0 {
				combined = append(combined, '\n')
			}
			start := len(combined)
			combined = append(combined, body...)
			s.Included = true
			s.Reason = "selected"
			s.Span = &SourceSpan{Start: start, End: len(combined)}
			r.Sources = append(r.Sources, s)
			digestSources = append(digestSources, s)
		}
	}
	state.combined = combined
	state.digestSources = digestSources
	state.selectedCount = selectedCount
	state.hadReadFailure = hadReadFailure
	state.truncated = truncated
	state.escaped = escaped
	return state
}

func finalizeEffectiveResult(r *EffectiveResult, state effectiveResolutionState) {
	r.Bytes = len(state.combined)
	switch {
	case state.escaped:
		r.Status = StatusOutsideRoot
	case state.truncated:
		r.Status = StatusTruncated
	case state.selectedCount == 0:
		r.Status = StatusUnknown
	case state.hadReadFailure:
		r.Status = StatusPartial
	case !r.Trusted:
		r.Status = StatusUntrusted
	default:
		r.Status = StatusComplete
	}
	if r.Status == StatusComplete {
		r.Instructions = string(state.combined)
		r.EffectiveSHA256 = digestEffective(state.digestSources)
	}
}

func hierarchy(root, targetDir string) ([]string, error) {
	rel, ok := relativeInside(root, targetDir)
	if !ok {
		return nil, fmt.Errorf("target directory escapes root")
	}
	dirs := []string{root}
	if rel == "." || rel == "" {
		return dirs, nil
	}
	cur := root
	for _, part := range strings.Split(rel, string(filepath.Separator)) {
		cur = filepath.Join(cur, part)
		dirs = append(dirs, cur)
	}
	return dirs, nil
}

func relativeInside(root, path string) (string, bool) {
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return "", false
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return rel, false
	}
	return rel, true
}

func slashClean(path string) string {
	path = filepath.ToSlash(filepath.Clean(path))
	if path == "" {
		return "."
	}
	return path
}

func digestEffective(sources []EffectiveSource) string {
	h := sha256.New()
	h.Write([]byte("fak-agents-effective-v1\x00"))
	var n [8]byte
	for _, source := range sources {
		for _, field := range []string{source.Path, source.Content} {
			binary.BigEndian.PutUint64(n[:], uint64(len(field)))
			h.Write(n[:])
			h.Write([]byte(field))
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}
