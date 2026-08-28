package adjudicator

import (
	"context"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// DevEditAttestation is host-installed provenance for one guarded development
// worker. It is never decoded from tool arguments or ToolCall.Meta.
type DevEditAttestation struct {
	TraceID      string
	Worktree     string
	Issue        string
	Lane         string
	Holder       string
	LaneOwnerPID int
	Paths        []string
	PolicyPath   string
	Verify       func(context.Context, DevEditAttestation) error
}

type devEditAttestationState struct{ att DevEditAttestation }

// SetDevEditAttestation installs launch-scoped development provenance. A nil or
// invalid attestation clears the grant. Verify runs on every protected write so
// a released or stale lease fails closed without restarting the guard.
func (a *Adjudicator) SetDevEditAttestation(att *DevEditAttestation) {
	if att == nil {
		a.devEdit.Store(nil)
		return
	}
	cp := *att
	cp.TraceID = strings.TrimSpace(cp.TraceID)
	cp.Worktree = cleanHostPath(cp.Worktree)
	cp.Issue = strings.TrimSpace(cp.Issue)
	cp.Lane = strings.TrimSpace(cp.Lane)
	cp.Holder = strings.TrimSpace(cp.Holder)
	cp.Paths = normalizeAttestedPaths(cp.Paths)
	cp.PolicyPath = normalizeAttestedPath(cp.PolicyPath)
	if cp.TraceID == "" || cp.Worktree == "" || cp.Issue == "" || cp.Lane == "" || cp.Holder == "" || cp.LaneOwnerPID <= 0 || len(cp.Paths) == 0 || cp.Verify == nil {
		a.devEdit.Store(nil)
		return
	}
	a.devEdit.Store(&devEditAttestationState{att: cp})
}

func cleanHostPath(p string) string {
	p = strings.TrimSpace(p)
	if p == "" {
		return ""
	}
	abs, err := filepath.Abs(p)
	if err != nil {
		return ""
	}
	return filepath.Clean(abs)
}

func normalizeAttestedPath(raw string) string {
	paths := normalizeAttestedPaths([]string{raw})
	if len(paths) == 1 {
		return paths[0]
	}
	return ""
}

func normalizeAttestedPaths(in []string) []string {
	out := make([]string, 0, len(in))
	seen := map[string]bool{}
	for _, raw := range in {
		p := strings.TrimPrefix(strings.ReplaceAll(strings.TrimSpace(raw), "\\", "/"), "./")
		p = strings.TrimSuffix(p, "/")
		if p == "" || filepath.IsAbs(raw) || strings.HasPrefix(p, "/") || p == ".." || strings.Contains(p, "../") {
			continue
		}
		if !seen[p] {
			seen[p] = true
			out = append(out, p)
		}
	}
	return out
}

func attestedPathMatch(target, pattern string) bool {
	target = strings.TrimPrefix(strings.ReplaceAll(target, "\\", "/"), "./")
	pattern = strings.TrimPrefix(strings.ReplaceAll(pattern, "\\", "/"), "./")
	if target == pattern {
		return true
	}
	if strings.HasSuffix(pattern, "/**") {
		base := strings.TrimSuffix(pattern, "/**")
		return target == base || strings.HasPrefix(target, base+"/")
	}
	ok, _ := filepath.Match(pattern, target)
	return ok
}

func isRuntimeGuardPolicyPath(path string) bool {
	path = strings.TrimPrefix(strings.ReplaceAll(path, "\\", "/"), "./")
	base := strings.ToLower(filepath.Base(path))
	return (strings.Contains(base, "guard") || strings.Contains(base, "policy")) &&
		(strings.HasSuffix(base, ".json") || strings.HasSuffix(base, ".toml") || strings.HasSuffix(base, ".yaml") || strings.HasSuffix(base, ".yml"))
}
func directDevEditTool(tool string) bool {
	switch strings.ToLower(strings.TrimSpace(tool)) {
	case "write_file", "write", "edit", "apply_patch":
		return true
	default:
		return false
	}
}
func (a *Adjudicator) devEditAttested(ctx context.Context, c *abi.ToolCall, target string) bool {
	st := a.devEdit.Load()
	if st == nil || c == nil || c.TraceID != st.att.TraceID {
		return false
	}
	rel := strings.TrimPrefix(strings.ReplaceAll(strings.TrimSpace(target), "\\", "/"), "./")
	if filepath.IsAbs(target) {
		abs := cleanHostPath(target)
		r, err := filepath.Rel(st.att.Worktree, abs)
		if err != nil || r == ".." || strings.HasPrefix(r, ".."+string(filepath.Separator)) {
			return false
		}
		rel = strings.ReplaceAll(r, "\\", "/")
	}
	if rel == "" || rel == "." || rel == ".." || strings.HasPrefix(rel, "../") || strings.Contains(rel, "/../") {
		return false
	}
	// Repository administration is never developable through this capability,
	// even when a malformed or over-broad lease row names it.
	if rel == ".git" || attestedPathMatch(rel, ".git/**") {
		return false
	}
	if rel == st.att.PolicyPath || isRuntimeGuardPolicyPath(rel) {
		return false
	}
	owned := false
	for _, p := range st.att.Paths {
		if attestedPathMatch(rel, p) {
			owned = true
			break
		}
	}
	return owned && st.att.Verify(ctx, st.att) == nil
}
