package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

type guardDOSLease struct {
	Lane   string   `json:"lane"`
	Holder string   `json:"holder"`
	Tree   []string `json:"tree"`
	Mode   string   `json:"mode"`
	PID    int      `json:"pid"`
}
type guardWorktreeOwner struct {
	Schema  string `json:"schema"`
	PID     int    `json:"pid"`
	LeaseID string `json:"lease_id"`
}

var (
	guardDevLeaseLive = func(ctx context.Context, workspace string) ([]byte, error) {
		cmd := exec.CommandContext(ctx, "dos", "lease-lane", "--workspace", workspace, "live")
		windowgate.ConfigureBackgroundCommand(cmd)
		return cmd.Output()
	}
	guardDevWorkspaceForWorktree = guardCommonWorkspace
)

// installGuardDevAttestation shifts SELF_MODIFY authorization to guard startup.
// Only a sanctioned detached worktree carrying a matching owner stamp and a live
// exclusive DOS lane can install the process-local grant. Model call fields are
// deliberately absent from this path.
func installGuardDevAttestation(trace, policyPath string) error {
	lane := firstGuardEnv(envMap(os.Environ()), "FAK_LANE", "DISPATCH_LANE")
	wt := firstGuardEnv(envMap(os.Environ()), "FAK_WORKTREE_DIR", "FLEET_WORKER_WORKTREE_DIR")
	issue := firstGuardEnv(envMap(os.Environ()), "FAK_ROOT_ISSUE", "DISPATCH_ISSUE", "FLEET_RESOLVE_ISSUE")
	if lane == "" || wt == "" || issue == "" || strings.TrimSpace(trace) == "" {
		adjudicator.Default.SetDevEditAttestation(nil)
		return nil
	}
	if _, err := strconv.Atoi(strings.TrimPrefix(issue, "#")); err != nil {
		return fmt.Errorf("dev attestation: invalid issue %q", issue)
	}
	issue = strings.TrimPrefix(issue, "#")
	absWT, err := filepath.Abs(wt)
	if err != nil {
		return err
	}
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	absCWD, err := filepath.Abs(cwd)
	if err != nil || !sameHostPath(absWT, absCWD) {
		return errors.New("dev attestation: worktree does not match guard cwd")
	}
	base := filepath.Base(absWT)
	if !strings.HasPrefix(base, "fak-worker-wt-"+sanitizeGuardLane(lane)+"-") {
		return errors.New("dev attestation: not a sanctioned worker worktree")
	}
	ownerPath := filepath.Join(filepath.Dir(absWT), ".fak-worker-owners", base+".json")
	ownerBytes, err := os.ReadFile(ownerPath)
	if err != nil {
		return fmt.Errorf("dev attestation: owner stamp: %w", err)
	}
	var owner guardWorktreeOwner
	if json.Unmarshal(ownerBytes, &owner) != nil || owner.Schema != "fak-worker-worktree-owner/1" || owner.PID <= 0 || (owner.LeaseID != lane && owner.LeaseID != "resolve-"+lane) {
		return errors.New("dev attestation: owner stamp does not match lane")
	}
	workspace := guardDevWorkspaceForWorktree(absWT)
	if workspace == "" {
		return errors.New("dev attestation: main workspace is not provable")
	}
	lease, err := matchingGuardDevLease(context.Background(), workspace, lane)
	if err != nil {
		return err
	}
	policyRel := relativeWithin(absWT, policyPath)
	att := adjudicator.DevEditAttestation{TraceID: trace, Worktree: absWT, Issue: issue, Lane: lane, Holder: lease.Holder, LaneOwnerPID: lease.PID, Paths: lease.Tree, PolicyPath: policyRel}
	att.Verify = func(ctx context.Context, current adjudicator.DevEditAttestation) error {
		live, err := matchingGuardDevLease(ctx, workspace, current.Lane)
		if err != nil {
			return err
		}
		if live.Holder != current.Holder || live.PID != current.LaneOwnerPID || !sameStringSet(live.Tree, current.Paths) {
			return errors.New("development lease changed")
		}
		return nil
	}
	adjudicator.Default.SetDevEditAttestation(&att)
	return nil
}

func matchingGuardDevLease(ctx context.Context, workspace, lane string) (guardDOSLease, error) {
	b, err := guardDevLeaseLive(ctx, workspace)
	if err != nil {
		return guardDOSLease{}, fmt.Errorf("dev attestation: DOS live lease: %w", err)
	}
	var rows []guardDOSLease
	if err := json.Unmarshal(b, &rows); err != nil {
		return guardDOSLease{}, fmt.Errorf("dev attestation: DOS lease JSON: %w", err)
	}
	var found *guardDOSLease
	for i := range rows {
		if rows[i].Lane == lane && rows[i].Mode == "exclusive" && rows[i].Holder != "" && rows[i].PID > 0 && len(rows[i].Tree) > 0 {
			if found != nil {
				return guardDOSLease{}, errors.New("dev attestation: ambiguous live lane")
			}
			found = &rows[i]
		}
	}
	if found == nil {
		return guardDOSLease{}, errors.New("dev attestation: matching live exclusive lane not found")
	}
	return *found, nil
}

func sanitizeGuardLane(lane string) string {
	var b strings.Builder
	for _, r := range lane {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '.' || r == '-' {
			b.WriteRune(r)
		} else {
			b.WriteByte('-')
		}
	}
	return strings.Trim(b.String(), "-")
}
func guardCommonWorkspace(wt string) string {
	cmd := exec.Command("git", "-C", wt, "rev-parse", "--path-format=absolute", "--git-common-dir")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	common := filepath.Clean(strings.TrimSpace(string(out)))
	if filepath.Base(common) != ".git" {
		return ""
	}
	return filepath.Dir(common)
}
func sameHostPath(a, b string) bool { return strings.EqualFold(filepath.Clean(a), filepath.Clean(b)) }
func relativeWithin(root, path string) string {
	if strings.TrimSpace(path) == "" {
		return ""
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return ""
	}
	r, err := filepath.Rel(root, abs)
	if err != nil || r == ".." || strings.HasPrefix(r, ".."+string(filepath.Separator)) {
		return ""
	}
	return strings.ReplaceAll(r, "\\", "/")
}
func sameStringSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	m := map[string]int{}
	for _, s := range a {
		m[s]++
	}
	for _, s := range b {
		m[s]--
	}
	for _, n := range m {
		if n != 0 {
			return false
		}
	}
	return true
}
