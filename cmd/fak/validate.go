// fak validate answers the shared-trunk question that neither a live-tree build nor
// ci-preflight can answer: does the committed tip plus only my explicit uncommitted
// delta pass affected-package build/vet and tests? Examples:
//
//	fak validate --mine internal/gitgate/gate.go --mine internal/gitgate/gate_test.go
//	fak validate --ref origin/main --mine cmd/fak/new_verb.go --json
//
// Ownership is deliberately explicit and repeatable; the verb never guesses from git
// status because this checkout contains concurrent peers' tracked and untracked WIP.
package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/affectedtests"
	"github.com/anthony-chaudhary/fak/internal/amdgpu"
	"github.com/anthony-chaudhary/fak/internal/validate"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const defaultValidateTimeout = 4 * time.Minute

var (
	validatePhaseHook   = func(context.Context, string) {}
	validateWSLLookPath = exec.LookPath
	validateWSLCommand  = runValidateWSLCapabilityCommand
)

type validatePhase struct {
	Name      string `json:"name"`
	Status    string `json:"status"`
	ElapsedMS int64  `json:"elapsed_ms"`
	Detail    string `json:"detail,omitempty"`
}

type validateOverlayProgress struct {
	Checked []string `json:"checked"`
	Skipped []string `json:"skipped"`
}

type validateWSLCapabilityVerdict struct {
	Status   string   `json:"status"`
	Identity string   `json:"identity,omitempty"`
	Required []string `json:"required"`
	Missing  []string `json:"missing"`
	Detail   string   `json:"detail,omitempty"`
	Cached   bool     `json:"cached"`
}

type validateResult struct {
	Schema          string                         `json:"schema"`
	Mode            string                         `json:"mode"`
	Ref             string                         `json:"ref"`
	Tip             string                         `json:"tip"`
	Mine            []string                       `json:"mine"`
	Tested          []string                       `json:"tested,omitempty"`
	Runner          string                         `json:"runner,omitempty"`
	TestRun         string                         `json:"test_run,omitempty"`
	TestScope       string                         `json:"test_scope,omitempty"`
	OK              bool                           `json:"ok"`
	Partial         bool                           `json:"partial"`
	TimedOut        bool                           `json:"timed_out"`
	Reason          string                         `json:"reason,omitempty"`
	TimeoutMS       int64                          `json:"timeout_ms"`
	ElapsedMS       int64                          `json:"elapsed_ms"`
	Phases          []validatePhase                `json:"phases"`
	SkippedPhases   []string                       `json:"skipped_phases"`
	Overlays        validateOverlayProgress        `json:"overlays"`
	WSLPreflight    *validateWSLCapabilityVerdict  `json:"wsl_preflight,omitempty"`
	StrixValidation *amdgpu.StrixValidationReceipt `json:"strix_validation,omitempty"`
	Failures        []ciPreflightFailure           `json:"failures"`
	SelectionAudit  *validateSelectionAudit        `json:"selection_audit,omitempty"`
}

type validateSelectionAudit struct {
	Base             string   `json:"base"`
	Head             string   `json:"head"`
	SelectedPackages []string `json:"selected_packages"`
	affectedtests.SelectionAudit
}

func cmdValidate(argv []string) { os.Exit(runValidate(os.Stdout, os.Stderr, argv)) }

// runValidate checks committed ref plus only explicitly-owned working-tree paths.
func runValidate(stdout, stderr io.Writer, argv []string) int {
	for _, arg := range argv {
		if strings.HasPrefix(arg, "--acceptance") || strings.HasPrefix(arg, "-acceptance") {
			return runValidateAcceptanceCLI(stdout, stderr, argv)
		}
	}
	return validate.RunWithHooks(stdout, stderr, argv, validate.Hooks{
		Phase:               validatePhaseHook,
		WSLLookPath:         validateWSLLookPath,
		WSLCommand:          validateWSLCommand,
		DiscoverStrixTarget: discoverStrixTargetFn,
		NewStrixAuthority:   newStrixControllerAuthorityFn,
		StrixAuthorityValid: strixControllerAuthorityValidFn,
		RunStrixValidation:  runStrixValidationFn,
		BuildStrixCandidate: buildStrixCandidateArchiveFn,
	})
}

func normalizeMinePaths(root string, raw []string) ([]string, error) {
	return normalizeMinePathsWithin(context.Background(), root, raw)
}

func normalizeMinePathsWithin(ctx context.Context, root string, raw []string) ([]string, error) {
	seen := map[string]bool{}
	var out []string
	rootAbs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	realRoot := rootAbs
	if resolved, rootErr := filepath.EvalSymlinks(rootAbs); rootErr == nil {
		realRoot = resolved
	}
	for _, value := range raw {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if strings.TrimSpace(value) == "" {
			return nil, fmt.Errorf("empty --mine path")
		}
		p := value
		if !filepath.IsAbs(p) {
			p = filepath.Join(rootAbs, p)
		}
		p, err = filepath.Abs(p)
		if err != nil {
			return nil, err
		}
		rel, err := filepath.Rel(rootAbs, p)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			// If containment fails with raw paths, try with canonicalized paths
			// (handles symlinked roots such as macOS /var -> /private/var).
			realP := p
			if resolved, evalErr := filepath.EvalSymlinks(p); evalErr == nil {
				realP = resolved
			} else if parent, evalErr := filepath.EvalSymlinks(filepath.Dir(p)); evalErr == nil {
				realP = filepath.Join(parent, filepath.Base(p))
			}
			inside, relErr := filepath.Rel(realRoot, realP)
			if relErr != nil || inside == ".." || strings.HasPrefix(inside, ".."+string(filepath.Separator)) {
				return nil, fmt.Errorf("--mine path %q escapes repo root", value)
			}
			rel = inside
		}
		rel = filepath.ToSlash(filepath.Clean(rel))
		if rel == "." {
			return nil, fmt.Errorf("--mine cannot name the repo root; list owned paths explicitly")
		}
		info, statErr := os.Stat(p)
		if statErr == nil && info.IsDir() {
			walkErr := filepath.WalkDir(p, func(child string, d os.DirEntry, walkErr error) error {
				if walkErr != nil {
					return walkErr
				}
				if err := ctx.Err(); err != nil {
					return err
				}
				if d.IsDir() {
					// An explicitly named hidden/scratch directory is owned input. Hidden or
					// underscore descendants of a broader directory request are generated
					// workspace state, not source, and can dwarf the actual overlay.
					if child != p && validateSkipWalkDir(d.Name()) {
						return filepath.SkipDir
					}
					return nil
				}
				childRel, err := filepath.Rel(rootAbs, child)
				if (err != nil || childRel == ".." || strings.HasPrefix(childRel, ".."+string(filepath.Separator))) && realRoot != rootAbs {
					childRel, err = filepath.Rel(realRoot, child)
				}
				if err != nil {
					return err
				}
				childRel = filepath.ToSlash(childRel)
				if !seen[childRel] {
					seen[childRel] = true
					out = append(out, childRel)
				}
				return nil
			})
			if walkErr != nil {
				return nil, walkErr
			}
			continue
		}
		if statErr != nil && !os.IsNotExist(statErr) {
			return nil, statErr
		}
		if !seen[rel] {
			seen[rel] = true
			out = append(out, rel)
		}
	}
	sort.Strings(out)
	return out, nil
}

func validateSkipWalkDir(name string) bool {
	return strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_")
}

// overlayMinePaths copies each owned working-tree path onto the materialized tip.
//
// The containment check canonicalizes both sides, the same both-sides discipline
// dispatchWitnessSamePath uses. EvalSymlinks(src) returns a fully resolved path, so
// measuring it against an unresolved srcRoot refuses honest owned paths on every host
// whose repo root is merely reachable through a symlink — macOS puts TMPDIR under /var,
// a symlink to /private/var, and the resolved file then reads as outside its own root.
//
// Resolving the root is best-effort on purpose: when EvalSymlinks cannot canonicalize it
// the raw spelling is kept rather than the check being skipped. A raw root can only
// refuse more than a canonical one — no canonical path lies under a symlinked spelling of
// a directory — so the fallback stays on the strict side, where an uncertain containment
// check belongs. Containment stays on filepath.Rel rather than a string prefix: Rel is
// separator-aware, so /a/bc reads as outside /a/b, and it is case-insensitive on Windows.
func overlayMinePaths(srcRoot, dstRoot string, paths []string) error {
	return overlayMinePathsWithin(context.Background(), srcRoot, dstRoot, paths, nil)
}

func overlayMinePathsWithin(ctx context.Context, srcRoot, dstRoot string, paths []string, checked func(string)) error {
	realRoot := srcRoot
	if resolved, rootErr := filepath.EvalSymlinks(srcRoot); rootErr == nil {
		realRoot = resolved
	}
	for _, rel := range paths {
		if err := ctx.Err(); err != nil {
			return err
		}
		src := filepath.Join(srcRoot, filepath.FromSlash(rel))
		dst := filepath.Join(dstRoot, filepath.FromSlash(rel))
		realSrc, evalErr := filepath.EvalSymlinks(src)
		if os.IsNotExist(evalErr) {
			if removeErr := os.RemoveAll(dst); removeErr != nil {
				return removeErr
			}
			if checked != nil {
				checked(rel)
			}
			continue
		}
		if evalErr != nil {
			return evalErr
		}
		inside, relErr := filepath.Rel(realRoot, realSrc)
		if relErr != nil || inside == ".." || strings.HasPrefix(inside, ".."+string(filepath.Separator)) {
			return fmt.Errorf("owned path %q resolves outside repo root", rel)
		}
		info, err := os.Stat(src)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		if err := copyValidateFileWithin(ctx, realSrc, dst, info.Mode().Perm()); err != nil {
			return err
		}
		if checked != nil {
			checked(rel)
		}
	}
	return nil
}

func copyValidateFileWithin(ctx context.Context, src, dst string, mode os.FileMode) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, mode)
	if err != nil {
		return err
	}
	closed := false
	defer func() {
		if !closed {
			_ = out.Close()
		}
	}()
	buf := make([]byte, 64*1024)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		n, readErr := in.Read(buf)
		if n > 0 {
			if _, err := out.Write(buf[:n]); err != nil {
				return err
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return readErr
		}
	}
	if err := out.Close(); err != nil {
		return err
	}
	closed = true
	return nil
}

func runValidateWSLCapabilityCommand(ctx context.Context, args ...string) ([]byte, error) {
	cmd := windowgate.CommandContext(ctx, "wsl.exe", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if exitErr, ok := err.(*exec.ExitError); ok && len(exitErr.Stderr) > 0 {
		out = append(out, exitErr.Stderr...)
	}
	return out, err
}
