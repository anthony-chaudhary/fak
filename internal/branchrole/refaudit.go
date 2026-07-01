package branchrole

import (
	"bufio"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	RefClassUnclassified      = "unclassified"
	RefClassDevelopmentSource = "development-source"
	RefClassWorkflowCovered   = "workflow-covered"
	RefClassAuditDoc          = "audit-doc"
	RefClassHistorical        = "historical"
	RefClassFixture           = "fixture"
	RefClassPublicGuard       = "public-link-guard"
	RefClassPublicFrontDoor   = "public-front-door"
)

var hardcodedRefPatterns = []*regexp.Regexp{
	regexp.MustCompile(`\borigin/(main|master)\b`),
	regexp.MustCompile(`\brefs/heads/(main|master)\b`),
	regexp.MustCompile(`github\.ref_name\s*==\s*['"](main|master)['"]`),
	regexp.MustCompile(`github\.ref\s*==\s*['"]refs/heads/(main|master)['"]`),
	regexp.MustCompile(`\bbranch\s*==\s*['"](main|master)['"]`),
	regexp.MustCompile(`['"]branch['"]\s*:\s*['"](main|master)['"]`),
	regexp.MustCompile(`\bgit\s+switch\s+(main|master)\b`),
	regexp.MustCompile(`\bon-master\b`),
	regexp.MustCompile(`\b--master-ref\b`),
	regexp.MustCompile(`\bworktree_master_ref\b`),
	regexp.MustCompile(`\bDEFAULT_WORKTREE_MASTER_REF\b`),
}

// RefFinding is one hard-coded branch reference classified by the branch-role audit.
type RefFinding struct {
	Path  string
	Line  int
	Class string
	Text  string
}

// AuditHardcodedRefs scans tracked-source-like files for main/master branch refs
// that are not covered by workflowaudit. It classifies known intentional families
// and leaves anything else unclassified so tests can fail closed.
func AuditHardcodedRefs(root string) ([]RefFinding, error) {
	if root == "" {
		var err error
		root, err = FindRoot("")
		if err != nil {
			return nil, err
		}
	}
	var findings []RefFinding
	// The audit walks the filesystem, so without ignore-awareness it descends
	// gitignored scratch (.fak, scratchpad, .dispatch-runs, tools/_registry, ...)
	// and flags refs in throwaway data as unclassified -- reddening the gate on any
	// working tree that has scratch dirs. Prune what git ignores; best-effort, so a
	// git failure just falls back to the static skip list below.
	ignored := gitIgnoredPaths(root)
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return relErr
		}
		rel = filepath.ToSlash(rel)
		if d.IsDir() {
			if rel != "." && ignored[rel] {
				return filepath.SkipDir
			}
			if skipRefAuditDir(rel) {
				return filepath.SkipDir
			}
			return nil
		}
		if ignored[rel] {
			return nil
		}
		if !scanRefAuditFile(rel) {
			return nil
		}
		rows, scanErr := scanHardcodedRefFile(path, rel)
		if scanErr != nil {
			return scanErr
		}
		findings = append(findings, rows...)
		return nil
	})
	return findings, err
}

// gitIgnoredPaths returns the set of repo-relative paths git ignores under root,
// collapsed to directories where git can (`ls-files --directory`). Best-effort:
// a git failure (not a repo, git missing) yields nil so the audit still runs,
// just without ignore-awareness. NUL-delimited so paths with odd characters are
// exact, not git-quoted.
func gitIgnoredPaths(root string) map[string]bool {
	cmd := exec.Command("git", "-C", root, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z")
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	set := map[string]bool{}
	for _, entry := range strings.Split(string(out), "\x00") {
		entry = strings.TrimSuffix(filepath.ToSlash(strings.TrimSpace(entry)), "/")
		if entry != "" {
			set[entry] = true
		}
	}
	return set
}

func scanHardcodedRefFile(path, rel string) ([]RefFinding, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []RefFinding
	// bufio.Scanner caps a token at 64 KiB and fails "token too long" on any
	// longer line; the audit walks generated data files (.json/.jsonl/.txt) that
	// can carry a single multi-KB line, so read with an unbounded bufio.Reader
	// instead and keep the newline-stripping / 1-based line-number semantics.
	reader := bufio.NewReader(f)
	lineNo := 0
	for {
		line, readErr := reader.ReadString('\n')
		if len(line) > 0 {
			lineNo++
			text := strings.TrimRight(line, "\r\n")
			if hardcodedRefLine(text) {
				out = append(out, RefFinding{
					Path:  rel,
					Line:  lineNo,
					Class: ClassifyHardcodedRef(rel, text),
					Text:  strings.TrimSpace(text),
				})
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				return out, nil
			}
			return out, readErr
		}
	}
}

func hardcodedRefLine(line string) bool {
	for _, pattern := range hardcodedRefPatterns {
		if pattern.MatchString(line) {
			return true
		}
	}
	return false
}

// ClassifyHardcodedRef classifies a single hard-coded branch ref occurrence.
func ClassifyHardcodedRef(path, line string) string {
	p := filepath.ToSlash(path)
	switch {
	case strings.HasPrefix(p, ".github/workflows/"):
		return RefClassWorkflowCovered
	case p == "docs/branch-regime-hardcoded-ref-audit.md" ||
		p == "docs/branch-regime.md" ||
		p == "docs/ci/workflow-branch-audit.md" ||
		strings.HasPrefix(p, "internal/branchrole/"):
		return RefClassAuditDoc
	case strings.HasPrefix(p, "docs/stable-releases/") ||
		strings.HasPrefix(p, "docs/_audits/") ||
		strings.HasPrefix(p, "docs/") ||
		p == ".github/branch-integration-closeout.md" ||
		p == "BENCHMARK-GOVERNANCE.md" ||
		p == "GPU.md":
		return RefClassHistorical
	case p == "AGENTS.md" ||
		p == "CONTRIBUTING.md" ||
		p == "tools/extend_preflight.py" ||
		p == "tools/fleet_control_pane.py" ||
		p == "tools/fleet_control_pane_test.py" ||
		p == "cmd/fak/affected.go" ||
		p == "cmd/fak/treedoctor.go" ||
		p == "internal/corelockaudit/corelockaudit.go" ||
		p == "internal/gitgate/gitgate.go" ||
		p == "internal/releasestatus/releasestatus.go" ||
		p == "internal/treedoctor/treedoctor.go" ||
		p == "internal/workerenvelope/workerenvelope.go" ||
		p == "tools/issue_resolve_witnessed.py" ||
		p == "tools/register_worktree_doctor.ps1" ||
		p == "tools/worktree_doctor.py":
		return RefClassDevelopmentSource
	case p == "cmd/fak/release.go" ||
		p == "cmd/fak/release_status.go" ||
		p == "cmd/fak/releasestale.go" ||
		p == "cmd/fak/selfupdate.go" ||
		p == "cmd/fak/usage.go" ||
		p == "internal/devindex/verbs.go" ||
		p == "internal/releasestale/releasestale.go" ||
		p == "internal/selfinstall/reap.go" ||
		p == "internal/selfinstall/selfinstall.go" ||
		strings.HasPrefix(p, "tools/dgx_") ||
		p == "tools/glm_witness_record.py" ||
		p == "tools/install_self_update_schedule.ps1" ||
		p == "tools/release_status.py" ||
		strings.HasPrefix(p, "tools/forge-rulesets/"):
		return RefClassPublicFrontDoor
	case strings.HasPrefix(p, "internal/workflowaudit/"):
		return RefClassFixture
	case p == "tools/demo_robustness_scorecard.py":
		return RefClassPublicGuard
	case strings.HasPrefix(p, "tools/bench_migrate") ||
		p == "tools/bench_node.README.md" ||
		p == "tools/gcp_bench.py" ||
		strings.HasPrefix(p, "tools/schemas/") ||
		strings.HasPrefix(p, "experiments/") ||
		strings.HasSuffix(p, "_test.go") ||
		strings.HasSuffix(p, "_test.py"):
		return RefClassFixture
	default:
		_ = line
		return RefClassUnclassified
	}
}

func skipRefAuditDir(rel string) bool {
	switch rel {
	case ".git", ".idea", ".vscode", ".claude", "node_modules", "vendor":
		return true
	default:
		return strings.HasPrefix(rel, "docs/vendor/")
	}
}

func scanRefAuditFile(rel string) bool {
	if rel == "llms-full.txt" {
		return false
	}
	switch filepath.Ext(rel) {
	case ".go", ".py", ".ps1", ".sh", ".md", ".yml", ".yaml", ".toml", ".txt", ".json":
		return true
	default:
		return false
	}
}
