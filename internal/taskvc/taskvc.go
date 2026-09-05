// Package taskvc binds the fleet's live Windows Scheduled Tasks to the installers
// that recreate them from version control (#3323).
//
// It prevents orphan tasks (untracked loops lost on reimage) and name drift
// (installer defaults diverging from live tasks). DeclaredTaskNames extracts
// literal task registrations, and Verify validates the declared inventory
// against tracked installers and scrubbed task-XML captures.
package taskvc

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ReasonOrphanTask marks an enabled task lacking version-controlled installer or capture.
const ReasonOrphanTask = "ORPHAN_FLEET_TASK"

// ReasonInstallerNameDrift marks an inventory row whose installer registers a different task name.
const ReasonInstallerNameDrift = "INSTALLER_NAME_DRIFT"

// ReasonMissingCapture marks an inventory row claiming an untracked exported task XML.
const ReasonMissingCapture = "MISSING_TASK_CAPTURE"

// CaptureDir is the repo-relative directory holding scrubbed task XML exports.
const CaptureDir = "tools/scheduled-tasks"

// Status represents the version-control coverage tier for a scheduled task.
type Status string

const (
	// StatusInstaller indicates a versioned installer statically registers this task name.
	StatusInstaller Status = "installer"
	// StatusXML indicates a scrubbed export of the task definition is tracked under CaptureDir.
	StatusXML Status = "xml"
	// StatusDrift indicates an installer exists but registers an unbindable or drifted name.
	StatusDrift Status = "drift"
	// StatusOrphan indicates a task has neither an installer nor an exported XML.
	StatusOrphan Status = "orphan"
)

// Coverage records the version-control claim for an enabled scheduled task.
type Coverage struct {
	Task      string
	Status    Status
	Installer string
	Capture   string
	Reason    string
}

// Offense describes an inventory row that fails verification against the repository.
type Offense struct {
	Task   string
	Reason string
	Detail string
}

// String formats the offense as a human-readable, reason-coded summary.
func (o Offense) String() string {
	return fmt.Sprintf("%s: %s (%s)", o.Task, o.Detail, o.Reason)
}

// taskNameDecl matches PowerShell parameter defaults assigning a static task name.
var taskNameDecl = regexp.MustCompile(`\$[A-Za-z0-9_]*TaskName\s*=\s*(?:'([^']*)'|"([^"]*)")`)

// DeclaredTaskNames extracts literal task names statically registered by an installer script.
// Dynamic or interpolated task names containing "$" are skipped.
func DeclaredTaskNames(src string) []string {
	seen := map[string]bool{}
	for _, m := range taskNameDecl.FindAllStringSubmatch(src, -1) {
		name := m[1]
		if name == "" {
			name = m[2]
		}
		name = strings.TrimSpace(name)
		if name == "" || strings.Contains(name, "$") {
			continue
		}
		seen[name] = true
	}
	names := make([]string, 0, len(seen))
	for n := range seen {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// Verify validates coverage claims against declared installer names and tracked captures,
// returning sorted offenses for invalid or incomplete inventory rows.
func Verify(inv []Coverage, declared map[string][]string, captured map[string]bool) []Offense {
	var offenses []Offense
	for _, c := range inv {
		if cap := strings.TrimSpace(c.Capture); cap != "" && !captured[cap] {
			offenses = append(offenses, Offense{
				Task:   c.Task,
				Reason: ReasonMissingCapture,
				Detail: fmt.Sprintf("claims exported task XML %s, which the tree does not track — refresh it with tools/capture_fleet_task_xml.ps1", cap),
			})
		}
		switch c.Status {
		case StatusInstaller:
			names, ok := declared[c.Installer]
			if !ok {
				offenses = append(offenses, Offense{
					Task:   c.Task,
					Reason: ReasonInstallerNameDrift,
					Detail: fmt.Sprintf("claims installer %s, which is not a tracked fleet installer", c.Installer),
				})
				continue
			}
			if !contains(names, c.Task) {
				offenses = append(offenses, Offense{
					Task:   c.Task,
					Reason: ReasonInstallerNameDrift,
					Detail: fmt.Sprintf("installer %s no longer declares this task (declares: %s) — a reinstall would create a second loop", c.Installer, strings.Join(names, ", ")),
				})
			}
		case StatusXML:
			if strings.TrimSpace(c.Capture) == "" {
				offenses = append(offenses, Offense{
					Task:   c.Task,
					Reason: ReasonMissingCapture,
					Detail: fmt.Sprintf("status %q names no exported task XML; capture it with tools/capture_fleet_task_xml.ps1", c.Status),
				})
			}
			if strings.TrimSpace(c.Reason) == "" {
				offenses = append(offenses, Offense{
					Task:   c.Task,
					Reason: ReasonOrphanTask,
					Detail: "an XML capture versions the task, not the script it launches; name what a restore would still be missing",
				})
			}
		case StatusDrift, StatusOrphan:
			if strings.TrimSpace(c.Reason) == "" {
				offenses = append(offenses, Offense{
					Task:   c.Task,
					Reason: ReasonOrphanTask,
					Detail: fmt.Sprintf("status %q carries no reason; name why the task is not rebuildable from the repo", c.Status),
				})
			}
		default:
			offenses = append(offenses, Offense{
				Task:   c.Task,
				Reason: ReasonOrphanTask,
				Detail: fmt.Sprintf("unknown coverage status %q", c.Status),
			})
		}
	}
	sort.Slice(offenses, func(i, j int) bool { return offenses[i].Task < offenses[j].Task })
	return offenses
}

func contains(hay []string, needle string) bool {
	for _, h := range hay {
		if h == needle {
			return true
		}
	}
	return false
}

// Uncovered returns inventory rows lacking both an installer and an exported task XML.
func Uncovered(inv []Coverage) []Coverage {
	var out []Coverage
	for _, c := range inv {
		if c.Status == StatusInstaller || strings.TrimSpace(c.Capture) != "" {
			continue
		}
		out = append(out, c)
	}
	return out
}

// ScanTree parses tracked installers and XML captures, verifying declared inventory in repoRoot.
func ScanTree(repoRoot string) ([]Offense, error) {
	declared, err := DeclaredByInstaller(repoRoot)
	if err != nil {
		return nil, err
	}
	captured, err := TrackedCaptures(repoRoot)
	if err != nil {
		return nil, err
	}
	return Verify(Inventory(), declared, captured), nil
}

// TrackedCaptures returns repo-relative paths of committed task XML files under CaptureDir.
func TrackedCaptures(repoRoot string) (map[string]bool, error) {
	paths, err := gitLsFiles(repoRoot, CaptureDir+"/*.xml")
	if err != nil {
		return nil, err
	}
	captured := make(map[string]bool, len(paths))
	for _, p := range paths {
		captured[p] = true
	}
	return captured, nil
}

// DeclaredByInstaller maps tracked installer script paths to their statically registered task names.
func DeclaredByInstaller(repoRoot string) (map[string][]string, error) {
	paths, err := trackedInstallers(repoRoot)
	if err != nil {
		return nil, err
	}
	declared := make(map[string][]string, len(paths))
	for _, p := range paths {
		src, err := readRepoFile(repoRoot, p)
		if err != nil {
			return nil, err
		}
		declared[p] = DeclaredTaskNames(src)
	}
	return declared, nil
}

// readRepoFile reads a repo-relative path under repoRoot.
func readRepoFile(repoRoot, rel string) (string, error) {
	b, err := os.ReadFile(filepath.Join(repoRoot, filepath.FromSlash(rel)))
	if err != nil {
		return "", fmt.Errorf("read fleet installer %s: %w", rel, err)
	}
	return string(b), nil
}

// trackedInstallers lists tracked PowerShell installer scripts under tools/.
func trackedInstallers(repoRoot string) ([]string, error) {
	return gitLsFiles(repoRoot, "tools/register_*.ps1", "tools/install_*.ps1")
}

// gitLsFiles lists tracked paths under repoRoot matching pathspecs, sorted.
func gitLsFiles(repoRoot string, pathspecs ...string) ([]string, error) {
	cmd := exec.Command("git", append([]string{"ls-files"}, pathspecs...)...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = repoRoot
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git ls-files %v in %s: %w", pathspecs, repoRoot, err)
	}
	var paths []string
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			paths = append(paths, line)
		}
	}
	sort.Strings(paths)
	return paths, nil
}
