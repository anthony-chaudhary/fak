// Package taskvc binds the fleet's live Windows Scheduled Tasks to the installers
// that recreate them from version control — the reboot-survival half of the fleet's
// always-on posture (#3323).
//
// The fleet runs its unattended garden as Scheduled Tasks (FleetResumeWatchdog,
// FleetStaleWorkGarden, FakLogvaultCapture, ...). Each is *supposed* to be
// reproducible from a versioned tools/register_*.ps1, so a reimaged host can be
// rebuilt from the repo. Two failure modes break that, and neither is visible to
// `fak schedscan` (which only reports live task HEALTH, never whether the task
// could be rebuilt at all):
//
//   - ORPHAN: an enabled fleet task with no installer in version control. If the
//     host is reimaged the loop is simply lost — nothing in the repo remembers it.
//   - NAME DRIFT: an installer exists, but the task name it registers no longer
//     matches the live task. Re-running the installer then spawns a SECOND,
//     differently-named loop instead of updating the existing one in place. This
//     is exactly what happened to FleetStaleWorkGarden, whose installer
//     (register_stale_work_watchdog.ps1) had drifted to a different -TaskName.
//
// taskvc makes both checkable. DeclaredTaskNames extracts the task names an
// installer script *statically* registers; Verify holds the declared Inventory
// (inventory.go) against those installers and refuses any row whose installer no
// longer declares its task. The name-drift class is caught permanently: delete or
// rename an installer's -TaskName default and the trunk guard reds.
//
// The parse is deliberately LITERAL-ONLY. A PowerShell single-quoted default
// ($TaskName = 'FleetScoutLoop') is a static, bindable fact; a name interpolated
// at runtime ($TaskName = "FleetSomething-$repoSlug") is not knowable without
// executing the script, so it is never treated as covering a task. That is a
// feature, not a parser gap: a templated installer genuinely CANNOT be proven to
// update a bare live task in place, so the inventory records the gap (StatusDrift)
// instead of pretending coverage. The cure is to give the installer a literal
// default — register_worktree_doctor.ps1 carried exactly that interpolated shape
// until #5409 pinned it to 'FleetWorktreeDoctor', which promoted its inventory row
// to StatusInstaller without this parse changing at all.
//
// Everything here is pure and cross-platform except ScanTree's `git ls-files`
// call; the classifier core is unit-tested on synthetic inputs, so the trunk
// guard runs identically on Linux CI and the Windows fleet host.
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

// ReasonOrphanTask is the closed-vocabulary refusal code for an enabled fleet task
// that has no installer in version control — the loop is lost on a reimage.
const ReasonOrphanTask = "ORPHAN_FLEET_TASK"

// ReasonInstallerNameDrift is the closed-vocabulary refusal code for an inventory
// row whose installer no longer declares that task name. Re-running such an
// installer creates a second task instead of updating the live one in place.
const ReasonInstallerNameDrift = "INSTALLER_NAME_DRIFT"

// ReasonMissingCapture is the closed-vocabulary refusal code for a row that claims
// an exported task XML the tree does not actually track. An untracked capture is
// the same lie as a deleted installer: the reimage it promises to survive would
// find nothing there.
const ReasonMissingCapture = "MISSING_TASK_CAPTURE"

// CaptureDir is where the scrubbed task XML exports live, repo-relative. Written
// by tools/capture_fleet_task_xml.ps1, which is also the only sanctioned way to
// refresh one: it strips the host SID, COMPUTERNAME\user and home paths, and
// refuses to write if any of them survive.
const CaptureDir = "tools/scheduled-tasks"

// Status is how one enabled fleet task is covered by version control.
type Status string

const (
	// StatusInstaller: a versioned tools/*.ps1 statically registers this exact
	// task name. Re-running it updates the live task in place. This is the only
	// status that means "rebuildable from the repo".
	StatusInstaller Status = "installer"
	// StatusXML: no installer, but a scrubbed export of the live task definition
	// is versioned under CaptureDir. The schedule, principal shape and exact
	// command line survive a reimage; whatever the command line POINTS AT does
	// not, unless it is itself in the repo. Weaker than StatusInstaller, which is
	// why Reason is still required — see Coverage.Reason.
	StatusXML Status = "xml"
	// StatusDrift: an installer for this loop exists but registers a DIFFERENT
	// (or interpolated, hence unbindable) name, so a reinstall duplicates it. The
	// inventory currently holds none — #5409 reconciled the last one — but the
	// status stays: the class recurs every time an installer's -TaskName default is
	// edited without its row.
	StatusDrift Status = "drift"
	// StatusOrphan: no installer AND no exported XML — the loop is simply lost on
	// a reimage. The inventory currently holds none; the gate exists to keep it
	// that way.
	StatusOrphan Status = "orphan"
)

// Coverage is one enabled fak-fleet Scheduled Task and its version-control claim.
// Installer is set only for StatusInstaller rows; Capture names the exported task
// XML for rows an installer does not cover; Reason is required for every status
// except StatusInstaller, so the residual gap is always named, never silently
// tolerated.
type Coverage struct {
	Task      string // live task name, e.g. "FleetStaleWorkGarden"
	Status    Status
	Installer string // repo-relative installer path, for StatusInstaller rows
	Capture   string // repo-relative exported task XML under CaptureDir
	Reason    string // why this task is not (or only partly) rebuildable
}

// Offense is one inventory row that does not hold up against the tree.
type Offense struct {
	Task   string
	Reason string // ReasonInstallerNameDrift | ReasonOrphanTask
	Detail string
}

// String renders the offense as a one-line, reason-coded report.
func (o Offense) String() string {
	return fmt.Sprintf("%s: %s (%s)", o.Task, o.Detail, o.Reason)
}

// taskNameDecl matches a PowerShell task-name parameter default. It deliberately
// accepts any $<Prefix>TaskName spelling, because installers that register more
// than one task name them separately ($CaptureTaskName / $VerifyTaskName in
// register_logvault_backup.ps1).
var taskNameDecl = regexp.MustCompile(`\$[A-Za-z0-9_]*TaskName\s*=\s*(?:'([^']*)'|"([^"]*)")`)

// DeclaredTaskNames returns the task names a fleet installer script statically
// registers, sorted and deduped.
//
// Only literal names count. An empty-string default and one interpolated at
// runtime ($TaskName = "FleetSomething-$repoSlug") are both skipped: neither is
// knowable without executing the script, so neither can be evidence that
// re-running the installer would update a given live task in place. The skip is
// keyed on the SHAPE of the value — an unresolved "$" — and never on a particular
// task, so an installer that trades its template for a literal binds immediately,
// with no change here.
func DeclaredTaskNames(src string) []string {
	seen := map[string]bool{}
	for _, m := range taskNameDecl.FindAllStringSubmatch(src, -1) {
		name := m[1]
		if name == "" {
			name = m[2] // double-quoted form
		}
		name = strings.TrimSpace(name)
		if name == "" || strings.Contains(name, "$") {
			continue // empty, or interpolated at runtime — not statically bindable
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

// Verify is the pure gate core: it holds each Coverage row against the task names
// the versioned installers actually declare, and returns one Offense per row that
// does not hold up, sorted for stable output.
//
// declared maps a repo-relative installer path to the task names it registers
// (i.e. DeclaredTaskNames applied to each tracked installer). captured is the set
// of exported task XML paths the tree actually tracks under CaptureDir.
//
//   - A StatusInstaller row must name an installer that EXISTS in declared and
//     declares that exact task — otherwise the installer was deleted, renamed, or
//     its -TaskName default drifted, and a reinstall would duplicate the loop.
//   - A StatusXML row must name a Capture the tree really tracks. A capture claim
//     is a reimage-survival promise; an untracked file cannot keep it.
//   - Any row that names a Capture must have it tracked, whatever its status — a
//     drift row leaning on a capture is making the same promise.
//   - Every row except StatusInstaller must carry a Reason. Naming the gap is the
//     point; an unexplained orphan is indistinguishable from an oversight, and an
//     XML capture is NOT full coverage (it versions the task, never the script the
//     task launches), so it has to say what it does not restore.
//
// Note the asymmetry: a drift/orphan row that turns out to be COVERED is not an
// offense here. Verify refuses overclaiming (a coverage claim that is not true),
// never underclaiming — so honestly recording a gap can never red the trunk.
func Verify(inv []Coverage, declared map[string][]string, captured map[string]bool) []Offense {
	var offenses []Offense
	for _, c := range inv {
		// A named capture must exist in the tree regardless of status: the row is
		// promising a reimage can read that file back.
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

// Uncovered returns the rows that would be LOST on a reimage: no installer, and no
// exported task XML either. This is the #3323 done condition as a predicate —
// "every enabled fak-fleet task has an installer or exported XML in the repo" is
// exactly len(Uncovered(Inventory())) == 0.
//
// It is deliberately separate from Verify. Verify asks "is this row's claim true?";
// Uncovered asks "is the claim good enough?" — a row can be perfectly honest about
// being an orphan and still fail the done condition.
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

// ScanTree parses every tracked fleet installer under tools/ in repoRoot, collects
// the tracked task-XML captures, and holds the declared Inventory against both,
// returning one Offense per row that does not hold up. Uses `git ls-files` (not a
// filesystem walk) so an untracked scratch script or a stray uncommitted export is
// correctly ignored — only files that would actually ship count.
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

// TrackedCaptures returns the set of repo-relative exported task XML paths tracked
// under CaptureDir. An export sitting in the working tree but never committed does
// NOT count: it would not survive the reimage the capture exists to survive.
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

// DeclaredByInstaller reads the tracked fleet installer scripts under tools/ and
// maps each repo-relative path to the task names it statically registers.
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

// trackedInstallers lists the tracked PowerShell installers under tools/. Both
// naming conventions in the tree are covered: the register_*.ps1 majority and the
// older install_*.ps1 spelling (install_self_update_schedule.ps1).
func trackedInstallers(repoRoot string) ([]string, error) {
	return gitLsFiles(repoRoot, "tools/register_*.ps1", "tools/install_*.ps1")
}

// gitLsFiles lists the tracked paths under repoRoot matching the given pathspecs,
// sorted. Shared by the installer scan and the capture scan so both agree on what
// "in version control" means.
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
