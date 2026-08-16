package main

import (
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"text/tabwriter"
)

// The scheduled-task LAUNCHER POSTURE audit (#2174).
//
// schedscan's health half answers "did the last run fail?". This half answers the
// operator question that survived #1456: WHICH live tasks still launch a
// shell/agent process WITHOUT the audited headless/process-isolated contract, so a
// console/TUI fault couples back to the operator desktop or the parent supervisor?
//
// The 2026-07-01 crash audit found the two ends of the spectrum side by side:
//   - FleetResumeWatchdog ran through `conhost.exe --headless powershell.exe ...`
//     — its own pseudoconsole, nothing attached to the operator's session.
//   - FleetStrandedRecovery ran as raw `powershell.exe -WindowStyle Hidden -File
//     %LOCALAPPDATA%\Fleet\stranded_recovery.ps1` — a bare shell, and a script no
//     repo installer can recreate. `-WindowStyle Hidden` hides a window; it does
//     not detach the console, and it says nothing about where the script came from.
//
// The contract has three independent axes, and a task is only "allowed" when all
// three hold. Collapsing them is what let the gap survive #1456: a task can be
// perfectly S4U and still launch a bare shell, and it can be perfectly shimmed and
// still be pinned to the operator's desktop session.
//
//  1. SESSION  — the principal's LogonType. Interactive* binds the task to the
//     operator's logged-on desktop: a fault there is on the operator's screen, and
//     with no session present Task Scheduler refuses the run outright (0x800710E0).
//     S4U / Password / ServiceAccount run in session 0, with no desktop to couple to.
//  2. LAUNCHER — the action's program. `conhost.exe --headless <prog>` gives the
//     wrapped program its own pseudoconsole; a raw powershell/cmd/python/bash action
//     inherits whatever console context the scheduler hands it.
//  3. PROVENANCE — where the launched script lives. A script under %LOCALAPPDATA%,
//     %TEMP% or the repo's untracked _scratch is not repo-owned: no installer
//     recreates it, so the audited launcher contract is unenforceable there. Those
//     are surfaced as operator warnings with remediation, never silently assumed OK.
//
// The classify/verdict core is pure and cross-platform. It runs over three
// interchangeable sources: the live Windows enumeration, a --from JSON snapshot,
// and --xml-dir, which audits the repo-owned task definitions under
// tools/scheduled-tasks/ on any OS — that last one is what makes the pass/fail
// report a thing CI can run.

const schedLauncherSchema = "fak.schedscan.launchers/1"

// schedLauncherInvariant is the sentence the whole report exists to defend, and it
// is emitted with every report (table header and JSON doc) so the posture verdict
// always travels with the reason anyone should care (#2170).
const schedLauncherInvariant = "#2170: a shell/console/TUI fault must not crash or block the program — " +
	"a scheduled task that is desktop-attached (Interactive logon) or launches a bare shell couples that fault " +
	"back to the operator desktop and to the parent fak supervisor instead of dying alone in session 0."

// Launcher classes — axis 2 of the contract.
const (
	// schedLauncherHeadless: `conhost.exe --headless <program> ...`. The approved
	// path; the wrapped program gets its own pseudoconsole.
	schedLauncherHeadless = "headless-shim"
	// schedLauncherRawShell: a bare shell/interpreter action — powershell, cmd,
	// python, bash and friends — with no headless shim in front of it.
	schedLauncherRawShell = "raw-shell"
	// schedLauncherProgram: a compiled program action (fak.exe and the like). Not a
	// shell, so there is no shell fault to blast back; allowed.
	schedLauncherProgram = "program"
	// schedLauncherUnknown: no action program at all (a malformed or non-Exec task).
	schedLauncherUnknown = "unknown"
)

// Session isolation — axis 1 of the contract.
const (
	schedSessionIsolated = "isolated"
	schedSessionDesktop  = "desktop-attached"
	schedSessionUnknown  = "unknown"
)

// Script provenance — axis 3 of the contract.
const (
	schedProvenanceInTree  = "in-tree"
	schedProvenanceOutTree = "out-of-tree"
	schedProvenanceNone    = "none"    // the action launches no script (a program action)
	schedProvenanceUnknown = "unknown" // a script path we cannot place (no repo root given)
)

// Contract verdicts.
const (
	schedVerdictPass   = "pass"
	schedVerdictWarn   = "warn"
	schedVerdictFail   = "fail"
	schedVerdictExempt = "exempt"
)

// schedLauncherShells are the interpreter executables that make an action a
// "raw shell" launch. Matched on the basename with any .exe suffix and directory
// stripped, so "C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe" and a
// bare `powershell` land on the same entry.
var schedLauncherShells = map[string]bool{
	"powershell": true, "pwsh": true, "cmd": true, "command": true,
	"python": true, "python3": true, "pythonw": true, "py": true,
	"bash": true, "sh": true, "zsh": true, "wsl": true, "busybox": true,
	"node": true, "perl": true, "ruby": true, "wscript": true, "cscript": true,
}

// schedLauncherExemptions records the tasks whose raw launcher is deliberate, with
// the reason that makes it safe. The acceptance contract for #2174 is explicit that
// a raw shell must be CONVERTED, EXEMPTED WITH A REASON, or WARNED — never silently
// tolerated — so an exemption lives here in version control where it can be argued
// with, and the report prints the reason next to the task instead of a bare "ok".
//
// Keyed by task name (the live TaskName, which is also the <URI> leaf of a capture).
var schedLauncherExemptions = map[string]string{
	"ClaudeAccountBackup": "raw python.exe is the launcher of record: the action runs an IN-TREE script " +
		"(tools/claude_account_backup.py) under an S4U principal, so there is no operator desktop to couple to " +
		"and no shell interpreter between the scheduler and the program — wrapping it in conhost would only add " +
		"the exit-code mask of #5095 without adding isolation",
}

// schedLauncherPosture is one task's launcher-contract record: the identity and
// operational fields an operator needs (name, action, working directory, last
// result, next run) joined to the three-axis classification and the pass/fail
// verdict, with every reason that produced it and the remediation for each.
type schedLauncherPosture struct {
	Name         string   `json:"name"`
	Path         string   `json:"path,omitempty"`
	Action       string   `json:"action,omitempty"`
	WorkingDir   string   `json:"working_dir,omitempty"`
	LogonType    string   `json:"logon_type,omitempty"`
	RunLevel     string   `json:"run_level,omitempty"`
	LastResult   string   `json:"last_result,omitempty"`
	NextRun      string   `json:"next_run,omitempty"`
	Class        string   `json:"class"`
	Wrapped      string   `json:"wrapped,omitempty"` // program behind a headless shim
	Session      string   `json:"session"`
	Provenance   string   `json:"provenance"`
	Script       string   `json:"script,omitempty"`
	Verdict      string   `json:"verdict"`
	Allowed      bool     `json:"allowed"` // allowed by the launcher contract (pass or exempt)
	Reasons      []string `json:"reasons,omitempty"`
	Remediations []string `json:"remediations,omitempty"`
}

// schedLauncherDoc is the fleet-wide pass/fail report.
type schedLauncherDoc struct {
	Schema       string                 `json:"schema"`
	GeneratedUTC string                 `json:"generated_utc"`
	OS           string                 `json:"os"`
	Source       string                 `json:"source"`
	RepoRoot     string                 `json:"repo_root,omitempty"`
	Filter       string                 `json:"filter,omitempty"`
	Invariant    string                 `json:"invariant"`
	Count        int                    `json:"count"`
	FailCount    int                    `json:"fail_count"`
	WarnCount    int                    `json:"warn_count"`
	Counts       map[string]int         `json:"counts"`
	Tasks        []schedLauncherPosture `json:"tasks"`
}

// schedExeBase reduces an action's program to a comparable basename: quoting and
// surrounding space stripped, directory dropped, .exe suffix removed, lowercased.
// "C:\WINDOWS\System32\conhost.exe" and `"conhost"` both reduce to "conhost".
func schedExeBase(execute string) string {
	exe := strings.ToLower(strings.Trim(strings.TrimSpace(execute), `"'`))
	if i := strings.LastIndexAny(exe, `\/`); i >= 0 {
		exe = exe[i+1:]
	}
	return strings.TrimSuffix(exe, ".exe")
}

// schedSplitArgs splits an action's argument string into tokens, honoring double
// quotes so a quoted program path with spaces ("C:\Program Files\Git\bin\bash.exe")
// stays one token. Task Scheduler stores the argument string verbatim, so this is
// the same split the launched process's own CRT would do for the leading tokens —
// which is all the classifier needs (it only inspects the first program token).
func schedSplitArgs(s string) []string {
	var out []string
	var cur strings.Builder
	inQuote := false
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	for _, r := range s {
		switch {
		case r == '"':
			inQuote = !inQuote
		case !inQuote && (r == ' ' || r == '\t' || r == '\n' || r == '\r'):
			flush()
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return out
}

// schedLauncherClassify places an action on the launcher axis and, for a headless
// shim, names the program it wraps.
//
// conhost WITHOUT --headless is deliberately NOT the approved path: plain
// `conhost.exe <prog>` allocates a real console window, which is precisely the
// desktop flash #1456 fixed. Only the --headless form gets the shim's credit, and
// the wrapped program is still reported so an operator can see what actually runs.
func schedLauncherClassify(execute, arguments string) (class, wrapped string) {
	base := schedExeBase(execute)
	if base == "" {
		return schedLauncherUnknown, ""
	}
	if base == "conhost" {
		args := schedSplitArgs(arguments)
		if len(args) > 0 && strings.EqualFold(args[0], "--headless") {
			if len(args) > 1 {
				wrapped = schedExeBase(args[1])
			}
			return schedLauncherHeadless, wrapped
		}
		// A console-allocating conhost launch is the #1456 desktop-flash shape.
		return schedLauncherRawShell, ""
	}
	if schedLauncherShells[base] {
		return schedLauncherRawShell, ""
	}
	return schedLauncherProgram, ""
}

// schedSessionIsolation places a task's principal on the session axis. Task
// Scheduler spells the interactive family several ways ("Interactive",
// "InteractiveToken", "InteractiveTokenOrPassword"); every one of them means the
// run needs a logged-on desktop session, so they all collapse to desktop-attached.
func schedSessionIsolation(logonType string) string {
	lt := strings.ToLower(strings.TrimSpace(logonType))
	switch {
	case lt == "":
		return schedSessionUnknown
	case strings.Contains(lt, "interactive"):
		return schedSessionDesktop
	case lt == "s4u" || lt == "password" || lt == "serviceaccount" || lt == "group" || lt == "none":
		return schedSessionIsolated
	default:
		return schedSessionUnknown
	}
}

// schedScriptPathRe finds the first script-looking path in an argument string. It
// stops at whitespace, quotes and shell metacharacters, which is enough to lift a
// path out of both the simple `-File "%LOCALAPPDATA%\Fleet\x.ps1"` form and the
// nested `-Command "... & 'python.exe' "C:\...\worktree_doctor.py" ..."` blobs the
// live fleet actually carries.
var schedScriptPathRe = regexp.MustCompile(`(?i)[^\s"';|&,]*\.(?:ps1|psm1|py|cmd|bat|sh|js|pl|rb)\b`)

// schedNormalizePath folds a path for prefix comparison: lowercased, backslashes
// turned into forward slashes, and an MSYS/Git-Bash drive prefix ("/c/work/fak")
// rewritten to its Windows spelling ("c:/work/fak") — the fleet's bash actions use
// the former to name paths inside the very repo the audit is running from.
func schedNormalizePath(p string) string {
	s := strings.ToLower(strings.ReplaceAll(strings.TrimSpace(p), `\`, "/"))
	if len(s) >= 3 && s[0] == '/' && s[2] == '/' && s[1] >= 'a' && s[1] <= 'z' {
		s = string(s[1]) + ":" + s[2:]
	}
	return strings.TrimSuffix(s, "/")
}

// schedVolatileRoots are path fragments that mark a script as NOT repo-owned even
// when it sits physically inside the checkout. _scratch is the sharp case: it is
// under the repo root but untracked, so a reimage restores the task and loses the
// script it points at.
var schedVolatileRoots = []string{
	"%localappdata%", "%appdata%", "%temp%", "%tmp%", "%userprofile%", "%programdata%", "%public%",
	"/appdata/local/temp/", "/appdata/local/", "/appdata/roaming/", "/_scratch/", "/temp/", "/tmp/",
}

// schedScriptProvenance places the launched script on the provenance axis relative
// to repoRoot. Volatile roots are tested FIRST so a script under the checkout's own
// untracked _scratch is called out-of-tree, which is what it is.
func schedScriptProvenance(arguments, repoRoot string) (provenance, script string) {
	m := schedScriptPathRe.FindString(arguments)
	if strings.TrimSpace(m) == "" {
		return schedProvenanceNone, ""
	}
	script = m
	norm := schedNormalizePath(m)
	for _, v := range schedVolatileRoots {
		if strings.Contains(norm, v) || strings.HasPrefix(norm, v) {
			return schedProvenanceOutTree, script
		}
	}
	root := schedNormalizePath(repoRoot)
	if root == "" {
		return schedProvenanceUnknown, script
	}
	if strings.HasPrefix(norm, root+"/") {
		return schedProvenanceInTree, script
	}
	return schedProvenanceOutTree, script
}

// schedLauncherAudit is the whole contract in one place: classify the three axes,
// then rank the findings. Severity order matters and is deliberate.
//
// A desktop-attached principal is the FAILURE: it is the only axis that both puts a
// fault on the operator's screen AND stops the task running at all when nobody is
// logged on (0x800710E0). Everything else is a WARNING — a raw shell or an
// out-of-tree script degrades the blast-radius guarantee without, by itself,
// stopping the loop. An exemption suppresses only the raw-shell finding it was
// written for; a desktop-attached exempted task still fails, because no
// hand-written reason makes an Interactive principal survive a locked screen.
func schedLauncherAudit(row schedScanTaskInfo, repoRoot string) schedLauncherPosture {
	class, wrapped := schedLauncherClassify(row.ActionExecute, row.ActionArguments)
	session := schedSessionIsolation(row.LogonType)
	provenance, script := schedScriptProvenance(row.ActionArguments, repoRoot)

	p := schedLauncherPosture{
		Name:       strings.TrimSpace(row.TaskName),
		Path:       strings.TrimSpace(row.TaskPath),
		Action:     strings.TrimSpace(strings.TrimSpace(row.ActionExecute) + " " + strings.TrimSpace(row.ActionArguments)),
		WorkingDir: strings.TrimSpace(row.ActionWorkingDirectory),
		LogonType:  strings.TrimSpace(row.LogonType),
		RunLevel:   strings.TrimSpace(row.RunLevel),
		NextRun:    strings.TrimSpace(row.NextRunTime),
		Class:      class,
		Wrapped:    wrapped,
		Session:    session,
		Provenance: provenance,
		Script:     script,
	}

	add := func(reason, fix string) {
		p.Reasons = append(p.Reasons, reason)
		if fix != "" {
			p.Remediations = append(p.Remediations, fix)
		}
	}

	verdict := schedVerdictPass
	worse := func(v string) {
		rank := map[string]int{schedVerdictPass: 0, schedVerdictExempt: 1, schedVerdictWarn: 2, schedVerdictFail: 3}
		if rank[v] > rank[verdict] {
			verdict = v
		}
	}

	if session == schedSessionDesktop {
		add("desktop-attached principal (LogonType="+p.LogonType+"): the task runs only inside the operator's logged-on session, "+
			"so a shell/TUI fault lands on the operator desktop and the run is refused (0x800710E0) whenever no session is present",
			"re-register the task with -LogonType S4U -RunLevel Limited -StartWhenAvailable (tools/migrate_fleet_tasks_to_s4u.ps1)")
		worse(schedVerdictFail)
	} else if session == schedSessionUnknown {
		add("principal logon type is unknown, so session isolation cannot be proven",
			"capture the task's Principal (LogonType) before trusting its posture")
		worse(schedVerdictWarn)
	}

	exemption, exempt := schedLauncherExemptions[p.Name]
	switch class {
	case schedLauncherRawShell:
		if exempt {
			add("raw-shell launcher EXEMPTED: "+exemption, "")
			worse(schedVerdictExempt)
		} else {
			add("raw shell launcher ("+schedExeBase(row.ActionExecute)+"): the action is not wrapped in the audited "+
				"`conhost.exe --headless` shim, so the launched shell inherits the scheduler's console context "+
				"(the FleetStrandedRecovery pattern from the 2026-07-01 crash audit)",
				"launch through `conhost.exe --headless <program> ...`, or add the task to schedLauncherExemptions with a reason")
			worse(schedVerdictWarn)
		}
	case schedLauncherUnknown:
		add("the task has no classifiable action program", "capture the task's first Exec action before trusting its posture")
		worse(schedVerdictWarn)
	}

	if provenance == schedProvenanceOutTree {
		add("out-of-tree script ("+script+"): no repo installer recreates it, so the audited launcher contract "+
			"cannot be enforced on the next reimage",
			"move the script into the repo and register it from a tools/register_*.ps1 installer, "+
				"or record the residual gap in internal/taskvc's inventory with a reason")
		worse(schedVerdictWarn)
	} else if provenance == schedProvenanceUnknown {
		add("script "+script+" could not be placed relative to the repo root", "pass --repo-root so provenance can be decided")
		worse(schedVerdictWarn)
	}

	p.Verdict = verdict
	p.Allowed = verdict == schedVerdictPass || verdict == schedVerdictExempt
	return p
}

// buildSchedLauncherDoc audits every filtered row and rolls the fleet up. Failing
// tasks sort first, then warnings, so the report is action-first like the health
// table. withHistory is false for a static source (--xml-dir): a versioned task
// DEFINITION carries no last-run result, and printing the zero value there would
// read as "last run succeeded", which is exactly the kind of trusted zero schedscan
// exists to refuse (#5095).
func buildSchedLauncherDoc(rows []schedScanTaskInfo, filter *regexp.Regexp, repoRoot, source, generatedUTC string, withHistory bool) schedLauncherDoc {
	doc := schedLauncherDoc{
		Schema:       schedLauncherSchema,
		GeneratedUTC: generatedUTC,
		OS:           runtime.GOOS,
		Source:       source,
		RepoRoot:     repoRoot,
		Invariant:    schedLauncherInvariant,
		Counts:       map[string]int{},
		Tasks:        []schedLauncherPosture{},
	}
	if filter != nil {
		doc.Filter = filter.String()
	}
	for _, row := range rows {
		if filter != nil && !filter.MatchString(row.TaskName) {
			continue
		}
		p := schedLauncherAudit(row, repoRoot)
		if withHistory {
			r := decodeSchedTaskResult(row.LastTaskResult)
			p.LastResult = r.Hex + " " + r.Message
		}
		doc.Tasks = append(doc.Tasks, p)
		doc.Counts[p.Verdict]++
	}
	doc.Count = len(doc.Tasks)
	doc.FailCount = doc.Counts[schedVerdictFail]
	doc.WarnCount = doc.Counts[schedVerdictWarn]
	rank := map[string]int{schedVerdictFail: 0, schedVerdictWarn: 1, schedVerdictExempt: 2, schedVerdictPass: 3}
	sort.SliceStable(doc.Tasks, func(i, j int) bool {
		if rank[doc.Tasks[i].Verdict] != rank[doc.Tasks[j].Verdict] {
			return rank[doc.Tasks[i].Verdict] < rank[doc.Tasks[j].Verdict]
		}
		return doc.Tasks[i].Name < doc.Tasks[j].Name
	})
	return doc
}

// schedTaskXMLDoc is the slice of a Windows Task Scheduler XML export the launcher
// audit needs. Go's decoder matches these local names inside the task namespace, so
// no namespace bookkeeping is needed; a task with several principals or actions
// contributes its first of each, which is the one the scheduler runs.
type schedTaskXMLDoc struct {
	XMLName    xml.Name `xml:"Task"`
	URI        string   `xml:"RegistrationInfo>URI"`
	LogonType  string   `xml:"Principals>Principal>LogonType"`
	RunLevel   string   `xml:"Principals>Principal>RunLevel"`
	UserID     string   `xml:"Principals>Principal>UserId"`
	Command    string   `xml:"Actions>Exec>Command"`
	Arguments  string   `xml:"Actions>Exec>Arguments"`
	WorkingDir string   `xml:"Actions>Exec>WorkingDirectory"`
}

// parseSchedTaskXML decodes one versioned task definition into the same row shape
// the live probe emits, so the audit core never learns there are two sources.
// fallbackName (the file's basename) names the task when the export carries no
// <URI>, which the scrubbing in tools/capture_fleet_task_xml.ps1 may strip.
func parseSchedTaskXML(b []byte, fallbackName string) (schedScanTaskInfo, error) {
	if len(b) >= 2 && ((b[0] == 0xFF && b[1] == 0xFE) || (b[0] == 0xFE && b[1] == 0xFF)) {
		return schedScanTaskInfo{}, fmt.Errorf("UTF-16 task XML: re-export it as UTF-8 with tools/capture_fleet_task_xml.ps1")
	}
	var d schedTaskXMLDoc
	if err := xml.Unmarshal(b, &d); err != nil {
		return schedScanTaskInfo{}, err
	}
	name := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(d.URI), `\`))
	if name == "" {
		name = fallbackName
	}
	return schedScanTaskInfo{
		TaskName:               name,
		State:                  "",
		LogonType:              strings.TrimSpace(d.LogonType),
		RunLevel:               strings.TrimSpace(d.RunLevel),
		UserId:                 strings.TrimSpace(d.UserID),
		ActionExecute:          strings.TrimSpace(d.Command),
		ActionArguments:        strings.TrimSpace(d.Arguments),
		ActionWorkingDirectory: strings.TrimSpace(d.WorkingDir),
	}, nil
}

// loadSchedTaskXMLDir reads every *.xml task definition in dir, sorted by name so
// the report is reproducible. A file that fails to parse is reported, not skipped:
// an unreadable task definition is itself a gap in the audit.
func loadSchedTaskXMLDir(dir string) ([]schedScanTaskInfo, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(ents))
	for _, e := range ents {
		if e.IsDir() || !strings.EqualFold(filepath.Ext(e.Name()), ".xml") {
			continue
		}
		names = append(names, e.Name())
	}
	sort.Strings(names)
	rows := make([]schedScanTaskInfo, 0, len(names))
	for _, n := range names {
		b, err := os.ReadFile(filepath.Join(dir, n))
		if err != nil {
			return nil, fmt.Errorf("%s: %w", n, err)
		}
		row, err := parseSchedTaskXML(b, strings.TrimSuffix(n, filepath.Ext(n)))
		if err != nil {
			return nil, fmt.Errorf("%s: %w", n, err)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

// emitSchedLauncherDoc writes the report and returns runSchedScan's exit code,
// keeping the same contract the health half already publishes: 0 clean, 1 an
// encode error, 3 a contract failure under --strict. Only FAIL latches --strict —
// a warning is a real finding an operator must triage (raw shell, out-of-tree
// script), but it does not stop the loop running, so gating CI on it would make the
// gate permanently red and therefore ignored.
func emitSchedLauncherDoc(stdout, stderr io.Writer, doc schedLauncherDoc, jsonOut, strict bool) int {
	if jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(doc); err != nil {
			fmt.Fprintf(stderr, "fak schedscan: encode: %v\n", err)
			return 1
		}
	} else {
		renderSchedLauncherTable(stdout, doc)
	}
	if strict && doc.FailCount > 0 {
		return 3
	}
	return 0
}

// renderSchedLauncherTable prints the operator view: the invariant the report
// defends, the rollup, one row per task with the contract fields, then the reasons
// and remediations for every task that is not a clean pass.
func renderSchedLauncherTable(w io.Writer, doc schedLauncherDoc) {
	fmt.Fprintf(w, "scheduled-task launcher posture: %d task(s), %d fail, %d warn  (%s)\n",
		doc.Count, doc.FailCount, doc.WarnCount, schedScanCountsLine(doc.Counts))
	fmt.Fprintf(w, "invariant: %s\n", schedLauncherInvariant)
	if doc.Count == 0 {
		fmt.Fprintln(w, "  (no tasks matched — try --all or a wider --filter)")
		return
	}
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "VERDICT\tALLOWED\tNAME\tCLASS\tSESSION\tSCRIPT\tWORKDIR\tLAST RESULT\tNEXT RUN")
	for _, t := range doc.Tasks {
		allowed := "no"
		if t.Allowed {
			allowed = "yes"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			strings.ToUpper(t.Verdict), allowed, t.Name, t.Class, t.Session, t.Provenance,
			truncateTableField(schedScanDash(t.WorkingDir), 24),
			truncateTableField(schedScanDash(t.LastResult), 34), schedScanShortTime(t.NextRun))
	}
	tw.Flush()
	for _, t := range doc.Tasks {
		if t.Verdict == schedVerdictPass {
			continue
		}
		fmt.Fprintf(w, "\n  %s [%s] %s\n", strings.ToUpper(t.Verdict), t.Name, t.Action)
		if t.WorkingDir != "" {
			fmt.Fprintf(w, "    working dir: %s\n", t.WorkingDir)
		}
		for _, r := range t.Reasons {
			fmt.Fprintf(w, "    why: %s\n", r)
		}
		for _, r := range t.Remediations {
			fmt.Fprintf(w, "    fix: %s\n", r)
		}
	}
}
