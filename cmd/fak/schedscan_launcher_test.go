package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// capturedTaskRepoRoot is the checkout the versioned task definitions under
// tools/scheduled-tasks/ were exported from. tools/capture_fleet_task_xml.ps1
// scrubs the host SID and home paths but deliberately keeps the repo path, so the
// captures still name it. The provenance axis needs a root to compare against, and
// pinning it here keeps the corpus witness host-independent: the classification is
// pure string work, so it decides the same way under Windows, WSL and CI.
const capturedTaskRepoRoot = `C:\work\fak`

func capturedTaskDir() string { return filepath.Join("..", "..", "tools", "scheduled-tasks") }

// TestSchedLauncherClassify pins the launcher axis against the action shapes the
// live fleet actually carries — every case below is copied from a versioned
// capture under tools/scheduled-tasks/ or from the 2026-07-01 crash audit.
//
// The load-bearing distinction is conhost WITH --headless versus conhost without:
// only the first gives the wrapped program its own pseudoconsole, and the second is
// the console-allocating shape #1456 fixed. A classifier that merges them would
// hand the approved-launcher credit to the exact pattern this audit exists to find.
func TestSchedLauncherClassify(t *testing.T) {
	cases := []struct {
		name        string
		exe, args   string
		wantClass   string
		wantWrapped string
	}{
		{"conhost headless powershell (FleetResumeWatchdog, the safe pattern)",
			`C:\WINDOWS\System32\conhost.exe`,
			`--headless powershell.exe -NoProfile -WindowStyle Hidden -File "%LOCALAPPDATA%\x\run.ps1"`,
			schedLauncherHeadless, "powershell"},
		{"conhost headless quoted bash (FakFleetJanitorHeadless)",
			"conhost.exe",
			`--headless "C:\Program Files\Git\bin\bash.exe" -lc "/c/work/fak/scripts/gcp-fleet-janitor.sh --live"`,
			schedLauncherHeadless, "bash"},
		{"conhost headless quoted python (UserSeatDrain-1010)",
			`C:\WINDOWS\System32\conhost.exe`,
			`--headless "C:\Program Files\Python313\python.exe" "C:\work\fak\tools\_watchdog\operator_user_seat_drain.py"`,
			schedLauncherHeadless, "python"},
		{"raw powershell hidden (FleetStrandedRecovery, the audited regression)",
			"powershell.exe",
			`-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%LOCALAPPDATA%\Fleet\stranded_recovery.ps1"`,
			schedLauncherRawShell, ""},
		{"raw python (ClaudeAccountBackup)",
			`C:\Program Files\Python313\python.exe`, `"C:\work\fak\tools\claude_account_backup.py" backup`,
			schedLauncherRawShell, ""},
		{"raw cmd", "cmd.exe", `/d /s /c "tick.cmd"`, schedLauncherRawShell, ""},
		{"conhost WITHOUT --headless still allocates a console", "conhost.exe", `cmd.exe /c echo hi`,
			schedLauncherRawShell, ""},
		{"a compiled program is not a shell", `C:\work\fak\fak.exe`, `schedscan --strict`, schedLauncherProgram, ""},
		{"no action program at all", "", "", schedLauncherUnknown, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gotClass, gotWrapped := schedLauncherClassify(c.exe, c.args)
			if gotClass != c.wantClass {
				t.Errorf("class = %q, want %q", gotClass, c.wantClass)
			}
			if gotWrapped != c.wantWrapped {
				t.Errorf("wrapped = %q, want %q", gotWrapped, c.wantWrapped)
			}
		})
	}
}

// TestSchedSessionIsolation pins the session axis. Task Scheduler spells the
// interactive family several ways and every one of them means "needs a logged-on
// desktop", which is both the blast-radius coupling and the 0x800710E0 refusal.
func TestSchedSessionIsolation(t *testing.T) {
	for _, lt := range []string{"Interactive", "InteractiveToken", "InteractiveTokenOrPassword", "interactive"} {
		if got := schedSessionIsolation(lt); got != schedSessionDesktop {
			t.Errorf("schedSessionIsolation(%q) = %q, want %q", lt, got, schedSessionDesktop)
		}
	}
	for _, lt := range []string{"S4U", "s4u", "Password", "ServiceAccount"} {
		if got := schedSessionIsolation(lt); got != schedSessionIsolated {
			t.Errorf("schedSessionIsolation(%q) = %q, want %q", lt, got, schedSessionIsolated)
		}
	}
	if got := schedSessionIsolation(""); got != schedSessionUnknown {
		t.Errorf("empty logon type = %q, want %q", got, schedSessionUnknown)
	}
}

// TestSchedScriptProvenance pins the provenance axis, including the two cases that
// make it worth having: an untracked _scratch script physically inside the checkout
// is NOT repo-owned, and an MSYS-style /c/work/... path names the same tree a
// Windows C:\work\... path does.
func TestSchedScriptProvenance(t *testing.T) {
	cases := []struct {
		name, args string
		wantProv   string
		wantScript string
	}{
		{"in-tree windows path", `"C:\work\fak\tools\claude_account_backup.py" backup`,
			schedProvenanceInTree, `C:\work\fak\tools\claude_account_backup.py`},
		{"in-tree msys path", `-lc "PROJECT=x /c/work/fak/scripts/gcp-fleet-janitor.sh --live"`,
			schedProvenanceInTree, "/c/work/fak/scripts/gcp-fleet-janitor.sh"},
		{"volatile user root", `-File "%LOCALAPPDATA%\Fleet\stranded_recovery.ps1"`,
			schedProvenanceOutTree, `%LOCALAPPDATA%\Fleet\stranded_recovery.ps1`},
		{"untracked _scratch inside the checkout is still out-of-tree",
			`-File "C:\work\fak\_scratch\run-meta-superloop-100.ps1"`,
			schedProvenanceOutTree, `C:\work\fak\_scratch\run-meta-superloop-100.ps1`},
		{"another checkout entirely", `"C:\work\fleet\tools\worktree_doctor.py" --repo x`,
			schedProvenanceOutTree, `C:\work\fleet\tools\worktree_doctor.py`},
		{"no script at all", `schedscan --strict`, schedProvenanceNone, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gotProv, gotScript := schedScriptProvenance(c.args, capturedTaskRepoRoot)
			if gotProv != c.wantProv {
				t.Errorf("provenance = %q, want %q", gotProv, c.wantProv)
			}
			if gotScript != c.wantScript {
				t.Errorf("script = %q, want %q", gotScript, c.wantScript)
			}
		})
	}
}

// TestSchedLauncherAuditCatchesTheAuditedRegressions is the acceptance witness for
// #2174: the audit must catch BOTH regression classes the ticket names, and must
// not cry wolf on the pattern that was already correct.
//
//   - #1456 class: a task pinned to the operator's desktop session. That is a FAIL
//     however cleanly it is shimmed — no headless wrapper makes an Interactive
//     principal survive a locked screen.
//   - FleetStrandedRecovery class: an S4U task that nonetheless launches a bare
//     shell at an out-of-tree script. windowgate's live classifier does not see this
//     one (it only inspects INTERACTIVE tasks, and it counts `-WindowStyle Hidden`
//     as windowless), which is exactly why the launcher axis is scored separately.
func TestSchedLauncherAuditCatchesTheAuditedRegressions(t *testing.T) {
	t.Run("desktop-attached task fails even behind a headless shim", func(t *testing.T) {
		p := schedLauncherAudit(schedScanTaskInfo{
			TaskName:        "FakMetaSuperloopNight100",
			LogonType:       "InteractiveToken",
			ActionExecute:   "conhost.exe",
			ActionArguments: `--headless powershell.exe -NoProfile -File "C:\work\fak\tools\x.ps1"`,
		}, capturedTaskRepoRoot)
		if p.Verdict != schedVerdictFail || p.Allowed {
			t.Fatalf("verdict = %q allowed = %v, want fail/not-allowed\n%+v", p.Verdict, p.Allowed, p)
		}
		if p.Class != schedLauncherHeadless || p.Session != schedSessionDesktop {
			t.Errorf("class/session = %q/%q, want %q/%q", p.Class, p.Session, schedLauncherHeadless, schedSessionDesktop)
		}
		if !strings.Contains(strings.Join(p.Remediations, " "), "S4U") {
			t.Errorf("remediation must name the S4U migration, got %q", p.Remediations)
		}
	})

	t.Run("S4U raw shell at an out-of-tree script is surfaced, not assumed safe", func(t *testing.T) {
		p := schedLauncherAudit(schedScanTaskInfo{
			TaskName:        "FleetStrandedRecovery",
			LogonType:       "S4U",
			ActionExecute:   "powershell.exe",
			ActionArguments: `-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%LOCALAPPDATA%\Fleet\stranded_recovery.ps1"`,
		}, capturedTaskRepoRoot)
		if p.Verdict != schedVerdictWarn || p.Allowed {
			t.Fatalf("verdict = %q allowed = %v, want warn/not-allowed\n%+v", p.Verdict, p.Allowed, p)
		}
		joined := strings.Join(p.Reasons, " | ")
		if !strings.Contains(joined, "raw shell launcher") {
			t.Errorf("missing the raw-shell finding: %s", joined)
		}
		if !strings.Contains(joined, "out-of-tree script") {
			t.Errorf("missing the out-of-tree finding: %s", joined)
		}
		if !strings.Contains(strings.Join(p.Remediations, " "), "--headless") {
			t.Errorf("remediation must name the headless shim, got %q", p.Remediations)
		}
	})

	t.Run("the approved pattern passes clean", func(t *testing.T) {
		p := schedLauncherAudit(schedScanTaskInfo{
			TaskName:        "FleetResumeWatchdog",
			LogonType:       "S4U",
			ActionExecute:   `C:\WINDOWS\System32\conhost.exe`,
			ActionArguments: `--headless powershell.exe -NoProfile -File "C:\work\fak\tools\resume_watchdog.ps1"`,
		}, capturedTaskRepoRoot)
		if p.Verdict != schedVerdictPass || !p.Allowed {
			t.Fatalf("verdict = %q allowed = %v, want pass/allowed\nreasons: %q", p.Verdict, p.Allowed, p.Reasons)
		}
	})

	t.Run("an exemption allows the task and carries its reason", func(t *testing.T) {
		name := "ClaudeAccountBackup"
		reason, ok := schedLauncherExemptions[name]
		if !ok {
			t.Fatalf("schedLauncherExemptions must keep at least one worked example so the "+
				"exempt-with-a-reason path stays exercised; %s is gone", name)
		}
		p := schedLauncherAudit(schedScanTaskInfo{
			TaskName:        name,
			LogonType:       "S4U",
			ActionExecute:   `C:\Program Files\Python313\python.exe`,
			ActionArguments: `"C:\work\fak\tools\claude_account_backup.py" backup`,
		}, capturedTaskRepoRoot)
		if p.Verdict != schedVerdictExempt || !p.Allowed {
			t.Fatalf("verdict = %q allowed = %v, want exempt/allowed\n%+v", p.Verdict, p.Allowed, p)
		}
		if !strings.Contains(strings.Join(p.Reasons, " "), reason) {
			t.Errorf("the exemption reason must travel with the row, got %q", p.Reasons)
		}
	})

	t.Run("an exemption cannot launder a desktop-attached principal", func(t *testing.T) {
		p := schedLauncherAudit(schedScanTaskInfo{
			TaskName:        "ClaudeAccountBackup",
			LogonType:       "Interactive",
			ActionExecute:   `C:\Program Files\Python313\python.exe`,
			ActionArguments: `"C:\work\fak\tools\claude_account_backup.py" backup`,
		}, capturedTaskRepoRoot)
		if p.Verdict != schedVerdictFail || p.Allowed {
			t.Fatalf("verdict = %q allowed = %v, want fail/not-allowed (an exemption covers the "+
				"launcher axis only)\n%+v", p.Verdict, p.Allowed, p)
		}
	})
}

// TestParseSchedTaskXML proves a versioned task definition decodes into the same
// row shape the live probe emits — that equivalence is what lets the pass/fail
// report run in CI on any OS instead of only on the one Windows box.
func TestParseSchedTaskXML(t *testing.T) {
	b, err := os.ReadFile(filepath.Join(capturedTaskDir(), "FleetStrandedRecovery.xml"))
	if err != nil {
		t.Fatalf("read capture: %v", err)
	}
	row, err := parseSchedTaskXML(b, "fallback")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if row.TaskName != "FleetStrandedRecovery" {
		t.Errorf("task name = %q, want FleetStrandedRecovery (taken from <URI>)", row.TaskName)
	}
	if row.LogonType != "S4U" {
		t.Errorf("logon type = %q, want S4U", row.LogonType)
	}
	if schedExeBase(row.ActionExecute) != "powershell" {
		t.Errorf("action execute = %q, want the raw powershell launcher", row.ActionExecute)
	}
	if !strings.Contains(row.ActionArguments, "stranded_recovery.ps1") {
		t.Errorf("action arguments = %q, want the recovery script", row.ActionArguments)
	}

	if _, err := parseSchedTaskXML([]byte{0xFF, 0xFE, 0x3C, 0x00}, "x"); err == nil {
		t.Error("a UTF-16 export must be refused with a re-export instruction, not silently parsed as empty")
	}
}

// TestSchedLauncherReportOverVersionedTaskDefinitions is the single command the
// acceptance contract asks for, run end to end:
//
//	fak schedscan --launchers --xml-dir tools/scheduled-tasks --strict
//
// It must produce a per-task pass/fail report over the repo-owned task definitions,
// exit 3 while any of them still violates the contract, and carry the #2170
// invariant the posture is defended for.
func TestSchedLauncherReportOverVersionedTaskDefinitions(t *testing.T) {
	dir := capturedTaskDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read capture dir: %v", err)
	}
	want := 0
	for _, e := range entries {
		if !e.IsDir() && strings.EqualFold(filepath.Ext(e.Name()), ".xml") {
			want++
		}
	}
	if want == 0 {
		t.Fatal("no versioned task definitions to audit")
	}

	var out, errBuf bytes.Buffer
	code := runSchedScan(&out, &errBuf, []string{"--launchers", "--json", "--strict",
		"--xml-dir", dir, "--repo-root", capturedTaskRepoRoot})
	if errBuf.Len() > 0 {
		t.Fatalf("stderr: %s", errBuf.String())
	}

	var doc schedLauncherDoc
	if err := json.Unmarshal(out.Bytes(), &doc); err != nil {
		t.Fatalf("decode report: %v\n%s", err, out.String())
	}
	if doc.Schema != schedLauncherSchema {
		t.Errorf("schema = %q, want %q", doc.Schema, schedLauncherSchema)
	}
	if doc.Count != want {
		t.Errorf("audited %d task definitions, want all %d (the Fak*/Fleet*/User* name filter "+
			"must not silently drop repo-owned captures)", doc.Count, want)
	}
	// The invariant this posture defends must ship with the report, not live only in
	// the ticket: an operator reading a FAIL has to know what it costs.
	if !strings.Contains(doc.Invariant, "#2170") {
		t.Errorf("report must link launcher posture to #2170's invariant, got %q", doc.Invariant)
	}
	if !strings.Contains(strings.ToLower(doc.Invariant), "crash") && !strings.Contains(strings.ToLower(doc.Invariant), "block") {
		t.Errorf("invariant must state the shell/TUI failure cannot crash or block the program: %q", doc.Invariant)
	}

	byName := map[string]schedLauncherPosture{}
	for _, p := range doc.Tasks {
		byName[p.Name] = p
		// A definition has no run history; reporting a decoded 0x0 there would read as
		// "the last run succeeded", the exact trusted zero schedscan refuses (#5095).
		if p.LastResult != "" {
			t.Errorf("%s: static definition must not report a last result, got %q", p.Name, p.LastResult)
		}
	}

	// Regression class 1 (#1456): every desktop-attached capture is a FAIL.
	desktop := 0
	for _, p := range doc.Tasks {
		if p.Session != schedSessionDesktop {
			continue
		}
		desktop++
		if p.Verdict != schedVerdictFail || p.Allowed {
			t.Errorf("%s: logon=%s verdict=%s allowed=%v — a desktop-attached task must fail the contract",
				p.Name, p.LogonType, p.Verdict, p.Allowed)
		}
	}
	if desktop == 0 {
		t.Log("no desktop-attached captures remain — the #1456 class is clean in version control")
	}

	// Regression class 2: the raw FleetStrandedRecovery launcher.
	if p, ok := byName["FleetStrandedRecovery"]; !ok {
		t.Error("FleetStrandedRecovery capture missing — the named regression is no longer under audit")
	} else if p.Class != schedLauncherRawShell || p.Allowed {
		t.Errorf("FleetStrandedRecovery: class=%s allowed=%v — the raw shell launcher must be surfaced",
			p.Class, p.Allowed)
	}

	// The approved pattern must still read as approved, or the gate is noise.
	if p, ok := byName["FakFleetJanitorHeadless"]; !ok {
		t.Error("FakFleetJanitorHeadless capture missing")
	} else if p.Class != schedLauncherHeadless || p.Session != schedSessionIsolated || !p.Allowed {
		t.Errorf("FakFleetJanitorHeadless: class=%s session=%s allowed=%v — S4U + conhost --headless "+
			"at an in-tree script is the approved path", p.Class, p.Session, p.Allowed)
	}

	// --strict is the pass/fail gate: exit 3 exactly when a definition fails.
	wantCode := 0
	if doc.FailCount > 0 {
		wantCode = 3
	}
	if code != wantCode {
		t.Errorf("exit code = %d, want %d (fail_count=%d)", code, wantCode, doc.FailCount)
	}
	if doc.FailCount+doc.WarnCount == 0 {
		t.Error("the audit found nothing at all in a corpus that provably contains the audited " +
			"regressions — the classifier is not wired to the report")
	}
}

// TestSchedLauncherTableCarriesTheContractFields pins the human report's payload:
// the acceptance contract names task name, action, working directory, last result,
// next run and the allow/deny verdict, and an operator gets the table, not the JSON.
func TestSchedLauncherTableCarriesTheContractFields(t *testing.T) {
	doc := buildSchedLauncherDoc([]schedScanTaskInfo{{
		TaskName:               "UserSeatDrain-1010",
		LogonType:              "InteractiveToken",
		LastTaskResult:         0x800710E0,
		NextRunTime:            "2026-08-12T10:30:00.0000000+00:00",
		ActionExecute:          `C:\WINDOWS\System32\conhost.exe`,
		ActionArguments:        `--headless "C:\Program Files\Python313\python.exe" "C:\work\fak\tools\_watchdog\operator_user_seat_drain.py"`,
		ActionWorkingDirectory: `C:\work\fak`,
	}}, nil, capturedTaskRepoRoot, "unit", "2026-08-12T00:00:00Z", true)

	var buf bytes.Buffer
	renderSchedLauncherTable(&buf, doc)
	got := buf.String()
	for _, want := range []string{
		"UserSeatDrain-1010",                   // task name
		"operator_user_seat_drain.py",          // action
		`working dir: C:\work\fak`,             // working directory
		"0x800710E0",                           // last result
		"2026-08-12 10:30",                     // next run
		"FAIL",                                 // whether the launcher contract allows it
		"#2170",                                // the invariant it is measured against
		"conhost.exe --headless",               // the remediation vocabulary
		"tools/migrate_fleet_tasks_to_s4u.ps1", // the concrete fix
	} {
		if !strings.Contains(got, want) {
			t.Errorf("launcher table missing %q\n---\n%s", want, got)
		}
	}
}
