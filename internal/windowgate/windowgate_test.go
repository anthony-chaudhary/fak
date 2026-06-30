package windowgate

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestPSInstallerRules(t *testing.T) {
	cases := []struct {
		name string
		src  string
		bad  bool
	}{
		{"s4u-clean",
			"$p = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U\n" +
				"Register-ScheduledTask -TaskName T -Action $a -Principal $p\n", false},
		{"schtasks-without-IT-fails",
			"schtasks /Create /TN T /SC MINUTE /MO 5 /TR \"python x.py\" /RL LIMITED /F\n", true},
		{"schtasks-system-clean",
			"schtasks /Create /TN T /SC MINUTE /MO 5 /RU SYSTEM /TR \"python x.py\" /RL LIMITED /F\n", false},
		{"conhost-headless-clean-even-if-interactive",
			"$a = New-ScheduledTaskAction -Execute conhost.exe -Argument '--headless powershell.exe -File w.ps1'\n" +
				"$p = New-ScheduledTaskPrincipal -LogonType Interactive\n" +
				"Register-ScheduledTask -TaskName T -Action $a -Principal $p\n", false},
		{"explicit-interactive-without-headless-fails",
			"$a = New-ScheduledTaskAction -Execute python.exe -Argument x.py\n" +
				"$p = New-ScheduledTaskPrincipal -LogonType Interactive\n" +
				"Register-ScheduledTask -TaskName T -Action $a -Principal $p\n", true},
		{"schtasks-with-IT-fails",
			"schtasks /Create /TN T /SC MINUTE /MO 5 /TR \"python x.py\" /IT /F\n", true},
		{"register-without-principal-defaults-interactive-fails",
			"$a = New-ScheduledTaskAction -Execute python.exe -Argument x.py\n" +
				"Register-ScheduledTask -TaskName T -Action $a\n", true},
		{"non-task-ps1-ignored",
			"Write-Output 'helper'\nStart-Process notepad -WindowStyle Hidden\n", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, bad := PSInstallerViolation(c.name+".ps1", c.src)
			if bad != c.bad {
				t.Fatalf("PSInstallerViolation bad=%v, want %v", bad, c.bad)
			}
		})
	}
}

func TestPSStartProcessRules(t *testing.T) {
	src := "Start-Process -FilePath powershell.exe -ArgumentList '-NoProfile'\n"
	if got := PSStartProcessViolations("tools/x.ps1", src); len(got) != 1 {
		t.Fatalf("PSStartProcessViolations = %d %v, want 1", len(got), got)
	}
	hidden := "Start-Process -FilePath powershell.exe `\n  -ArgumentList '-NoProfile' `\n  -WindowStyle Hidden\n"
	if got := PSStartProcessViolations("tools/x.ps1", hidden); len(got) != 0 {
		t.Fatalf("hidden Start-Process should be clean, got %v", got)
	}
	noNewWindow := "Start-Process -FilePath curl.exe -ArgumentList '-L' -NoNewWindow\n"
	if got := PSStartProcessViolations("tools/x.ps1", noNewWindow); len(got) != 0 {
		t.Fatalf("NoNewWindow Start-Process should be clean, got %v", got)
	}
	comment := "<# Start-Process in docs #>\n# Start-Process also ignored\n"
	if got := PSStartProcessViolations("tools/x.ps1", comment); len(got) != 0 {
		t.Fatalf("comments should not be executable Start-Process findings, got %v", got)
	}
}

func TestClassifyLiveScheduledTasks(t *testing.T) {
	rep := ClassifyLiveScheduledTasks([]LiveScheduledTask{
		{
			TaskPath: "\\", TaskName: "Visible", State: "Ready", LogonType: "InteractiveToken",
			Execute: "cmd.exe", Arguments: "/c C:\\work\\fak\\tools\\tick.bat",
		},
		{
			TaskPath: "\\", TaskName: "Headless", State: "Ready", LogonType: "3",
			Execute: "conhost.exe", Arguments: "--headless powershell.exe -WindowStyle Hidden -File C:\\work\\fak\\tools\\tick.ps1",
		},
		{
			TaskPath: "\\", TaskName: "PythonW", State: "Ready", LogonType: "3",
			Execute: "C:\\Python313\\pythonw.exe", Arguments: "C:\\work\\fak\\tools\\guard.py",
		},
		{
			TaskPath: "\\", TaskName: "DisabledVisible", State: "Disabled", LogonType: "3",
			Execute: "bash.exe", Arguments: "-lc C:\\work\\fak\\scripts\\tick.sh",
		},
		{
			TaskPath: "\\", TaskName: "OffDesktop", State: "Ready", LogonType: "S4U",
			Execute: "cmd.exe", Arguments: "/c C:\\work\\fak\\tools\\tick.bat",
		},
		{
			TaskPath: "\\GoogleUserPEH\\", TaskName: "ChromeHelper", State: "Ready", LogonType: "InteractiveToken",
			Execute:   `"C:\Program Files\Google\Chrome\Application\PlatformExperienceHelper\platform_experience_helper.exe"`,
			Arguments: "--chrome-upload-metrics",
		},
	})
	if len(rep.Violations) != 1 {
		t.Fatalf("violations = %d %v, want exactly the visible interactive task", len(rep.Violations), rep.Violations)
	}
	if len(rep.Watchlist) != 3 {
		t.Fatalf("watchlist = %d %v, want hidden/headless/pythonw/disabled rows", len(rep.Watchlist), rep.Watchlist)
	}
}

func TestClassifyVisibleWindows(t *testing.T) {
	rep := ClassifyVisibleWindows([]VisibleWindow{
		{
			PID: 1, Name: "powershell", Title: "worker", Path: `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`,
			CommandLine: `powershell.exe -File C:\work\fak\tools\tick.ps1`,
		},
		{
			PID: 2, Name: "WindowsTerminal", Title: "dev shell",
			CommandLine: `"C:\Program Files\WindowsTerminal.exe"`,
		},
		{
			PID: 3, Name: "chrome", Title: "Apply",
			CommandLine:       `chrome.exe --remote-debugging-port=9223 --window-position=-32000,-32000 --user-data-dir=C:\Users\USER\AppData\Local\Chrome-CDP-Apply-anthony-1 --single-argument https://example.test/oauth?state=abc&code_challenge=def`,
			ParentPID:         26560,
			ParentName:        "python.exe",
			ParentCommandLine: `python.exe C:\work\job\scripts\run_apply_next_xco_tick.py --profile anthony --keep-going`,
		},
		{
			PID: 4, Name: "Slack", Title: "fleet-status",
		},
	})
	if len(rep.Violations) != 1 {
		t.Fatalf("visible violations = %d %v, want repo-owned powershell only", len(rep.Violations), rep.Violations)
	}
	if len(rep.Watchlist) != 2 {
		t.Fatalf("visible watchlist = %d %v, want terminal + browser automation", len(rep.Watchlist), rep.Watchlist)
	}
	if len(rep.Findings) != 3 {
		t.Fatalf("visible findings = %d %+v, want structured rows for classified windows", len(rep.Findings), rep.Findings)
	}
	var browser *VisibleWindowFinding
	for i := range rep.Findings {
		if rep.Findings[i].Category == "browser_automation" {
			browser = &rep.Findings[i]
			break
		}
	}
	if browser == nil {
		t.Fatalf("missing browser automation finding: %+v", rep.Findings)
	}
	if browser.Browser == nil || browser.Browser.RemoteDebuggingPort != "9223" ||
		browser.Browser.Profile != "Chrome-CDP-Apply-anthony-1" || !browser.Browser.Offscreen {
		t.Fatalf("browser details = %+v, want port/profile/offscreen attribution", browser.Browser)
	}
	if browser.ParentPID != 26560 || browser.ParentName != "python.exe" ||
		!strings.Contains(browser.ParentCommandLine, "run_apply_next_xco_tick.py") {
		t.Fatalf("parent attribution = pid %d name %q cmd %q", browser.ParentPID, browser.ParentName, browser.ParentCommandLine)
	}
	if !strings.Contains(browser.Message, "profile=Chrome-CDP-Apply-anthony-1") ||
		!strings.Contains(browser.Message, "parent=python.exe[26560]") {
		t.Fatalf("browser message lacks attribution hints: %s", browser.Message)
	}
	for _, row := range append(rep.Violations, rep.Watchlist...) {
		if strings.Contains(row, "state=abc") || strings.Contains(row, "code_challenge=def") || strings.Contains(row, "https://example.test") {
			t.Fatalf("visible window row leaked URL credentials: %s", row)
		}
	}
	for _, finding := range rep.Findings {
		if strings.Contains(finding.CommandLine, "state=abc") || strings.Contains(finding.CommandLine, "code_challenge=def") || strings.Contains(finding.CommandLine, "https://example.test") {
			t.Fatalf("visible window finding leaked URL credentials: %+v", finding)
		}
	}
}

func TestPySpawnRules(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want int
	}{
		{"non-optin-ignored",
			"import subprocess\nsubprocess.run(['git', 'status'])\n", 0},
		{"optin-missing-flag-flagged",
			"no_window_creationflags = 1\nimport subprocess\nsubprocess.run(['taskkill', '/PID', '1'])\n", 1},
		{"optin-with-flag-clean",
			"from x import no_window_creationflags\nimport subprocess\n" +
				"subprocess.run(['git', 'log'], creationflags=no_window_creationflags())\n", 0},
		{"kwargs-splat-counts",
			"no_window_creationflags = 1\nimport subprocess\nkw = {}\nsubprocess.Popen(['claude'], **kw)\n", 0},
		{"posix-only-exempt",
			"no_window_creationflags = 1\nimport subprocess\nsubprocess.run(['pgrep', '-fa', 'w'])\n", 0},
		{"nested-join-paren-balanced",
			"no_window_creationflags = 1\nimport subprocess\n" +
				"subprocess.run(['pgrep', '-fa', '|'.join(M)])\n", 0},
		{"string-with-paren-does-not-confuse",
			"no_window_creationflags = 1\nimport subprocess\n" +
				"subprocess.run(['echo', 'a)b'], creationflags=cf)\n", 0},
		{"multiline-call-with-flag-clean",
			"no_window_creationflags = 1\nimport subprocess\n" +
				"subprocess.run(\n  ['git', 'log'],\n  capture_output=True,\n  creationflags=cf,\n)\n", 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := PySpawnViolations(c.name+".py", c.src)
			if len(got) != c.want {
				t.Fatalf("PySpawnViolations = %d %v, want %d", len(got), got, c.want)
			}
		})
	}
}

func TestEverySpawnConstructorChecked(t *testing.T) {
	for _, ctor := range []string{"run", "Popen", "call", "check_call", "check_output"} {
		src := "no_window_creationflags = 1\nimport subprocess\nsubprocess." + ctor + "(['git', 'x'])\n"
		if got := PySpawnViolations("m.py", src); len(got) != 1 {
			t.Errorf("ctor %s: got %d violations, want 1", ctor, len(got))
		}
	}
}

func TestPySpawnCandidatesSurfaceNonOptInConsoleTools(t *testing.T) {
	src := "import subprocess\nsubprocess.run(['gh', 'issue', 'list'])\n"
	if got := PySpawnCandidates("m.py", src); len(got) != 1 {
		t.Fatalf("PySpawnCandidates = %d %v, want 1 advisory row", len(got), got)
	}
	clean := "import subprocess\nsubprocess.run(['gh', 'issue', 'list'], creationflags=no_window_creationflags())\n"
	if got := PySpawnCandidates("m.py", clean); len(got) != 0 {
		t.Fatalf("suppressed candidate should be clean, got %v", got)
	}
	optIn := "no_window_creationflags = 1\nimport subprocess\nsubprocess.run(['gh', 'issue', 'list'])\n"
	if got := PySpawnCandidates("m.py", optIn); len(got) != 0 {
		t.Fatalf("opt-in module belongs to hard violation path, got advisory %v", got)
	}
	if got := PySpawnCandidates("tools/m_test.py", src); len(got) != 0 {
		t.Fatalf("test fixtures should not enter the operator watchlist, got %v", got)
	}
	if got := PySpawnCandidates("examples/demo.py", src); len(got) != 0 {
		t.Fatalf("manual examples should not enter the operator watchlist, got %v", got)
	}
	installed := "import subprocess\nfrom dispatch_worker import install_no_window_subprocess_defaults\ninstall_no_window_subprocess_defaults(subprocess)\nsubprocess.run(['gh'])\n"
	if got := PySpawnCandidates("tools/m.py", installed); len(got) != 0 {
		t.Fatalf("installer-covered module should be clean, got %v", got)
	}
	spacedInstalled := "import subprocess\ninstall_no_window_subprocess_defaults( subprocess )\nsubprocess.run(['gh'])\n"
	if got := PySpawnCandidates("tools/m.py", spacedInstalled); len(got) != 0 {
		t.Fatalf("spaced installer call should be clean, got %v", got)
	}
}

func TestPySpawnCandidatesIgnoreStringLiterals(t *testing.T) {
	src := "cell = '''\nsubprocess.run([\"git\", \"status\"])\n'''\n# subprocess.run(['gh'])\n"
	if got := PySpawnCandidates("tools/gen.py", src); len(got) != 0 {
		t.Fatalf("string/comment subprocess text should not be executable watchlist, got %v", got)
	}
}

func TestGoDispatchExecRules(t *testing.T) {
	cases := []struct {
		name string
		rel  string
		src  string
		want int
	}{
		{"non-dispatch-go-ignored",
			"cmd/fak/serve.go",
			"package main\nimport \"os/exec\"\nfunc f(){ cmd := exec.Command(\"git\"); _, _ = cmd.Output() }\n",
			0},
		{"dispatch-helper-missing-hook-flagged",
			"cmd/fak/dispatch_tick.go",
			"package main\nimport \"os/exec\"\nfunc f(){\n cmd := exec.Command(\"git\", \"status\")\n cmd.Dir = \".\"\n _, _ = cmd.Output()\n}\n",
			1},
		{"dispatch-helper-hook-clean",
			"cmd/fak/dispatch_tick.go",
			"package main\nimport \"os/exec\"\nfunc f(){\n cmd := exec.CommandContext(ctx, \"git\", \"status\")\n cmd.Dir = \".\"\n configureDispatchHelperCommand(cmd)\n _, _ = cmd.CombinedOutput()\n}\n",
			0},
		{"dispatch-worker-spawn-clean",
			"cmd/fak/dispatch_tick.go",
			"package main\nimport \"os/exec\"\nfunc f(){\n cmd := exec.Command(exe, args...)\n configureDispatchSpawn(cmd)\n _ = cmd.Start()\n}\n",
			0},
		{"gardenbundle-background-helper-missing-hook-flagged",
			"internal/gardenbundle/gardenbundle.go",
			"package gardenbundle\nimport \"os/exec\"\nfunc f(){\n cmd := exec.Command(\"git\", \"rev-parse\", \"HEAD\")\n _, _ = cmd.Output()\n}\n",
			1},
		{"generic-background-hook-clean",
			"internal/gardenbundle/gardenbundle.go",
			"package gardenbundle\nimport (\n \"os/exec\"\n \"github.com/anthony-chaudhary/fak/internal/windowgate\"\n)\nfunc f(){\n cmd := exec.Command(\"git\", \"rev-parse\", \"HEAD\")\n windowgate.ConfigureBackgroundCommand(cmd)\n _, _ = cmd.Output()\n}\n",
			0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := GoExecViolations(c.rel, c.src)
			if len(got) != c.want {
				t.Fatalf("GoExecViolations = %d %v, want %d", len(got), got, c.want)
			}
		})
	}
}

func TestGoExecCandidatesSurfaceLiteralConsoleTools(t *testing.T) {
	src := "package main\nimport \"os/exec\"\nfunc f(){\n cmd := exec.Command(\"gh\", \"issue\", \"list\")\n _, _ = cmd.Output()\n}\n"
	got := GoExecCandidates("cmd/fak/feature.go", src)
	if len(got) != 1 {
		t.Fatalf("GoExecCandidates = %d %v, want 1 advisory row", len(got), got)
	}
	clean := "package main\nimport \"os/exec\"\nfunc f(){\n cmd := exec.Command(\"gh\", \"issue\", \"list\")\n windowgate.ConfigureBackgroundCommand(cmd)\n _, _ = cmd.Output()\n}\n"
	if got := GoExecCandidates("cmd/fak/feature.go", clean); len(got) != 0 {
		t.Fatalf("configured candidate should be clean, got %v", got)
	}
}

func TestGoExecCandidatesSurfaceInlineConsoleTools(t *testing.T) {
	src := "package main\nimport \"os/exec\"\nfunc f(){\n _, _ = exec.Command(\"git\", \"rev-parse\", \"HEAD\").Output()\n}\n"
	got := GoExecCandidates("internal/x/x.go", src)
	if len(got) != 1 {
		t.Fatalf("GoExecCandidates inline = %d %v, want 1 advisory row", len(got), got)
	}
}

func TestScanTreeIncludesUntrackedGoFiles(t *testing.T) {
	root := t.TempDir()
	runGit(t, root, "init", "-q")
	if err := os.MkdirAll(filepath.Join(root, "cmd", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	src := "package main\nimport \"os/exec\"\nfunc f(){\n cmd := exec.Command(\"gh\", \"issue\", \"list\")\n _, _ = cmd.Output()\n}\n"
	if err := os.WriteFile(filepath.Join(root, "cmd", "fak", "untracked.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	rep, err := ScanTree(root)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(rep.GoCandidates) != 1 {
		t.Fatalf("GoCandidates = %d %v, want the untracked cmd/fak helper observed", len(rep.GoCandidates), rep.GoCandidates)
	}
}

// TestTrackedTreeHasNoPopups is the live trunk guard: the real repo's tracked
// and untracked worktree .ps1 task installers, window-suppressing .py modules,
// and hard-ratcheted Go helpers must be clean.
func TestTrackedTreeHasNoPopups(t *testing.T) {
	rep, err := ScanTree(repoRoot(t))
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	for _, v := range rep.PSInstallers {
		t.Errorf("task-installer popup: %s", v)
	}
	for _, v := range rep.PSStartProcesses {
		t.Errorf("Start-Process popup: %s", v)
	}
	for _, v := range rep.PySpawns {
		t.Errorf("unsuppressed spawn: %s", v)
	}
	for _, v := range rep.GoExecs {
		t.Errorf("go exec popup: %s", v)
	}
	for _, v := range rep.GoCandidates {
		t.Errorf("go exec watchlist: %s", v)
	}
	if !rep.OK() || len(rep.GoCandidates) > 0 {
		t.Errorf("fix: make the installer off-desktop (S4U) or headless (conhost --headless); " +
			"flag Python spawns with creationflags=no_window_creationflags(); configure Go helper execs")
	}
}

func runGit(t *testing.T, root string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	ConfigureBackgroundCommand(cmd)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod found above %s", dir)
		}
		dir = parent
	}
}
