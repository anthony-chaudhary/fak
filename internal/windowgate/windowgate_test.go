package windowgate

import (
	"os"
	"path/filepath"
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
		{"schtasks-without-IT-clean",
			"schtasks /Create /TN T /SC MINUTE /MO 5 /TR \"python x.py\" /RL LIMITED /F\n", false},
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

// TestTrackedTreeHasNoPopups is the live trunk guard: the real repo's tracked
// .ps1 task installers and window-suppressing .py modules must be clean.
func TestTrackedTreeHasNoPopups(t *testing.T) {
	rep, err := ScanTree(repoRoot(t))
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	for _, v := range rep.PSInstallers {
		t.Errorf("task-installer popup: %s", v)
	}
	for _, v := range rep.PySpawns {
		t.Errorf("unsuppressed spawn: %s", v)
	}
	for _, v := range rep.GoExecs {
		t.Errorf("go exec popup: %s", v)
	}
	if !rep.OK() {
		t.Errorf("fix: make the installer off-desktop (S4U) or headless (conhost --headless); " +
			"flag Python spawns with creationflags=no_window_creationflags(); configure Go dispatch execs")
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
