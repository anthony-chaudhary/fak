package windowgate

import (
	"strings"
	"testing"
)

// benchSink prevents compiler dead-code elimination across benchmark iterations.
var benchSink int

func BenchmarkScanContent(b *testing.B) {
	cases := []struct {
		name string
		rel  string
		src  string
	}{
		{
			name: "SafePowerShell",
			rel:  "tools/safe_task.ps1",
			src: `$p = New-ScheduledTaskPrincipal -UserId SYSTEM -LogonType ServiceAccount
$a = New-ScheduledTaskAction -Execute "fak.exe" -Argument "serve"
Register-ScheduledTask -TaskName "FakService" -Action $a -Principal $p
Start-Process -FilePath "fak.exe" -ArgumentList "status" -WindowStyle Hidden -NoNewWindow
`,
		},
		{
			name: "UnsafePowerShellInstaller",
			rel:  "tools/unsafe_installer.ps1",
			src: `$a = New-ScheduledTaskAction -Execute "fak.exe" -Argument "serve"
Register-ScheduledTask -TaskName "FakService" -Action $a
`,
		},
		{
			name: "UnsafePowerShellStartProcess",
			rel:  "tools/unsafe_startproc.ps1",
			src: "Write-Output \"launching agent\"\n" +
				"Start" + "-Process -FilePath \"notepad.exe\" -ArgumentList \"log.txt\"\n",
		},
		{
			name: "SafePython",
			rel:  "tools/dispatch_safe.py",
			src: `import subprocess
from tools.window_suppression import no_window_creationflags

def run_worker():
    subprocess.run(["git", "status"], creationflags=no_window_creationflags(), check=True)
`,
		},
		{
			name: "UnsafePython",
			rel:  "tools/dispatch_unsafe.py",
			src: `import subprocess
from tools.window_suppression import no_window_creationflags

def run_worker():
    subprocess.run(["git", "status"], check=True)
`,
		},
		{
			name: "SafeGoSource",
			rel:  "cmd/fak/dispatch_safe.go",
			src: `package fak

import (
	"os/exec"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func runWorker() error {
	cmd := exec.Command("git", "status")
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.Run()
}
`,
		},
		{
			name: "UnsafeGoSource",
			rel:  "cmd/fak/dispatch_unsafe.go",
			src: `package fak

import "os/exec"

func runWorker() error {
	cmd := exec.Command("git", "status")
	return cmd.Run()
}
`,
		},
	}

	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			localSink := 0
			for i := 0; i < b.N; i++ {
				findings := ScanContent(c.rel, c.src)
				localSink += len(findings)
			}
			benchSink = localSink
		})
	}
}

func BenchmarkScanScript(b *testing.B) {
	scriptBodies := []struct {
		name string
		rel  string
		src  string
	}{
		{
			name: "PowerShellClean",
			rel:  "tools/clean.ps1",
			src: `Write-Output "Starting diagnostic..."
$items = Get-ChildItem -Path "." -Filter "*.go"
foreach ($item in $items) {
    Write-Output $item.FullName
}
`,
		},
		{
			name: "PowerShellInteractiveInstaller",
			rel:  "tools/setup.ps1",
			src: `$p = New-ScheduledTaskPrincipal -LogonType Interactive
$a = New-ScheduledTaskAction -Execute "python.exe" -Argument "run.py"
Register-ScheduledTask -TaskName "PeriodicRun" -Action $a -Principal $p
`,
		},
		{
			name: "PowerShellUnsuppressedStartProcess",
			rel:  "tools/watcher.ps1",
			src:  "Start" + "-Process -FilePath powershell.exe -ArgumentList \"-NoProfile -File worker.ps1\"\n",
		},
		{
			name: "PowerShellBlockCommentsProse",
			rel:  "tools/doc_exporter.ps1",
			src: "<#\n" +
				"Restore instruction:\n" +
				"  Register-ScheduledTask -TaskName T -Xml (Get-Content -Raw t.xml)\n" +
				"  Start" + "-Process notepad.exe\n" +
				"#>\n" +
				"Write-Output \"Document exporter loaded.\"\n" +
				"# Register-ScheduledTask -TaskName Commented -Action $a\n",
		},
		{
			name: "PythonClean",
			rel:  "tools/formatter.py",
			src: `import os
import sys

def format_all():
    print("Formatting repository files...")
`,
		},
		{
			name: "PythonOptInSuppressed",
			rel:  "tools/sync_worker.py",
			src: `import subprocess
from tools.window_suppression import no_window_creationflags

def sync():
    p = subprocess.Popen(["git", "pull"], creationflags=no_window_creationflags())
    p.wait()
`,
		},
		{
			name: "PythonOptInUnsuppressed",
			rel:  "tools/sync_worker.py",
			src: `import subprocess
from tools.window_suppression import no_window_creationflags

def sync():
    p = subprocess.Popen(["git", "pull"])
    p.wait()
`,
		},
		{
			name: "LargeScriptMultiline",
			rel:  "tools/big_deploy.ps1",
			src: strings.Repeat(`Write-Output "step"
$a = New-ScheduledTaskAction -Execute "fak.exe"
# Register-ScheduledTask -TaskName FalseAlarm
Start-Process -FilePath "fak.exe" -ArgumentList "status" -WindowStyle Hidden -NoNewWindow
`, 25),
		},
	}

	for _, sc := range scriptBodies {
		b.Run(sc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			localSink := 0
			for i := 0; i < b.N; i++ {
				violations := ScanScript(sc.rel, sc.src)
				localSink += len(violations)
			}
			benchSink = localSink
		})
	}
}

func BenchmarkGoExecScan(b *testing.B) {
	goSources := []struct {
		name string
		rel  string
		src  string
	}{
		{
			name: "SafeConfiguredBackground",
			rel:  "cmd/fak/dispatch_probe.go",
			src: `package fak

import (
	"os/exec"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func runProbe() ([]byte, error) {
	cmd := exec.Command("git", "rev-parse", "HEAD")
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.Output()
}
`,
		},
		{
			name: "SafeWindowgateConstructor",
			rel:  "cmd/fak/dispatch_probe.go",
			src: `package fak

import "github.com/anthony-chaudhary/fak/internal/windowgate"

func runProbe() ([]byte, error) {
	cmd := windowgate.Command("git", "rev-parse", "HEAD")
	return cmd.Output()
}
`,
		},
		{
			name: "UnsafeUnsuppressedHelper",
			rel:  "cmd/fak/dispatch_runner.go",
			src: `package fak

import "os/exec"

func runRunner() ([]byte, error) {
	cmd := exec.Command("git", "rev-parse", "HEAD")
	return cmd.Output()
}
`,
		},
		{
			name: "UnsafeInlineTerm",
			rel:  "cmd/fak/dispatch_inline.go",
			src: `package fak

import "os/exec"

func runInline() error {
	return exec.Command("git", "fetch").Run()
}
`,
		},
		{
			name: "CleanNonBackgroundGo",
			rel:  "internal/windowgate/clean.go",
			src: `package windowgate

import "fmt"

func ProcessData(items []string) int {
	count := 0
	for _, item := range items {
		if len(item) > 0 {
			count++
		}
	}
	return count
}
`,
		},
		{
			name: "LargeGoFileMultiFunc",
			rel:  "cmd/fak/dispatch_complex.go",
			src: `package fak

import (
	"os/exec"
	"strings"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Helper 1: safe
func helper1() error {
	cmd := exec.Command("git", "status")
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.Run()
}

// Helper 2: comments and strings with exec.Command
func helper2() string {
	msg := "exec.Command(\"git\")"
	// exec.Command in comment
	return strings.ToUpper(msg)
}

// Helper 3: unsafe helper
func helper3() ([]byte, error) {
	cmd := exec.Command("fak.exe", "status")
	return cmd.CombinedOutput()
}

// Helper 4: safe helper
func helper4() error {
	cmd := exec.Command("go", "test")
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.Run()
}
`,
		},
	}

	for _, tc := range goSources {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			localSink := 0
			for i := 0; i < b.N; i++ {
				findings := ScanGoFileForExec(tc.rel, tc.src)
				localSink += len(findings)
			}
			benchSink = localSink
		})
	}
}
