//go:build windows

package main

import (
	"bytes"
	"os/exec"
	"strings"
	"testing"
)

func TestHostdiagWindowsCollectorAvoidsReadOnlyPIDVariable(t *testing.T) {
	if strings.Contains(hostdiagEventPS, "$pid=") || strings.Contains(hostdiagEventPS, "process_id=$pid") {
		t.Fatal("collector assigns PowerShell's read-only $PID automatic variable")
	}
	if !strings.Contains(hostdiagEventPS, "$processId=0") || !strings.Contains(hostdiagEventPS, "process_id=$processId") {
		t.Fatal("collector does not use bounded processId local")
	}
}
func TestHostdiagWindowsCollectorParses(t *testing.T) {
	script := strings.Replace(hostdiagEventPS, "__MILLIS__", "60000", 1)
	command := `$source=[Console]::In.ReadToEnd(); $errors=$null; [System.Management.Automation.Language.Parser]::ParseInput($source,[ref]$null,[ref]$errors)>$null; if($errors.Count){$errors|ForEach-Object{$_.Message}; exit 1}`
	cmd := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command)
	cmd.Stdin = strings.NewReader(script)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("collector PowerShell does not parse: %v: %s", err, stderr.String())
	}
}

func TestHostdiagWindowsCollectorBoundsHostLifecycleEvidence(t *testing.T) {
	for _, want := range []string{
		"ProviderName='User32';Id=1074", "event_name='HOST_RESTART_INITIATED'",
		"ProviderName='EventLog';Id=6008", "event_name='HOST_UNEXPECTED_SHUTDOWN'",
		"ProviderName='Microsoft-Windows-Kernel-Power';Id=41", "event_name='HOST_UNCLEAN_RESTART'",
	} {
		if !strings.Contains(hostdiagEventPS, want) {
			t.Fatalf("collector missing %q", want)
		}
	}
	for _, name := range []string{"HOST_RESTART_INITIATED", "HOST_UNEXPECTED_SHUTDOWN", "HOST_UNCLEAN_RESTART"} {
		marker := "event_name='" + name + "'"
		at := strings.Index(hostdiagEventPS, marker)
		start := strings.LastIndex(hostdiagEventPS[:at], "[pscustomobject]")
		end := strings.Index(hostdiagEventPS[at:], "}")
		row := hostdiagEventPS[start : at+end+1]
		if strings.Contains(row, "message=") {
			t.Fatalf("%s retained raw message: %s", name, row)
		}
	}
}

func TestHostdiagWindowsCollectorRetainsBoundedGenericApplicationCrashes(t *testing.T) {
	genericBranch := "elseif(-not [string]::IsNullOrWhiteSpace($app) -and -not [string]::IsNullOrWhiteSpace($module) -and -not [string]::IsNullOrWhiteSpace($exception))"
	if !strings.Contains(hostdiagEventPS, genericBranch) {
		t.Fatal("collector does not require the generic application crash identity fields")
	}
	marker := "event_name='WINDOWS_APPLICATION_PROCESS_CRASH'"
	if strings.Count(hostdiagEventPS, marker) != 1 {
		t.Fatalf("generic application crash mapping count = %d, want 1", strings.Count(hostdiagEventPS, marker))
	}
	at := strings.Index(hostdiagEventPS, marker)
	start := strings.LastIndex(hostdiagEventPS[:at], "[pscustomobject]")
	end := strings.Index(hostdiagEventPS[at:], "}")
	row := hostdiagEventPS[start : at+end+1]
	for _, want := range []string{
		"record_id=[string]$_.RecordId", "report_id=[string]$fields.IntegratorReportId", "app=$app", "application_fault=$fault",
	} {
		if !strings.Contains(row, want) {
			t.Fatalf("generic application crash row missing %q: %s", want, row)
		}
	}
	for _, forbidden := range []string{"message=", "process_id=", "process_start_ms="} {
		if strings.Contains(row, forbidden) {
			t.Fatalf("generic application crash row retained %q: %s", forbidden, row)
		}
	}
}

func TestHostdiagWindowsCollectorKeepsSpecializedCrashesUnduplicated(t *testing.T) {
	for _, name := range []string{"POWERSHELL_PROCESS_CRASH", "WINDOWS_SHELL_PROCESS_CRASH"} {
		if count := strings.Count(hostdiagEventPS, "event_name='"+name+"'"); count != 1 {
			t.Fatalf("%s mapping count = %d, want 1", name, count)
		}
	}
	specialized := strings.Index(hostdiagEventPS, "if($app -ieq 'pwsh.exe'")
	generic := strings.Index(hostdiagEventPS, "elseif(-not [string]::IsNullOrWhiteSpace($app)")
	if specialized < 0 || generic < specialized {
		t.Fatal("generic mapping does not follow the specialized mutually exclusive branch")
	}
}

func TestHostdiagWindowsCollectorRetainsBoundedApplicationHangs(t *testing.T) {
	if count := strings.Count(hostdiagEventPS, "ProviderName='Application Hang';Id=1002"); count != 1 {
		t.Fatalf("Application Hang/1002 query count = %d, want 1", count)
	}
	marker := "event_name='WINDOWS_APPLICATION_HANG'"
	if count := strings.Count(hostdiagEventPS, marker); count != 1 {
		t.Fatalf("application hang mapping count = %d, want 1", count)
	}
	at := strings.Index(hostdiagEventPS, marker)
	start := strings.LastIndex(hostdiagEventPS[:at], "[pscustomobject]")
	end := strings.Index(hostdiagEventPS[at:], "}")
	row := hostdiagEventPS[start : at+end+1]
	for _, want := range []string{
		"record_id=[string]$_.RecordId", "report_id=$report", "app=$app", "application_hang=$hang",
	} {
		if !strings.Contains(row, want) {
			t.Fatalf("application hang row missing %q: %s", want, row)
		}
	}
	for _, forbidden := range []string{"message=", "process_id=", "process_start_ms=", "command", "path="} {
		if strings.Contains(strings.ToLower(row), forbidden) {
			t.Fatalf("application hang row retained %q: %s", forbidden, row)
		}
	}
	for _, class := range []string{"Unknown", "Cross-process"} {
		if !strings.Contains(hostdiagEventPS, "'"+class+"'") {
			t.Fatalf("collector does not retain bounded hang class %q", class)
		}
	}
}

func TestHostdiagWindowsCollectorRejectsMalformedApplicationHangs(t *testing.T) {
	for _, required := range []string{"$app", "$version", "$report", "$hangClass"} {
		if !strings.Contains(hostdiagEventPS, "-not [string]::IsNullOrWhiteSpace("+required+")") {
			t.Fatalf("collector does not require application hang field %s", required)
		}
	}
	if !strings.Contains(hostdiagEventPS, "default {''}") {
		t.Fatal("collector does not reject unrecognized hang classifications")
	}
}
