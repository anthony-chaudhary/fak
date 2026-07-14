package repoguard

import "testing"

func TestForegroundPowerShellInventory(t *testing.T) {
	cases := []struct {
		name string
		cmd  string
		want bool
	}{
		{"unbounded events", `Get-WinEvent -LogName System`, true},
		{"unbounded cim", `Get-CimInstance Win32_Process`, true},
		{"unbounded wmi", `Get-WmiObject Win32_Service`, true},
		{"events max", `Get-WinEvent -LogName System -MaxEvents 100`, false},
		{"events server filter", `Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddHours(-1)}`, false},
		{"cim filter", `Get-CimInstance Win32_Process -Filter "Name='git.exe'"`, false},
		{"cim query", `Get-CimInstance -Query 'select * from Win32_Process where Name="git.exe"'`, false},
		{"pipeline bound", `Get-CimInstance Win32_Process | Select-Object -First 20`, false},
		{"background job", `Start-Job { Get-WinEvent -LogName System }`, false},
		{"as job", `Get-CimInstance Win32_Process -AsJob`, false},
		{"ordinary command", `Get-Process git`, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyForegroundPowerShellInventory(tc.cmd)
			if (len(got) > 0) != tc.want {
				t.Fatalf("findings=%+v wantFinding=%v", got, tc.want)
			}
			if tc.want && (got[0].Reason != ReasonForegroundPowerShellInventory || got[0].Fix == "") {
				t.Fatalf("finding is not actionable: %+v", got[0])
			}
		})
	}
}
