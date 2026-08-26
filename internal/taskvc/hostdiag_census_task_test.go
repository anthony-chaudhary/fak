package taskvc

import (
	"encoding/xml"
	"os"
	"strings"
	"testing"
)

type hostdiagCensusTask struct {
	Principals struct {
		Principal struct {
			LogonType string `xml:"LogonType"`
		} `xml:"Principal"`
	} `xml:"Principals"`
	Settings struct {
		ExecutionTimeLimit      string `xml:"ExecutionTimeLimit"`
		MultipleInstancesPolicy string `xml:"MultipleInstancesPolicy"`
	} `xml:"Settings"`
	Triggers struct {
		TimeTrigger struct {
			Repetition struct {
				Interval string `xml:"Interval"`
			} `xml:"Repetition"`
		} `xml:"TimeTrigger"`
	} `xml:"Triggers"`
	Actions struct {
		Exec struct {
			Command   string `xml:"Command"`
			Arguments string `xml:"Arguments"`
		} `xml:"Exec"`
	} `xml:"Actions"`
}

func TestHostdiagCensusTaskContract(t *testing.T) {
	raw, err := os.ReadFile("../../tools/scheduled-tasks/FakHostdiagCensus.xml")
	if err != nil {
		t.Fatal(err)
	}
	var task hostdiagCensusTask
	if err := xml.Unmarshal(raw, &task); err != nil {
		t.Fatalf("parse task XML: %v", err)
	}
	if got, want := task.Principals.Principal.LogonType, "S4U"; got != want {
		t.Errorf("LogonType = %q, want %q", got, want)
	}
	if got, want := task.Triggers.TimeTrigger.Repetition.Interval, "PT5M"; got != want {
		t.Errorf("interval = %q, want %q", got, want)
	}
	if got, want := task.Settings.MultipleInstancesPolicy, "IgnoreNew"; got != want {
		t.Errorf("MultipleInstancesPolicy = %q, want %q", got, want)
	}
	if got, want := task.Settings.ExecutionTimeLimit, "PT2M"; got != want {
		t.Errorf("ExecutionTimeLimit = %q, want %q", got, want)
	}
	if got, want := task.Actions.Exec.Command, `%USERPROFILE%\bin\fak.exe`; got != want {
		t.Errorf("command = %q, want %q", got, want)
	}
	for _, want := range []string{
		"hostdiag census",
		`--ledger "%LOCALAPPDATA%\fak-hostdiag\hostdiag.jsonl"`,
		"--max-bytes 16777216",
	} {
		if !strings.Contains(task.Actions.Exec.Arguments, want) {
			t.Errorf("arguments missing %q: %q", want, task.Actions.Exec.Arguments)
		}
	}
	for _, forbidden := range []string{"powershell", "pwsh", "cmd.exe", "--since", "correlate"} {
		if strings.Contains(strings.ToLower(task.Actions.Exec.Command+" "+task.Actions.Exec.Arguments), forbidden) {
			t.Errorf("task action contains forbidden %q", forbidden)
		}
	}
}
