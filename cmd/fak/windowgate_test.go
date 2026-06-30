package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func TestBuildWindowgatePayloadKeepsCandidatesAdvisoryByDefault(t *testing.T) {
	rep := windowgate.Report{
		PyCandidates:      []string{"tools/x.py:1: subprocess gh launch is on the desktop-popup watchlist"},
		GoCandidates:      []string{"cmd/fak/x.go:1: exec.Command(\"gh\", \"issue\") reaches cmd.Output()"},
		PyExplicitModules: []string{"tools/flagged.py"},
		PyDefaultModules:  []string{"tools/defaulted.py"},
	}
	p := buildWindowgatePayload("root", rep, false)
	if !p.OK || p.Verdict != "OK" || p.Finding != "no_desktop_popup_watchlist" {
		t.Fatalf("payload = ok %v verdict %q finding %q, want advisory OK watchlist", p.OK, p.Verdict, p.Finding)
	}
	if p.Counts["py_watchlist"] != 1 || p.Counts["go_watchlist"] != 1 || len(p.Watchlist) != 2 {
		t.Fatalf("watchlist not surfaced: %+v", p)
	}
	if p.Tools["gh"] != 2 {
		t.Fatalf("tool summary = %+v, want gh=2", p.Tools)
	}
	if p.Files["tools/x.py"] != 1 || p.Files["cmd/fak/x.go"] != 1 {
		t.Fatalf("file summary = %+v, want both watchlist files", p.Files)
	}
	if p.Dirs["tools"] != 1 || p.Dirs["cmd/fak"] != 1 {
		t.Fatalf("dir summary = %+v, want tools=1 cmd/fak=1", p.Dirs)
	}
	if p.Suppression["py_explicit_modules"] != 1 || p.Suppression["py_default_modules"] != 1 {
		t.Fatalf("suppression summary = %+v, want explicit/defaulted adoption", p.Suppression)
	}
}

func TestBuildWindowgatePayloadHardViolationFails(t *testing.T) {
	rep := windowgate.Report{GoExecs: []string{"internal/gardenbundle/gardenbundle.go:1: missing hook"}}
	p := buildWindowgatePayload("root", rep, false)
	if p.OK || p.Verdict != "ACTION" || p.Finding != "no_desktop_popup_regression" {
		t.Fatalf("payload = ok %v verdict %q finding %q, want hard ACTION", p.OK, p.Verdict, p.Finding)
	}
	if !strings.Contains(p.NextAction, "ConfigureBackgroundCommand") {
		t.Fatalf("next action does not point at runtime helper: %q", p.NextAction)
	}
}

func TestBuildWindowgatePayloadCanFailCandidates(t *testing.T) {
	rep := windowgate.Report{GoCandidates: []string{"cmd/fak/x.go:1: gh helper"}}
	p := buildWindowgatePayload("root", rep, true)
	if p.OK || p.Verdict != "ACTION" || p.Finding != "no_desktop_popup_watchlist" {
		t.Fatalf("payload = ok %v verdict %q finding %q, want candidate ACTION", p.OK, p.Verdict, p.Finding)
	}
}

func TestAttachLiveTaskPayloadFailsVisibleTasks(t *testing.T) {
	p := buildWindowgatePayload("root", windowgate.Report{}, false)
	attachLiveTaskPayload(&p, windowgate.LiveTaskReport{
		Scanned:    2,
		Violations: []string{"\\Visible: cmd.exe can flash"},
		Watchlist:  []string{"\\Hidden: review child spawns"},
	}, false)
	if p.OK || p.Verdict != "ACTION" || p.Finding != "no_desktop_popup_live_task_regression" {
		t.Fatalf("payload = ok %v verdict %q finding %q, want live-task ACTION", p.OK, p.Verdict, p.Finding)
	}
	if p.LiveTasks == nil || p.LiveTasks.Scanned != 2 || len(p.LiveTasks.Violations) != 1 || len(p.LiveTasks.Watchlist) != 1 {
		t.Fatalf("live task payload not surfaced: %+v", p.LiveTasks)
	}
}

func TestAttachLiveTaskPayloadCanFailWatchlist(t *testing.T) {
	p := buildWindowgatePayload("root", windowgate.Report{}, false)
	attachLiveTaskPayload(&p, windowgate.LiveTaskReport{
		Scanned:   1,
		Watchlist: []string{"\\Hidden: review child spawns"},
	}, true)
	if p.OK || p.Verdict != "ACTION" || p.Finding != "no_desktop_popup_live_task_watchlist" {
		t.Fatalf("payload = ok %v verdict %q finding %q, want live watchlist ACTION", p.OK, p.Verdict, p.Finding)
	}
}
