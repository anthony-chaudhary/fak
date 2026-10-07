package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestWatchdogDarwinUnitsHaveEmitter pins that every launchd unit watchdog-autoheal
// expects on darwin is actually produced by something in the tree. A unit nothing
// emits is a silent no-op for the watchdog (not-installed is never restarted), which
// is how com.fleet.stale-work-garden went unwatched. Each unit must be either rendered
// by a Go emitter or shipped as a tracked tools/<label>.plist that the mac node
// installer installs.
//
// fak-test:runtime fast est=50ms
func TestWatchdogDarwinUnitsHaveEmitter(t *testing.T) {
	goEmitters := map[string]func() (string, error){
		gardenLaunchdLabel: func() (string, error) {
			return gardenRenderDarwinPlist("/opt/fak & co/bin/fak", "/work/<repo>", time.Hour, true)
		},
	}
	installer, err := os.ReadFile(filepath.Join("..", "..", "tools", "install-mac-node.sh"))
	if err != nil {
		t.Fatalf("read mac node installer: %v", err)
	}

	units := 0
	for _, svc := range watchdogAutohealServicesForGOOS("darwin") {
		if svc.Manager != "launchd" {
			continue
		}
		units++
		if base := filepath.Base(filepath.ToSlash(svc.UnitPath)); base != svc.Unit+".plist" {
			t.Errorf("%s: UnitPath %q does not name %s.plist", svc.ID, svc.UnitPath, svc.Unit)
		}
		var plist string
		if emit, ok := goEmitters[svc.Unit]; ok {
			if plist, err = emit(); err != nil {
				t.Errorf("%s: Go emitter failed: %v", svc.Unit, err)
				continue
			}
		} else {
			raw, err := os.ReadFile(filepath.Join("..", "..", "tools", svc.Unit+".plist"))
			if err != nil {
				t.Errorf("%s: no Go emitter and no tracked tools/%s.plist template: %v", svc.Unit, svc.Unit, err)
				continue
			}
			if !strings.Contains(string(installer), `install_plist "`+svc.Unit+`"`) {
				t.Errorf("%s: tracked template is not installed by tools/install-mac-node.sh", svc.Unit)
			}
			plist = string(raw)
		}
		if !strings.Contains(plist, "<string>"+svc.Unit+"</string>") {
			t.Errorf("%s: emitted plist does not carry Label %q:\n%s", svc.Unit, svc.Unit, plist)
		}
		assertWellFormedXML(t, plist)
	}
	if units == 0 {
		t.Fatal("darwin projection has no launchd units")
	}
}

// fak-test:runtime fast est=50ms
func TestGardenRenderDarwinPlistEscapesAndCarriesSchedule(t *testing.T) {
	plist, err := gardenRenderDarwinPlist("/opt/fak & co/bin/fak", "/work/<repo>", 2*time.Hour, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"<string>/opt/fak &amp; co/bin/fak</string>",
		"<string>garden</string>",
		"<string>watchdog</string>",
		"<string>/work/&lt;repo&gt;</string>",
		"<string>--live</string>",
		"<integer>7200</integer>",
		"<string>" + gardenLaunchdLabel + "</string>",
	} {
		if !strings.Contains(plist, want) {
			t.Errorf("garden plist missing %q:\n%s", want, plist)
		}
	}
	quiet, err := gardenRenderDarwinPlist("fak", "/r", time.Hour, false)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(quiet, "--live") {
		t.Errorf("non-live garden plist must not pass --live:\n%s", quiet)
	}
	assertWellFormedXML(t, plist)
}
