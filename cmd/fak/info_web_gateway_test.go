package main

import (
	"errors"
	"os/exec"
	"reflect"
	"strings"
	"testing"
)

func TestInfoWebGatewayButtonCapturedRender(t *testing.T) {
	bar := buildInfoTabBar(viewOverview, false)
	const button = "[w web gateway]"
	if !strings.Contains(bar.text, button) {
		t.Fatalf("interactive TUI must render a dedicated web-gateway button:\n%s", bar.text)
	}

	var region infoTabRegion
	found := false
	for _, candidate := range bar.regions {
		if candidate.term == "web-gateway" {
			region, found = candidate, true
			break
		}
	}
	if !found {
		t.Fatalf("rendered button has no clickable region: %#v", bar.regions)
	}
	state := applyInfoClick(infoViewState{}, region.start, 1)
	if !state.launchWeb {
		t.Fatal("clicking the web-gateway button did not request a launch")
	}
}

func TestInfoWebGatewayKeyboardShortcut(t *testing.T) {
	for _, key := range []byte{'w', 'W'} {
		var scanner infoInputScanner
		event := scanner.step(key)
		if event.Kind != infoInputLaunchHarnessWeb {
			t.Fatalf("key %q decoded as %v, want web-gateway launch", key, event.Kind)
		}
		if state := applyInfoInput(infoViewState{}, event); !state.launchWeb {
			t.Fatalf("key %q did not request a launch", key)
		}
	}
}

func TestLaunchInfoHarnessWebUsesShippedCommand(t *testing.T) {
	var got []string
	err := launchInfoHarnessWeb("fak-test", func(cmd *exec.Cmd) error {
		got = append([]string(nil), cmd.Args...)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"fak-test", "harness", "web"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("launch command = %q, want %q", got, want)
	}

	boom := errors.New("boom")
	if err := launchInfoHarnessWeb("fak-test", func(*exec.Cmd) error { return boom }); !errors.Is(err, boom) {
		t.Fatalf("launch error = %v, want wrapped %v", err, boom)
	}
}
