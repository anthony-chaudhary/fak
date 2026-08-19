package main

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"reflect"
	"strings"
	"testing"
	"time"
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

func TestStartInfoHarnessWebReopensReadyGateway(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer server.Close()

	oldURL, oldClient, oldOpen := infoHarnessWebURL, infoHarnessWebHTTPClient, infoHarnessWebOpenURL
	infoHarnessWebURL = server.URL
	infoHarnessWebHTTPClient = server.Client()
	var opened string
	infoHarnessWebOpenURL = func(raw string) error { opened = raw; return nil }
	t.Cleanup(func() {
		infoHarnessWebURL, infoHarnessWebHTTPClient, infoHarnessWebOpenURL = oldURL, oldClient, oldOpen
	})

	endpoint, err := startInfoHarnessWeb()
	if err != nil {
		t.Fatal(err)
	}
	if endpoint != server.URL || opened != server.URL {
		t.Fatalf("endpoint/opened = %q/%q, want existing %q", endpoint, opened, server.URL)
	}
}
func TestLaunchInfoHarnessWebWaitsForReadyAndOpensBrowser(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, "<!doctype html><title>fak local harness</title>")
	}))
	defer server.Close()

	var gotArgs []string
	var opened string
	endpoint, err := launchInfoHarnessWeb("fak-test", server.URL, server.Client(), func(raw string) error {
		opened = raw
		return nil
	}, func(cmd *exec.Cmd) error {
		gotArgs = append([]string(nil), cmd.Args...)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if endpoint != server.URL || opened != server.URL {
		t.Fatalf("endpoint/opened = %q/%q, want %q", endpoint, opened, server.URL)
	}
	want := []string{"fak-test", "harness", "web"}
	if !reflect.DeepEqual(gotArgs, want) {
		t.Fatalf("launch command = %q, want %q", gotArgs, want)
	}
}

func TestLaunchInfoHarnessWebReportsStartAndBrowserFailures(t *testing.T) {
	boom := errors.New("boom")
	_, err := launchInfoHarnessWeb("fak-test", "http://127.0.0.1:1", &http.Client{Timeout: time.Millisecond}, func(string) error { return nil }, func(*exec.Cmd) error { return boom })
	if !errors.Is(err, boom) {
		t.Fatalf("launch error = %v, want wrapped %v", err, boom)
	}

	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer server.Close()
	_, err = launchInfoHarnessWeb("fak-test", server.URL, server.Client(), func(string) error { return boom }, func(*exec.Cmd) error { return nil })
	if !errors.Is(err, boom) || !strings.Contains(err.Error(), "open web gateway") {
		t.Fatalf("browser error = %v, want wrapped open failure", err)
	}
}

func TestInfoWebGatewayLaunchNoticeCapturedRender(t *testing.T) {
	state := infoViewState{launchNotice: "web gateway: opened http://127.0.0.1:8787"}
	got := renderGuardInfoInteractiveBlock(state, guardInfoVars{}, newGuardInfoTrend(1), 100, 8)
	if !strings.Contains(got, state.launchNotice) {
		t.Fatalf("launch result missing from TUI render:\n%s", got)
	}
}
