//go:build darwin

package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// syntheticIncumbentDeps builds observation deps from canned outputs, keyed by
// the state each test needs. lsof raw uses the exact BSD -Fpcftn record shape
// the live parser consumes.
func syntheticIncumbentDeps(lsofRaw string, launchdPrint func(args ...string) ([]byte, error), psCommand string, healthStatus, modelsStatus int, modelsBody string) incumbentObservationDeps {
	return incumbentObservationDeps{
		lookupUID: func() (int, error) { return 501, nil },
		runLsof: func(_ context.Context, port int) ([]byte, error) {
			if lsofRaw == "" {
				return nil, nil
			}
			return []byte(strings.ReplaceAll(lsofRaw, "PORT", fmt.Sprint(port))), nil
		},
		runLaunchd: func(_ context.Context, args ...string) ([]byte, error) {
			return launchdPrint(args...)
		},
		readPS: func(_ context.Context, pid int) (modelCanaryProcessIdentity, error) {
			if psCommand == "" {
				return modelCanaryProcessIdentity{}, errors.New("no ps output")
			}
			return modelCanaryProcessIdentity{
				PID:        pid,
				StartedAt:  time.Now().UTC().Format(time.RFC3339Nano),
				ArgvSHA256: digestBytes([]byte(psCommand)),
			}, nil
		},
		probeHTTP: func(_ context.Context, url string) (int, error) {
			if strings.HasSuffix(url, "/health") {
				return healthStatus, nil
			}
			return modelsStatus, nil
		},
		probeHTTPBody: func(_ context.Context, url string) (int, []byte, error) {
			if strings.HasSuffix(url, "/v1/models") {
				return modelsStatus, []byte(modelsBody), nil
			}
			return healthStatus, nil, nil
		},
		now: func() time.Time { return time.Date(2026, 9, 2, 12, 0, 0, 0, time.UTC) },
	}
}

const incumbentSyntheticLsof = "p709\ncllama-server\nf3\ntIPv4\nnlocalhost:PORT\n"

// launchctlAbsentError synthesizes the live launchctl absent-job failure: a real
// nonzero ExitError carrying the "Could not find service" diagnostic.
func launchctlAbsentError(t *testing.T, label string) error {
	t.Helper()
	cmd := exec.Command("sh", "-c", "echo 'Bad request.'; echo 'Could not find service "+label+" in domain for user gui: 501' >&2; exit 113")
	var combined strings.Builder
	cmd.Stdout, cmd.Stderr = &combined, &combined
	runErr := cmd.Run()
	if runErr == nil {
		t.Fatal("synthetic launchctl failure did not fail")
	}
	return fmt.Errorf("launchctl print gui/501/%s: %w: %s", label, runErr, strings.TrimSpace(combined.String()))
}

// incumbentSyntheticDomainOutput mirrors the live `launchctl print gui/<uid>`
// services-table rows the owner scanner parses (pid, state marker, label).
const incumbentSyntheticDomainOutput = `gui/501 = {
	type = login
	active count = 3
	services = {
		   709      - 	com.fak.qwen36-kernel
		     1      - 	com.apple.launchd.shutdown
		   0      - 	com.apple.example
	}
}
com.apple.example = {
	state = not running
}
`

func TestParseIncumbentDomainOwnerResolvesSingleRow(t *testing.T) {
	label, ok := parseIncumbentDomainOwner([]byte(incumbentSyntheticDomainOutput), 709)
	if !ok || label != "com.fak.qwen36-kernel" {
		t.Fatalf("owner = %q ok=%v, want com.fak.qwen36-kernel", label, ok)
	}
	if _, ok := parseIncumbentDomainOwner([]byte(incumbentSyntheticDomainOutput), 424242); ok {
		t.Fatal("absent pid resolved as owned")
	}
	if _, ok := parseIncumbentDomainOwner([]byte(incumbentSyntheticDomainOutput), 0); ok {
		t.Fatal("pid 0 resolved as owned")
	}
}

func TestObserveIncumbentPreflightClassifiesLiveHoldState(t *testing.T) {
	// The observed #9714 state: healthy incumbent on 8090, expected job absent,
	// alternate supervisor row in the domain table.
	command := "/opt/llama-server --model /models/Qwen3.6-27B.q4_k_m.gguf --alias qwen3.6-27b --port 8090"
	deps := syntheticIncumbentDeps(incumbentSyntheticLsof,
		func(args ...string) ([]byte, error) {
			joined := strings.Join(args, " ")
			if strings.HasPrefix(joined, "print gui/501/") {
				// launchctl reports the absent job through a nonzero exit; the
				// live runner wraps that into the error this branch returns.
				return nil, launchctlAbsentError(t, incumbentDefaultLabel)
			}
			if joined == "print gui/501" {
				return []byte(incumbentSyntheticDomainOutput), nil
			}
			return nil, fmt.Errorf("unexpected launchctl call: %s", joined)
		},
		command, 200, 200,
		`{"data":[{"id":"`+incumbentDefaultAlias+`"}]}`)
	receipt, err := observeIncumbentPreflight(context.Background(), deps, incumbentDefaultPort, incumbentDefaultLabel)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Verdict != incumbentVerdictJobAbsent || receipt.Reason != "alternate_launchd_supervisor_owns_incumbent" {
		t.Fatalf("verdict = %s/%s", receipt.Verdict, receipt.Reason)
	}
	if receipt.ExpectedOwner.Label != incumbentDefaultLabel {
		t.Fatalf("expected owner label drift: %+v", receipt.ExpectedOwner)
	}
	if !receipt.Incumbent.ListenerPresent || receipt.Incumbent.ListenerPID != 709 {
		t.Fatalf("listener drift: %+v", receipt.Incumbent)
	}
	if receipt.Incumbent.CommandSHA256 == "" {
		t.Fatal("listener command digest was not observed")
	}
	wantOwner := digestBytes([]byte("com.fak.qwen36-kernel"))
	if !receipt.Incumbent.OwnerResolved || receipt.Incumbent.OwnerLabelSHA256 != wantOwner {
		t.Fatalf("owner drift: resolved=%v digest=%s want=%s", receipt.Incumbent.OwnerResolved, receipt.Incumbent.OwnerLabelSHA256, wantOwner)
	}
	if !receipt.ReadOnly || receipt.Schema != incumbentPreflightSchema {
		t.Fatalf("receipt shape drift: %+v", receipt)
	}
	if receipt.ObservedAt == "" {
		t.Fatal("observed_at missing on a classified receipt")
	}
}

func TestObserveIncumbentPreflightOwnedState(t *testing.T) {
	command := "/opt/llama-server --alias qwen3.6-27b --port 8090"
	// Override the expected identity so the synthetic command can reach the
	// owned verdict; production keeps the preserved #9714 constant.
	prev := incumbentExpectedCommandSHA256
	incumbentExpectedCommandSHA256 = digestBytes([]byte(command))
	defer func() { incumbentExpectedCommandSHA256 = prev }()
	deps := syntheticIncumbentDeps(incumbentSyntheticLsof,
		func(args ...string) ([]byte, error) {
			joined := strings.Join(args, " ")
			if strings.HasPrefix(joined, "print gui/501/") {
				return []byte("gui/501/com.fak.qwen36-model = {\n\tpid = 709\n\tpath = /Library/LaunchAgents/com.fak.qwen36-model.plist\n}\n"), nil
			}
			return nil, fmt.Errorf("unexpected launchctl call: %s", joined)
		},
		command, 200, 200,
		`{"data":[{"id":"`+incumbentDefaultAlias+`"}]}`)
	receipt, err := observeIncumbentPreflight(context.Background(), deps, incumbentDefaultPort, incumbentDefaultLabel)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Verdict != incumbentVerdictOwned {
		t.Fatalf("owned state classified %s/%s: %+v", receipt.Verdict, receipt.Reason, receipt)
	}
	if !receipt.ExpectedOwner.JobBindsListener || receipt.ExpectedOwner.PlistPath == "" {
		t.Fatalf("owned binding drift: %+v", receipt.ExpectedOwner)
	}
}

func TestObserveIncumbentPreflightObservationFailuresStayTyped(t *testing.T) {
	deps := syntheticIncumbentDeps("", func(args ...string) ([]byte, error) {
		return nil, launchctlAbsentError(t, incumbentDefaultLabel)
	}, "", 0, 0, "")
	receipt, err := observeIncumbentPreflight(context.Background(), deps, incumbentDefaultPort, incumbentDefaultLabel)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Verdict != incumbentVerdictJobAbsent || receipt.Incumbent.ListenerPresent {
		t.Fatalf("no-listener state must be a typed absence: %+v", receipt)
	}

	failingDeps := syntheticIncumbentDeps("", func(args ...string) ([]byte, error) {
		return nil, errors.New("launchctl unavailable")
	}, "", 0, 0, "")
	failingDeps.runLsof = func(_ context.Context, _ int) ([]byte, error) { return nil, errors.New("lsof permission denied") }
	receipt, err = observeIncumbentPreflight(context.Background(), failingDeps, incumbentDefaultPort, incumbentDefaultLabel)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Verdict != incumbentVerdictFailed || !strings.Contains(receipt.Reason, "observe listener") {
		t.Fatalf("tool failure must stay typed: %+v", receipt)
	}

	launchdToolFailure := syntheticIncumbentDeps("", func(args ...string) ([]byte, error) {
		return nil, errors.New("launchctl exploded")
	}, "", 0, 0, "")
	receipt, err = observeIncumbentPreflight(context.Background(), launchdToolFailure, incumbentDefaultPort, incumbentDefaultLabel)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Verdict != incumbentVerdictFailed || !strings.Contains(receipt.Reason, "read expected launchd job") {
		t.Fatalf("launchctl tool failure must stay typed: %+v", receipt)
	}
}

func TestParseIncumbentPlistLabel(t *testing.T) {
	body := renderIncumbentPlist(incumbentRenderSpec{Label: incumbentDefaultLabel, Program: []string{"/bin/x"}, ThrottleInterl: 10})
	label, err := parseIncumbentPlistLabel([]byte(body))
	if err != nil || label != incumbentDefaultLabel {
		t.Fatalf("label = %q err = %v", label, err)
	}
	if _, err := parseIncumbentPlistLabel([]byte("<plist></plist>")); err == nil {
		t.Fatal("plist without Label admitted")
	}
}

func TestInstallIncumbentFailsClosed(t *testing.T) {
	dir := t.TempDir()
	plist := filepath.Join(dir, "com.fak.qwen36-model.plist")
	body := renderIncumbentPlist(incumbentRenderSpec{Label: incumbentDefaultLabel, Program: []string{"/bin/x"}, ThrottleInterl: 10})
	if err := os.WriteFile(plist, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	// State 1: alternate supervisor owns the healthy incumbent -> refuse, never
	// bootstrap, never signal.
	command := "/opt/llama-server --model /models/Qwen3.6-27B.q4_k_m.gguf --alias qwen3.6-27b --port 8090"
	holdDeps := syntheticIncumbentDeps(incumbentSyntheticLsof,
		func(args ...string) ([]byte, error) {
			joined := strings.Join(args, " ")
			if strings.HasPrefix(joined, "print gui/501/") {
				return nil, launchctlAbsentError(t, incumbentDefaultLabel)
			}
			if joined == "print gui/501" {
				return []byte(incumbentSyntheticDomainOutput), nil
			}
			// A bootstrap attempt in this state is the bug the test exists for.
			return nil, fmt.Errorf("MUTATION ATTEMPTED: %s", joined)
		},
		command, 200, 200, `{"data":[{"id":"`+incumbentDefaultAlias+`"}]}`)
	outcome, err := installIncumbent(context.Background(), holdDeps, incumbentInstallInput{Plist: plist, Execute: true, ReadyTimeout: time.Second})
	if err != nil {
		t.Fatalf("refusals must be typed outcomes, not errors: %v", err)
	}
	if outcome.Admitted || outcome.Executed != true {
		t.Fatalf("outcome drift: %+v", outcome)
	}
	if !strings.Contains(outcome.Refusal, "alternate_launchd_supervisor_owns_incumbent") {
		t.Fatalf("refusal = %q", outcome.Refusal)
	}

	// State 2: nothing running -> dry-run plans but does not execute.
	idleDeps := syntheticIncumbentDeps("", func(args ...string) ([]byte, error) {
		joined := strings.Join(args, " ")
		if strings.HasPrefix(joined, "print gui/501/") {
			return nil, launchctlAbsentError(t, incumbentDefaultLabel)
		}
		return nil, fmt.Errorf("unexpected launchctl call: %s", joined)
	}, "", 0, 0, "")
	outcome, err = installIncumbent(context.Background(), idleDeps, incumbentInstallInput{Plist: plist, Execute: false, ReadyTimeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	if !outcome.Admitted || outcome.Executed {
		t.Fatalf("dry-run must plan without executing: %+v", outcome)
	}
	if len(outcome.Plan) == 0 || !strings.Contains(outcome.Plan[1], "bootstrap") {
		t.Fatalf("dry-run plan missing the bootstrap step: %+v", outcome.Plan)
	}

	// State 3: execute with no incumbent -> bootstrap is planned and admitted.
	var bootstrapped bool
	execDeps := idleDeps
	execDeps.runLaunchd = func(_ context.Context, args ...string) ([]byte, error) {
		joined := strings.Join(args, " ")
		if strings.HasPrefix(joined, "print gui/501/") {
			if bootstrapped {
				return []byte("gui/501/com.fak.qwen36-model = {\n\tpid = 900\n\tpath = " + plist + "\n}\n"), nil
			}
			return nil, launchctlAbsentError(t, incumbentDefaultLabel)
		}
		if strings.HasPrefix(joined, "bootstrap gui/501") {
			bootstrapped = true
			return nil, nil
		}
		return nil, fmt.Errorf("unexpected launchctl call: %s", joined)
	}
	execDeps.readPS = func(_ context.Context, pid int) (modelCanaryProcessIdentity, error) {
		argv := []string{"/bin/x"}
		return modelCanaryProcessIdentity{
			PID:        pid,
			StartedAt:  time.Now().UTC().Format(time.RFC3339Nano),
			ArgvSHA256: digestBytes([]byte(strings.Join(argv, " "))),
		}, nil
	}
	// The bootstrapped command /bin/x cannot bind the preserved identity, so the
	// readiness poll must time out rather than declare ownership of a different
	// command. Use a tiny timeout and expect the typed failure.
	outcome, err = installIncumbent(context.Background(), execDeps, incumbentInstallInput{Plist: plist, Execute: true, ReadyTimeout: 1200 * time.Millisecond})
	if err == nil || !strings.Contains(err.Error(), "failed to become") {
		t.Fatalf("bootstrapped foreign command must not pass readiness: outcome=%+v err=%v", outcome, err)
	}
	if !bootstrapped {
		t.Fatal("execute never issued the bootstrap")
	}
}
