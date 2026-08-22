package main

import (
	"bytes"
	"io"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

func TestAccountsLaunchPersistsClaudeActivationBeforeChild(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)
	oldBroker := launchSpawnBroker
	oldRun := accountsLaunchRun
	var attempt launchBrokerAttempt
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant {
		attempt = a
		return allowLaunchBrokerGrant(a, "activation-test")
	}
	accountsLaunchRun = func(_, _ io.Writer, _ []string, _ []string) launchRunResult {
		receipts, err := ultracodebench.ReadActivations(home, attempt.Metadata.AgentRunID)
		if err != nil {
			t.Fatal(err)
		}
		if len(receipts) != 1 || receipts[0].Harness != "claude" || !receipts[0].Injected || receipts[0].State() != ultracodebench.ActivationUnknown {
			t.Fatalf("pre-spawn activation=%+v", receipts)
		}
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() {
		launchSpawnBroker = oldBroker
		accountsLaunchRun = oldRun
	})
	var stdout, stderr bytes.Buffer
	if code := runAccounts(&stdout, &stderr, []string{"launch", "--name", "gem8-seat", "--ultracode=on", "--registry", regPath, "--home", home}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
}

func TestAccountsActivationDistinguishesUnsupportedAndExplicitOff(t *testing.T) {
	home := t.TempDir()
	if err := persistAccountsUltracodeActivation(home, "run-codex", "codex", "on", true); err != nil {
		t.Fatal(err)
	}
	if err := persistAccountsUltracodeActivation(home, "run-off", "claude", "off", false); err != nil {
		t.Fatal(err)
	}
	unsupported, err := ultracodebench.ReadActivations(home, "run-codex")
	if err != nil {
		t.Fatal(err)
	}
	off, err := ultracodebench.ReadActivations(home, "run-off")
	if err != nil {
		t.Fatal(err)
	}
	if len(unsupported) != 1 || unsupported[0].State() != ultracodebench.ActivationDegraded || unsupported[0].Injected {
		t.Fatalf("unsupported=%+v", unsupported)
	}
	if len(off) != 1 || off[0].State() != ultracodebench.ActivationInactive || off[0].Injected {
		t.Fatalf("off=%+v", off)
	}
}
