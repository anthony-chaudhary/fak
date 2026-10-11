package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

const opsJournalTestObservationEnv = "FAK_OPS_JOURNAL_TEST_OBSERVATION"

// This fixture consumes the Ops handoff; it does not qualify guard registration.
func init() {
	observation := os.Getenv(opsJournalTestObservationEnv)
	if observation == "" || len(os.Args) < 2 || os.Args[1] != "guard" {
		return
	}
	path := strings.TrimSpace(os.Getenv("FAK_SESSION_JOURNAL"))
	if path != "" {
		if err := sessionjournal.Append(path, sessionjournal.Event{
			Kind: sessionjournal.KindOpen, ID: "ops-journal-handoff", TS: "2026-10-10T00:00:00Z",
		}); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
	}
	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	data, err := json.Marshal(map[string]string{"home": os.Getenv("HOME"), "journal": path, "cwd": cwd})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := os.WriteFile(observation, data, 0600); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	fmt.Println("{\"type\":\"step_finish\",\"part\":{\"reason\":\"stop\"}}")
	os.Exit(0)
}

// fak-test:justify why=regression when=changed:cmd/fak/**
// fak-test:runtime medium est=3s lane=default
func TestOpsRunSessionJournalOverride(t *testing.T) {
	root := t.TempDir()
	workspace := filepath.Join(root, "workspace")
	durable := filepath.Join(root, "durable")
	for _, path := range []string{workspace, filepath.Join(durable, "alias")} {
		if err := os.MkdirAll(path, 0700); err != nil {
			t.Fatal(err)
		}
	}
	prompt := filepath.Join(root, "prompt.txt")
	receipt := filepath.Join(root, "receipt.json")
	observation := filepath.Join(root, "observation.json")
	if err := os.WriteFile(prompt, []byte("inspect journal handoff\n"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(opsJournalTestObservationEnv, observation)
	forbidden := []string{
		"FAK_SESSION_JOURNAL_REGISTER", "FAK_SESSION_REGISTRY", "FAK_REGISTRATION_ID",
		"FAK_PARENT_REGISTRATION_ID", "FAK_ATTEMPT_ID", "FAK_PARENT_ATTEMPT_ID",
		"FAK_ROOT_REGISTRATION_ID", "FAK_GUARD_SESSIONS_TOKEN", "FAK_SLACK_OUTBOX_DIR",
		"FAK_OPS_UNDECLARED_VARIABLE",
	}
	for _, key := range forbidden {
		t.Setenv(key, "forbidden-journal-test-sentinel")
	}
	t.Setenv("FAK_SESSION_JOURNAL", "")
	if err := os.Unsetenv("FAK_SESSION_JOURNAL"); err != nil {
		t.Fatal(err)
	}
	gateway := newOpsRunQualifiedGateway(t)
	old := opsRunExecute
	t.Cleanup(func() { opsRunExecute = old })
	var forwarded map[string]string
	opsRunExecute = func(ctx context.Context, out, errOut io.Writer, argv, env []string, p []byte) (int, bool, bool, []opsRunLifecycleRecord) {
		forwarded = make(map[string]string, len(env))
		for _, item := range env {
			key, value, ok := strings.Cut(item, "=")
			if ok {
				forwarded[strings.ToUpper(key)] = value
			}
		}
		if len(argv) < 2 || argv[1] != "guard" {
			t.Errorf("Ops handoff did not select guard")
		}
		return executeOpsRun(ctx, out, errOut, argv, env, p)
	}
	type identity struct {
		Source      string `json:"session_journal_source"`
		Destination string `json:"session_journal_destination_digest"`
		Config      string `json:"effective_config_digest"`
		GuardMode   string `json:"guard_mode_requested"`
	}
	run := func(name, override, wantPath string) identity {
		t.Helper()
		if override == "" {
			if err := os.Unsetenv("FAK_SESSION_JOURNAL"); err != nil {
				t.Fatal(err)
			}
		} else if err := os.Setenv("FAK_SESSION_JOURNAL", override); err != nil {
			t.Fatal(err)
		}
		var stderr bytes.Buffer
		code := runOpsRun(io.Discard, &stderr, []string{
			"--workspace", workspace, "--prompt-file", prompt, "--receipt", receipt,
			"--provider", "openai", "--model", "fixture", "--base-url", gateway.URL + "/v1",
			"--api-key-env", opsJournalTestObservationEnv, "--guard-mode", "enforce", "--timeout", "5s",
		})
		if code != 0 {
			t.Fatalf("%s: Ops exit=%d stderr=%s", name, code, stderr.String())
		}
		for _, key := range forbidden {
			if _, exists := forwarded[key]; exists {
				t.Errorf("%s: forbidden variable %s reached Ops handoff", name, key)
			}
		}
		path, exists := forwarded["FAK_SESSION_JOURNAL"]
		source, destination := "platform_default", "unknown"
		if wantPath == "" {
			if exists {
				t.Errorf("%s: unset journal override was synthesized", name)
			}
		} else {
			source = "environment_override"
			normalized := strings.TrimSpace(path)
			if !filepath.IsAbs(normalized) {
				normalized = filepath.Join(workspace, normalized)
			}
			normalized, err := filepath.Abs(filepath.Clean(normalized))
			if err != nil {
				t.Fatal(err)
			}
			if !exists || normalized != wantPath {
				t.Errorf("%s: journal handoff=%q, want destination %q", name, path, wantPath)
			}
			sum := sha256.Sum256([]byte(wantPath))
			destination = "sha256:" + hex.EncodeToString(sum[:])
		}
		data, err := os.ReadFile(receipt)
		if err != nil {
			t.Fatal(err)
		}
		var got struct {
			Identity identity `json:"launch_identity"`
		}
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatal(err)
		}
		if got.Identity.Source != source || got.Identity.Destination != destination {
			t.Errorf("%s: journal receipt source=%q destination=%q, want %q %q",
				name, got.Identity.Source, got.Identity.Destination, source, destination)
		}
		if got.Identity.GuardMode != "enforce" {
			t.Errorf("%s: guard request=%q, want enforce", name, got.Identity.GuardMode)
		}
		if !strings.HasPrefix(got.Identity.Config, "sha256:") {
			t.Errorf("%s: effective config digest missing", name)
		}
		observedData, err := os.ReadFile(observation)
		if err != nil {
			t.Fatal(err)
		}
		var observed map[string]string
		if err := json.Unmarshal(observedData, &observed); err != nil {
			t.Fatal(err)
		}
		if observed["cwd"] != workspace {
			t.Errorf("%s: child cwd=%q, want %q", name, observed["cwd"], workspace)
		}
		home := observed["home"]
		if home == "" || home == root || home == workspace || home == durable {
			t.Fatalf("%s: child home was not isolated", name)
		}
		if _, err := os.Stat(home); !errors.Is(err, os.ErrNotExist) {
			t.Errorf("%s: ephemeral Ops home survived cleanup: %v", name, err)
		}
		if wantPath != "" {
			events := sessionjournal.LoadFile(wantPath)
			if len(events) == 0 || events[len(events)-1].ID != "ops-journal-handoff" {
				t.Errorf("%s: durable child event missing after Ops home cleanup", name)
			}
			if strings.Contains(string(data), wantPath) {
				t.Errorf("%s: receipt exposed journal destination", name)
			}
		}
		return got.Identity
	}
	run("unset", "", "")
	firstPath := filepath.Join(durable, "first.jsonl")
	firstOverride := "  " + durable + string(os.PathSeparator) + "alias" + string(os.PathSeparator) + ".." + string(os.PathSeparator) + "first.jsonl  "
	if err := os.Setenv("FAK_SESSION_JOURNAL", firstOverride); err != nil {
		t.Fatal(err)
	}
	defaultEnv, cleanup, err := opsRunChildEnvironment("{}", "")
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range defaultEnv {
		key, _, _ := strings.Cut(item, "=")
		for _, forbiddenKey := range append([]string{"FAK_SESSION_JOURNAL"}, forbidden...) {
			if strings.EqualFold(key, forbiddenKey) {
				t.Errorf("default child environment inherited forbidden %s", forbiddenKey)
			}
		}
	}
	cleanup()
	first := run("override", firstOverride, firstPath)
	if err := sessionjournal.Append(firstPath, sessionjournal.Event{
		Kind: sessionjournal.KindBeat, ID: "content-change", TS: "2026-10-10T00:00:01Z",
	}); err != nil {
		t.Fatal(err)
	}
	same := run("changed_contents", firstOverride, firstPath)
	if first.Destination != same.Destination {
		t.Error("journal contents changed destination digest")
	}
	secondPath := filepath.Join(durable, "second.jsonl")
	secondRelative, err := filepath.Rel(workspace, secondPath)
	if err != nil {
		t.Fatal(err)
	}
	second := run("changed_destination", secondRelative, secondPath)
	if first.Destination == second.Destination || first.Config == second.Config {
		t.Error("changed journal destination did not change destination and effective config digests")
	}
	run("unset_again", "", "")
}
