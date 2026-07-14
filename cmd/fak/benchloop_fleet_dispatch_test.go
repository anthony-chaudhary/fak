package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestBenchFleetDispatchClaimsOnceAndWritesWitness(t *testing.T) {
	dir := t.TempDir()
	q := filepath.Join(dir, "requests")
	os.MkdirAll(q, 0755)
	req := benchFleetRequest{Schema: "fak.bench-fleet.request.v1", ID: "abc", Machine: "gcp-g2-l4", Command: "echo ok", State: "queued"}
	if err := writeBenchFleetRequest(filepath.Join(q, "gcp-abc.json"), req); err != nil {
		t.Fatal(err)
	}
	fake := func(name string, args ...string) ([]byte, int, error) {
		return []byte("FAK_BENCH_NODE=fak-cuda-build-l4\ngpu=NVIDIA L4\n"), 0, nil
	}
	var out, errOut bytes.Buffer
	if code := runBenchFleetDispatchWithExec(&out, &errOut, []string{"--queue", q, "--json"}, fake); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got benchFleetDispatchReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Claimed != 1 || got.Succeeded != 1 {
		t.Fatalf("report=%+v", got)
	}
	out.Reset()
	if code := runBenchFleetDispatchWithExec(&out, &errOut, []string{"--queue", q, "--json"}, fake); code != 0 {
		t.Fatal(code)
	}
	json.Unmarshal(out.Bytes(), &got)
	if got.Claimed != 0 {
		t.Fatalf("duplicate claim: %+v", got)
	}
	if _, err := os.Stat(filepath.Join(dir, "witnesses", "abc.json")); err != nil {
		t.Fatal(err)
	}
}

func TestBenchFleetRoutesUnavailableNodesToTypedWait(t *testing.T) {
	_, _, route, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: "node-macos-a"})
	if err == nil || route != "mac" || state != "waiting_session" {
		t.Fatalf("route=%s state=%s err=%v", route, state, err)
	}
}

func TestBenchFleetFailedExecutionRemainsWitnessedAndNotReclaimed(t *testing.T) {
	dir := t.TempDir()
	q := filepath.Join(dir, "requests")
	if err := os.MkdirAll(q, 0o755); err != nil {
		t.Fatal(err)
	}
	req := benchFleetRequest{Schema: "fak.bench-fleet.request.v1", ID: "bad", Machine: "gcp-g2-l4", Command: "false", State: "queued"}
	if err := writeBenchFleetRequest(filepath.Join(q, "bad.json"), req); err != nil {
		t.Fatal(err)
	}
	fake := func(string, ...string) ([]byte, int, error) { return []byte("remote failure"), 7, errors.New("exit 7") }
	var out, errOut bytes.Buffer
	if code := runBenchFleetDispatchWithExec(&out, &errOut, []string{"--queue", q, "--json"}, fake); code != 1 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	got, err := readBenchFleetRequest(filepath.Join(q, "bad.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got.State != "failed" {
		t.Fatalf("state=%s", got.State)
	}
	out.Reset()
	if code := runBenchFleetDispatchWithExec(&out, &errOut, []string{"--queue", q, "--json"}, fake); code != 0 {
		t.Fatal(code)
	}
	var report benchFleetDispatchReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Claimed != 0 {
		t.Fatalf("failed request reclaimed: %+v", report)
	}
}

func TestBenchFleetDGXCredentialFailureIsTypedWaiting(t *testing.T) {
	root := t.TempDir()
	bridge := filepath.Join(root, "..", "fak-private", ".dgxbridge-verify")
	if err := os.MkdirAll(bridge, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bridge, "dgxbridge-fresh.exe"), []byte("stub"), 0o755); err != nil {
		t.Fatal(err)
	}
	w := executeBenchFleetRequest(root, benchFleetRequest{ID: "dgx", Machine: "a100", Command: "echo ok"}, func(string, ...string) ([]byte, int, error) {
		return []byte("no Slack channel set"), 1, errors.New("exit 1")
	})
	if w.State != "waiting_credentials" {
		t.Fatalf("witness=%+v", w)
	}
}

func TestBenchFleetRejectsEmptyRemoteSuccess(t *testing.T) {
	w := executeBenchFleetRequest(t.TempDir(), benchFleetRequest{ID: "empty", Machine: "gcp-g2-l4", Command: "true"}, func(string, ...string) ([]byte, int, error) { return nil, 0, nil })
	if w.State != "failed" || w.Error != "remote witness missing FAK_BENCH_NODE marker" {
		t.Fatalf("witness=%+v", w)
	}
}
