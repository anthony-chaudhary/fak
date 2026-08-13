package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestSessionMoveAdmitAndVerifyCLI(t *testing.T) {
	destination := gateway.SessionPlacement{Provider: "provider-b", AccountRef: "account-ref-b", Model: "model-b", Compute: "compute-b", Capabilities: []string{"tools"}, ContextLimit: 128000, BudgetAvailable: 10, ComputeAvailable: true, CacheLineage: "cache-b"}
	checkpoint := gateway.SessionMoveCheckpoint{Schema: "fak.session-move.v1", SessionID: "logical-cli", SourceEpoch: "epoch-a", EventHead: 7, Source: gateway.SessionPlacement{Provider: "a", AccountRef: "a", Model: "a", Compute: "a", ComputeAvailable: true}, Destination: destination, Terminal: []byte("exact terminal\r\n"), CreatedAt: time.Unix(1, 0).UTC()}
	// Obtain a canonically-digested checkpoint through the destination-independent
	// rejection/repair path used by production serialization.
	checkpoint = gateway.FinalizeSessionMoveCheckpoint(checkpoint)
	in, _ := json.Marshal(checkpoint)
	var receipt bytes.Buffer
	args := []string{"admit", "--provider", destination.Provider, "--account-ref", destination.AccountRef, "--model", destination.Model, "--compute", destination.Compute, "--capabilities", "tools", "--context-limit", "128000", "--budget", "10", "--cache-lineage", "cache-b"}
	if code := runSessionMove(&receipt, &bytes.Buffer{}, args); code == 0 {
		t.Fatal("admit unexpectedly read process stdin")
	}
	if code := runSessionMoveAdmit(bytes.NewReader(in), &receipt, &bytes.Buffer{}, args[1:]); code != 0 {
		t.Fatalf("admit code=%d", code)
	}
	path := filepath.Join(t.TempDir(), "receipt.json")
	if err := os.WriteFile(path, receipt.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runSessionMoveVerify(bytes.NewReader(in), &out, &errb, []string{"--receipt", path}); code != 0 {
		t.Fatalf("verify code=%d err=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "MOVE RECEIPT PASS session=logical-cli") {
		t.Fatalf("out=%s", out.String())
	}
}

func TestSessionMoveCheckpointRestoreAndVerifyCLI(t *testing.T) {
	var providerCalled bool
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		providerCalled = true
		if r.URL.Path != "/api/generate" {
			t.Errorf("path=%s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"model-b","response":"RESTORED","done":true}`)
	}))
	defer provider.Close()
	flags := []string{"--session-id", "logical-portable", "--source-epoch", "epoch-source", "--event-head", "9", "--source-provider", "provider-a", "--source-account-ref", "account-a", "--source-model", "model-a", "--source-compute", "local:workstation", "--provider", "provider-b", "--account-ref", "account-b", "--model", "model-b", "--compute", "node:test", "--capabilities", "tools", "--context-limit", "32768", "--budget", "4", "--cache-lineage", "cache-b"}
	var checkpoint bytes.Buffer
	if code := runSessionMoveCheckpoint(strings.NewReader("exact transcript\r\n"), &checkpoint, &bytes.Buffer{}, flags); code != 0 {
		t.Fatalf("checkpoint code=%d", code)
	}
	var parsed gateway.SessionMoveCheckpoint
	if err := json.Unmarshal(checkpoint.Bytes(), &parsed); err != nil {
		t.Fatal(err)
	}
	if parsed.Digest == "" || string(parsed.Terminal) != "exact transcript\r\n" {
		t.Fatalf("checkpoint=%+v", parsed)
	}
	restoreFlags := append([]string{}, flags[14:]...)
	restoreFlags = append(restoreFlags, "--provider-endpoint", provider.URL, "--state-file", filepath.Join(t.TempDir(), "restored", "checkpoint.json"))
	var receipt bytes.Buffer
	var errb bytes.Buffer
	if code := runSessionMoveRestore(bytes.NewReader(checkpoint.Bytes()), &receipt, &errb, restoreFlags); code != 0 {
		t.Fatalf("restore code=%d err=%s", code, errb.String())
	}
	if !providerCalled {
		t.Fatal("provider continuation was not called")
	}
	var restored gateway.SessionMoveDestinationReceipt
	if err := json.Unmarshal(receipt.Bytes(), &restored); err != nil {
		t.Fatal(err)
	}
	if restored.Phase != gateway.SessionMoveRestored || restored.DestinationEpoch == "" || restored.DestinationEpoch == parsed.SourceEpoch || restored.ProviderWitness == nil {
		t.Fatalf("receipt=%+v", restored)
	}
	path := filepath.Join(t.TempDir(), "receipt.json")
	if err := os.WriteFile(path, receipt.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if code := runSessionMoveVerify(bytes.NewReader(checkpoint.Bytes()), &out, &errb, []string{"--receipt", path}); code != 0 {
		t.Fatalf("verify code=%d err=%s", code, errb.String())
	}
}
