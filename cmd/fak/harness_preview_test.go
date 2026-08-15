package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func TestHarnessPreviewCLIRepeatedLockWritesNoBytes(t *testing.T) {
	path := writePreviewLock(t, harnessresolve.Lock{Schema: harnessresolve.LockSchema, ID: "sha256:same"})
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"preview", "--current", path, "--candidate", path})
	if code != 0 || out.Len() != 0 || errb.Len() != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, out.String(), errb.String())
	}
}

func TestHarnessPreviewCLIHeadlessFailsClosedWithRecovery(t *testing.T) {
	current := writePreviewLock(t, harnessresolve.Lock{Schema: harnessresolve.LockSchema, ID: "sha256:old"})
	candidate := writePreviewLock(t, harnessresolve.Lock{Schema: harnessresolve.LockSchema, ID: "sha256:new", Assets: []harnesscompose.EffectiveAsset{{Kind: "tool", ID: "deploy", Source: "task:release"}}})
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"preview", "--current", current, "--candidate", candidate, "--headless"})
	if code != 3 || errb.Len() != 0 {
		t.Fatalf("code=%d stderr=%q", code, errb.String())
	}
	var got struct {
		RequiresDecision bool `json:"requires_decision"`
		Changes          []struct {
			Reason     string `json:"reason"`
			Layer      string `json:"layer"`
			Capability string `json:"capability"`
		} `json:"changes"`
		Recovery []struct {
			ID string `json:"id"`
		} `json:"recovery"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !got.RequiresDecision || len(got.Changes) != 1 || got.Changes[0].Reason != "privilege-widening" || got.Changes[0].Layer != "task:release" || got.Changes[0].Capability != "tool:deploy" {
		t.Fatalf("unexpected decision: %s", out.String())
	}
	if len(got.Recovery) != 3 || got.Recovery[2].ID != "keep-current" {
		t.Fatalf("missing machine recovery: %s", out.String())
	}
}

func TestHarnessPreviewCLICapturedRenderHasOneBoundedBlock(t *testing.T) {
	candidate := writePreviewLock(t, harnessresolve.Lock{Schema: harnessresolve.LockSchema, ID: "sha256:legal", Assets: []harnesscompose.EffectiveAsset{{Kind: "instruction", ID: "citations", Source: "domain:legal"}}})
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"preview", "--candidate", candidate, "--candidate-domain", "legal", "--view", "tui"})
	want := "HARNESS PREVIEW | decision required\n- novel-domain | domain:legal | domain:legal\n  switch contextual defaults from stock to legal; choice: keep the current lock\nchoices: approve-once | remember | keep-current\n"
	if code != 3 || out.String() != want || strings.Contains(out.String(), "\x1b") || errb.Len() != 0 {
		t.Fatalf("code=%d\nwant=%q\n got=%q\nstderr=%q", code, want, out.String(), errb.String())
	}
}

func writePreviewLock(t *testing.T, lock harnessresolve.Lock) string {
	t.Helper()
	lock.ID = ""
	raw, err := json.Marshal(lock)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	lock.ID = "sha256:" + hex.EncodeToString(sum[:])
	raw, err = json.Marshal(lock)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "product.lock.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}
