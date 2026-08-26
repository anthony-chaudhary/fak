package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessartifact"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func TestHarnessInspectCapturedTrustButVerifyRender(t *testing.T) {
	path := writePreviewLock(t, harnessresolve.Lock{
		Schema:      harnessresolve.LockSchema,
		Environment: harnessresolve.Environment{OS: "linux", Arch: "amd64", Contract: "fak.harness/v1"},
		Budget:      harnessresolve.Budget{ContextTokens: 16000, MemoryMiB: 512, Workers: 4},
		Components: []harnessresolve.LockedComponent{{
			ID: "agent-runtime", Version: "1.2.0", Source: "registry:fak", Reason: "supplies agent", Provides: []string{"agent"},
		}},
		Assets: []harnesscompose.EffectiveAsset{
			{Kind: "instruction", ID: "support-tone", Source: "team:support", Value: "concise", Mandatory: true},
			{Kind: "policy", ID: "tools", Source: "company:security", Grants: []string{"search_kb"}, Denies: []string{"refund_payment"}, Locked: true},
			{Kind: "tool", ID: "search_kb", Source: "task:ticket"},
		},
	})
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"inspect", "--lock", path})
	for _, want := range []string{
		"HARNESS INSPECT | VERIFIED",
		"environment: linux/amd64 | contract fak.harness/v1",
		"agent-runtime@1.2.0 | from registry:fak | supplies agent | provides agent",
		"instruction:support-tone | from team:support | mandatory | concise",
		"policy:tools | from company:security | locked by source | grants search_kb | denies refund_payment",
		"tool:search_kb | from task:ticket | changeable by re-resolve",
		"compare a candidate: fak harness preview --current " + path,
		"change the harness: edit the product manifest or layer selection, then run fak harness resolve",
	} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q in captured render:\n%s", want, out.String())
		}
	}
	if code != 0 || errb.Len() != 0 {
		t.Fatalf("code=%d stderr=%q", code, errb.String())
	}
}

func TestHarnessInspectJSONIsMachineReadable(t *testing.T) {
	path := writePreviewLock(t, harnessresolve.Lock{Schema: harnessresolve.LockSchema, Assets: []harnesscompose.EffectiveAsset{{Kind: "workflow", ID: "review", Source: "repo:defaults"}}})
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"inspect", "--lock", path, "--json"})
	var got struct {
		Schema   string                                         `json:"schema"`
		Verified bool                                           `json:"verified"`
		Assets   []struct{ Capability, Source, Control string } `json:"assets"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if code != 0 || errb.Len() != 0 || got.Schema != "fak-harness-inspection/1" || !got.Verified || len(got.Assets) != 1 || got.Assets[0].Capability != "workflow:review" || got.Assets[0].Source != "repo:defaults" {
		t.Fatalf("code=%d stderr=%q report=%+v", code, errb.String(), got)
	}
}

func TestHarnessInspectLifecycleTamperRefusesWithoutSecretLeak(t *testing.T) {
	dir := t.TempDir()
	lockPath := writePreviewLock(t, harnessresolve.Lock{Schema: harnessresolve.LockSchema})
	receiptPath := filepath.Join(dir, "lifecycle.json")
	receipt := harnessartifact.ModelLifecycleReceipt{
		Declaration: harnessartifact.LifecycleIdentity{ID: "decl-1"}, Artifact: harnessartifact.LifecycleIdentity{ID: "artifact-1"},
		Runtime: harnessartifact.LifecycleIdentity{ID: "runtime-1"}, Admission: harnessartifact.LifecycleIdentity{ID: "admission-1"},
		Process: harnessartifact.LifecycleIdentity{ID: "process-1"}, Readiness: harnessartifact.LifecycleIdentity{ID: "ready-1"}, Stop: harnessartifact.LifecycleIdentity{ID: "stop-1"},
		HealthURL: "http://user:inspectsecret@127.0.0.1/health?token=inspectsecret", State: "ready",
	}
	if err := harnessartifact.WriteModelLifecycleReceipt(receiptPath, receipt); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatal(err)
	}
	at := bytes.Index(raw, []byte("runtime-1"))
	raw[at] = 'R'
	if err := os.WriteFile(receiptPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runHarnessInspect(&stdout, &stderr, []string{"--lock", lockPath, "--lifecycle-receipt", receiptPath})
	captured := stdout.String() + stderr.String()
	if code == 0 || !strings.Contains(captured, "LIFECYCLE_RECEIPT_TAMPERED") {
		t.Fatalf("code=%d output=%s", code, captured)
	}
	if strings.Contains(captured, "inspectsecret") {
		t.Fatalf("secret leaked: %s", captured)
	}
}
