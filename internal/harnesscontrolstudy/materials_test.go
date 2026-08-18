package harnesscontrolstudy

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

const controlTaskDigest = "b82b6d9e17853a40fa37ffa4b8d78da53bc42bab3f6d6ffbe21ba876cf2d7fa3"

func TestPairedStudyMaterialsAreSelfContainedAndVerified(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	study := filepath.Join(root, "docs", "benchmarks", "harness-control-study")

	readme := mustRead(t, filepath.Join(study, "README.md"))
	for _, b := range readme {
		if b < 0x20 && b != '\n' && b != '\r' {
			t.Fatalf("study README contains control byte 0x%02x", b)
		}
	}
	for _, want := range []string{"while `rows` was still empty", "normalized `task-card.md`"} {
		if !strings.Contains(string(readme), want) {
			t.Fatalf("study README missing correction text %q", want)
		}
	}

	canonicalTask := mustRead(t, filepath.Join(study, "task-card.md"))
	if got := sha256Hex(canonicalTask); got != controlTaskDigest {
		t.Fatalf("task digest = %s, want %s", got, controlTaskDigest)
	}
	for _, arm := range []string{"default-control", "scratch"} {
		armDir := filepath.Join(study, "materials", arm)
		if got := mustRead(t, filepath.Join(armDir, "task-card.md")); !slices.Equal(got, canonicalTask) {
			t.Fatalf("%s task card differs from preregistered bytes", arm)
		}
		card := string(mustRead(t, filepath.Join(armDir, "arm-card.md")))
		if !strings.Contains(card, "pinned `fak` binary") || !strings.Contains(card, "Stop the clock") {
			t.Fatalf("%s arm card lacks pinned-binary or clock boundary", arm)
		}
		if !strings.Contains(card, "`task-card.md`") || strings.Contains(card, "../") {
			t.Fatalf("%s arm card must reference only its local task card", arm)
		}
	}

	scratchEntries, err := os.ReadDir(filepath.Join(study, "materials", "scratch"))
	if err != nil {
		t.Fatal(err)
	}
	var scratchNames []string
	for _, entry := range scratchEntries {
		scratchNames = append(scratchNames, entry.Name())
	}
	slices.Sort(scratchNames)
	if want := []string{"arm-card.md", "task-card.md"}; !slices.Equal(scratchNames, want) {
		t.Fatalf("scratch bundle leaks product material: got %v want %v", scratchNames, want)
	}

	defaultDir := filepath.Join(study, "materials", "default-control")
	manifest, err := harnessresolve.Parse(mustRead(t, filepath.Join(defaultDir, "product.json")))
	if err != nil {
		t.Fatalf("parse default manifest: %v", err)
	}
	result, err := harnessresolve.Resolve(context.Background(), manifest, []string{"default"}, harnessresolve.Environment{OS: "linux", Arch: "amd64", Contract: "v1"})
	if err != nil {
		t.Fatalf("resolve default product: %v", err)
	}
	var captured harnessresolve.Lock
	if err := json.Unmarshal(mustRead(t, filepath.Join(defaultDir, "product.lock.json")), &captured); err != nil {
		t.Fatalf("parse captured lock: %v", err)
	}
	if err := harnessresolve.VerifyLock(captured); err != nil {
		t.Fatalf("verify captured lock: %v", err)
	}
	componentRaw := mustRead(t, filepath.Join(defaultDir, "kernel-component.txt"))
	if got, want := "sha256:"+sha256Hex(componentRaw), captured.Components[0].Digest; got != want {
		t.Fatalf("component digest = %s, want %s", got, want)
	}

	if result.Lock.ID != captured.ID {
		t.Fatalf("captured lock drift: generated %s captured %s", result.Lock.ID, captured.ID)
	}
	for capability, want := range map[string]string{"instruction:response-style": "concise", "tool:search_kb": "available"} {
		if got := assetValue(captured, capability); got != want {
			t.Fatalf("%s = %q, want %q", capability, got, want)
		}
	}
	if got := policyShape(captured, "tools"); got != "grant=search_kb deny=shell" {
		t.Fatalf("policy:tools = %q", got)
	}
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func sha256Hex(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func assetValue(lock harnessresolve.Lock, capability string) string {
	for _, asset := range lock.Assets {
		if asset.Kind+":"+asset.ID == capability {
			return asset.Value
		}
	}
	return ""
}

func policyShape(lock harnessresolve.Lock, id string) string {
	for _, asset := range lock.Assets {
		if asset.Kind == "policy" && asset.ID == id {
			return fmt.Sprintf("grant=%s deny=%s", strings.Join(asset.Grants, ","), strings.Join(asset.Denies, ","))
		}
	}
	return ""
}
