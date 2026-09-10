package agent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/codetools"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

type ownedReadResult struct {
	Content string `json:"content"`
	Version string `json:"version"`
}

func runOwnedRead(t *testing.T, catalog []ToolDef, path string) (ownedReadResult, ArmMetrics) {
	t.Helper()
	planner := &recordingCodePlanner{turns: []*Completion{
		toolCallTurn(codetools.ToolRead, mustSelfcheckArgs(map[string]any{"file_path": path})),
		{Message: Message{Content: "done"}},
	}}
	metrics, err := RunArm(context.Background(), planner, "read the workspace file", true, 3, nil,
		WithToolCatalog(catalog))
	if err != nil {
		t.Fatalf("RunArm Read(%q): %v", path, err)
	}
	var got ownedReadResult
	receipt := lastResultFromMessages(planner.messages)
	if err := json.Unmarshal([]byte(receipt), &got); err != nil {
		t.Fatalf("decode Read(%q) receipt %q: %v", path, receipt, err)
	}
	if got.Version == "" {
		t.Fatalf("Read(%q) returned no guarded-mutation version: %s", path, receipt)
	}
	return got, metrics
}

func TestOwnedReadDoesNotCrossArmedWorkspaceRoots(t *testing.T) {
	DisarmCodeTools()
	vdso.Default.BumpWorld() // Isolate this regression from process-global entries seeded by other tests.
	t.Cleanup(func() {
		DisarmCodeTools()
		vdso.Default.BumpWorld()
	})

	const rel = "scope-probe.txt"
	rootA := t.TempDir()
	rootB := t.TempDir()
	markerA := "workspace-alpha-71b28643"
	markerB := "workspace-beta-9c85460e"
	if err := os.WriteFile(filepath.Join(rootA, rel), []byte(markerA), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(rootB, rel), []byte(markerB), 0o644); err != nil {
		t.Fatal(err)
	}

	catalogA, err := ArmCodeTools(rootA)
	if err != nil {
		t.Fatalf("ArmCodeTools(rootA): %v", err)
	}
	first, firstMetrics := runOwnedRead(t, catalogA, rel)
	if first.Content != markerA {
		t.Fatalf("rootA Read content = %q, want %q (metrics=%+v)", first.Content, markerA, firstMetrics)
	}
	DisarmCodeTools()

	catalogB, err := ArmCodeTools(rootB)
	if err != nil {
		t.Fatalf("ArmCodeTools(rootB): %v", err)
	}
	second, secondMetrics := runOwnedRead(t, catalogB, rel)
	if second.Content != markerB {
		t.Fatalf("rootB Read content = %q, want %q; previous workspace content was %q (metrics=%+v, CallMeta(Read)=%v)",
			second.Content, markerB, markerA, secondMetrics, codetools.CallMeta(codetools.ToolRead, ""))
	}
	if second.Version == first.Version {
		t.Fatalf("rootB Read version = rootA version %q despite distinct bytes", second.Version)
	}
}

func TestOwnedReadObservesExternalPeerUpdateWithoutWorldBump(t *testing.T) {
	DisarmCodeTools()
	vdso.Default.BumpWorld() // Isolate this regression; there is no bump between the two Reads below.
	t.Cleanup(func() {
		DisarmCodeTools()
		vdso.Default.BumpWorld()
	})

	const rel = "scope-probe.txt"
	root := t.TempDir()
	path := filepath.Join(root, rel)
	markerA := "peer-before-4e56c1ad"
	markerB := "peer-after-a3d70982"
	if err := os.WriteFile(path, []byte(markerA), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := ArmCodeTools(root)
	if err != nil {
		t.Fatalf("ArmCodeTools: %v", err)
	}
	first, firstMetrics := runOwnedRead(t, catalog, rel)
	if first.Content != markerA {
		t.Fatalf("initial Read content = %q, want %q (metrics=%+v)", first.Content, markerA, firstMetrics)
	}

	// This write models another process changing the workspace. It emits no kernel
	// mutation event and deliberately does not call vdso.Default.BumpWorld.
	if err := os.WriteFile(path, []byte(markerB), 0o644); err != nil {
		t.Fatal(err)
	}
	second, secondMetrics := runOwnedRead(t, catalog, rel)
	if second.Content != markerB {
		t.Fatalf("post-peer-update Read content = %q, want %q (metrics=%+v, CallMeta(Read)=%v)",
			second.Content, markerB, secondMetrics, codetools.CallMeta(codetools.ToolRead, ""))
	}
	if second.Version == first.Version {
		t.Fatalf("post-peer-update Read version = stale version %q", second.Version)
	}
}

func TestOwnedReadExplicitNonIdempotentMetaOverridesDynamicPromotion(t *testing.T) {
	DisarmCodeTools()
	vdso.Default.BumpWorld() // Isolate this regression from process-global entries seeded by other tests.
	vdso.Default.PromoteReadOnly(codetools.ToolRead)
	t.Cleanup(func() {
		DisarmCodeTools()
		vdso.Default.InvalidatePromoted(codetools.ToolRead)
		vdso.Default.BumpWorld()
	})
	if !vdso.Default.IsPromotedReadOnly(codetools.ToolRead) {
		t.Fatal("test setup did not dynamically promote Read")
	}
	if meta := codetools.CallMeta(codetools.ToolRead, ""); meta["readOnlyHint"] != "true" || meta["idempotentHint"] != "" {
		t.Fatalf("Read metadata = %v, want explicit read-only with omitted idempotency", meta)
	}

	const rel = "scope-probe.txt"
	root := t.TempDir()
	path := filepath.Join(root, rel)
	markerA := "promoted-before-236c94fd"
	markerB := "promoted-after-b4076e1a"
	if err := os.WriteFile(path, []byte(markerA), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := ArmCodeTools(root)
	if err != nil {
		t.Fatalf("ArmCodeTools: %v", err)
	}
	first, firstMetrics := runOwnedRead(t, catalog, rel)
	if first.Content != markerA {
		t.Fatalf("initial promoted Read content = %q, want %q (metrics=%+v)", first.Content, markerA, firstMetrics)
	}

	// Dynamic promotion must not override this call's explicit omission of the
	// idempotency claim. A peer write emits no kernel mutation or world bump.
	if err := os.WriteFile(path, []byte(markerB), 0o644); err != nil {
		t.Fatal(err)
	}
	second, secondMetrics := runOwnedRead(t, catalog, rel)
	if second.Content != markerB {
		t.Fatalf("promoted post-peer-update Read content = %q, want %q (metrics=%+v, CallMeta(Read)=%v)",
			second.Content, markerB, secondMetrics, codetools.CallMeta(codetools.ToolRead, ""))
	}
	if second.Version == first.Version {
		t.Fatalf("promoted post-peer-update Read version = stale version %q", second.Version)
	}
}
