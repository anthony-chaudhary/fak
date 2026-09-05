package gym

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sandbox"
)

// TestGymSubTenMillisecondReset verifies that an arena populated with 1,000 files
// resets to pristine baseline in strictly under 10 milliseconds without polluting the base workspace.
func TestGymSubTenMillisecondReset(t *testing.T) {
	ctx := context.Background()
	baseDir := t.TempDir()

	// 1. Seed base workspace with a pristine anchor file
	seedFile := filepath.Join(baseDir, "anchor_base.txt")
	if err := os.WriteFile(seedFile, []byte("pristine-base-workspace-anchor"), 0644); err != nil {
		t.Fatalf("failed to write base anchor file: %v", err)
	}

	cfg := Config{
		BaseDir:       baseDir,
		WorkspaceName: "bench-sub10ms-reset",
		PinnedPTY:     true,
	}

	arena, err := Create(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to create gym arena: %v", err)
	}
	defer arena.Destroy()

	// 2. Simulate agent generating 1,000 files and writing data inside the gym
	for i := 0; i < 1000; i++ {
		fname := fmt.Sprintf("agent_generated_file_%04d.txt", i)
		fpath := filepath.Join(arena.Path(), fname)
		payload := fmt.Sprintf("agent-generated-payload-content-block-%04d", i)
		if err := os.WriteFile(fpath, []byte(payload), 0644); err != nil {
			t.Fatalf("failed writing simulated agent file %d: %v", i, err)
		}
	}

	// Verify the 1,000 files exist in the gym arena
	arenaEntries, err := os.ReadDir(arena.Path())
	if err != nil {
		t.Fatalf("failed reading arena directory: %v", err)
	}
	if len(arenaEntries) != 1001 {
		t.Fatalf("expected 1001 entries in arena (1 anchor + 1000 created), found %d", len(arenaEntries))
	}

	// 3. Measure time to Reset()
	start := time.Now()
	if err := arena.Reset(ctx); err != nil {
		t.Fatalf("arena.Reset failed: %v", err)
	}
	resetDuration := time.Since(start)

	t.Logf("Observed Reset() duration for 1,000 files: %v", resetDuration)

	// 4. Assert reset duration is < 10ms
	if resetDuration >= 10*time.Millisecond {
		t.Errorf("Reset() exceeded sub-10ms requirement: took %v (must be < 10ms)", resetDuration)
	}

	// 5. Assert the base workspace has zero residual files created by the agent
	baseEntries, err := os.ReadDir(baseDir)
	if err != nil {
		t.Fatalf("failed reading base directory: %v", err)
	}
	if len(baseEntries) != 1 || baseEntries[0].Name() != "anchor_base.txt" {
		t.Errorf("base workspace corrupted! expected 1 entry (anchor_base.txt), found %d", len(baseEntries))
		for _, e := range baseEntries {
			t.Logf("  residual entry found in base: %s", e.Name())
		}
	}

	// Verify arena is also restored to pristine state (only anchor_base.txt)
	postResetArenaEntries, err := os.ReadDir(arena.Path())
	if err != nil {
		t.Fatalf("failed reading post-reset arena directory: %v", err)
	}
	if len(postResetArenaEntries) != 1 || postResetArenaEntries[0].Name() != "anchor_base.txt" {
		t.Errorf("post-reset arena not pristine: expected 1 entry, found %d", len(postResetArenaEntries))
	}
}

// TestGymForkAndPromote verifies branching into a child arena, making changes,
// and promoting those modifications back to the pristine base workspace.
func TestGymForkAndPromote(t *testing.T) {
	ctx := context.Background()
	baseDir := t.TempDir()

	baseFile := filepath.Join(baseDir, "document.md")
	if err := os.WriteFile(baseFile, []byte("# Version 1.0 (Original)\n"), 0644); err != nil {
		t.Fatalf("failed writing initial base document: %v", err)
	}

	cfg := Config{
		BaseDir:       baseDir,
		WorkspaceName: "parent-arena",
	}

	parent, err := Create(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to create parent arena: %v", err)
	}
	defer parent.Destroy()

	// 1. Fork arena into child arena
	child, err := parent.Fork(ctx, "canary-feature-branch")
	if err != nil {
		t.Fatalf("failed to fork child arena: %v", err)
	}
	defer child.Destroy()

	// 2. Modify existing file and create new file in child
	childModifiedDoc := filepath.Join(child.Path(), "document.md")
	if err := os.WriteFile(childModifiedDoc, []byte("# Version 2.0 (Child Promoted)\n"), 0644); err != nil {
		t.Fatalf("failed modifying document in child: %v", err)
	}

	childNewFeature := filepath.Join(child.Path(), "feature.go")
	if err := os.WriteFile(childNewFeature, []byte("package main\n\nfunc Feature() bool { return true }\n"), 0644); err != nil {
		t.Fatalf("failed creating feature file in child: %v", err)
	}

	// 3. Verify pristine base has not yet received child changes
	currentBaseBytes, err := os.ReadFile(baseFile)
	if err != nil {
		t.Fatalf("failed reading base file: %v", err)
	}
	if string(currentBaseBytes) != "# Version 1.0 (Original)\n" {
		t.Fatalf("base workspace was polluted before promote: %s", string(currentBaseBytes))
	}

	// 4. Promote child changes back to base
	if err := child.Promote(ctx); err != nil {
		t.Fatalf("child.Promote failed: %v", err)
	}

	// 5. Verifies pristine base receives the promoted files
	promotedDocBytes, err := os.ReadFile(baseFile)
	if err != nil {
		t.Fatalf("failed reading promoted base document: %v", err)
	}
	if string(promotedDocBytes) != "# Version 2.0 (Child Promoted)\n" {
		t.Errorf("expected promoted content '# Version 2.0 (Child Promoted)\\n', got %q", string(promotedDocBytes))
	}

	promotedFeatureBytes, err := os.ReadFile(filepath.Join(baseDir, "feature.go"))
	if err != nil {
		t.Fatalf("failed reading promoted feature file in base: %v", err)
	}
	if string(promotedFeatureBytes) != "package main\n\nfunc Feature() bool { return true }\n" {
		t.Errorf("unexpected content in promoted feature file: %q", string(promotedFeatureBytes))
	}
}

// TestCanaryExecution runs a live canary task inside an ephemeral gym arena,
// asserting task success, sub-10ms reset, and clean teardown without base pollution.
func TestCanaryExecution(t *testing.T) {
	ctx := context.Background()
	baseDir := t.TempDir()

	// Initial configuration in base workspace
	cfgFile := filepath.Join(baseDir, "app.json")
	if err := os.WriteFile(cfgFile, []byte(`{"env":"production","status":"healthy"}`), 0644); err != nil {
		t.Fatalf("failed writing initial app.json: %v", err)
	}

	runner := NewCanaryRunner(Config{
		BaseDir:       baseDir,
		WorkspaceName: "canary-verification",
		PinnedPTY:     true,
	})

	// Run canary task with programmatic action
	task := CanaryTask{
		Name: "canary-simulation-pass",
		Action: func(arena *Arena) error {
			// Read base file to verify fall-through read
			readContent, err := os.ReadFile(filepath.Join(arena.Path(), "app.json"))
			if err != nil {
				return fmt.Errorf("failed reading fallthrough base file: %w", err)
			}
			if len(readContent) == 0 {
				return fmt.Errorf("read empty content from app.json")
			}

			// Generate ephemeral test results inside arena
			reportFile := filepath.Join(arena.Path(), "test_results.xml")
			return os.WriteFile(reportFile, []byte("<testsuites><testsuite name=\"canary\" tests=\"1\" passes=\"1\"/></testsuites>"), 0644)
		},
	}

	verdict, err := runner.RunTask(ctx, task)
	if err != nil {
		t.Fatalf("runner.RunTask failed: %v", err)
	}

	if !verdict.Passed {
		t.Errorf("expected canary verdict to pass, but failed with: %v", verdict.Error)
	}

	t.Logf("Canary ResetDuration: %v", verdict.ResetDuration)
	if verdict.ResetDuration >= 10*time.Millisecond {
		t.Errorf("canary ResetDuration >= 10ms: got %v", verdict.ResetDuration)
	}

	// Verify modified artifacts were recorded
	foundReport := false
	for _, art := range verdict.ArtifactsModified {
		if art == "test_results.xml" {
			foundReport = true
			break
		}
	}
	if !foundReport {
		t.Errorf("expected 'test_results.xml' in ArtifactsModified, got: %v", verdict.ArtifactsModified)
	}

	// Verify clean teardown: base directory remains untouched
	baseEntries, err := os.ReadDir(baseDir)
	if err != nil {
		t.Fatalf("failed reading baseDir: %v", err)
	}
	if len(baseEntries) != 1 || baseEntries[0].Name() != "app.json" {
		t.Errorf("base workspace was polluted by canary! found entries: %v", baseEntries)
	}
}

// TestCanaryExecutionWithCommand tests CanaryRunner with an actual sandbox command dispatch.
func TestCanaryExecutionWithCommand(t *testing.T) {
	ctx := context.Background()
	baseDir := t.TempDir()

	runner := NewCanaryRunner(Config{
		BaseDir:       baseDir,
		WorkspaceName: "canary-cmd-test",
		PinnedPTY:     true,
	})

	req := sandbox.ExecutionRequest{
		Command: "go",
		Argv:    []string{"version"},
	}

	verdict, err := runner.Run(ctx, req)
	if err != nil {
		t.Fatalf("runner.Run failed: %v", err)
	}

	if !verdict.Passed {
		t.Fatalf("canary execution command failed: %v", verdict.Error)
	}

	if verdict.ResetDuration >= 10*time.Millisecond {
		t.Errorf("expected ResetDuration < 10ms, got %v", verdict.ResetDuration)
	}
}
