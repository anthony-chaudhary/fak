package vdso

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// helper to emit a Claude-native EvComplete event
func emitComplete(v *VDSO, tool string, argsJSON string, resultPayload string) {
	c := &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(argsJSON), Len: int64(len(argsJSON))},
		Meta: map[string]string{
			"readOnlyHint":   "true",
			"idempotentHint": "true",
		},
	}
	res := &abi.Result{
		Call:    c,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(resultPayload), Len: int64(len(resultPayload))},
		Status:  abi.StatusOK,
	}
	v.Emit(abi.Event{
		Kind:   abi.EvComplete,
		Call:   c,
		Result: res,
	})
}

// TestProactive_DeterministicFileReadServedInline verifies that deterministic file reads
// are intercepted and served inline from the vDSO cache with 0ms model latency, 0 remote tokens,
// and advanced turn state.
func TestProactive_DeterministicFileReadServedInline(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	testFile := filepath.Join(dir, "target.go")
	content := "package main\n\nfunc Answer() int { return 42 }\n"
	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	expectedResult := fmt.Sprintf(`{"content":%q}`, content)
	argsJSON := fmt.Sprintf(`{"filePath":%q}`, testFile)

	cases := []struct {
		name string
		turn TurnState
	}{
		{
			name: "PlanStep with plain read path",
			turn: TurnState{
				TurnIndex: 1,
				PlanStep:  "Read " + testFile,
			},
		},
		{
			name: "PlanStep with backticks",
			turn: TurnState{
				TurnIndex: 2,
				PlanStep:  "read `" + testFile + "`",
			},
		},
		{
			name: "PlanStep with inspect keyword",
			turn: TurnState{
				TurnIndex: 3,
				PlanStep:  "Inspect `" + testFile + "`",
			},
		},
		{
			name: "PlanStep with cat keyword",
			turn: TurnState{
				TurnIndex: 4,
				PlanStep:  "cat " + testFile,
			},
		},
		{
			name: "PlanStep with next step phrasing",
			turn: TurnState{
				TurnIndex: 5,
				PlanStep:  "Next step: read `" + testFile + "` to verify implementation.",
			},
		},
		{
			name: "PreviousOutput phrasing when PlanStep empty",
			turn: TurnState{
				TurnIndex:      6,
				PreviousOutput: "I will now read `" + testFile + "`.",
			},
		},
		{
			name: "Explicit TargetTool and TargetPath",
			turn: TurnState{
				TurnIndex:  7,
				TargetTool: "Read",
				TargetPath: testFile,
			},
		},
		{
			name: "JSON tool invocation in PlanStep",
			turn: TurnState{
				TurnIndex: 8,
				PlanStep:  fmt.Sprintf(`{"tool":"Read","filePath":%q}`, testFile),
			},
		},
		{
			name: "PlanStep solely the backticked file path",
			turn: TurnState{
				TurnIndex: 9,
				PlanStep:  "`" + testFile + "`",
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := New(64)
			emitComplete(v, ToolClaudeRead, argsJSON, expectedResult)

			interceptor := NewProactiveInterceptor(WithVDSO(v))
			result, ok := interceptor.Evaluate(ctx, tc.turn)
			if !ok || result == nil {
				t.Fatalf("expected proactive inline hit, got ok=%v, result=%v", ok, result)
			}

			if !result.ServedInline {
				t.Errorf("ServedInline = false, want true")
			}
			if result.ModelLatency != 0 {
				t.Errorf("ModelLatency = %v, want 0", result.ModelLatency)
			}
			if result.RemoteTokens != 0 {
				t.Errorf("RemoteTokens = %v, want 0", result.RemoteTokens)
			}
			if result.Tool != ToolClaudeRead {
				t.Errorf("Tool = %q, want %q", result.Tool, ToolClaudeRead)
			}
			if result.Path != testFile {
				t.Errorf("Path = %q, want %q", result.Path, testFile)
			}

			// Verify TurnState advanced directly
			if result.TurnState.TurnIndex != tc.turn.TurnIndex+1 {
				t.Errorf("TurnIndex = %d, want %d", result.TurnState.TurnIndex, tc.turn.TurnIndex+1)
			}
			if result.TurnState.PreviousOutput != expectedResult {
				t.Errorf("PreviousOutput = %q, want %q", result.TurnState.PreviousOutput, expectedResult)
			}
			if result.TurnState.PlanStep != "" {
				t.Errorf("PlanStep = %q, want empty (consumed)", result.TurnState.PlanStep)
			}
			if result.TurnState.Metadata["proactive_interception"] != "true" {
				t.Errorf("Metadata[proactive_interception] = %q, want true", result.TurnState.Metadata["proactive_interception"])
			}

			evals, hits, falls := interceptor.Stats()
			if evals != 1 || hits != 1 || falls != 0 {
				t.Errorf("Stats = (%d, %d, %d), want (1, 1, 0)", evals, hits, falls)
			}
		})
	}
}

// TestProactive_GlobAndGrepServedInline verifies Glob and Grep tools are served inline from vDSO cache.
func TestProactive_GlobAndGrepServedInline(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	file1 := filepath.Join(dir, "f1.go")
	_ = os.WriteFile(file1, []byte("func TestA() {}"), 0644)

	v := New(64)
	interceptor := NewProactiveInterceptor(WithVDSO(v))

	// 1. Glob inline serve
	globArgs := fmt.Sprintf(`{"pattern":"*.go","path":%q}`, dir)
	globResult := `{"matches":["f1.go"]}`
	emitComplete(v, ToolClaudeGlob, globArgs, globResult)

	turnGlob := TurnState{
		TurnIndex: 1,
		PlanStep:  fmt.Sprintf("Glob `*.go` in `%s`", dir),
	}
	resGlob, ok := interceptor.Evaluate(ctx, turnGlob)
	if !ok || resGlob == nil {
		t.Fatalf("glob inline evaluate failed")
	}
	if resGlob.Tool != ToolClaudeGlob {
		t.Errorf("Tool = %q, want %q", resGlob.Tool, ToolClaudeGlob)
	}
	if resGlob.Pattern != "*.go" {
		t.Errorf("Pattern = %q, want *.go", resGlob.Pattern)
	}

	// 2. Grep inline serve
	grepArgs := fmt.Sprintf(`{"pattern":"TestA","path":%q}`, dir)
	grepResult := `{"matches":[{"path":"f1.go","line":1}]}`
	emitComplete(v, ToolClaudeGrep, grepArgs, grepResult)

	turnGrep := TurnState{
		TurnIndex: 2,
		PlanStep:  fmt.Sprintf("Grep `TestA` in `%s`", dir),
	}
	resGrep, ok := interceptor.Evaluate(ctx, turnGrep)
	if !ok || resGrep == nil {
		t.Fatalf("grep inline evaluate failed")
	}
	if resGrep.Tool != ToolClaudeGrep {
		t.Errorf("Tool = %q, want %q", resGrep.Tool, ToolClaudeGrep)
	}
	if resGrep.Pattern != "TestA" {
		t.Errorf("Pattern = %q, want TestA", resGrep.Pattern)
	}
}

// TestProactive_StaleOrMissingFilesFallThrough verifies that stale files (modified on disk after caching),
// missing/deleted files, and non-existent files fall through to the remote model.
func TestProactive_StaleOrMissingFilesFallThrough(t *testing.T) {
	ctx := context.Background()

	t.Run("stale file modified on disk falls through", func(t *testing.T) {
		dir := t.TempDir()
		fPath := filepath.Join(dir, "stale_check.go")
		initialContent := "package v1\n"
		if err := os.WriteFile(fPath, []byte(initialContent), 0644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}

		v := New(64)
		argsJSON := fmt.Sprintf(`{"filePath":%q}`, fPath)
		emitComplete(v, ToolClaudeRead, argsJSON, initialContent)

		interceptor := NewProactiveInterceptor(WithVDSO(v))
		turn := TurnState{TurnIndex: 1, PlanStep: "Read " + fPath}

		// Initial evaluate hits inline
		res, ok := interceptor.Evaluate(ctx, turn)
		if !ok || res == nil {
			t.Fatalf("initial evaluation should hit inline")
		}

		// Update file on disk (stale)
		time.Sleep(15 * time.Millisecond) // ensure mtime tick
		updatedContent := "package v2 // updated\n"
		if err := os.WriteFile(fPath, []byte(updatedContent), 0644); err != nil {
			t.Fatalf("WriteFile update: %v", err)
		}

		// Subsequent evaluate MUST detect stale disk state and fall through
		resAfter, okAfter := interceptor.Evaluate(ctx, turn)
		if okAfter || resAfter != nil {
			t.Fatalf("expected stale file to fall through to remote model, got ok=%v, res=%v", okAfter, resAfter)
		}
	})

	t.Run("missing file deleted on disk falls through", func(t *testing.T) {
		dir := t.TempDir()
		fPath := filepath.Join(dir, "delete_check.go")
		if err := os.WriteFile(fPath, []byte("transient"), 0644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}

		v := New(64)
		argsJSON := fmt.Sprintf(`{"filePath":%q}`, fPath)
		emitComplete(v, ToolClaudeRead, argsJSON, "transient")

		interceptor := NewProactiveInterceptor(WithVDSO(v))
		turn := TurnState{TurnIndex: 1, PlanStep: "Read `" + fPath + "`"}

		// Initial evaluate hits
		if _, ok := interceptor.Evaluate(ctx, turn); !ok {
			t.Fatalf("initial evaluation should hit inline")
		}

		// Delete file
		if err := os.Remove(fPath); err != nil {
			t.Fatalf("Remove: %v", err)
		}

		// Next evaluate MUST detect missing file on disk and fall through
		resAfter, okAfter := interceptor.Evaluate(ctx, turn)
		if okAfter || resAfter != nil {
			t.Fatalf("expected deleted file to fall through to remote model, got ok=%v", okAfter)
		}
	})

	t.Run("nonexistent file falls through", func(t *testing.T) {
		v := New(64)
		interceptor := NewProactiveInterceptor(WithVDSO(v))

		nonexistent := "/tmp/definitely_not_existing_path_fak_9999.go"
		turn := TurnState{TurnIndex: 1, PlanStep: "Read " + nonexistent}

		res, ok := interceptor.Evaluate(ctx, turn)
		if ok || res != nil {
			t.Fatalf("expected nonexistent file to fall through to remote model, got ok=%v", ok)
		}
	})

	t.Run("mutating operations fall through", func(t *testing.T) {
		v := New(64)
		interceptor := NewProactiveInterceptor(WithVDSO(v))

		mutatingSteps := []string{
			"Edit `internal/vdso/proactive.go`",
			"Write new file `foo.txt`",
			"Delete file `bar.txt`",
			"Update `config.json` with new port",
			"rm -rf scratch",
			"Modify `example.go`",
		}

		for _, step := range mutatingSteps {
			turn := TurnState{TurnIndex: 1, PlanStep: step}
			res, ok := interceptor.Evaluate(ctx, turn)
			if ok || res != nil {
				t.Errorf("step %q should fall through (mutating action), got ok=%v", step, ok)
			}
		}
	})

	t.Run("ambiguous multiple files fall through", func(t *testing.T) {
		v := New(64)
		interceptor := NewProactiveInterceptor(WithVDSO(v))

		ambiguousSteps := []string{
			"Read `fileA.go` and `fileB.go`",
			"Compare `foo.txt` with `bar.txt`",
			"Check whether `a.go` or `b.go` has the fix",
		}

		for _, step := range ambiguousSteps {
			turn := TurnState{TurnIndex: 1, PlanStep: step}
			res, ok := interceptor.Evaluate(ctx, turn)
			if ok || res != nil {
				t.Errorf("step %q should fall through (ambiguous multiple targets), got ok=%v", step, ok)
			}
		}
	})

	t.Run("conversational prose without target falls through", func(t *testing.T) {
		v := New(64)
		interceptor := NewProactiveInterceptor(WithVDSO(v))

		proseSteps := []string{
			"I will analyze the user requirements and formulate a plan.",
			"Let's summarize the benchmark findings.",
			"Everything looks clean and ready for deployment.",
		}

		for _, step := range proseSteps {
			turn := TurnState{TurnIndex: 1, PlanStep: step}
			res, ok := interceptor.Evaluate(ctx, turn)
			if ok || res != nil {
				t.Errorf("prose %q should fall through, got ok=%v", step, ok)
			}
		}
	})
}

// TestProactive_ZeroChoiceDiskRead verifies that when WithZeroChoiceDiskRead is enabled,
// an unambiguous file present on disk but not yet in vDSO is deterministically read,
// cached, and returned inline.
func TestProactive_ZeroChoiceDiskRead(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	testFile := filepath.Join(dir, "disk_only.txt")
	content := "deterministic content from disk\n"
	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	v := New(64)
	interceptor := NewProactiveInterceptor(
		WithVDSO(v),
		WithZeroChoiceDiskRead(true),
	)

	turn := TurnState{
		TurnIndex: 1,
		PlanStep:  "Read `" + testFile + "`",
	}

	res, ok := interceptor.Evaluate(ctx, turn)
	if !ok || res == nil {
		t.Fatalf("expected zero-choice disk read to succeed inline, got ok=%v", ok)
	}
	if string(res.Content) != content {
		t.Errorf("Content = %q, want %q", string(res.Content), content)
	}

	// Verify it populated vDSO cache so subsequent standard lookup hits tier-2
	call := &abi.ToolCall{
		Tool: ToolClaudeRead,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(fmt.Sprintf(`{"filePath":%q}`, testFile))},
	}
	cachedRes, hit := v.Lookup(ctx, call)
	if !hit || cachedRes == nil {
		t.Fatalf("vDSO should now cache the zero-choice read result")
	}
}

// TestProactive_WorkDirResolution verifies relative path resolution against TurnState.WorkDir.
func TestProactive_WorkDirResolution(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	relName := "nested/file.go"
	fullPath := filepath.Join(dir, relName)
	if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	content := "package nested\n"
	if err := os.WriteFile(fullPath, []byte(content), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	v := New(64)
	argsJSON := fmt.Sprintf(`{"filePath":%q}`, relName)
	emitComplete(v, ToolClaudeRead, argsJSON, content)

	interceptor := NewProactiveInterceptor(WithVDSO(v), WithWorkDir(dir))

	turn := TurnState{
		TurnIndex: 1,
		WorkDir:   dir,
		PlanStep:  "Read `" + relName + "`",
	}

	res, ok := interceptor.Evaluate(ctx, turn)
	if !ok || res == nil {
		t.Fatalf("expected relative path to resolve and hit inline, got ok=%v", ok)
	}
	if string(res.Content) != content {
		t.Errorf("Content = %q, want %q", string(res.Content), content)
	}
}

// TestProactive_ConcurrencyHighChurn verifies concurrency safety under high churn
// with concurrent reads, file modifications, cache updates, and missing file probes.
func TestProactive_ConcurrencyHighChurn(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	const numFiles = 8
	filePaths := make([]string, numFiles)
	for i := 0; i < numFiles; i++ {
		p := filepath.Join(dir, fmt.Sprintf("churn_%d.go", i))
		c := fmt.Sprintf("package churn%d\n", i)
		if err := os.WriteFile(p, []byte(c), 0644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		filePaths[i] = p
	}

	v := New(128)
	interceptor := NewProactiveInterceptor(WithVDSO(v))

	// Prime cache
	for i, p := range filePaths {
		args := fmt.Sprintf(`{"filePath":%q}`, p)
		emitComplete(v, ToolClaudeRead, args, fmt.Sprintf("package churn%d\n", i))
	}

	var wg sync.WaitGroup
	workers := 40
	iterations := 100

	for w := 0; w < workers; w++ {
		wg.Add(1)
		workerID := w
		go func() {
			defer wg.Done()
			for it := 0; it < iterations; it++ {
				fileIdx := (workerID + it) % numFiles
				targetPath := filePaths[fileIdx]

				switch workerID % 4 {
				case 0:
					// Read evaluation
					turn := TurnState{
						TurnIndex: it,
						PlanStep:  "Read `" + targetPath + "`",
					}
					_, _ = interceptor.Evaluate(ctx, turn)

				case 1:
					// File churn on disk
					if it%10 == 0 {
						newContent := fmt.Sprintf("// churn iteration %d\n", it)
						_ = os.WriteFile(targetPath, []byte(newContent), 0644)
					}

				case 2:
					// Cache re-emit
					args := fmt.Sprintf(`{"filePath":%q}`, targetPath)
					emitComplete(v, ToolClaudeRead, args, fmt.Sprintf("// re-emitted %d\n", it))

				case 3:
					// Non-existent or mutating probe
					if it%2 == 0 {
						turn := TurnState{
							TurnIndex: it,
							PlanStep:  "Edit `nonexistent_file.go`",
						}
						_, _ = interceptor.Evaluate(ctx, turn)
					} else {
						turn := TurnState{
							TurnIndex: it,
							PlanStep:  "Read `/tmp/fak_missing_churn_test.txt`",
						}
						_, _ = interceptor.Evaluate(ctx, turn)
					}
				}
			}
		}()
	}

	wg.Wait()

	evals, hits, falls := interceptor.Stats()
	if evals == 0 {
		t.Fatalf("expected non-zero evaluations, got 0")
	}
	t.Logf("Concurrency test completed: evals=%d, hits=%d, falls=%d", evals, hits, falls)
}
