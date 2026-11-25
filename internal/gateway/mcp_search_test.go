//go:build wip_mcp_search

package gateway

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestMCPGrepAndGlob(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})

	// 1. Create a temporary workspace root with sample files
	root := t.TempDir()
	subdir := filepath.Join(root, "subdir")
	if err := os.MkdirAll(subdir, 0o755); err != nil {
		t.Fatal(err)
	}
	fileA := filepath.Join(root, "fileA.txt")
	fileB := filepath.Join(subdir, "fileB.go")
	if err := os.WriteFile(fileA, []byte("hello world\nalpha beta\nneedle in fileA\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fileB, []byte("package main\n\nfunc needle() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Register confined search engines to the workspace root
	if _, err := RegisterSearchEngines(root); err != nil {
		t.Fatalf("RegisterSearchEngines: %v", err)
	}

	srv, err := New(Config{EngineID: "codetools.grep", Model: "test-model", VDSO: true, NativeCodeWorkspace: root})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer srv.Close()
	ctx := context.Background()

	// -------------------------------------------------------------
	// 1. Single fak_grep call via handleMethod / dispatchRPC (<10ms)
	// -------------------------------------------------------------
	grepRPC := `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"fak_grep","arguments":{"pattern":"needle"}}}`
	startGrep := time.Now()
	respGrep := srv.dispatchRPC(ctx, []byte(grepRPC))
	grepDur := time.Since(startGrep)
	if respGrep == nil || respGrep.Error != nil {
		t.Fatalf("dispatchRPC fak_grep failed: %v", respGrep)
	}
	if grepDur > 100*time.Millisecond {
		t.Errorf("fak_grep took %v, expected fast execution (<10ms on warm disk)", grepDur)
	}

	var grepSyscall FakSearchResponse
	decodeMCPText(t, respGrep.Result, &grepSyscall)
	if grepSyscall.Receipt == nil {
		// Also inspect inside Result.Content if top-level Receipt is empty
		var body map[string]any
		if grepSyscall.Result != nil {
			_ = json.Unmarshal([]byte(grepSyscall.Result.Content), &body)
			if r, ok := body["receipt"].(map[string]any); ok {
				grepSyscall.Receipt = r
			}
		}
	}
	if grepSyscall.Receipt == nil {
		t.Fatalf("fak_grep response missing fak-search-receipt/1 receipt: %+v", grepSyscall)
	}
	if schema, _ := grepSyscall.Receipt["schema"].(string); schema != "fak-search-receipt/1" {
		t.Errorf("receipt schema = %q, want fak-search-receipt/1", schema)
	}
	if outcome, _ := grepSyscall.Receipt["outcome"].(string); outcome != "executed_search" && outcome != "verified_fresh_reuse" {
		t.Errorf("receipt outcome = %q, want executed_search", outcome)
	}
	if witness, _ := grepSyscall.Receipt["witness"].(string); witness != "codetools" && witness != "vdso" {
		t.Errorf("receipt witness = %q, want codetools", witness)
	}
	if conf, _ := grepSyscall.Receipt["confinement_verified"].(bool); !conf {
		t.Errorf("receipt confinement_verified = false, want true")
	}
	matchesCount := 0
	if mc, ok := grepSyscall.Receipt["matches"].(float64); ok {
		matchesCount = int(mc)
	} else if mc, ok := grepSyscall.Receipt["matches"].(int); ok {
		matchesCount = mc
	}
	if matchesCount != 2 {
		t.Errorf("receipt matches = %d, want 2 (needle in fileA and fileB)", matchesCount)
	}

	// -------------------------------------------------------------
	// 2. Single fak_glob call via handleMethod / dispatchRPC (<10ms)
	// -------------------------------------------------------------
	globRPC := `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fak_glob","arguments":{"pattern":"**/*.go"}}}`
	startGlob := time.Now()
	respGlob := srv.dispatchRPC(ctx, []byte(globRPC))
	globDur := time.Since(startGlob)
	if respGlob == nil || respGlob.Error != nil {
		t.Fatalf("dispatchRPC fak_glob failed: %v", respGlob)
	}
	if globDur > 100*time.Millisecond {
		t.Errorf("fak_glob took %v, expected fast execution (<10ms on warm disk)", globDur)
	}

	var globSyscall FakSearchResponse
	decodeMCPText(t, respGlob.Result, &globSyscall)
	if globSyscall.Receipt == nil {
		var body map[string]any
		if globSyscall.Result != nil {
			_ = json.Unmarshal([]byte(globSyscall.Result.Content), &body)
			if r, ok := body["receipt"].(map[string]any); ok {
				globSyscall.Receipt = r
			}
		}
	}
	if globSyscall.Receipt == nil {
		t.Fatalf("fak_glob response missing fak-search-receipt/1 receipt: %+v", globSyscall)
	}
	if schema, _ := globSyscall.Receipt["schema"].(string); schema != "fak-search-receipt/1" {
		t.Errorf("receipt schema = %q, want fak-search-receipt/1", schema)
	}
	if conf, _ := globSyscall.Receipt["confinement_verified"].(bool); !conf {
		t.Errorf("receipt confinement_verified = false, want true")
	}
	filesCount := 0
	if fc, ok := globSyscall.Receipt["files"].(float64); ok {
		filesCount = int(fc)
	} else if fc, ok := globSyscall.Receipt["files"].(int); ok {
		filesCount = fc
	}
	if filesCount != 1 {
		t.Errorf("receipt files = %d, want 1 (*.go -> fileB.go)", filesCount)
	}

	// -------------------------------------------------------------
	// 3. Path confinement: calling with path "../../" outside workspace
	// -------------------------------------------------------------
	escapeGrepRPC := `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fak_grep","arguments":{"pattern":"needle","path":"../../"}}}`
	respEscapeGrep := srv.dispatchRPC(ctx, []byte(escapeGrepRPC))
	if respEscapeGrep != nil && respEscapeGrep.Error == nil {
		var escapeSyscall FakSearchResponse
		decodeMCPText(t, respEscapeGrep.Result, &escapeSyscall)
		var confVerified bool = true
		if escapeSyscall.Receipt != nil {
			confVerified, _ = escapeSyscall.Receipt["confinement_verified"].(bool)
		} else if escapeSyscall.Result != nil {
			var body map[string]any
			_ = json.Unmarshal([]byte(escapeSyscall.Result.Content), &body)
			if r, ok := body["receipt"].(map[string]any); ok {
				confVerified, _ = r["confinement_verified"].(bool)
			}
		}
		if confVerified && (escapeSyscall.Result == nil || escapeSyscall.Result.Status != "ERROR") {
			t.Errorf("path confinement failed: expected error or confinement_verified=false for path ../../")
		}
	}

	escapeGlobRPC := `{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"fak_glob","arguments":{"pattern":"*.go","path":"../../"}}}`
	respEscapeGlob := srv.dispatchRPC(ctx, []byte(escapeGlobRPC))
	if respEscapeGlob != nil && respEscapeGlob.Error == nil {
		var escapeSyscall FakSearchResponse
		decodeMCPText(t, respEscapeGlob.Result, &escapeSyscall)
		var confVerified bool = true
		if escapeSyscall.Receipt != nil {
			confVerified, _ = escapeSyscall.Receipt["confinement_verified"].(bool)
		} else if escapeSyscall.Result != nil {
			var body map[string]any
			_ = json.Unmarshal([]byte(escapeSyscall.Result.Content), &body)
			if r, ok := body["receipt"].(map[string]any); ok {
				confVerified, _ = r["confinement_verified"].(bool)
			}
		}
		if confVerified && (escapeSyscall.Result == nil || escapeSyscall.Result.Status != "ERROR") {
			t.Errorf("path confinement failed: expected error or confinement_verified=false for path ../../")
		}
	}

	// -------------------------------------------------------------
	// 4. Batch form support (#11500)
	// -------------------------------------------------------------
	// 4a. fak_grep with patterns: []string
	batchPatternsGrepRPC := `{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"fak_grep","arguments":{"patterns":["world","package"]}}}`
	respBatchGrep := srv.dispatchRPC(ctx, []byte(batchPatternsGrepRPC))
	if respBatchGrep == nil || respBatchGrep.Error != nil {
		t.Fatalf("dispatchRPC batch fak_grep patterns failed: %v", respBatchGrep)
	}
	var batchGrepResp FakSearchBatchResponse
	decodeMCPText(t, respBatchGrep.Result, &batchGrepResp)
	if batchGrepResp.ItemCount != 2 || len(batchGrepResp.Results) != 2 {
		t.Fatalf("expected 2 batch results, got %d", len(batchGrepResp.Results))
	}
	if batchGrepResp.Results[0].Receipt == nil || batchGrepResp.Results[1].Receipt == nil {
		t.Fatalf("batch items missing receipts: %+v", batchGrepResp)
	}

	// 4b. fak_grep with queries: []mcpGrepQuery
	batchQueriesGrepRPC := `{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"fak_grep","arguments":{"queries":[{"pattern":"world"},{"pattern":"needle","glob":"*.go"}]}}}`
	respQueriesGrep := srv.dispatchRPC(ctx, []byte(batchQueriesGrepRPC))
	if respQueriesGrep == nil || respQueriesGrep.Error != nil {
		t.Fatalf("dispatchRPC batch fak_grep queries failed: %v", respQueriesGrep)
	}
	var queriesGrepResp FakSearchBatchResponse
	decodeMCPText(t, respQueriesGrep.Result, &queriesGrepResp)
	if queriesGrepResp.ItemCount != 2 || len(queriesGrepResp.Results) != 2 {
		t.Fatalf("expected 2 query batch results, got %d", len(queriesGrepResp.Results))
	}

	// 4c. fak_glob with patterns: []string
	batchGlobRPC := `{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"fak_glob","arguments":{"patterns":["*.txt","**/*.go"]}}}`
	respBatchGlob := srv.dispatchRPC(ctx, []byte(batchGlobRPC))
	if respBatchGlob == nil || respBatchGlob.Error != nil {
		t.Fatalf("dispatchRPC batch fak_glob patterns failed: %v", respBatchGlob)
	}
	var batchGlobResp FakSearchBatchResponse
	decodeMCPText(t, respBatchGlob.Result, &batchGlobResp)
	if batchGlobResp.ItemCount != 2 || len(batchGlobResp.Results) != 2 {
		t.Fatalf("expected 2 batch glob results, got %d", len(batchGlobResp.Results))
	}
	if batchGlobResp.Results[0].Receipt == nil || batchGlobResp.Results[1].Receipt == nil {
		t.Fatalf("batch glob items missing receipts: %+v", batchGlobResp)
	}
}
