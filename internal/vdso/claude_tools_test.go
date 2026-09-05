package vdso

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// claudeCall builds a ToolCall for a Claude-native tool without explicit hints.
func claudeCall(tool string, args string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args), Len: int64(len(args))},
	}
}

// claudeCompleteEvent builds an EvComplete event for a Claude-native call.
func claudeCompleteEvent(c *abi.ToolCall, result string) abi.Event {
	return abi.Event{
		Kind: abi.EvComplete,
		Call: c,
		Result: &abi.Result{
			Call:    c,
			Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(result), Len: int64(len(result))},
			Status:  abi.StatusOK,
		},
	}
}

// TestClaudeRead_CacheHitAndBitIdentity verifies that a Claude-native Read tool call
// is cached on EvComplete and served on subsequent Lookup with bit-identical output.
func TestClaudeRead_CacheHitAndBitIdentity(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)

	dir := t.TempDir()
	filePath := filepath.Join(dir, "example.go")
	content := "package main\n\nfunc Hello() string { return \"world\" }\n"
	if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	callCamel := claudeCall("Read", fmt.Sprintf(`{"filePath":%q}`, filePath))
	expectedResult := fmt.Sprintf(`{"content":%q}`, content)

	// Turn 1: Cold lookup misses
	ctx := context.Background()
	if _, ok := v.Lookup(ctx, callCamel); ok {
		t.Fatalf("first lookup should miss on empty cache")
	}

	// Turn 1 completes and fills vDSO
	v.Emit(claudeCompleteEvent(callCamel, expectedResult))

	// Turn 2: Lookup with camelCase filePath hits cache
	res, ok := v.Lookup(ctx, callCamel)
	if !ok || res == nil {
		t.Fatalf("second lookup missed; want cache hit")
	}
	if res.Meta["served_by"] != "vdso" || res.Meta["tier"] != "2" {
		t.Fatalf("res.Meta = %+v; want served_by:vdso, tier:2", res.Meta)
	}
	if string(res.Payload.Inline) != expectedResult {
		t.Fatalf("bit-identity mismatch: got %q, want %q", string(res.Payload.Inline), expectedResult)
	}

	// Turn 3: Lookup with snake_case file_path also hits (same file entity and canonicalization)
	callSnake := claudeCall("Read", fmt.Sprintf(`{"file_path":%q}`, filePath))
	v.Emit(claudeCompleteEvent(callSnake, expectedResult))
	resSnake, ok := v.Lookup(ctx, callSnake)
	if !ok || resSnake == nil {
		t.Fatalf("snake_case lookup missed; want cache hit")
	}
	if string(resSnake.Payload.Inline) != expectedResult {
		t.Fatalf("bit-identity mismatch: got %q, want %q", string(resSnake.Payload.Inline), expectedResult)
	}
}

// TestClaudeRead_WindowSlicesAreIndependent verifies that Read with offset/limit windows
// cache distinct slices independently without cross-slice collisions.
func TestClaudeRead_WindowSlicesAreIndependent(t *testing.T) {
	v := New(64)
	dir := t.TempDir()
	filePath := filepath.Join(dir, "lines.txt")
	content := "line1\nline2\nline3\nline4\nline5\n"
	if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	slice1 := claudeCall("Read", fmt.Sprintf(`{"filePath":%q,"offset":1,"limit":2}`, filePath))
	slice2 := claudeCall("Read", fmt.Sprintf(`{"filePath":%q,"offset":3,"limit":2}`, filePath))

	v.Emit(claudeCompleteEvent(slice1, `{"lines":["line1","line2"]}`))
	v.Emit(claudeCompleteEvent(slice2, `{"lines":["line3","line4"]}`))

	ctx := context.Background()
	r1, ok1 := v.Lookup(ctx, slice1)
	if !ok1 || string(r1.Payload.Inline) != `{"lines":["line1","line2"]}` {
		t.Fatalf("slice1 lookup failed: ok=%v, payload=%q", ok1, string(r1.Payload.Inline))
	}

	r2, ok2 := v.Lookup(ctx, slice2)
	if !ok2 || string(r2.Payload.Inline) != `{"lines":["line3","line4"]}` {
		t.Fatalf("slice2 lookup failed: ok=%v, payload=%q", ok2, string(r2.Payload.Inline))
	}
}

// TestClaudeRead_InvalidationOnFileModificationOnDisk tests that modifying a file on disk
// causes the next Lookup to strictly invalidate the cached entry (content hash and mtime changed).
func TestClaudeRead_InvalidationOnFileModificationOnDisk(t *testing.T) {
	v := New(64)
	dir := t.TempDir()
	filePath := filepath.Join(dir, "config.json")
	initialContent := `{"version":1,"env":"dev"}`
	if err := os.WriteFile(filePath, []byte(initialContent), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	call := claudeCall("Read", fmt.Sprintf(`{"filePath":%q}`, filePath))
	v.Emit(claudeCompleteEvent(call, initialContent))

	ctx := context.Background()
	// Confirm cache hit before modification
	res, ok := v.Lookup(ctx, call)
	if !ok || string(res.Payload.Inline) != initialContent {
		t.Fatalf("pre-modification lookup should hit")
	}

	// External modification on disk (outside fak)
	updatedContent := `{"version":2,"env":"prod"}`
	if err := os.WriteFile(filePath, []byte(updatedContent), 0644); err != nil {
		t.Fatalf("WriteFile update: %v", err)
	}
	newMtime := time.Now().Add(2 * time.Second)
	if err := os.Chtimes(filePath, newMtime, newMtime); err != nil {
		t.Fatalf("Chtimes: %v", err)
	}

	// Next lookup must detect mtime/content-hash change, evict entry, and return miss
	if _, ok := v.Lookup(ctx, call); ok {
		t.Fatalf("lookup hit on modified file; want strict invalidation (miss)")
	}

	misses := v.MissReasons()
	if misses[MissWitnessRevoked] == 0 && misses[MissNotCached] == 0 {
		t.Fatalf("expected MissWitnessRevoked or MissNotCached, got: %+v", misses)
	}
}

// TestClaudeRead_InvalidationOnFileDeletionOnDisk tests that deleting a file on disk
// causes the next Lookup to strictly invalidate the cached entry.
func TestClaudeRead_InvalidationOnFileDeletionOnDisk(t *testing.T) {
	v := New(64)
	dir := t.TempDir()
	filePath := filepath.Join(dir, "transient.txt")
	if err := os.WriteFile(filePath, []byte("temporary data"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	call := claudeCall("Read", fmt.Sprintf(`{"filePath":%q}`, filePath))
	v.Emit(claudeCompleteEvent(call, `{"data":"temporary data"}`))

	ctx := context.Background()
	if _, ok := v.Lookup(ctx, call); !ok {
		t.Fatalf("pre-deletion lookup should hit")
	}

	// External deletion on disk
	if err := os.Remove(filePath); err != nil {
		t.Fatalf("Remove: %v", err)
	}

	// Next lookup must detect file does not exist, evict entry, and return miss
	if _, ok := v.Lookup(ctx, call); ok {
		t.Fatalf("lookup hit on deleted file; want strict invalidation (miss)")
	}
}

// TestClaudeRead_CausalRevocationOnWriteOrEdit tests that tool writes via Write or Edit
// trigger causal revocation (Revoke), evicting cached entries and advancing the trust epoch.
func TestClaudeRead_CausalRevocationOnWriteOrEdit(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)

	dir := t.TempDir()
	filePathA := filepath.Join(dir, "target.go")
	filePathB := filepath.Join(dir, "sibling.go")

	_ = os.WriteFile(filePathA, []byte("package target\n"), 0644)
	_ = os.WriteFile(filePathB, []byte("package sibling\n"), 0644)

	callA := claudeCall("Read", fmt.Sprintf(`{"filePath":%q}`, filePathA))
	callB := claudeCall("Read", fmt.Sprintf(`{"filePath":%q}`, filePathB))

	v.Emit(claudeCompleteEvent(callA, "package target\n"))
	v.Emit(claudeCompleteEvent(callB, "package sibling\n"))

	ctx := context.Background()
	if _, ok := v.Lookup(ctx, callA); !ok {
		t.Fatalf("read A should hit initially")
	}
	if _, ok := v.Lookup(ctx, callB); !ok {
		t.Fatalf("read B should hit initially")
	}

	// Subscribe to the coherence integrity bus to verify Revocation event emission
	var receivedRevocations []Revocation
	unsub := v.SubscribeRevocations(func(r Revocation) {
		receivedRevocations = append(receivedRevocations, r)
	})
	defer unsub()

	initialTrustEpoch := v.TrustEpoch()

	// Tool-mediated write to target.go via Edit
	editCall := claudeCall("Edit", fmt.Sprintf(`{"filePath":%q,"oldString":"target","newString":"updated"}`, filePathA))
	v.Emit(claudeCompleteEvent(editCall, `{"ok":true}`))

	// Trust epoch must have advanced
	if v.TrustEpoch() <= initialTrustEpoch {
		t.Fatalf("trustEpoch did not advance after Edit: before=%d, now=%d", initialTrustEpoch, v.TrustEpoch())
	}

	// Coherence bus must have received Revocation
	if len(receivedRevocations) == 0 {
		t.Fatalf("expected Revocation event on coherence bus, got none")
	}

	// Read A must now MISS because it was causally revoked
	if _, ok := v.Lookup(ctx, callA); ok {
		t.Fatalf("read A hit after Edit on target.go; want causal revocation (miss)")
	}

	// Read B (sibling file) must remain warm (unaffected)
	resB, okB := v.Lookup(ctx, callB)
	if !okB || resB == nil {
		t.Fatalf("read B missed after Edit on target.go; sibling file should stay cached")
	}

	// Now execute Write on sibling.go
	writeCall := claudeCall("Write", fmt.Sprintf(`{"filePath":%q,"content":"package sibling2\n"}`, filePathB))
	v.Emit(claudeCompleteEvent(writeCall, `{"ok":true}`))

	// Read B must now MISS as well
	if _, ok := v.Lookup(ctx, callB); ok {
		t.Fatalf("read B hit after Write on sibling.go; want causal revocation (miss)")
	}
}

// TestClaudeGlob_CacheHitAndDirectoryInvalidation tests Glob caching and invalidation
// when files within the directory change or are created.
func TestClaudeGlob_CacheHitAndDirectoryInvalidation(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)

	dir := t.TempDir()
	subDir := filepath.Join(dir, "src")
	if err := os.MkdirAll(subDir, 0755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	file1 := filepath.Join(subDir, "a.go")
	if err := os.WriteFile(file1, []byte("package src"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	globCall := claudeCall("Glob", fmt.Sprintf(`{"path":%q,"pattern":"*.go"}`, subDir))
	result1 := fmt.Sprintf(`{"files":[%q]}`, file1)

	ctx := context.Background()
	if _, ok := v.Lookup(ctx, globCall); ok {
		t.Fatalf("cold glob should miss")
	}

	v.Emit(claudeCompleteEvent(globCall, result1))

	// Second lookup hits
	res, ok := v.Lookup(ctx, globCall)
	if !ok || string(res.Payload.Inline) != result1 {
		t.Fatalf("glob lookup failed: ok=%v, payload=%q", ok, string(res.Payload.Inline))
	}

	// Writing a new file in subDir via Write tool must invalidate Glob(subDir)
	file2 := filepath.Join(subDir, "b.go")
	writeCall := claudeCall("Write", fmt.Sprintf(`{"filePath":%q,"content":"package src"}`, file2))
	v.Emit(claudeCompleteEvent(writeCall, `{"ok":true}`))

	if _, ok := v.Lookup(ctx, globCall); ok {
		t.Fatalf("Glob should miss after Write to file in directory")
	}
}

// TestClaudeGrep_CacheHitAndInvalidation tests Grep caching and invalidation when
// file content in the target directory changes.
func TestClaudeGrep_CacheHitAndInvalidation(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)

	dir := t.TempDir()
	pkgDir := filepath.Join(dir, "pkg")
	if err := os.MkdirAll(pkgDir, 0755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	file := filepath.Join(pkgDir, "handler.go")
	_ = os.WriteFile(file, []byte("func HandleRequest() {}\n"), 0644)

	grepCall := claudeCall("Grep", fmt.Sprintf(`{"path":%q,"pattern":"HandleRequest","include":"*.go"}`, pkgDir))
	grepResult := fmt.Sprintf(`{"matches":[{"file":%q,"line":1}]}`, file)

	ctx := context.Background()
	if _, ok := v.Lookup(ctx, grepCall); ok {
		t.Fatalf("cold grep should miss")
	}

	v.Emit(claudeCompleteEvent(grepCall, grepResult))

	// Cache hit
	res, ok := v.Lookup(ctx, grepCall)
	if !ok || string(res.Payload.Inline) != grepResult {
		t.Fatalf("grep cache hit failed: ok=%v", ok)
	}

	// Edit the file via Edit tool
	editCall := claudeCall("Edit", fmt.Sprintf(`{"filePath":%q,"oldString":"HandleRequest","newString":"ServeHTTP"}`, file))
	v.Emit(claudeCompleteEvent(editCall, `{"ok":true}`))

	// Grep must miss
	if _, ok := v.Lookup(ctx, grepCall); ok {
		t.Fatalf("grep should miss after file in directory was edited")
	}
}

// TestClaudeBash_ReadOnlyCommandsCacheAndMutatingCommandsInvalidate tests read-only Bash
// command caching (cat, head, ls) and invalidation by mutating Bash commands.
func TestClaudeBash_ReadOnlyCommandsCacheAndMutatingCommandsInvalidate(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)

	dir := t.TempDir()
	file := filepath.Join(dir, "test.txt")
	_ = os.WriteFile(file, []byte("first line\nsecond line\n"), 0644)

	cleanFile := filepath.ToSlash(file)
	catCall := claudeCall("Bash", fmt.Sprintf(`{"command":"cat %s"}`, cleanFile))
	expected := "first line\nsecond line\n"

	ctx := context.Background()
	if _, ok := v.Lookup(ctx, catCall); ok {
		t.Fatalf("cold cat should miss")
	}

	v.Emit(claudeCompleteEvent(catCall, expected))

	// Cache hit
	res, ok := v.Lookup(ctx, catCall)
	if !ok || res == nil || string(res.Payload.Inline) != expected {
		t.Fatalf("cat lookup failed: ok=%v, payload=%v", ok, res)
	}

	// External write via mutating bash command (e.g. echo append)
	mutCall := claudeCall("Bash", fmt.Sprintf(`{"command":"echo third >> %s"}`, cleanFile))
	v.Emit(claudeCompleteEvent(mutCall, `{"ok":true}`))

	// catCall must be invalidated
	if _, ok := v.Lookup(ctx, catCall); ok {
		t.Fatalf("catCall should miss after mutating bash write")
	}
}

// TestClaudeSpeculatable tests Speculatable with Claude-native read tools.
func TestClaudeSpeculatable(t *testing.T) {
	cases := []struct {
		name string
		call *abi.ToolCall
		want bool
	}{
		{"Read is speculatable", claudeCall("Read", `{"filePath":"a.go"}`), true},
		{"Grep is speculatable", claudeCall("Grep", `{"pattern":"foo"}`), true},
		{"Glob is speculatable", claudeCall("Glob", `{"pattern":"*.go"}`), true},
		{"Read-only Bash is speculatable", claudeCall("Bash", `{"command":"cat a.go"}`), true},
		{"Mutating Bash is NOT speculatable", claudeCall("Bash", `{"command":"rm a.go"}`), false},
		{"Write is NOT speculatable", claudeCall("Write", `{"filePath":"a.go","content":"x"}`), false},
		{"Edit is NOT speculatable", claudeCall("Edit", `{"filePath":"a.go","oldString":"a","newString":"b"}`), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reason, ok := Speculatable(tc.call)
			if ok != tc.want {
				t.Fatalf("Speculatable(%s) = %v (reason %s); want %v", tc.call.Tool, ok, reason, tc.want)
			}
		})
	}
}

// TestClaudeArgumentExtraction verifies helper extractions across tool varieties.
func TestClaudeArgumentExtraction(t *testing.T) {
	// Read with camelCase filePath
	args1 := []byte(`{"filePath":"/path/to/main.go","offset":10,"limit":50}`)
	if got := ExtractToolPath(args1); got != "/path/to/main.go" {
		t.Errorf("ExtractToolPath(filePath) = %q, want /path/to/main.go", got)
	}

	// Read with snake_case file_path
	args2 := []byte(`{"file_path":"src/index.ts"}`)
	if got := ExtractToolPath(args2); got != "src/index.ts" {
		t.Errorf("ExtractToolPath(file_path) = %q, want src/index.ts", got)
	}

	// Glob with pattern and directory path
	args3 := []byte(`{"path":"./internal","pattern":"**/*.go"}`)
	if got := ExtractToolDirectory(args3); got != "internal" {
		t.Errorf("ExtractToolDirectory = %q, want internal", got)
	}
	if got := ExtractToolPattern(args3); got != "**/*.go" {
		t.Errorf("ExtractToolPattern = %q, want **/*.go", got)
	}

	// Grep with regex and include
	args4 := []byte(`{"path":"cmd","pattern":"func [A-Z]","include":"*.go"}`)
	if got := ExtractToolDirectory(args4); got != "cmd" {
		t.Errorf("ExtractToolDirectory = %q, want cmd", got)
	}
	if got := ExtractToolPattern(args4); got != "func [A-Z]" {
		t.Errorf("ExtractToolPattern = %q, want func [A-Z]", got)
	}
	if got := ExtractToolInclude(args4); got != "*.go" {
		t.Errorf("ExtractToolInclude = %q, want *.go", got)
	}

	// Bash with command
	args5 := []byte(`{"command":"git status"}`)
	if got := ExtractToolCommand(args5); got != "git status" {
		t.Errorf("ExtractToolCommand = %q, want git status", got)
	}
}

// Benchmarks

func BenchmarkClaudeReadCacheHit(b *testing.B) {
	v := New(1024)
	dir := b.TempDir()
	filePath := filepath.Join(dir, "bench.txt")
	content := strings.Repeat("hello world benchmark line\n", 50)
	_ = os.WriteFile(filePath, []byte(content), 0644)

	call := claudeCall("Read", fmt.Sprintf(`{"filePath":%q}`, filePath))
	v.Emit(claudeCompleteEvent(call, content))

	ctx := context.Background()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res, ok := v.Lookup(ctx, call)
		if !ok || res == nil {
			b.Fatalf("lookup failed")
		}
	}
}

func BenchmarkClaudeGlobCacheHit(b *testing.B) {
	v := New(1024)
	dir := b.TempDir()
	for i := 0; i < 10; i++ {
		_ = os.WriteFile(filepath.Join(dir, fmt.Sprintf("f%d.go", i)), []byte("package main"), 0644)
	}

	call := claudeCall("Glob", fmt.Sprintf(`{"path":%q,"pattern":"*.go"}`, dir))
	v.Emit(claudeCompleteEvent(call, `{"count":10}`))

	ctx := context.Background()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res, ok := v.Lookup(ctx, call)
		if !ok || res == nil {
			b.Fatalf("lookup failed")
		}
	}
}

func BenchmarkClaudeGrepCacheHit(b *testing.B) {
	v := New(1024)
	dir := b.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\nfunc main() {}\n"), 0644)

	call := claudeCall("Grep", fmt.Sprintf(`{"path":%q,"pattern":"func main","include":"*.go"}`, dir))
	v.Emit(claudeCompleteEvent(call, `{"matches":[{"line":2}]}`))

	ctx := context.Background()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res, ok := v.Lookup(ctx, call)
		if !ok || res == nil {
			b.Fatalf("lookup failed")
		}
	}
}
