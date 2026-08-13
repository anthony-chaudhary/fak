package codetools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// engines_test.go — behavior tests for the three read engines: what they do when allowed,
// and what they refuse. Each denial case names the closed code, so a regression that
// changes WHY a call was refused fails as loudly as one that stops refusing it.

// decodeResult parses an engine payload into a map, failing the test on malformed JSON —
// the result shape is part of the contract, not an implementation detail.
func decodeResult(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("engine payload is not JSON: %v (%s)", err, string(b))
	}
	return m
}

// errCode extracts the closed refusal code from an engine error payload.
func errCode(t *testing.T, b []byte) string {
	t.Helper()
	m := decodeResult(t, b)
	e, ok := m["error"].(map[string]any)
	if !ok {
		t.Fatalf("payload carries no error object: %s", string(b))
	}
	code, _ := e["code"].(string)
	return code
}

// argsOf marshals a typed argument struct the way a caller would put it on a ToolCall.
func argsOf(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return b
}

// mustWrite seeds a fixture file, creating parents.
func mustWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// deadCtx is an already-canceled context: the state a queued tool call finds itself in
// after the loop that proposed it has been terminated.
func deadCtx() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return ctx
}

func TestReadReturnsFileContent(t *testing.T) {
	ts, dir := newTestToolset(t)
	mustWrite(t, filepath.Join(dir, "a.txt"), "alpha\nbeta\ngamma")
	out, isErr := ts.read(context.Background(), argsOf(t, ReadArgs{FilePath: "a.txt"}))
	if isErr {
		t.Fatalf("Read failed: %s", string(out))
	}
	if got := decodeResult(t, out)["content"]; got != "alpha\nbeta\ngamma" {
		t.Fatalf("content = %q", got)
	}
}

func TestReadWindowsByLine(t *testing.T) {
	ts, dir := newTestToolset(t)
	mustWrite(t, filepath.Join(dir, "a.txt"), "l1\nl2\nl3\nl4")
	out, isErr := ts.read(context.Background(), argsOf(t, ReadArgs{FilePath: "a.txt", Offset: 2, Limit: 2}))
	if isErr {
		t.Fatalf("Read failed: %s", string(out))
	}
	if got := decodeResult(t, out)["content"]; got != "l2\nl3" {
		t.Fatalf("windowed content = %q, want %q", got, "l2\nl3")
	}
}

func TestReadRefusesTraversalAndMissingFile(t *testing.T) {
	ts, _ := newTestToolset(t)
	out, isErr := ts.read(context.Background(), argsOf(t, ReadArgs{FilePath: "../escape.txt"}))
	if !isErr || errCode(t, out) != CodePathEscape {
		t.Fatalf("traversal Read = %s, want PATH_ESCAPE", string(out))
	}
	out, isErr = ts.read(context.Background(), argsOf(t, ReadArgs{FilePath: "nope.txt"}))
	if !isErr || errCode(t, out) != CodeNotFound {
		t.Fatalf("missing Read = %s, want NOT_FOUND", string(out))
	}
}

func TestReadRefusesADirectory(t *testing.T) {
	ts, dir := newTestToolset(t)
	if err := os.MkdirAll(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	out, isErr := ts.read(context.Background(), argsOf(t, ReadArgs{FilePath: "sub"}))
	if !isErr || errCode(t, out) != CodeIsDir {
		t.Fatalf("directory Read = %s, want IS_DIR", string(out))
	}
}

func TestReadIsBoundedBySize(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir, Limits: Limits{MaxReadBytes: 8}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	mustWrite(t, filepath.Join(dir, "big.txt"), strings.Repeat("x", 100))
	out, isErr := ts.read(context.Background(), argsOf(t, ReadArgs{FilePath: "big.txt"}))
	if isErr {
		t.Fatalf("Read failed: %s", string(out))
	}
	m := decodeResult(t, out)
	if m["truncated"] != true {
		t.Fatalf("oversize Read did not report truncation: %s", string(out))
	}
	if got := m["content"].(string); len(got) != 8 {
		t.Fatalf("content length = %d, want 8", len(got))
	}
}

func TestReadRejectsUnknownArgument(t *testing.T) {
	ts, _ := newTestToolset(t)
	out, isErr := ts.read(context.Background(), []byte(`{"filePath":"a.txt"}`))
	if !isErr || errCode(t, out) != CodeMalformed {
		t.Fatalf("unknown-field Read = %s, want MALFORMED", string(out))
	}
}

// TestReadIsCancellationAware pins that a canceled context stops the call BEFORE it
// touches the disk, so a terminated session's queued read cannot still run.
func TestReadIsCancellationAware(t *testing.T) {
	ts, dir := newTestToolset(t)
	mustWrite(t, filepath.Join(dir, "a.txt"), "content")
	out, isErr := ts.read(deadCtx(), argsOf(t, ReadArgs{FilePath: "a.txt"}))
	if !isErr || errCode(t, out) != CodeCanceled {
		t.Fatalf("canceled Read = %s, want CANCELED", string(out))
	}
	if strings.Contains(string(out), "content") {
		t.Fatalf("canceled Read still returned file bytes: %s", string(out))
	}
}

func TestGrepFindsMatchesWithoutAShell(t *testing.T) {
	ts, dir := newTestToolset(t)
	mustWrite(t, filepath.Join(dir, "a.go"), "package main\nfunc Target() {}\n")
	mustWrite(t, filepath.Join(dir, "b.txt"), "Target in a text file\n")
	out, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: `func Target`}))
	if isErr {
		t.Fatalf("Grep failed: %s", string(out))
	}
	m := decodeResult(t, out)
	if m["match_count"] != float64(1) {
		t.Fatalf("match_count = %v, want 1: %s", m["match_count"], string(out))
	}
	// The glob filter narrows candidates by base name before any file is opened.
	out, _ = ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "Target", Glob: "*.txt"}))
	m = decodeResult(t, out)
	if m["match_count"] != float64(1) {
		t.Fatalf("globbed match_count = %v, want 1: %s", m["match_count"], string(out))
	}
	rows := m["matches"].([]any)
	if row := rows[0].(map[string]any); row["file"] != "b.txt" {
		t.Fatalf("globbed match file = %v, want b.txt", row["file"])
	}
}

func TestGrepIsBoundedAndRefusesBadPattern(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir, Limits: Limits{MaxMatches: 2}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	mustWrite(t, filepath.Join(dir, "many.txt"), "hit\nhit\nhit\nhit\n")
	out, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "hit"}))
	if isErr {
		t.Fatalf("Grep failed: %s", string(out))
	}
	m := decodeResult(t, out)
	if m["match_count"] != float64(2) || m["truncated"] != true {
		t.Fatalf("bounded Grep = %s, want 2 matches and truncated", string(out))
	}
	out, isErr = ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "("}))
	if !isErr || errCode(t, out) != CodeMalformed {
		t.Fatalf("bad-pattern Grep = %s, want MALFORMED", string(out))
	}
}

// TestGrepBoundsASingleMatchLine pins that one pathological minified line cannot flood a
// result that is otherwise within its row bound.
func TestGrepBoundsASingleMatchLine(t *testing.T) {
	ts, dir := newTestToolset(t)
	mustWrite(t, filepath.Join(dir, "min.js"), strings.Repeat("a", 4000)+"needle")
	out, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "needle"}))
	if isErr {
		t.Fatalf("Grep failed: %s", string(out))
	}
	rows := decodeResult(t, out)["matches"].([]any)
	if len(rows) != 1 {
		t.Fatalf("match rows = %d, want 1", len(rows))
	}
	if text := rows[0].(map[string]any)["text"].(string); len(text) > maxMatchLineBytes {
		t.Fatalf("match line = %d bytes, want <= %d", len(text), maxMatchLineBytes)
	}
}

func TestGrepRefusesEscapingSearchRoot(t *testing.T) {
	ts, _ := newTestToolset(t)
	out, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "x", Path: ".."}))
	if !isErr || errCode(t, out) != CodePathEscape {
		t.Fatalf("escaping Grep = %s, want PATH_ESCAPE", string(out))
	}
}

func TestGrepIsCancellationAware(t *testing.T) {
	ts, dir := newTestToolset(t)
	mustWrite(t, filepath.Join(dir, "a.txt"), "hit\n")
	out, isErr := ts.grep(deadCtx(), argsOf(t, GrepArgs{Pattern: "hit"}))
	if !isErr || errCode(t, out) != CodeCanceled {
		t.Fatalf("canceled Grep = %s, want CANCELED", string(out))
	}
}

func TestGlobMatchesRelativePaths(t *testing.T) {
	ts, dir := newTestToolset(t)
	mustWrite(t, filepath.Join(dir, "a.go"), "x")
	mustWrite(t, filepath.Join(dir, "sub", "b.go"), "x")
	mustWrite(t, filepath.Join(dir, "sub", "c.md"), "x")
	out, isErr := ts.glob(context.Background(), argsOf(t, GlobArgs{Pattern: "**/*.go"}))
	if isErr {
		t.Fatalf("Glob failed: %s", string(out))
	}
	m := decodeResult(t, out)
	if m["count"] != float64(2) {
		t.Fatalf("count = %v, want 2: %s", m["count"], string(out))
	}
}

func TestGlobIsBoundedByEntries(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir, Limits: Limits{MaxEntries: 2}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for _, n := range []string{"a.go", "b.go", "c.go", "d.go"} {
		mustWrite(t, filepath.Join(dir, n), "x")
	}
	out, isErr := ts.glob(context.Background(), argsOf(t, GlobArgs{Pattern: "*.go"}))
	if isErr {
		t.Fatalf("Glob failed: %s", string(out))
	}
	m := decodeResult(t, out)
	if m["count"] != float64(2) || m["truncated"] != true {
		t.Fatalf("bounded Glob = %s, want 2 entries and truncated", string(out))
	}
}

// TestGlobSkipsDotDirectories pins that a search never walks into the trees whose
// contents an agent grepping its workspace must not pull into the loop's context window.
func TestGlobSkipsDotDirectories(t *testing.T) {
	ts, dir := newTestToolset(t)
	mustWrite(t, filepath.Join(dir, ".git", "objects", "deadbeef"), "binary")
	mustWrite(t, filepath.Join(dir, "keep.txt"), "x")
	out, _ := ts.glob(context.Background(), argsOf(t, GlobArgs{Pattern: "**/*"}))
	files := decodeResult(t, out)["files"].([]any)
	for _, f := range files {
		if strings.HasPrefix(f.(string), ".git/") {
			t.Fatalf("Glob walked into .git: %v", files)
		}
	}
}

// TestSearchDoesNotFollowSymlinkedFiles pins the walk-side half of confinement: a link
// planted inside the tree must not widen a search into the host filesystem.
func TestSearchDoesNotFollowSymlinkedFiles(t *testing.T) {
	ts, dir := newTestToolset(t)
	outside := t.TempDir()
	mustWrite(t, filepath.Join(outside, "secret.txt"), "classified needle")
	if err := os.Symlink(filepath.Join(outside, "secret.txt"), filepath.Join(dir, "link.txt")); err != nil {
		t.Skip("symlink creation requires privilege on this host")
	}
	out, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "needle"}))
	if isErr {
		t.Fatalf("Grep failed: %s", string(out))
	}
	if strings.Contains(string(out), "classified") {
		t.Fatalf("Grep followed a symlink out of the workspace: %s", string(out))
	}
}

func TestGlobIsCancellationAware(t *testing.T) {
	ts, dir := newTestToolset(t)
	mustWrite(t, filepath.Join(dir, "a.go"), "x")
	out, isErr := ts.glob(deadCtx(), argsOf(t, GlobArgs{Pattern: "*.go"}))
	if !isErr || errCode(t, out) != CodeCanceled {
		t.Fatalf("canceled Glob = %s, want CANCELED", string(out))
	}
}

// TestCatalogAndCacheScopeAgreeOnWriteShape pins the one-source-of-truth contract: the
// catalog's ReadOnly bit is what CallMeta stamps onto the vDSO scope, so a catalog and a
// cache key can never disagree about whether a tool mutates.
func TestCatalogAndCacheScopeAgreeOnWriteShape(t *testing.T) {
	for _, d := range Catalog() {
		if !d.ReadOnly {
			t.Fatalf("catalog entry %q is write-shaped; this slice ships read tools only", d.Name)
		}
		meta := CallMeta(d.Name, "")
		if meta["readOnlyHint"] != "true" || meta["idempotentHint"] != "true" {
			t.Fatalf("CallMeta(%q) = %v, want the read-only/idempotent hints", d.Name, meta)
		}
		if _, ok := meta["destructive"]; ok {
			t.Fatalf("CallMeta(%q) marked a read tool destructive: %v", d.Name, meta)
		}
	}
	if got := CallMeta(ToolRead, "tenant-a")["principal"]; got != "tenant-a" {
		t.Fatalf("CallMeta principal = %q, want tenant-a", got)
	}
}
