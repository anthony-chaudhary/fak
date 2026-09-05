package codetools

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestGrepOversizedRecord(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{
		Root: dir,
		Limits: Limits{
			MaxMatches: 5,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ts.RegisterEngines()

	engine := abi.Engine(EngineGrep)
	if engine == nil {
		t.Fatalf("registered engine %q not found", EngineGrep)
	}

	execute := func(args GrepArgs) ([]byte, bool) {
		t.Helper()
		body := argsOf(t, args)
		call := &abi.ToolCall{
			Tool:   ToolGrep,
			Engine: EngineGrep,
			Args:   abi.Ref{Kind: abi.RefInline, Inline: body},
			Meta:   CallMeta(ToolGrep, "test"),
		}
		res, err := engine.Complete(context.Background(), call)
		if err != nil {
			t.Fatalf("Complete failed: %v", err)
		}
		return bytesOf(context.Background(), res.Payload), res.Status == abi.StatusError
	}

	// 1. Temporary file with serialized record > 65536 bytes (e.g. 70,000 bytes)
	longLine := strings.Repeat("A", 35000) + "NEEDLE_KEY" + strings.Repeat("B", 35000)
	mustWrite(t, filepath.Join(dir, "oversized_record.json"), `{"record":"`+longLine+`"}`+"\n")

	out, isErr := execute(GrepArgs{Pattern: "NEEDLE_KEY"})
	if isErr {
		t.Fatalf("Grep oversized record failed: %s", string(out))
	}
	m := decodeResult(t, out)
	if m["match_count"] != float64(1) {
		t.Fatalf("oversized match_count = %v, want 1", m["match_count"])
	}
	if m["truncated"] != true {
		t.Fatalf("oversized truncated = %v, want true", m["truncated"])
	}
	if m["truncation_reason"] != "line_width" {
		t.Fatalf("oversized truncation_reason = %v, want line_width", m["truncation_reason"])
	}
	rows := m["matches"].([]any)
	if len(rows) != 1 {
		t.Fatalf("len(matches) = %d, want 1", len(rows))
	}
	row := rows[0].(map[string]any)
	text := row["text"].(string)
	if len(text) != maxMatchLineBytes {
		t.Fatalf("oversized text len = %d, want %d (bounded output)", len(text), maxMatchLineBytes)
	}
	if row["truncated"] != true {
		t.Fatalf("match record truncated = %v, want true", row["truncated"])
	}

	// 2. Ordinary hits (untruncated lines, useful output)
	mustWrite(t, filepath.Join(dir, "ordinary.txt"), "short ordinary match alpha\nshort ordinary match beta\n")
	out, isErr = execute(GrepArgs{Pattern: "short ordinary match"})
	if isErr {
		t.Fatalf("Grep ordinary failed: %s", string(out))
	}
	m = decodeResult(t, out)
	if m["match_count"] != float64(2) {
		t.Fatalf("ordinary match_count = %v, want 2", m["match_count"])
	}
	if m["truncated"] != false {
		t.Fatalf("ordinary truncated = %v, want false", m["truncated"])
	}
	if m["truncation_reason"] != "" {
		t.Fatalf("ordinary truncation_reason = %v, want empty", m["truncation_reason"])
	}
	for i, r := range m["matches"].([]any) {
		rm := r.(map[string]any)
		if rm["truncated"] != false {
			t.Fatalf("ordinary row %d truncated = %v, want false", i, rm["truncated"])
		}
	}

	// 3. Match cap (MaxMatches limit)
	mustWrite(t, filepath.Join(dir, "many_hits.txt"), "hit_cap 1\nhit_cap 2\nhit_cap 3\nhit_cap 4\n")
	out, isErr = execute(GrepArgs{Pattern: "hit_cap", MaxMatches: 2})
	if isErr {
		t.Fatalf("Grep match cap failed: %s", string(out))
	}
	m = decodeResult(t, out)
	if m["match_count"] != float64(2) {
		t.Fatalf("match cap count = %v, want 2", m["match_count"])
	}
	if m["truncated"] != true {
		t.Fatalf("match cap truncated = %v, want true", m["truncated"])
	}
	if m["truncation_reason"] != "match_limit" {
		t.Fatalf("match cap truncation_reason = %v, want match_limit", m["truncation_reason"])
	}

	// 4. Outside-workspace target asserting confinement
	out, isErr = execute(GrepArgs{Pattern: "secret", Path: ".." + string(filepath.Separator) + "outside"})
	if !isErr {
		t.Fatalf("Grep outside workspace succeeded, want error: %s", string(out))
	}
	if code := errCode(t, out); code != CodePathEscape {
		t.Fatalf("outside target code = %v, want %s", code, CodePathEscape)
	}

	outsideCall := &abi.ToolCall{
		Tool:   ToolGrep,
		Engine: EngineGrep,
		Args:   abi.Ref{Kind: abi.RefInline, Inline: argsOf(t, GrepArgs{Pattern: "secret", Path: ".." + string(filepath.Separator) + "outside"})},
		Meta:   CallMeta(ToolGrep, "test"),
	}
	v := ts.Adjudicate(context.Background(), outsideCall)
	if v.Kind != abi.VerdictDeny || v.Meta["code"] != CodePathEscape {
		t.Fatalf("Adjudicate outside workspace = %+v, want DENY %s", v, CodePathEscape)
	}
}

func TestGrepFileContentMaxReadBytesTruncation(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{
		Root: dir,
		Limits: Limits{
			MaxReadBytes: 1024,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ts.RegisterEngines()
	engine := abi.Engine(EngineGrep)
	if engine == nil {
		t.Fatalf("registered engine %q not found", EngineGrep)
	}

	execute := func(args GrepArgs) ([]byte, bool) {
		t.Helper()
		body := argsOf(t, args)
		call := &abi.ToolCall{
			Tool:   ToolGrep,
			Engine: EngineGrep,
			Args:   abi.Ref{Kind: abi.RefInline, Inline: body},
			Meta:   CallMeta(ToolGrep, "test"),
		}
		res, err := engine.Complete(context.Background(), call)
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		return bytesOf(context.Background(), res.Payload), res.Status == abi.StatusError
	}

	// Write a file > 1024 bytes (e.g. 2048 bytes) with match in head and match in tail
	content := "head_needle match\n" + strings.Repeat("x\n", 600) + "tail_needle match\n"
	mustWrite(t, filepath.Join(dir, "large_file.txt"), content)

	out, isErr := execute(GrepArgs{Pattern: "head_needle"})
	if isErr {
		t.Fatalf("Grep large file failed: %s", string(out))
	}
	m := decodeResult(t, out)
	if m["match_count"] != float64(1) {
		t.Fatalf("match_count = %v, want 1", m["match_count"])
	}
	if m["truncated"] != true {
		t.Fatalf("truncated = %v, want true", m["truncated"])
	}
	if m["truncation_reason"] != "file_size" {
		t.Fatalf("truncation_reason = %v, want file_size", m["truncation_reason"])
	}
	rows := m["matches"].([]any)
	if len(rows) != 1 {
		t.Fatalf("len(matches) = %d, want 1", len(rows))
	}
	row := rows[0].(map[string]any)
	if row["truncated"] != false {
		t.Fatalf("row truncated = %v, want false", row["truncated"])
	}

	// Tail match is beyond MaxReadBytes (1024 bytes), so not found, but truncated is true
	out, isErr = execute(GrepArgs{Pattern: "tail_needle"})
	if isErr {
		t.Fatalf("Grep tail_needle failed: %s", string(out))
	}
	m = decodeResult(t, out)
	if m["match_count"] != float64(0) {
		t.Fatalf("tail_needle match_count = %v, want 0", m["match_count"])
	}
	if m["truncated"] != true {
		t.Fatalf("truncated = %v, want true", m["truncated"])
	}
	if m["truncation_reason"] != "file_size" {
		t.Fatalf("truncation_reason = %v, want file_size", m["truncation_reason"])
	}
}

func TestGrepLongLine100KB(t *testing.T) {
	ts, dir := newTestToolset(t)
	// 100,000 bytes line
	longLine := strings.Repeat("X", 99990) + "needle" + strings.Repeat("Y", 4)
	mustWrite(t, filepath.Join(dir, "long100k.txt"), longLine+"\n")

	out, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "needle"}))
	if isErr {
		t.Fatalf("Grep failed: %s", string(out))
	}
	m := decodeResult(t, out)
	if m["match_count"] != float64(1) {
		t.Fatalf("match_count = %v, want 1", m["match_count"])
	}
	if m["truncated"] != true {
		t.Fatalf("truncated = %v, want true", m["truncated"])
	}
	if m["truncation_reason"] != "line_width" {
		t.Fatalf("truncation_reason = %v, want line_width", m["truncation_reason"])
	}
	rows := m["matches"].([]any)
	if len(rows) != 1 {
		t.Fatalf("len(matches) = %d, want 1", len(rows))
	}
	row := rows[0].(map[string]any)
	text := row["text"].(string)
	if len(text) != maxMatchLineBytes {
		t.Fatalf("text len = %d, want %d", len(text), maxMatchLineBytes)
	}
	if row["truncated"] != true {
		t.Fatalf("row truncated = %v, want true", row["truncated"])
	}
}

func TestGrepHitCapLimit(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir, Limits: Limits{MaxMatches: 2}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	mustWrite(t, filepath.Join(dir, "hits.txt"), "target1\ntarget2\ntarget3\ntarget4\n")

	out, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "target"}))
	if isErr {
		t.Fatalf("Grep failed: %s", string(out))
	}
	m := decodeResult(t, out)
	if m["match_count"] != float64(2) {
		t.Fatalf("match_count = %v, want 2", m["match_count"])
	}
	if m["truncated"] != true {
		t.Fatalf("truncated = %v, want true", m["truncated"])
	}
	if m["truncation_reason"] != "match_limit" {
		t.Fatalf("truncation_reason = %v, want match_limit", m["truncation_reason"])
	}

	// Precedence test: oversized line + hit-cap limit -> match_limit takes precedence
	mustWrite(t, filepath.Join(dir, "precedence.txt"), strings.Repeat("A", 600)+"target_p\ntarget_p\ntarget_p\n")
	out, isErr = ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "target_p"}))
	if isErr {
		t.Fatalf("Grep failed: %s", string(out))
	}
	m = decodeResult(t, out)
	if m["match_count"] != float64(2) {
		t.Fatalf("match_count = %v, want 2", m["match_count"])
	}
	if m["truncated"] != true {
		t.Fatalf("truncated = %v, want true", m["truncated"])
	}
	if m["truncation_reason"] != "match_limit" {
		t.Fatalf("truncation_reason = %v, want match_limit", m["truncation_reason"])
	}
}

func TestGlobTruncationReporting(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir, Limits: Limits{MaxEntries: 2}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for _, n := range []string{"a.go", "b.go", "c.go", "d.go"} {
		mustWrite(t, filepath.Join(dir, n), "package main\n")
	}

	out, isErr := ts.glob(context.Background(), argsOf(t, GlobArgs{Pattern: "*.go"}))
	if isErr {
		t.Fatalf("Glob failed: %s", string(out))
	}
	m := decodeResult(t, out)
	if m["count"] != float64(2) {
		t.Fatalf("count = %v, want 2", m["count"])
	}
	if m["truncated"] != true {
		t.Fatalf("truncated = %v, want true", m["truncated"])
	}
	if m["truncation_reason"] != "match_limit" {
		t.Fatalf("truncation_reason = %v, want match_limit", m["truncation_reason"])
	}

	// Not truncated when under limit
	out, isErr = ts.glob(context.Background(), argsOf(t, GlobArgs{Pattern: "a.go"}))
	if isErr {
		t.Fatalf("Glob failed: %s", string(out))
	}
	m = decodeResult(t, out)
	if m["count"] != float64(1) {
		t.Fatalf("count = %v, want 1", m["count"])
	}
	if m["truncated"] != false {
		t.Fatalf("truncated = %v, want false", m["truncated"])
	}
	if m["truncation_reason"] != "" {
		t.Fatalf("truncation_reason = %v, want empty", m["truncation_reason"])
	}
}
