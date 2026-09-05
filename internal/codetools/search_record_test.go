package codetools

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

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

func TestGrepWalkBudgetPrecedence(t *testing.T) {
	t.Run("oversized_line_then_walk_budget", func(t *testing.T) {
		dir := t.TempDir()
		ts, err := New(Config{Root: dir, Limits: Limits{MaxWalkFiles: 2}})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		// File 1: oversized line (> 512 bytes) with match
		longLine := strings.Repeat("A", 600) + "needle"
		mustWrite(t, filepath.Join(dir, "01_oversized.txt"), longLine+"\n")
		// File 2: normal match
		mustWrite(t, filepath.Join(dir, "02_second.txt"), "normal needle\n")
		// File 3: forces walk budget exhaustion (visited > MaxWalkFiles)
		mustWrite(t, filepath.Join(dir, "03_third.txt"), "another needle\n")

		out, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "needle"}))
		if isErr {
			t.Fatalf("Grep failed: %s", string(out))
		}
		m := decodeResult(t, out)
		if m["truncated"] != true {
			t.Fatalf("truncated = %v, want true", m["truncated"])
		}
		if m["truncation_reason"] != "walk_budget" {
			t.Fatalf("truncation_reason = %v, want walk_budget", m["truncation_reason"])
		}
		rows := m["matches"].([]any)
		if len(rows) != 2 {
			t.Fatalf("len(matches) = %d, want 2", len(rows))
		}
		row0 := rows[0].(map[string]any)
		if row0["truncated"] != true {
			t.Fatalf("row0 truncated = %v, want true", row0["truncated"])
		}
	})

	t.Run("file_size_limit_then_walk_budget", func(t *testing.T) {
		dir := t.TempDir()
		ts, err := New(Config{Root: dir, Limits: Limits{MaxReadBytes: 200, MaxWalkFiles: 2}})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		// File 1: exceeds MaxReadBytes (file_size limit)
		largeContent := "needle in head\n" + strings.Repeat("x\n", 200)
		mustWrite(t, filepath.Join(dir, "01_large.txt"), largeContent)
		// File 2: normal match
		mustWrite(t, filepath.Join(dir, "02_second.txt"), "normal needle\n")
		// File 3: forces walk budget exhaustion
		mustWrite(t, filepath.Join(dir, "03_third.txt"), "another needle\n")

		out, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "needle"}))
		if isErr {
			t.Fatalf("Grep failed: %s", string(out))
		}
		m := decodeResult(t, out)
		if m["truncated"] != true {
			t.Fatalf("truncated = %v, want true", m["truncated"])
		}
		if m["truncation_reason"] != "walk_budget" {
			t.Fatalf("truncation_reason = %v, want walk_budget", m["truncation_reason"])
		}
	})
}

func TestSnapToRuneBoundaryMultiByte(t *testing.T) {
	t.Run("direct", func(t *testing.T) {
		cases := []struct {
			name     string
			input    string
			maxBytes int
			wantEnd  string
		}{
			{
				name:     "cjk_split_2_bytes_in",
				input:    strings.Repeat("a", 510) + "世" + "suffix",
				maxBytes: 512,
				wantEnd:  strings.Repeat("a", 510),
			},
			{
				name:     "cjk_split_1_byte_in",
				input:    strings.Repeat("a", 511) + "世" + "suffix",
				maxBytes: 512,
				wantEnd:  strings.Repeat("a", 511),
			},
			{
				name:     "cjk_exact_boundary",
				input:    strings.Repeat("a", 509) + "世" + "suffix",
				maxBytes: 512,
				wantEnd:  strings.Repeat("a", 509) + "世",
			},
			{
				name:     "emoji_split_3_bytes_in",
				input:    strings.Repeat("a", 509) + "🎉" + "suffix",
				maxBytes: 512,
				wantEnd:  strings.Repeat("a", 509),
			},
			{
				name:     "emoji_split_2_bytes_in",
				input:    strings.Repeat("a", 510) + "🎉" + "suffix",
				maxBytes: 512,
				wantEnd:  strings.Repeat("a", 510),
			},
			{
				name:     "emoji_split_1_byte_in",
				input:    strings.Repeat("a", 511) + "🎉" + "suffix",
				maxBytes: 512,
				wantEnd:  strings.Repeat("a", 511),
			},
			{
				name:     "emoji_exact_boundary",
				input:    strings.Repeat("a", 508) + "🎉" + "suffix",
				maxBytes: 512,
				wantEnd:  strings.Repeat("a", 508) + "🎉",
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				got := snapToRuneBoundary(tc.input, tc.maxBytes)
				if !utf8.ValidString(got) {
					t.Fatalf("snapToRuneBoundary() produced invalid UTF-8: %q", got)
				}
				if len(got) > tc.maxBytes {
					t.Fatalf("len(got) = %d, want <= %d", len(got), tc.maxBytes)
				}
				if strings.ContainsRune(got, utf8.RuneError) {
					t.Fatalf("snapToRuneBoundary() produced replacement character: %q", got)
				}
				if got != tc.wantEnd {
					t.Fatalf("got %q (len %d), want %q (len %d)", got, len(got), tc.wantEnd, len(tc.wantEnd))
				}
			})
		}
	})

	t.Run("grep_match_text", func(t *testing.T) {
		dir := t.TempDir()
		ts, err := New(Config{Root: dir})
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		// 1. CJK character crossing 512-byte boundary
		cjkLine := strings.Repeat("A", 510) + "世" + "match_target_cjk" + strings.Repeat("B", 50)
		mustWrite(t, filepath.Join(dir, "cjk.txt"), cjkLine+"\n")

		// 2. Emoji character crossing 512-byte boundary
		emojiLine := strings.Repeat("C", 510) + "🎉" + "match_target_emoji" + strings.Repeat("D", 50)
		mustWrite(t, filepath.Join(dir, "emoji.txt"), emojiLine+"\n")

		// Test CJK match
		out, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "match_target_cjk"}))
		if isErr {
			t.Fatalf("Grep cjk failed: %s", string(out))
		}
		m := decodeResult(t, out)
		rows := m["matches"].([]any)
		if len(rows) != 1 {
			t.Fatalf("cjk matches count = %d, want 1", len(rows))
		}
		row := rows[0].(map[string]any)
		matchText := row["text"].(string)
		if !utf8.ValidString(matchText) {
			t.Fatalf("cjk match.Text is not valid UTF-8: %q", matchText)
		}
		if len(matchText) > maxMatchLineBytes {
			t.Fatalf("len(cjk match.Text) = %d, want <= %d", len(matchText), maxMatchLineBytes)
		}
		if strings.ContainsRune(matchText, utf8.RuneError) {
			t.Fatalf("cjk match.Text contains replacement character: %q", matchText)
		}

		// Test Emoji match
		out, isErr = ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "match_target_emoji"}))
		if isErr {
			t.Fatalf("Grep emoji failed: %s", string(out))
		}
		m = decodeResult(t, out)
		rows = m["matches"].([]any)
		if len(rows) != 1 {
			t.Fatalf("emoji matches count = %d, want 1", len(rows))
		}
		row = rows[0].(map[string]any)
		matchText = row["text"].(string)
		if !utf8.ValidString(matchText) {
			t.Fatalf("emoji match.Text is not valid UTF-8: %q", matchText)
		}
		if len(matchText) > maxMatchLineBytes {
			t.Fatalf("len(emoji match.Text) = %d, want <= %d", len(matchText), maxMatchLineBytes)
		}
		if strings.ContainsRune(matchText, utf8.RuneError) {
			t.Fatalf("emoji match.Text contains replacement character: %q", matchText)
		}
	})
}
