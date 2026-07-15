package transcript_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume/transcript"
)

// --- Record.Role -----------------------------------------------------------

func TestRecordRole(t *testing.T) {
	cases := []struct {
		name string
		rec  transcript.Record
		want string
	}{
		{
			name: "message role wins over type",
			rec: transcript.Record{
				Type:    "user",
				Message: &transcript.Message{Role: "assistant"},
			},
			want: "assistant",
		},
		{
			name: "message present but role empty falls back to type",
			rec: transcript.Record{
				Type:    "summary",
				Message: &transcript.Message{Role: ""},
			},
			want: "summary",
		},
		{
			name: "no message falls back to type",
			rec:  transcript.Record{Type: "progress"},
			want: "progress",
		},
		{
			name: "no message, no type",
			rec:  transcript.Record{},
			want: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.rec.Role(); got != tc.want {
				t.Errorf("Role() = %q, want %q", got, tc.want)
			}
		})
	}
}

// --- Record.IsSynthetic ------------------------------------------------------

func TestRecordIsSynthetic(t *testing.T) {
	cases := []struct {
		name string
		rec  transcript.Record
		want bool
	}{
		{"nil message", transcript.Record{}, false},
		{"synthetic model", transcript.Record{Message: &transcript.Message{Model: "<synthetic>"}}, true},
		{"real model", transcript.Record{Message: &transcript.Message{Model: "claude-3-opus"}}, false},
		{"empty model", transcript.Record{Message: &transcript.Message{Model: ""}}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.rec.IsSynthetic(); got != tc.want {
				t.Errorf("IsSynthetic() = %v, want %v", got, tc.want)
			}
		})
	}
}

// --- Record.Text -------------------------------------------------------------

func rawString(s string) json.RawMessage {
	b, err := json.Marshal(s)
	if err != nil {
		panic(err)
	}
	return json.RawMessage(b)
}

func TestRecordText(t *testing.T) {
	cases := []struct {
		name string
		rec  transcript.Record
		want string
	}{
		{
			name: "nil message yields empty string",
			rec:  transcript.Record{},
			want: "",
		},
		{
			name: "bare string content verbatim",
			rec: transcript.Record{
				Message: &transcript.Message{Content: rawString("hello world")},
			},
			want: "hello world",
		},
		{
			name: "block list joins text blocks with newline",
			rec: transcript.Record{
				Message: &transcript.Message{
					Content: json.RawMessage(`[{"type":"text","text":"line one"},{"type":"text","text":"line two"}]`),
				},
			},
			want: "line one\nline two",
		},
		{
			name: "tool_result blocks are excluded from Text",
			rec: transcript.Record{
				Message: &transcript.Message{
					Content: json.RawMessage(`[{"type":"text","text":"said this"},{"type":"tool_result","content":"tool output"}]`),
				},
			},
			want: "said this",
		},
		{
			name: "empty text blocks are skipped",
			rec: transcript.Record{
				Message: &transcript.Message{
					Content: json.RawMessage(`[{"type":"text","text":""},{"type":"text","text":"kept"}]`),
				},
			},
			want: "kept",
		},
		{
			name: "tool_use only block yields empty text",
			rec: transcript.Record{
				Message: &transcript.Message{
					Content: json.RawMessage(`[{"type":"tool_use","name":"Bash"}]`),
				},
			},
			want: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.rec.Text(); got != tc.want {
				t.Errorf("Text() = %q, want %q", got, tc.want)
			}
		})
	}
}

// --- Record.TextWithToolResults ------------------------------------------------

func TestRecordTextWithToolResults(t *testing.T) {
	cases := []struct {
		name string
		rec  transcript.Record
		want string
	}{
		{
			name: "nil message yields empty string",
			rec:  transcript.Record{},
			want: "",
		},
		{
			name: "bare string content verbatim",
			rec: transcript.Record{
				Message: &transcript.Message{Content: rawString("plain text")},
			},
			want: "plain text",
		},
		{
			name: "text and tool_result joined with spaces",
			rec: transcript.Record{
				Message: &transcript.Message{
					Content: json.RawMessage(`[{"type":"text","text":"parked on background task"},{"type":"tool_result","content":"still running"}]`),
				},
			},
			want: "parked on background task still running",
		},
		{
			name: "nested tool_result block list recursively flattened",
			rec: transcript.Record{
				Message: &transcript.Message{
					Content: json.RawMessage(`[{"type":"tool_result","content":[{"type":"text","text":"nested a"},{"type":"text","text":"nested b"}]}]`),
				},
			},
			// flattenContent always joins with a space, at every recursion level —
			// unlike Text(), which joins top-level text blocks with a newline.
			want: "nested a nested b",
		},
		{
			name: "empty tool_result content contributes nothing",
			rec: transcript.Record{
				Message: &transcript.Message{
					Content: json.RawMessage(`[{"type":"text","text":"only this"},{"type":"tool_result","content":""}]`),
				},
			},
			want: "only this",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.rec.TextWithToolResults(); got != tc.want {
				t.Errorf("TextWithToolResults() = %q, want %q", got, tc.want)
			}
		})
	}
}

// --- Record.LastToolUseName ----------------------------------------------------

func TestRecordLastToolUseName(t *testing.T) {
	cases := []struct {
		name string
		rec  transcript.Record
		want string
	}{
		{"nil message", transcript.Record{}, ""},
		{
			name: "no tool_use blocks",
			rec: transcript.Record{
				Message: &transcript.Message{
					Content: json.RawMessage(`[{"type":"text","text":"hi"}]`),
				},
			},
			want: "",
		},
		{
			name: "single tool_use block",
			rec: transcript.Record{
				Message: &transcript.Message{
					Content: json.RawMessage(`[{"type":"tool_use","name":"Read"}]`),
				},
			},
			want: "Read",
		},
		{
			name: "returns the LAST tool_use block, not the first",
			rec: transcript.Record{
				Message: &transcript.Message{
					Content: json.RawMessage(`[{"type":"tool_use","name":"Read"},{"type":"text","text":"thinking"},{"type":"tool_use","name":"Bash"}]`),
				},
			},
			want: "Bash",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.rec.LastToolUseName(); got != tc.want {
				t.Errorf("LastToolUseName() = %q, want %q", got, tc.want)
			}
		})
	}
}

// --- Record.HasToolResult -------------------------------------------------------

func TestRecordHasToolResult(t *testing.T) {
	cases := []struct {
		name string
		rec  transcript.Record
		want bool
	}{
		{"nil message", transcript.Record{}, false},
		{
			name: "no tool_result block",
			rec: transcript.Record{
				Message: &transcript.Message{
					Content: json.RawMessage(`[{"type":"text","text":"hi"}]`),
				},
			},
			want: false,
		},
		{
			name: "has tool_result block",
			rec: transcript.Record{
				Message: &transcript.Message{
					Content: json.RawMessage(`[{"type":"tool_use","name":"Bash"},{"type":"tool_result","content":"done"}]`),
				},
			},
			want: true,
		},
		{
			name: "bare string content never has a tool_result",
			rec: transcript.Record{
				Message: &transcript.Message{Content: rawString("no blocks here")},
			},
			want: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.rec.HasToolResult(); got != tc.want {
				t.Errorf("HasToolResult() = %v, want %v", got, tc.want)
			}
		})
	}
}

// --- Parse ---------------------------------------------------------------------

func TestParse(t *testing.T) {
	input := strings.Join([]string{
		`{"type":"user","uuid":"u1","timestamp":"2026-06-23T10:00:00Z","message":{"role":"user","content":"hi"}}`,
		``,                // blank line skipped
		`   `,             // whitespace-only line skipped
		`not json at all`, // malformed line skipped
		`{"type":"assistant","uuid":"u2","timestamp":"2026-06-23T10:01:00Z","message":{"role":"assistant","content":"hello back"}}`,
	}, "\n")

	recs := transcript.Parse(strings.NewReader(input))
	if len(recs) != 2 {
		t.Fatalf("Parse() returned %d records, want 2 (got: %+v)", len(recs), recs)
	}
	if recs[0].UUID != "u1" || recs[0].Text() != "hi" {
		t.Errorf("first record = %+v, want uuid u1 text hi", recs[0])
	}
	if recs[1].UUID != "u2" || recs[1].Text() != "hello back" {
		t.Errorf("second record = %+v, want uuid u2 text 'hello back'", recs[1])
	}
}

func TestParseEmptyInput(t *testing.T) {
	recs := transcript.Parse(strings.NewReader(""))
	if len(recs) != 0 {
		t.Fatalf("Parse(\"\") = %d records, want 0", len(recs))
	}
}

// --- LoadFile / LoadFileTail -----------------------------------------------------

func writeTranscript(t *testing.T, lines []string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "transcript.jsonl")
	content := strings.Join(lines, "\n") + "\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	return path
}

func TestLoadFile(t *testing.T) {
	path := writeTranscript(t, []string{
		`{"type":"user","uuid":"a","message":{"role":"user","content":"first"}}`,
		`{"type":"assistant","uuid":"b","message":{"role":"assistant","content":"second"}}`,
	})

	recs := transcript.LoadFile(path)
	if len(recs) != 2 {
		t.Fatalf("LoadFile() returned %d records, want 2", len(recs))
	}
	if recs[0].Text() != "first" || recs[1].Text() != "second" {
		t.Errorf("LoadFile() texts = %q, %q; want first, second", recs[0].Text(), recs[1].Text())
	}
}

func TestLoadFileMissing(t *testing.T) {
	dir := t.TempDir()
	recs := transcript.LoadFile(filepath.Join(dir, "does-not-exist.jsonl"))
	if recs != nil {
		t.Errorf("LoadFile(missing) = %+v, want nil", recs)
	}
}

func TestLoadFileTailWholeFileWhenNonPositive(t *testing.T) {
	path := writeTranscript(t, []string{
		`{"type":"user","uuid":"a","message":{"role":"user","content":"first"}}`,
		`{"type":"assistant","uuid":"b","message":{"role":"assistant","content":"second"}}`,
	})

	recs := transcript.LoadFileTail(path, 0)
	if len(recs) != 2 {
		t.Fatalf("LoadFileTail(path, 0) returned %d records, want 2 (whole file)", len(recs))
	}

	recsNeg := transcript.LoadFileTail(path, -5)
	if len(recsNeg) != 2 {
		t.Fatalf("LoadFileTail(path, -5) returned %d records, want 2 (whole file)", len(recsNeg))
	}
}

func TestLoadFileTailReadsOnlyTail(t *testing.T) {
	// Build a file where the last line is long and clearly identifiable, and the
	// earlier lines are irrelevant filler. A small tailBytes should seek into the
	// middle of an earlier line (producing one skipped malformed fragment) and still
	// recover the final, complete record intact.
	lines := []string{
		`{"type":"user","uuid":"first","message":{"role":"user","content":"this is the first record and it is intentionally long enough to push the tail seek past it entirely so only later records remain readable"}}`,
		`{"type":"assistant","uuid":"last","message":{"role":"assistant","content":"tail record"}}`,
	}
	path := writeTranscript(t, lines)

	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	lastLineLen := int64(len(lines[1])) + 1 // +1 newline
	tailBytes := lastLineLen + 5            // just a few bytes into the prior line
	if tailBytes >= fi.Size() {
		t.Fatalf("test fixture too small: tailBytes=%d size=%d", tailBytes, fi.Size())
	}

	recs := transcript.LoadFileTail(path, tailBytes)
	if len(recs) != 1 {
		t.Fatalf("LoadFileTail() returned %d records, want 1 (only the tail record survives); got %+v", len(recs), recs)
	}
	if recs[0].UUID != "last" || recs[0].Text() != "tail record" {
		t.Errorf("LoadFileTail() record = %+v, want uuid=last text='tail record'", recs[0])
	}
}

func TestLoadFileTailMissing(t *testing.T) {
	dir := t.TempDir()
	recs := transcript.LoadFileTail(filepath.Join(dir, "nope.jsonl"), 100)
	if recs != nil {
		t.Errorf("LoadFileTail(missing) = %+v, want nil", recs)
	}
}

// --- LastTimestamp ---------------------------------------------------------------

func TestLastTimestamp(t *testing.T) {
	cases := []struct {
		name string
		recs []transcript.Record
		want string
	}{
		{"empty slice", nil, ""},
		{
			name: "no record carries a timestamp",
			recs: []transcript.Record{{Type: "user"}, {Type: "assistant"}},
			want: "",
		},
		{
			name: "returns last record's timestamp",
			recs: []transcript.Record{
				{Timestamp: "2026-06-23T10:00:00Z"},
				{Timestamp: "2026-06-23T10:05:00Z"},
			},
			want: "2026-06-23T10:05:00Z",
		},
		{
			name: "skips trailing records with no timestamp",
			recs: []transcript.Record{
				{Timestamp: "2026-06-23T10:00:00Z"},
				{Timestamp: "2026-06-23T10:05:00Z"},
				{Type: "summary"}, // no timestamp
			},
			want: "2026-06-23T10:05:00Z",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := transcript.LastTimestamp(tc.recs); got != tc.want {
				t.Errorf("LastTimestamp() = %q, want %q", got, tc.want)
			}
		})
	}
}

// --- UUIDSet -----------------------------------------------------------------------

func TestUUIDSet(t *testing.T) {
	recs := []transcript.Record{
		{UUID: "a"},
		{UUID: "b"},
		{UUID: "a"}, // duplicate
		{UUID: ""},  // blank skipped
	}
	got := transcript.UUIDSet(recs)
	want := map[string]bool{"a": true, "b": true}
	if len(got) != len(want) {
		t.Fatalf("UUIDSet() = %v, want %v", got, want)
	}
	for k := range want {
		if !got[k] {
			t.Errorf("UUIDSet() missing key %q", k)
		}
	}
	if got[""] {
		t.Errorf("UUIDSet() should not include the blank uuid")
	}
}

func TestUUIDSetEmpty(t *testing.T) {
	got := transcript.UUIDSet(nil)
	if len(got) != 0 {
		t.Errorf("UUIDSet(nil) = %v, want empty map", got)
	}
}

// --- TerminalText ------------------------------------------------------------------

func TestTerminalText(t *testing.T) {
	cases := []struct {
		name string
		recs []transcript.Record
		want string
	}{
		{"empty slice", nil, ""},
		{
			name: "single user record",
			recs: []transcript.Record{
				{Type: "user", Message: &transcript.Message{Role: "user", Content: rawString("hello")}},
			},
			want: "hello",
		},
		{
			name: "ignores trailing control/metadata records",
			recs: []transcript.Record{
				{Type: "user", Message: &transcript.Message{Role: "user", Content: rawString("real turn")}},
				{Type: "summary", Message: &transcript.Message{Role: "summary", Content: rawString("a summary banner")}},
			},
			want: "real turn",
		},
		{
			name: "picks the LAST assistant turn over an earlier user turn",
			recs: []transcript.Record{
				{Type: "user", Message: &transcript.Message{Role: "user", Content: rawString("question")}},
				{Type: "assistant", Message: &transcript.Message{Role: "assistant", Content: rawString("answer")}},
			},
			want: "answer",
		},
		{
			name: "falls back to record Type when message role empty",
			recs: []transcript.Record{
				{Type: "assistant", Message: &transcript.Message{Role: "", Content: rawString("typed turn")}},
			},
			want: "typed turn",
		},
		{
			name: "no user/assistant records at all yields empty",
			recs: []transcript.Record{
				{Type: "summary", Message: &transcript.Message{Role: "summary", Content: rawString("just a banner")}},
			},
			want: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := transcript.TerminalText(tc.recs); got != tc.want {
				t.Errorf("TerminalText() = %q, want %q", got, tc.want)
			}
		})
	}
}

// --- ParseTime ---------------------------------------------------------------------

func TestParseTime(t *testing.T) {
	t.Run("empty string yields zero time", func(t *testing.T) {
		if got := transcript.ParseTime(""); !got.IsZero() {
			t.Errorf("ParseTime(\"\") = %v, want zero time", got)
		}
	})

	t.Run("garbage string yields zero time", func(t *testing.T) {
		if got := transcript.ParseTime("not-a-timestamp"); !got.IsZero() {
			t.Errorf("ParseTime(garbage) = %v, want zero time", got)
		}
	})

	t.Run("RFC3339 plain", func(t *testing.T) {
		got := transcript.ParseTime("2026-06-23T10:05:00Z")
		if got.IsZero() {
			t.Fatal("ParseTime() returned zero time for a valid RFC3339 timestamp")
		}
		want := time.Date(2026, 6, 23, 10, 5, 0, 0, time.UTC)
		if !got.Equal(want) {
			t.Errorf("ParseTime() = %v, want %v", got, want)
		}
	})

	t.Run("RFC3339Nano with fractional seconds", func(t *testing.T) {
		got := transcript.ParseTime("2026-06-23T10:05:00.123456789Z")
		if got.IsZero() {
			t.Fatal("ParseTime() returned zero time for a valid RFC3339Nano timestamp")
		}
		want := time.Date(2026, 6, 23, 10, 5, 0, 123456789, time.UTC)
		if !got.Equal(want) {
			t.Errorf("ParseTime() = %v, want %v", got, want)
		}
	})

	t.Run("numeric offset", func(t *testing.T) {
		got := transcript.ParseTime("2026-06-23T10:05:00-07:00")
		if got.IsZero() {
			t.Fatal("ParseTime() returned zero time for a timestamp with a numeric offset")
		}
		if !got.Equal(time.Date(2026, 6, 23, 17, 5, 0, 0, time.UTC)) {
			t.Errorf("ParseTime() = %v, want 2026-06-23T17:05:00Z (UTC-normalized)", got)
		}
	})
}

func TestRecordToolUsesPreservesNativeInput(t *testing.T) {
	rec := transcript.Record{Message: &transcript.Message{Role: "assistant", Content: json.RawMessage(`[
		{"type":"text","text":"I need one decision"},
		{"type":"tool_use","name":"Read","input":{"path":"README.md"}},
		{"type":"tool_use","name":"AskUserQuestion","input":{"questions":[{"question":"Which isolation?","options":[{"label":"Explicit paths","description":"Commit owned files"},{"label":"Wait","description":"Wait for peers"}]}]}}
	]`)}}
	uses := rec.ToolUses()
	if len(uses) != 2 || uses[0].Name != "Read" || uses[1].Name != "AskUserQuestion" {
		t.Fatalf("uses=%+v", uses)
	}
	var input struct {
		Questions []struct {
			Question string `json:"question"`
			Options  []struct {
				Label string `json:"label"`
			} `json:"options"`
		} `json:"questions"`
	}
	if err := json.Unmarshal(uses[1].Input, &input); err != nil {
		t.Fatal(err)
	}
	if len(input.Questions) != 1 || input.Questions[0].Question != "Which isolation?" || len(input.Questions[0].Options) != 2 {
		t.Fatalf("input=%+v", input)
	}
	last, ok := rec.LastToolUse()
	if !ok || last.Name != uses[1].Name || string(last.Input) != string(uses[1].Input) {
		t.Fatalf("last=%+v ok=%v", last, ok)
	}
	uses[1].Input[0] = 'X'
	lastAgain, _ := rec.LastToolUse()
	if len(lastAgain.Input) == 0 || lastAgain.Input[0] != '{' {
		t.Fatal("ToolUses returned aliased input bytes")
	}
}

func TestRecordToolUsesIgnoresMalformedAndNonToolBlocks(t *testing.T) {
	rec := transcript.Record{Message: &transcript.Message{Content: json.RawMessage(`[
		{"type":"tool_use","name":"","input":{"ignored":true}},
		{"type":"tool_result","name":"AskUserQuestion","content":"not an input"},
		{"type":"tool_use","name":"ExitPlanMode","input":{"plan":"inspect"}}
	]`)}}
	uses := rec.ToolUses()
	if len(uses) != 1 || uses[0].Name != "ExitPlanMode" {
		t.Fatalf("uses=%+v", uses)
	}
}
