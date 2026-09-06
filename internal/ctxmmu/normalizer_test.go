package ctxmmu_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

func TestTimestampNormalization(t *testing.T) {
	norm := ctxmmu.NewNormalizer("$SESSION_START_TIME", "")

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "ISO8601 UTC Z",
			input:    "Started at 2026-09-05T14:30:00Z successfully",
			expected: "Started at $SESSION_START_TIME successfully",
		},
		{
			name:     "ISO8601 with fractional seconds",
			input:    "Event time: 2026-09-05T14:30:00.123456789Z logged",
			expected: "Event time: $SESSION_START_TIME logged",
		},
		{
			name:     "ISO8601 with positive offset",
			input:    "Timestamp 2026-09-05T14:30:00+02:00 recorded",
			expected: "Timestamp $SESSION_START_TIME recorded",
		},
		{
			name:     "ISO8601 with negative offset",
			input:    "Timestamp 2026-09-05T14:30:00-07:00 recorded",
			expected: "Timestamp $SESSION_START_TIME recorded",
		},
		{
			name:     "Standard date-time space separated",
			input:    "Log header: 2026-09-05 14:30:00 server boot",
			expected: "Log header: $SESSION_START_TIME server boot",
		},
		{
			name:     "Unix date UTC",
			input:    "Date is Sat Sep 05 14:30:00 UTC 2026 done",
			expected: "Date is $SESSION_START_TIME done",
		},
		{
			name:     "Unix date without zone",
			input:    "Date: Sat Sep  5 14:30:00 2026",
			expected: "Date: $SESSION_START_TIME",
		},
		{
			name:     "RFC1123 format",
			input:    "HTTP Date: Sat, 05 Sep 2026 14:30:00 GMT",
			expected: "HTTP Date: $SESSION_START_TIME",
		},
		{
			name:     "Environment banner Today's date",
			input:    "Today's date: Sat Sep 05 2026",
			expected: "Today's date: $SESSION_START_TIME",
		},
		{
			name:     "Bare date Sat Sep 05 2026",
			input:    "Session banner: Sat Sep 05 2026",
			expected: "Session banner: $SESSION_START_TIME",
		},
		{
			name:     "Labeled epoch seconds",
			input:    "timestamp: 1725537600",
			expected: "timestamp: $SESSION_START_TIME",
		},
		{
			name:     "Labeled epoch milliseconds",
			input:    "epoch=1725537600000",
			expected: "epoch=$SESSION_START_TIME",
		},
		{
			name:     "Bracketed epoch",
			input:    "Build [1725537600] completed",
			expected: "Build [$SESSION_START_TIME] completed",
		},
		{
			name:     "At epoch",
			input:    "Snapshot created @1725537600",
			expected: "Snapshot created @$SESSION_START_TIME",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := norm.NormalizeHeader(tc.input)
			if got != tc.expected {
				t.Errorf("NormalizeHeader(%q) = %q, want %q", tc.input, got, tc.expected)
			}
		})
	}
}

func TestTimestampNormalizationCustomSessionStart(t *testing.T) {
	customStart := "2026-09-05T00:00:00Z"
	got := ctxmmu.CanonicalizeHeader("Run at 2026-09-05T18:45:00Z on Sat Sep 05 2026", customStart, "")
	expected := "Run at 2026-09-05T00:00:00Z on 2026-09-05T00:00:00Z"
	if got != expected {
		t.Errorf("CanonicalizeHeader with custom start got %q, want %q", got, expected)
	}
}

func TestWorkspacePathCanonicalization(t *testing.T) {
	wsRootWin := `C:\Users\devuser\OneDrive\Desktop\work\fak`
	normWin := ctxmmu.NewNormalizer("", wsRootWin)

	testsWin := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "Windows path with backslashes",
			input:    `Working directory: C:\Users\devuser\OneDrive\Desktop\work\fak\internal\ctxmmu\normalizer.go`,
			expected: `Working directory: $WORKSPACE/internal/ctxmmu/normalizer.go`,
		},
		{
			name:     "Windows path with forward slashes",
			input:    `Reading file C:/Users/devuser/OneDrive/Desktop/work/fak/cmd/fak/main.go here`,
			expected: `Reading file $WORKSPACE/cmd/fak/main.go here`,
		},
		{
			name:     "Windows JSON escaped backslashes",
			input:    `{"path": "C:\\Users\\devuser\\OneDrive\\Desktop\\work\\fak\\go.mod"}`,
			expected: `{"path": "$WORKSPACE/go.mod"}`,
		},
		{
			name:     "Windows exact root directory",
			input:    `Workspace root: C:\Users\devuser\OneDrive\Desktop\work\fak`,
			expected: `Workspace root: $WORKSPACE`,
		},
		{
			name:     "Windows lowercase drive letter",
			input:    `Path: c:\users\devuser\onedrive\desktop\work\fak\README.md`,
			expected: `Path: $WORKSPACE/README.md`,
		},
	}

	for _, tc := range testsWin {
		t.Run(tc.name, func(t *testing.T) {
			got := normWin.NormalizeHeader(tc.input)
			if got != tc.expected {
				t.Errorf("NormalizeHeader(%q) = %q, want %q", tc.input, got, tc.expected)
			}
		})
	}

	// Unix path tests
	wsRootUnix := "/home/runner/work/fak"
	normUnix := ctxmmu.NewNormalizer("", wsRootUnix)

	testsUnix := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "Unix path with subdirectories",
			input:    "Compiling /home/runner/work/fak/internal/ctxmmu/normalizer.go now",
			expected: "Compiling $WORKSPACE/internal/ctxmmu/normalizer.go now",
		},
		{
			name:     "Unix exact root directory",
			input:    "Workspace: /home/runner/work/fak",
			expected: "Workspace: $WORKSPACE",
		},
	}

	for _, tc := range testsUnix {
		t.Run(tc.name, func(t *testing.T) {
			got := normUnix.NormalizeHeader(tc.input)
			if got != tc.expected {
				t.Errorf("NormalizeHeader(%q) = %q, want %q", tc.input, got, tc.expected)
			}
		})
	}
}

func TestVirtualRootCustomization(t *testing.T) {
	norm := ctxmmu.NewNormalizer("", "/home/runner/work/fak").WithVirtualRoot("/repo")
	got := norm.NormalizeHeader("File at /home/runner/work/fak/pkg/abi/types.go")
	expected := "File at /repo/pkg/abi/types.go"
	if got != expected {
		t.Errorf("WithVirtualRoot(/repo) got %q, want %q", got, expected)
	}
}

func TestPIDAndTempFileScrubbing(t *testing.T) {
	norm := ctxmmu.NewNormalizer("", "")

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "PID space separated",
			input:    "Process PID 12345 started",
			expected: "Process PID [PID] started",
		},
		{
			name:     "PID colon separated",
			input:    "Worker pid: 67890 active",
			expected: "Worker pid: [PID] active",
		},
		{
			name:     "PID equals separated",
			input:    "daemon pid=4321",
			expected: "daemon pid=[PID]",
		},
		{
			name:     "Process ID colon separated",
			input:    "Process ID: 9999 terminated",
			expected: "Process ID: [PID] terminated",
		},
		{
			name:     "Process space separated",
			input:    "Running process 5555",
			expected: "Running process [PID]",
		},
		{
			name:     "Bracketed PID",
			input:    "[PID: 54321] task spawned",
			expected: "[PID: [PID]] task spawned",
		},
		{
			name:     "Parenthesized pid",
			input:    "command (pid 8888) completed",
			expected: "command (pid [PID]) completed",
		},
		{
			name:     "Single digit PID",
			input:    "supervisor pid 1",
			expected: "supervisor pid [PID]",
		},
		{
			name:     "Unix opencode temp file",
			input:    "Output written to /tmp/opencode/cmd-87654.out",
			expected: "Output written to /tmp/opencode/[TMP]",
		},
		{
			name:     "Unix general temp file",
			input:    "Scratch at /tmp/scratch-99.tmp",
			expected: "Scratch at /tmp/[TMP]",
		},
		{
			name:     "Unix var/tmp file",
			input:    "Log at /var/tmp/temp-123.log",
			expected: "Log at /tmp/[TMP]",
		},
		{
			name:     "Windows opencode temp file",
			input:    `Temp path: C:\Users\devuser\AppData\Local\Temp\opencode\sub-987.tmp`,
			expected: `Temp path: /tmp/opencode/[TMP]`,
		},
		{
			name:     "Windows general temp file",
			input:    `Junk at C:\Users\devuser\AppData\Local\Temp\junk-456.tmp`,
			expected: `Junk at /tmp/[TMP]`,
		},
		{
			name:     "Windows opencode temp directory itself",
			input:    `Temp dir C:\Users\devuser\AppData\Local\Temp\opencode`,
			expected: `Temp dir /tmp/opencode`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := norm.NormalizeHeader(tc.input)
			if got != tc.expected {
				t.Errorf("NormalizeHeader(%q) = %q, want %q", tc.input, got, tc.expected)
			}
		})
	}
}

func TestToolOutputDurationNormalization(t *testing.T) {
	norm := ctxmmu.NewNormalizer("", "")
	input := "Command completed in duration: 45ms, elapsed: 1.25s, took 300ms"
	got := norm.NormalizeToolOutput("bash", input)
	expected := "Command completed in duration: [DURATION], elapsed: [DURATION], took [DURATION]"
	if got != expected {
		t.Errorf("NormalizeToolOutput duration got %q, want %q", got, expected)
	}
}

func TestLCPAndPrefixDivergenceHelpers(t *testing.T) {
	// Exact match
	s1 := "hello world"
	s2 := "hello world"
	if lcp := ctxmmu.ComputeLCP(s1, s2); lcp != len(s1) {
		t.Errorf("ComputeLCP(%q, %q) = %d, want %d", s1, s2, lcp, len(s1))
	}
	if div := ctxmmu.PrefixDivergenceRatio(s1, s2); div != 0.0 {
		t.Errorf("PrefixDivergenceRatio(%q, %q) = %f, want 0.0", s1, s2, div)
	}
	if pres := ctxmmu.LCPPreservationRatio(s1, s2); pres != 1.0 {
		t.Errorf("LCPPreservationRatio(%q, %q) = %f, want 1.0", s1, s2, pres)
	}

	// Extension: s2 extends s1
	s1 = "prefix block "
	s2 = "prefix block with appended turn content"
	if lcp := ctxmmu.ComputeLCP(s1, s2); lcp != len(s1) {
		t.Errorf("ComputeLCP(%q, %q) = %d, want %d", s1, s2, lcp, len(s1))
	}
	if div := ctxmmu.PrefixDivergenceRatio(s1, s2); div != 0.0 {
		t.Errorf("PrefixDivergenceRatio(%q, %q) = %f, want 0.0", s1, s2, div)
	}
	if pres := ctxmmu.LCPPreservationRatio(s1, s2); pres != 1.0 {
		t.Errorf("LCPPreservationRatio(%q, %q) = %f, want 1.0", s1, s2, pres)
	}

	// Complete divergence
	s1 = "alpha"
	s2 = "beta"
	if lcp := ctxmmu.ComputeLCP(s1, s2); lcp != 0 {
		t.Errorf("ComputeLCP(%q, %q) = %d, want 0", s1, s2, lcp)
	}
	if div := ctxmmu.PrefixDivergenceRatio(s1, s2); div != 1.0 {
		t.Errorf("PrefixDivergenceRatio(%q, %q) = %f, want 1.0", s1, s2, div)
	}

	// Empty strings
	if div := ctxmmu.PrefixDivergenceRatio("", ""); div != 0.0 {
		t.Errorf("PrefixDivergenceRatio for empty strings = %f, want 0.0", div)
	}
	if div := ctxmmu.PrefixDivergenceRatio("", "content"); div != 1.0 {
		t.Errorf("PrefixDivergenceRatio empty vs non-empty = %f, want 1.0", div)
	}
}

func TestLCPPreservationAcrossTurns(t *testing.T) {
	// Simulate 5 subagent turns with volatile timestamps, PIDs, workspace paths, and temp files.
	wsRoot := `C:\Users\devuser\OneDrive\Desktop\work\fak`
	norm := ctxmmu.NewNormalizer("$SESSION_START_TIME", wsRoot)

	type turnData struct {
		timestamp string
		pid       int
		tempFile  string
		duration  string
		command   string
		output    string
	}

	turns := []turnData{
		{
			timestamp: "2026-09-05T10:00:01Z",
			pid:       1001,
			tempFile:  `C:\Users\devuser\AppData\Local\Temp\opencode\proc-1001.tmp`,
			duration:  "15ms",
			command:   "git status",
			output:    "On branch main, working tree clean",
		},
		{
			timestamp: "2026-09-05T10:01:22Z",
			pid:       1042,
			tempFile:  `C:\Users\devuser\AppData\Local\Temp\opencode\proc-1042.tmp`,
			duration:  "85ms",
			command:   "go build ./...",
			output:    "build completed successfully",
		},
		{
			timestamp: "2026-09-05T10:02:45Z",
			pid:       1088,
			tempFile:  `C:\Users\devuser\AppData\Local\Temp\opencode\proc-1088.tmp`,
			duration:  "340ms",
			command:   "go test ./internal/ctxmmu/...",
			output:    "PASS ok github.com/anthony-chaudhary/fak/internal/ctxmmu",
		},
		{
			timestamp: "2026-09-05T10:03:10Z",
			pid:       1125,
			tempFile:  `C:\Users\devuser\AppData\Local\Temp\opencode\proc-1125.tmp`,
			duration:  "42ms",
			command:   "git diff",
			output:    "diff --git a/internal/ctxmmu/normalizer.go",
		},
		{
			timestamp: "2026-09-05T10:04:55Z",
			pid:       1190,
			tempFile:  `C:\Users\devuser\AppData\Local\Temp\opencode\proc-1190.tmp`,
			duration:  "120ms",
			command:   "git log -1",
			output:    "commit e8f31a2 perf(ctxmmu): canonicalize shell headers",
		},
	}

	renderTurnPrompt := func(upToTurn int, canonicalize bool) string {
		var sb strings.Builder
		// Immutable system header with environment injection
		rawSysHeader := fmt.Sprintf(
			"System Instructions for Subagent\n"+
				"Workspace root: %s\n"+
				"Temp directory: %s\n"+
				"Environment: win32\n"+
				"Session started: %s\n"+
				"Agent PID: %d\n\n",
			wsRoot,
			`C:\Users\devuser\AppData\Local\Temp\opencode`,
			turns[0].timestamp,
			turns[0].pid,
		)
		if canonicalize {
			sb.WriteString(norm.NormalizeHeader(rawSysHeader))
		} else {
			sb.WriteString(rawSysHeader)
		}

		// Interactive turns history
		for i := 0; i <= upToTurn; i++ {
			td := turns[i]
			rawTurn := fmt.Sprintf(
				"--- Turn %d ---\n"+
					"Command: %s (PID %d @ %s in %s\\internal\\ctxmmu)\n"+
					"TempFile: %s\n"+
					"Duration: duration: %s\n"+
					"Output:\n%s\n\n",
				i+1,
				td.command,
				td.pid,
				td.timestamp,
				wsRoot,
				td.tempFile,
				td.duration,
				td.output,
			)
			if canonicalize {
				sb.WriteString(norm.NormalizeToolOutput("bash", rawTurn))
			} else {
				sb.WriteString(rawTurn)
			}
		}
		return sb.String()
	}

	// 1. Verify that when canonicalized, consecutive turns preserve >= 98% of previous turn prompt
	for i := 0; i < len(turns)-1; i++ {
		canonCurrent := renderTurnPrompt(i, true)
		canonNext := renderTurnPrompt(i+1, true)

		lcp := ctxmmu.ComputeLCP(canonCurrent, canonNext)
		pres := ctxmmu.LCPPreservationRatio(canonCurrent, canonNext)
		div := ctxmmu.PrefixDivergenceRatio(canonCurrent, canonNext)

		if pres < 0.98 {
			t.Errorf("Turns %d -> %d: LCP preservation ratio = %.4f (< 0.98), lcp=%d, lenCurrent=%d",
				i+1, i+2, pres, lcp, len(canonCurrent))
		}
		if div > 0.02 {
			t.Errorf("Turns %d -> %d: prefix divergence ratio = %.4f (> 0.02)", i+1, i+2, div)
		}
		// In fact, the canonicalized current turn prompt is 100% byte-identical to the prefix of next turn!
		if lcp != len(canonCurrent) {
			t.Errorf("Turns %d -> %d: expected full prefix match (%d bytes), got %d",
				i+1, i+2, len(canonCurrent), lcp)
		}
	}

	// 2. Demonstrate that volatile headers alone across turns 1-5 have 100% LCP when canonicalized
	for i := 0; i < len(turns); i++ {
		for j := i + 1; j < len(turns); j++ {
			hdrA := fmt.Sprintf("Header @ %s PID %d in %s temp %s",
				turns[i].timestamp, turns[i].pid, wsRoot, turns[i].tempFile)
			hdrB := fmt.Sprintf("Header @ %s PID %d in %s temp %s",
				turns[j].timestamp, turns[j].pid, wsRoot, turns[j].tempFile)

			// Uncanonicalized headers diverge almost immediately on timestamp/pid
			rawLCP := ctxmmu.ComputeLCP(hdrA, hdrB)
			rawPres := ctxmmu.LCPPreservationRatio(hdrA, hdrB)
			if rawPres >= 0.98 {
				t.Fatalf("Uncanonicalized headers unexpectedly matched with high preservation: %f", rawPres)
			}
			_ = rawLCP

			// Canonicalized headers match 100%
			canonA := norm.NormalizeHeader(hdrA)
			canonB := norm.NormalizeHeader(hdrB)
			canonPres := ctxmmu.LCPPreservationRatio(canonA, canonB)
			if canonPres < 0.98 {
				t.Errorf("Canonicalized headers %d and %d: LCP preservation = %.4f, want >= 0.98",
					i+1, j+1, canonPres)
			}
			if canonA != canonB {
				t.Errorf("Canonicalized headers differed:\nA: %q\nB: %q", canonA, canonB)
			}
		}
	}
}

func TestMMUAdmissionNormalizer(t *testing.T) {
	wsRoot := `C:\Users\devuser\OneDrive\Desktop\work\fak`
	norm := ctxmmu.NewNormalizer("$SESSION_START_TIME", wsRoot)
	m := ctxmmu.New().WithNormalizer(norm)

	ctx := context.Background()
	call := &abi.ToolCall{
		Tool: "bash",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"cmd":"test"}`)},
	}

	rawOutput := fmt.Sprintf(
		"[bash PID 4321 @ 2026-09-05T12:00:00Z in %s\\internal\\ctxmmu]\n"+
			"Temp created at C:\\Users\\devuser\\AppData\\Local\\Temp\\opencode\\exec.tmp\n"+
			"Result: ok (duration: 45ms)\n",
		wsRoot,
	)

	res := &abi.Result{
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(rawOutput), Len: int64(len(rawOutput))},
	}

	v := m.Admit(ctx, call, res)
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("Admit verdict want Allow, got %v", v.Kind)
	}

	if res.Meta == nil || res.Meta["canonicalized"] != "true" {
		t.Errorf("Expected res.Meta[canonicalized] == true, got %v", res.Meta)
	}

	gotBytes := res.Payload.Inline
	gotStr := string(gotBytes)

	expectedOutput := norm.NormalizeToolOutput("bash", rawOutput)
	if gotStr != expectedOutput {
		t.Errorf("Admit output mismatch:\nGot:      %q\nExpected: %q", gotStr, expectedOutput)
	}

	// Verify specific volatile artifacts were canonicalized
	if strings.Contains(gotStr, "2026-09-05T12:00:00Z") {
		t.Errorf("Volatile timestamp was not canonicalized: %s", gotStr)
	}
	if strings.Contains(gotStr, "PID 4321") {
		t.Errorf("PID was not canonicalized: %s", gotStr)
	}
	if strings.Contains(gotStr, wsRoot) {
		t.Errorf("Workspace root was not canonicalized: %s", gotStr)
	}
	if strings.Contains(gotStr, `C:\Users\devuser\AppData\Local\Temp\opencode\exec.tmp`) {
		t.Errorf("Temp file was not canonicalized: %s", gotStr)
	}
	if !strings.Contains(gotStr, "$WORKSPACE/internal/ctxmmu") {
		t.Errorf("Expected $WORKSPACE/internal/ctxmmu in output: %s", gotStr)
	}
	if !strings.Contains(gotStr, "/tmp/opencode/[TMP]") {
		t.Errorf("Expected /tmp/opencode/[TMP] in output: %s", gotStr)
	}
	if !strings.Contains(gotStr, "duration: [DURATION]") {
		t.Errorf("Expected duration: [DURATION] in output: %s", gotStr)
	}
}
