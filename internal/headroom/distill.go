package headroom

import (
	"bytes"
	"context"
	"fmt"
	"strings"
)

// DistillName is the tool-aware built-in output filter. It is opt-in through the
// ordinary Compressor registry and falls back to the native structural compressor
// whenever no trusted built-in filter matches.
const DistillName = "distill"

type distillCompressor struct {
	native Compressor
}

func (distillCompressor) Name() string { return DistillName }

func (c distillCompressor) Compress(ctx context.Context, in Input) (Output, error) {
	if body, dropped, filter, ok := applyDistillFilter(in); ok {
		hint := fmt.Sprintf("[fak distill: %d %s dropped]", dropped, filter.dropLabel)
		body = append(body, '\n')
		body = append(body, hint...)
		if len(body) < len(in.Bytes) { // never-worse invariant
			return Output{
				Bytes:      body,
				Compressed: true,
				Codec:      filter.codec,
				OrigLen:    len(in.Bytes),
				NewLen:     len(body),
				Status:     "saved",
				Reason:     filter.reason,
			}, nil
		}
	}
	fallback := c.native
	if fallback == nil {
		fallback = nativeCompressor{}
	}
	return fallback.Compress(ctx, in)
}

// A distillFilter is a trusted, compiled-in filter selected by both producer and
// detected content kind. Keeping this registry internal prevents project output
// from supplying the rules that decide which of its own bytes the model sees.
type distillFilter struct {
	codec     string
	dropLabel string
	reason    string
	matches   func(Input) bool
	apply     func([]byte) ([]byte, int, bool)
}

var distillFilters = []distillFilter{
	{codec: "go-test-distill", dropLabel: "routine go test line(s)", reason: "built-in go test filter removed routine passing output and preserved errors", matches: matchesGoTest, apply: applyGoTestFilter},
	{codec: "golangci-lint-distill", dropLabel: "routine golangci-lint line(s)", reason: "built-in golangci-lint filter removed routine info output and preserved diagnostics", matches: matchesGolangCILint, apply: applyGolangCILintFilter},
	{codec: "package-manager-distill", dropLabel: "routine npm/pnpm line(s)", reason: "built-in npm/pnpm filter removed routine progress and preserved warnings and failures", matches: matchesPackageManager, apply: applyPackageManagerFilter},
	{codec: "git-status-distill", dropLabel: "routine git status advice line(s)", reason: "built-in git status filter removed routine advice and preserved repository state", matches: matchesGitStatus, apply: applyGitStatusFilter},
	{codec: "git-log-distill", dropLabel: "routine git log date line(s)", reason: "built-in git log filter removed routine date metadata and preserved commit identities and subjects", matches: matchesGitLog, apply: applyGitLogFilter},
}

func applyDistillFilter(in Input) ([]byte, int, distillFilter, bool) {
	for _, filter := range distillFilters {
		if filter.matches(in) {
			body, dropped, ok := filter.apply(in.Bytes)
			return body, dropped, filter, ok
		}
	}
	return nil, 0, distillFilter{}, false
}

func matchesGoTest(in Input) bool {
	kind := in.Kind
	if kind == KindUnknown {
		kind = Detect(in.Bytes)
	}
	if kind != KindText && kind != KindLog {
		return false
	}
	lowerTool := strings.ToLower(in.Tool)
	if !strings.Contains(lowerTool, "bash") && !strings.Contains(lowerTool, "shell") && !strings.Contains(lowerTool, "go test") {
		return false
	}
	// The result itself must carry Go test's stable record vocabulary; the tool
	// name alone is not enough to apply a lossy classifier.
	return bytes.Contains(in.Bytes, []byte("--- PASS:")) || bytes.Contains(in.Bytes, []byte("--- FAIL:"))
}

// applyGoTestFilter drops only routine passing-test records. FAIL records,
// continuation lines, diagnostics, package summaries, and unknown lines are kept.
// That conservative rule is the error-preservation invariant: classification
// uncertainty spends tokens rather than hiding evidence.
func applyGoTestFilter(raw []byte) ([]byte, int, bool) {
	records := groupDistillRecords(raw, 64*1024, classifyGoTestLine)
	out := make([]distillRecord, 0, len(records))
	dropped := 0
	for _, record := range records {
		if record.Kind == recordRoutine && !record.ForceKeep {
			dropped++
			continue
		}
		out = append(out, record)
	}
	if dropped == 0 {
		return nil, 0, false
	}
	return bytes.TrimRight(joinDistillRecords(out), "\r\n"), dropped, true
}

func classifyGoTestLine(line []byte) recordLineKind {
	trimmed := strings.TrimSpace(string(line))
	switch {
	case strings.HasPrefix(trimmed, "--- PASS:"), strings.HasPrefix(trimmed, "=== RUN"), strings.HasPrefix(trimmed, "=== PAUSE"), strings.HasPrefix(trimmed, "=== CONT"):
		return recordRoutine
	case strings.HasPrefix(trimmed, "--- FAIL:"), trimmed == "FAIL", strings.HasPrefix(trimmed, "FAIL\t"):
		return recordError
	case len(line) > 0 && (line[0] == ' ' || line[0] == '\t'):
		return recordContinuation
	default:
		return recordUnknown
	}
}
func init() { Register(distillCompressor{}) }

func appendRestoreHint(body []byte, origin string) []byte {
	hint := "[restore original: fak_context_restore {\"origin\":\"" + origin + "\"}]"
	out := make([]byte, 0, len(body)+1+len(hint))
	out = append(out, body...)
	if len(out) > 0 && out[len(out)-1] != '\n' {
		out = append(out, '\n')
	}
	return append(out, hint...)
}
