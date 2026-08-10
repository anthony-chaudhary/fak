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
	if body, dropped, ok := applyDistillFilter(in); ok {
		hint := fmt.Sprintf("[fak distill: %d routine go test line(s) dropped]", dropped)
		body = append(body, '\n')
		body = append(body, hint...)
		if len(body) < len(in.Bytes) { // never-worse invariant
			return Output{
				Bytes:      body,
				Compressed: true,
				Codec:      "go-test-distill",
				OrigLen:    len(in.Bytes),
				NewLen:     len(body),
				Status:     "saved",
				Reason:     "built-in go test filter removed routine passing output and preserved errors",
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
	matches func(Input) bool
	apply   func([]byte) ([]byte, int, bool)
}

var distillFilters = []distillFilter{
	{matches: matchesGoTest, apply: applyGoTestFilter},
}

func applyDistillFilter(in Input) ([]byte, int, bool) {
	for _, filter := range distillFilters {
		if filter.matches(in) {
			return filter.apply(in.Bytes)
		}
	}
	return nil, 0, false
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
	lines := strings.Split(string(raw), "\n")
	out := make([]string, 0, len(lines))
	dropped := 0
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "--- PASS:") || strings.HasPrefix(trimmed, "=== RUN") || strings.HasPrefix(trimmed, "=== PAUSE") || strings.HasPrefix(trimmed, "=== CONT") {
			dropped++
			continue
		}
		out = append(out, line)
	}
	if dropped == 0 {
		return nil, 0, false
	}
	body := bytes.TrimRight([]byte(strings.Join(out, "\n")), "\n")
	return body, dropped, true
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
