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
	if body, dropped, ok := applyGoTestFilter(in.Tool, in.Bytes); ok {
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

// applyGoTestFilter drops only routine passing-test records. FAIL records,
// continuation lines, diagnostics, package summaries, and unknown lines are kept.
// That conservative rule is the error-preservation invariant: classification
// uncertainty spends tokens rather than hiding evidence.
func applyGoTestFilter(tool string, raw []byte) ([]byte, int, bool) {
	if !isGoTestTool(tool, raw) {
		return nil, 0, false
	}
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

func isGoTestTool(tool string, raw []byte) bool {
	lowerTool := strings.ToLower(tool)
	if !strings.Contains(lowerTool, "bash") && !strings.Contains(lowerTool, "shell") && !strings.Contains(lowerTool, "go test") {
		return false
	}
	// The result itself must carry Go test's stable record vocabulary; the tool
	// name alone is not enough to apply a lossy classifier.
	return bytes.Contains(raw, []byte("--- PASS:")) || bytes.Contains(raw, []byte("--- FAIL:"))
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
