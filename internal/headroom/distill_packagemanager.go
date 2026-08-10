package headroom

import (
	"bytes"
	"strings"
)

func matchesPackageManager(in Input) bool {
	kind := in.Kind
	if kind == KindUnknown {
		kind = Detect(in.Bytes)
	}
	if kind != KindText && kind != KindLog && kind != KindCode {
		return false
	}
	tool := strings.ToLower(in.Tool)
	if !strings.Contains(tool, "npm") && !strings.Contains(tool, "pnpm") && !strings.Contains(tool, "bash") && !strings.Contains(tool, "shell") {
		return false
	}
	lower := bytes.ToLower(in.Bytes)
	return bytes.Contains(lower, []byte("npm ")) || bytes.Contains(lower, []byte("npm error")) || bytes.Contains(lower, []byte("npm warn")) || bytes.Contains(lower, []byte("pnpm")) || bytes.Contains(lower, []byte("progress: resolved"))
}

func applyPackageManagerFilter(raw []byte) ([]byte, int, bool) {
	records := groupDistillRecords(raw, 64*1024, classifyPackageManagerLine)
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

func classifyPackageManagerLine(line []byte) recordLineKind {
	text := string(line)
	trimmed := strings.TrimSpace(text)
	lower := strings.ToLower(trimmed)
	switch {
	case strings.HasPrefix(lower, "npm error"), strings.HasPrefix(lower, "npm err!"), strings.HasPrefix(lower, "npm warn"), strings.HasPrefix(lower, "warn "), strings.HasPrefix(lower, "err_pnpm_"), strings.HasPrefix(lower, " err_pnpm_"), strings.Contains(lower, "failed"):
		return recordError
	case strings.HasPrefix(lower, "progress: resolved "), strings.HasPrefix(lower, "npm http fetch "), strings.HasPrefix(lower, "npm timing "), strings.HasPrefix(lower, "npm verbose "):
		return recordRoutine
	case len(text) > 0 && (text[0] == ' ' || text[0] == '\t'):
		return recordContinuation
	default:
		return recordUnknown
	}
}
