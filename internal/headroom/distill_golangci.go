package headroom

import (
	"bytes"
	"strings"
)

func matchesGolangCILint(in Input) bool {
	kind := in.Kind
	if kind == KindUnknown {
		kind = Detect(in.Bytes)
	}
	if kind != KindText && kind != KindLog && kind != KindCode {
		return false
	}
	tool := strings.ToLower(in.Tool)
	if !strings.Contains(tool, "golangci-lint") && !strings.Contains(tool, "bash") && !strings.Contains(tool, "shell") {
		return false
	}
	lower := bytes.ToLower(in.Bytes)
	return bytes.Contains(lower, []byte("golangci-lint")) || bytes.Contains(lower, []byte("level=info")) || bytes.Contains(lower, []byte("level=warning")) || bytes.Contains(lower, []byte("level=error"))
}

func applyGolangCILintFilter(raw []byte) ([]byte, int, bool) {
	records := groupDistillRecords(raw, 64*1024, classifyGolangCILintLine)
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

func classifyGolangCILintLine(line []byte) recordLineKind {
	text := string(line)
	trimmed := strings.TrimSpace(text)
	lower := strings.ToLower(trimmed)
	switch {
	case strings.HasPrefix(lower, "level=info "):
		return recordRoutine
	case strings.HasPrefix(lower, "level=warning "), strings.HasPrefix(lower, "level=error "):
		return recordError
	case looksGolangCIDiagnostic(trimmed):
		return recordError
	case len(text) > 0 && (text[0] == ' ' || text[0] == '\t'):
		return recordContinuation
	default:
		return recordUnknown
	}
}

func looksGolangCIDiagnostic(line string) bool {
	// Stable text output: path:line:column: message (linter). Requiring both a
	// numeric location and trailing linter keeps arbitrary colon-rich logs out.
	open := strings.LastIndex(line, " (")
	if open < 0 || !strings.HasSuffix(line, ")") {
		return false
	}
	prefix := line[:open]
	parts := strings.SplitN(prefix, ":", 4)
	if len(parts) != 4 {
		return false
	}
	return allDigits(parts[1]) && allDigits(parts[2]) && strings.TrimSpace(parts[3]) != ""
}

func allDigits(value string) bool {
	if value == "" {
		return false
	}
	for _, r := range value {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}
