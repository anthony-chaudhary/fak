package headroom

import (
	"bytes"
	"strings"
)

func gitCompatibleKind(in Input) bool {
	kind := in.Kind
	if kind == KindUnknown {
		kind = Detect(in.Bytes)
	}
	return kind == KindText || kind == KindLog || kind == KindCode
}

func gitCompatibleTool(tool string) bool {
	tool = strings.ToLower(tool)
	return strings.Contains(tool, "git") || strings.Contains(tool, "bash") || strings.Contains(tool, "shell") || strings.Contains(tool, "powershell")
}

func matchesGitStatus(in Input) bool {
	if !gitCompatibleKind(in) || !gitCompatibleTool(in.Tool) {
		return false
	}
	body := string(in.Bytes)
	return strings.Contains(body, "On branch ") || strings.Contains(body, "Changes not staged for commit:") || strings.Contains(body, "Changes to be committed:") || strings.Contains(body, "Untracked files:") || looksPorcelainStatus(in.Bytes)
}

func looksPorcelainStatus(raw []byte) bool {
	lines := bytes.Split(bytes.TrimRight(raw, "\r\n"), []byte("\n"))
	if len(lines) == 0 || len(lines) > 100000 {
		return false
	}
	matched := 0
	for _, line := range lines {
		line = bytes.TrimSuffix(line, []byte("\r"))
		if len(line) < 3 || (line[2] != ' ' && line[2] != '\t') || !validStatusCode(line[0]) || !validStatusCode(line[1]) {
			return false
		}
		matched++
	}
	return matched > 0
}

func validStatusCode(value byte) bool {
	return strings.ContainsRune(" MADRCU?!", rune(value))
}

func applyGitStatusFilter(raw []byte) ([]byte, int, bool) {
	return dropDistillRecords(raw, classifyGitStatusLine)
}

func classifyGitStatusLine(line []byte) recordLineKind {
	trimmed := strings.TrimSpace(string(line))
	switch {
	case strings.HasPrefix(trimmed, "(use \"git add "), strings.HasPrefix(trimmed, "(use \"git restore "), strings.HasPrefix(trimmed, "(use \"git rm "):
		return recordRoutine
	default:
		return recordUnknown
	}
}

func matchesGitLog(in Input) bool {
	if !gitCompatibleKind(in) || !gitCompatibleTool(in.Tool) {
		return false
	}
	body := string(in.Bytes)
	return strings.HasPrefix(body, "commit ") && strings.Contains(body, "\nAuthor:") && strings.Contains(body, "\nDate:")
}

func applyGitLogFilter(raw []byte) ([]byte, int, bool) {
	return dropDistillRecords(raw, classifyGitLogLine)
}

func classifyGitLogLine(line []byte) recordLineKind {
	if strings.HasPrefix(string(line), "Date:   ") {
		return recordRoutine
	}
	return recordUnknown
}

func dropDistillRecords(raw []byte, classify recordLineClassifier) ([]byte, int, bool) {
	records := groupDistillRecords(raw, 64*1024, classify)
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
