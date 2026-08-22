package trajectory

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var auditClaudePytestFixturePath = regexp.MustCompile(`^C--Users-.+-AppData-Local-Temp-pytest-of-.+-pytest-[0-9]+-test_(?:main_json_runs_and_emits_0|route_findings_off_writes0|route_findings_on_appends0)-ws/00000000-aaaa-bbbb-cccc-000000000000\.jsonl$`)

// auditIsClaudePytestFixture recognizes the complete hermetic DOS test shape
// that is written beneath Claude's default projects root. Every mismatch falls
// through to normal parsing so missing provider usage remains a refusal.
func auditIsClaudePytestFixture(path, rel string) bool {
	rel = filepath.ToSlash(rel)
	if !auditClaudePytestFixturePath.MatchString(rel) {
		return false
	}
	parts := strings.Split(rel, "/")
	if len(parts) != 2 {
		return false
	}

	file, err := os.Open(path)
	if err != nil {
		return false
	}
	defer file.Close()

	var records []map[string]any
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 4096), 64*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record map[string]any
		if json.Unmarshal([]byte(line), &record) != nil {
			return false
		}
		records = append(records, record)
	}
	if scanner.Err() != nil || len(records) != 5 || !auditClaudePytestUser(records[0], parts[0]) {
		return false
	}
	for i, timestamp := range []string{
		"2026-06-01T14:00:00.000Z",
		"2026-06-01T14:01:00.000Z",
		"2026-06-01T14:02:00.000Z",
		"2026-06-01T14:03:00.000Z",
	} {
		if !auditClaudePytestAssistant(records[i+1], timestamp) {
			return false
		}
	}
	return true
}

func auditClaudePytestUser(record map[string]any, project string) bool {
	if !auditExactKeys(record, "type", "timestamp", "cwd", "gitBranch", "message") ||
		record["type"] != "user" || record["timestamp"] != "2026-06-01T14:00:00.000Z" || record["gitBranch"] != "feat/x" {
		return false
	}
	cwd, ok := record["cwd"].(string)
	if !ok || strings.NewReplacer(":", "-", `\`, "-", "/", "-").Replace(cwd) != project {
		return false
	}
	message, ok := record["message"].(map[string]any)
	if !ok || !auditExactKeys(message, "content") {
		return false
	}
	content, ok := message["content"].([]any)
	if !ok || len(content) != 1 {
		return false
	}
	block, ok := content[0].(map[string]any)
	return ok && auditExactKeys(block, "type", "text") && block["type"] == "text" && block["text"] == "do the thing"
}

func auditClaudePytestAssistant(record map[string]any, timestamp string) bool {
	if !auditExactKeys(record, "type", "timestamp", "version", "gitBranch", "cwd", "message") ||
		record["type"] != "assistant" || record["timestamp"] != timestamp || record["version"] != "1.0.0" ||
		record["gitBranch"] != "feat/x" || record["cwd"] != "/ws" {
		return false
	}
	message, ok := record["message"].(map[string]any)
	if !ok || !auditExactKeys(message, "usage", "content") {
		return false
	}
	usage, ok := message["usage"].(map[string]any)
	if !ok || len(usage) != 0 {
		return false
	}
	content, ok := message["content"].([]any)
	if !ok || len(content) != 1 {
		return false
	}
	block, ok := content[0].(map[string]any)
	if !ok || !auditExactKeys(block, "type", "name", "input") || block["type"] != "tool_use" || block["name"] != "Read" {
		return false
	}
	input, ok := block["input"].(map[string]any)
	return ok && auditExactKeys(input, "file_path") && input["file_path"] == "/ws/a.py"
}

func auditExactKeys(object map[string]any, keys ...string) bool {
	if len(object) != len(keys) {
		return false
	}
	for _, key := range keys {
		if _, ok := object[key]; !ok {
			return false
		}
	}
	return true
}
