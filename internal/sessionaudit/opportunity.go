package sessionaudit

import (
	"bufio"
	"bytes"
	"encoding/json"
	"sort"
	"strings"
)

// ToolOpportunityRow attributes tool-result volume to the command/tool that
// produced it. EstCompressible is a conservative structural estimate: bytes in
// repeated whole lines beyond their first occurrence. It is an opportunity
// signal, not a promised distiller saving.
type ToolOpportunityRow struct {
	Tool            string `json:"tool"`
	Calls           int64  `json:"calls"`
	RawBytes        int64  `json:"raw_bytes"`
	EstCompressible int64  `json:"est_compressible"`
}

type opportunityCall struct {
	tool string
	key  string
}

// OpportunityByTool deterministically scans a Claude-style transcript JSONL,
// pairs tool_result blocks with their tool_use ids, and ranks output-volume
// buckets. Malformed records and unpaired results are ignored rather than
// guessed. Ties are ordered by bucket name.
func OpportunityByTool(transcript []byte) []ToolOpportunityRow {
	calls := make(map[string]opportunityCall)
	rows := make(map[string]ToolOpportunityRow)
	scan := bufio.NewScanner(bytes.NewReader(transcript))
	// Session records can contain large tool results; match Analyze's practical
	// transcript posture rather than Scanner's small default token limit.
	scan.Buffer(make([]byte, 64*1024), 64*1024*1024)
	for scan.Scan() {
		var record transcriptRecord
		if json.Unmarshal(scan.Bytes(), &record) != nil || len(record.Message.Content) == 0 {
			continue
		}
		var blocks []contentBlock
		if json.Unmarshal(record.Message.Content, &blocks) != nil {
			continue
		}
		for _, block := range blocks {
			switch block.Type {
			case "tool_use":
				if block.ID == "" {
					continue
				}
				calls[block.ID] = opportunityCall{tool: block.Name, key: opportunityBucket(block.Name, block.Input)}
			case "tool_result":
				call, ok := calls[block.ToolUseID]
				if !ok || call.key == "" {
					continue
				}
				body := textBytes(block.Content)
				row := rows[call.key]
				row.Tool = call.key
				row.Calls++
				row.RawBytes += int64(len(body))
				row.EstCompressible += repeatedLineBytes(body)
				rows[call.key] = row
				delete(calls, block.ToolUseID)
			}
		}
	}
	out := make([]ToolOpportunityRow, 0, len(rows))
	for _, row := range rows {
		out = append(out, row)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].EstCompressible != out[j].EstCompressible {
			return out[i].EstCompressible > out[j].EstCompressible
		}
		if out[i].RawBytes != out[j].RawBytes {
			return out[i].RawBytes > out[j].RawBytes
		}
		return out[i].Tool < out[j].Tool
	})
	return out
}

func opportunityBucket(tool string, input json.RawMessage) string {
	name := strings.TrimSpace(tool)
	lower := strings.ToLower(name)
	if lower != "bash" && lower != "shell" && lower != "powershell" {
		return lower
	}
	var args struct {
		Command string `json:"command"`
	}
	if json.Unmarshal(input, &args) != nil {
		return lower
	}
	if token := leadingCommandToken(args.Command); token != "" {
		return token
	}
	return lower
}

func leadingCommandToken(command string) string {
	command = strings.TrimSpace(command)
	for strings.HasPrefix(command, "$") {
		// Skip a leading shell assignment such as FOO=bar, but retain commands
		// whose executable genuinely starts with '$'.
		break
	}
	fields := strings.Fields(command)
	for _, field := range fields {
		field = strings.Trim(field, "'\"")
		if strings.Contains(field, "=") && !strings.ContainsAny(field, `/\\`) {
			continue
		}
		field = strings.TrimSuffix(strings.TrimSuffix(field, ".exe"), ".EXE")
		return strings.ToLower(field)
	}
	return ""
}

func textBytes(raw json.RawMessage) []byte {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return []byte(s)
	}
	var parts []json.RawMessage
	if json.Unmarshal(raw, &parts) == nil {
		var out []byte
		for _, part := range parts {
			out = append(out, textBytes(part)...)
		}
		return out
	}
	var object map[string]json.RawMessage
	if json.Unmarshal(raw, &object) == nil {
		if text, ok := object["text"]; ok {
			return textBytes(text)
		}
		if content, ok := object["content"]; ok {
			return textBytes(content)
		}
	}
	return nil
}

func repeatedLineBytes(body []byte) int64 {
	seen := make(map[string]bool)
	var saved int64
	for _, line := range bytes.SplitAfter(body, []byte("\n")) {
		if len(line) == 0 {
			continue
		}
		key := string(line)
		if seen[key] {
			saved += int64(len(line))
		} else {
			seen[key] = true
		}
	}
	return saved
}
