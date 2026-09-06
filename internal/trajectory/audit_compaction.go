package trajectory

// AuditCompactionEvent measures one file's explicit transcript content, not
// provider prompt bytes or tokens. Events are emitted on raw session rows;
// histories from separate fragments are never stitched into a denominator.
type AuditCompactionEvent struct {
	Line          int      `json:"line"`
	Unit          string   `json:"unit"`
	Origin        string   `json:"origin"`
	BeforeBytes   *int64   `json:"before_bytes"`
	AfterBytes    *int64   `json:"after_bytes"`
	SummaryBytes  *int64   `json:"summary_bytes"`
	RetainedRatio *float64 `json:"retained_ratio"`
	Coverage      string   `json:"coverage"`
}

type auditCompactionState struct {
	bytes      int64
	seen       bool
	incomplete bool
}

func (s *auditCompactionState) observe(record map[string]any, line int, row *AuditTranscriptRow) {
	payload, _ := record["payload"].(map[string]any)
	switch record["type"] {
	case "response_item":
		n, ok := auditCompactionItemBytes(payload)
		s.bytes += n
		s.seen = true
		s.incomplete = s.incomplete || !ok
	case "compacted":
		event := AuditCompactionEvent{Line: line, Unit: "reconstructed_transcript_content_utf8_bytes", Origin: "codex_compacted_event_engine_unobserved", Coverage: "complete"}
		// A summary may also occur as a preceding response_item. Report its
		// explicit bytes separately, without guessing which prior item it replaces.
		if n, ok := auditCompactionStringBytes(payload["message"]); ok {
			event.SummaryBytes = &n
		}
		if s.seen && !s.incomplete {
			n := s.bytes
			event.BeforeBytes = &n
		} else {
			event.Coverage = "incomplete_before_history"
		}
		history, ok := payload["replacement_history"].([]any)
		next := auditCompactionState{seen: ok, incomplete: !ok}
		for _, raw := range history {
			item, _ := raw.(map[string]any)
			n, supported := auditCompactionItemBytes(item)
			next.bytes += n
			next.incomplete = next.incomplete || !supported
		}
		if next.seen && !next.incomplete {
			n := next.bytes
			event.AfterBytes = &n
		} else if !ok {
			event.Coverage = "missing_or_malformed_replacement_history"
		} else {
			event.Coverage = "unsupported_replacement_history"
		}
		if event.BeforeBytes != nil && event.AfterBytes != nil {
			if *event.BeforeBytes > 0 {
				ratio := float64(*event.AfterBytes) / float64(*event.BeforeBytes)
				event.RetainedRatio = &ratio
			} else {
				event.Coverage = "empty_before_history"
			}
		}
		row.CompactionEvents = append(row.CompactionEvents, event)
		// A replacement supersedes all prior explicit history. Missing history
		// poisons subsequent denominators until a complete replacement arrives.
		*s = next
	}
}

func auditCompactionItemBytes(item map[string]any) (int64, bool) {
	switch item["type"] {
	case "message":
		switch item["role"] {
		case "user", "assistant", "developer", "system":
			return auditCompactionTextBytes(item["content"])
		}
	case "function_call":
		return auditCompactionStringBytes(item["arguments"])
	case "custom_tool_call":
		return auditCompactionStringBytes(item["input"])
	case "function_call_output", "custom_tool_call_output":
		return auditCompactionTextBytes(item["output"])
	}
	// Opaque reasoning, images, and unknown items have no complete observable
	// text representation. In particular, do not count their JSON as content.
	return 0, false
}

func auditCompactionStringBytes(value any) (int64, bool) {
	text, ok := value.(string)
	return int64(len(text)), ok
}

func auditCompactionTextBytes(value any) (int64, bool) {
	if text, ok := value.(string); ok {
		return int64(len(text)), true
	}
	blocks, ok := value.([]any)
	if !ok {
		return 0, false
	}
	var total int64
	complete := true
	for _, raw := range blocks {
		block, _ := raw.(map[string]any)
		switch block["type"] {
		case "text", "input_text", "output_text":
			n, supported := auditCompactionStringBytes(block["text"])
			total += n
			complete = complete && supported
		default:
			complete = false
		}
	}
	return total, complete
}
