package trajectory

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// langfuseTestSessionJSONL is a minimal fak-trajectory-audit/1 JSONL corpus:
// one summary row, one source_denominator row, and two session rows. The
// sessions carry distinct tool distributions and models so the projection's
// typed observations, scores, and session grouping are all exercised.
const langfuseTestSessionJSONL = `{"schema":"fak-trajectory-audit/1","kind":"summary","sources":1,"sessions":2}
{"schema":"fak-trajectory-audit/1","kind":"source_denominator","source":"claude","root":"claude/projects","root_present":true,"files_discovered":2,"files_scanned":2,"records":40,"record_types":{},"token_semantics":"message usage buckets are disjoint"}
{"schema":"fak-trajectory-audit/1","kind":"session","source":"claude","session_id":"sess-a","source_path":"projects/sess-a.jsonl","models":["claude-sonnet-4","claude-haiku-4"],"tokens":{"input_tokens":1000,"output_tokens":200,"cache_create_tokens":50,"cache_read_tokens":30},"tool_calls":10,"tool_errors":2,"tool_distribution":[{"name":"Read","bytes":400,"share":0.5,"calls":6},{"name":"Bash","bytes":400,"share":0.5,"calls":4}]}
{"schema":"fak-trajectory-audit/1","kind":"session","source":"claude","session_id":"sess-b","source_path":"projects/sess-b.jsonl","models":["claude-sonnet-4"],"tokens":{"input_tokens":500,"output_tokens":100,"cache_create_tokens":0,"cache_read_tokens":0},"tool_calls":0,"tool_errors":0,"tool_distribution":[]}`

// langfuseRecordsByType groups the export's records by their Go-level Type.
func langfuseRecordsByType(t *testing.T, export LangfuseExport) map[string][]LangfuseExportRecord {
	t.Helper()
	byType := map[string][]LangfuseExportRecord{}
	for _, rec := range export.Records {
		byType[rec.Type] = append(byType[rec.Type], rec)
	}
	return byType
}

// langfuseFindRecord returns the first record of the given Go-level type whose
// Name matches, failing the test if none exists.
func langfuseFindRecord(t *testing.T, export LangfuseExport, recType, name string) LangfuseExportRecord {
	t.Helper()
	for _, rec := range export.Records {
		if rec.Type == recType && rec.Name == name {
			return rec
		}
	}
	t.Fatalf("no %s record named %q in export", recType, name)
	return LangfuseExportRecord{}
}

func TestLangfuseExport(t *testing.T) {
	t.Run("HappyPathExportShape", func(t *testing.T) {
		export, err := ExportLangfuse(strings.NewReader(langfuseTestSessionJSONL))
		if err != nil {
			t.Fatalf("ExportLangfuse: %v", err)
		}
		byType := langfuseRecordsByType(t, export)
		if len(byType["trace"]) != 2 {
			t.Fatalf("traces = %d, want 2", len(byType["trace"]))
		}
		if len(byType["observation"]) == 0 {
			t.Fatalf("no observations emitted")
		}
		if len(byType["score"]) == 0 {
			t.Fatalf("no scores emitted")
		}
		// The receipt is the final record and carries the input digest.
		last := export.Records[len(export.Records)-1]
		if last.Type != "langfuse_export_receipt" {
			t.Fatalf("final record type = %q, want langfuse_export_receipt", last.Type)
		}
		if export.Receipt.Sessions != 2 || export.Receipt.Traces != 2 {
			t.Fatalf("receipt sessions/traces = %d/%d, want 2/2", export.Receipt.Sessions, export.Receipt.Traces)
		}
		if export.Receipt.InputRecords != 4 {
			t.Fatalf("receipt input_records = %d, want 4", export.Receipt.InputRecords)
		}
		if !strings.HasPrefix(export.Receipt.SourceDigest, "sha256:") {
			t.Fatalf("receipt source_digest = %q, want sha256: prefix", export.Receipt.SourceDigest)
		}
		// Every record must serialize to valid JSON with a "type" key.
		for _, rec := range export.Records {
			encoded, err := json.Marshal(rec)
			if err != nil {
				t.Fatalf("marshal %s record: %v", rec.Type, err)
			}
			var wire map[string]any
			if err := json.Unmarshal(encoded, &wire); err != nil {
				t.Fatalf("unmarshal wire: %v", err)
			}
			if _, ok := wire["type"]; !ok {
				t.Fatalf("wire record missing type key: %s", encoded)
			}
		}
	})

	t.Run("ParentObservationLinkage", func(t *testing.T) {
		export, err := ExportLangfuse(strings.NewReader(langfuseTestSessionJSONL))
		if err != nil {
			t.Fatalf("ExportLangfuse: %v", err)
		}
		// The agent observation is the root of each trace; tool and generation
		// observations must point at it via parentObservationId.
		agent := langfuseFindRecord(t, export, "observation", "fak.agent")
		if agent.ParentID != "" {
			t.Fatalf("agent observation has parent %q, want none (root)", agent.ParentID)
		}
		tool := langfuseFindRecord(t, export, "observation", "Read")
		if tool.ParentID != agent.ID {
			t.Fatalf("tool observation parent = %q, want agent id %q", tool.ParentID, agent.ID)
		}
		if tool.TraceID != agent.TraceID {
			t.Fatalf("tool observation trace = %q, want agent trace %q", tool.TraceID, agent.TraceID)
		}
		gen := langfuseFindRecord(t, export, "observation", "generate_content claude-sonnet-4")
		if gen.ParentID != agent.ID {
			t.Fatalf("generation observation parent = %q, want agent id %q", gen.ParentID, agent.ID)
		}
	})

	t.Run("TypedObservations", func(t *testing.T) {
		export, err := ExportLangfuse(strings.NewReader(langfuseTestSessionJSONL))
		if err != nil {
			t.Fatalf("ExportLangfuse: %v", err)
		}
		// sess-a has two tools and two models; sess-b has one model and no tools.
		agentCount, toolCount, genCount := 0, 0, 0
		for _, rec := range export.Records {
			if rec.Type != "observation" {
				continue
			}
			switch rec.ObservationType {
			case "agent":
				agentCount++
			case "tool":
				toolCount++
			case "generation":
				genCount++
			default:
				t.Fatalf("unexpected observation type %q", rec.ObservationType)
			}
		}
		if agentCount != 2 {
			t.Fatalf("agent observations = %d, want 2", agentCount)
		}
		if toolCount != 2 { // Read + Bash, both from sess-a
			t.Fatalf("tool observations = %d, want 2", toolCount)
		}
		if genCount != 3 { // 2 models (sess-a) + 1 model (sess-b)
			t.Fatalf("generation observations = %d, want 3", genCount)
		}
		// A generation observation carries its model and the exact usage buckets.
		gen := langfuseFindRecord(t, export, "observation", "generate_content claude-sonnet-4")
		if gen.Model != "claude-sonnet-4" {
			t.Fatalf("generation model = %q, want claude-sonnet-4", gen.Model)
		}
		usage, ok := gen.Usage.(map[string]int64)
		if !ok {
			t.Fatalf("generation usage = %T, want map[string]int64", gen.Usage)
		}
		if usage["input"] != 1000 || usage["output"] != 200 {
			t.Fatalf("generation usage = %v, want input=1000 output=200", usage)
		}
	})

	t.Run("Scores", func(t *testing.T) {
		export, err := ExportLangfuse(strings.NewReader(langfuseTestSessionJSONL))
		if err != nil {
			t.Fatalf("ExportLangfuse: %v", err)
		}
		// sess-a: 2 tool errors over 10 calls => fraction 0.2; accounted tokens
		// 1000+200+50+30 = 1280. sess-b: no tool calls => no tool-error score,
		// but still an accounted-tokens score of 600.
		var toolError, accounted []LangfuseExportRecord
		for _, rec := range export.Records {
			if rec.Type != "score" {
				continue
			}
			switch rec.ScoreName {
			case "fak.trajectory.tool_error_fraction":
				toolError = append(toolError, rec)
			case "fak.trajectory.accounted_tokens":
				accounted = append(accounted, rec)
			}
		}
		if len(toolError) != 1 {
			t.Fatalf("tool_error_fraction scores = %d, want 1 (only sess-a has tool calls)", len(toolError))
		}
		if got := *toolError[0].Value; got != 0.2 {
			t.Fatalf("tool_error_fraction value = %v, want 0.2", got)
		}
		if len(accounted) != 2 {
			t.Fatalf("accounted_tokens scores = %d, want 2", len(accounted))
		}
		values := map[float64]bool{}
		for _, rec := range accounted {
			values[*rec.Value] = true
		}
		if !values[1280] || !values[600] {
			t.Fatalf("accounted_tokens values = %v, want {1280, 600}", values)
		}
	})

	t.Run("SessionGrouping", func(t *testing.T) {
		export, err := ExportLangfuse(strings.NewReader(langfuseTestSessionJSONL))
		if err != nil {
			t.Fatalf("ExportLangfuse: %v", err)
		}
		// Every trace and observation for a session shares that session's
		// sessionId, and the two sessions stay disjoint.
		sessions := map[string]map[string]bool{} // sessionId -> set of traceIds
		for _, rec := range export.Records {
			if rec.Type == "trace" {
				sessions[rec.SessionID] = map[string]bool{rec.ID: true}
				continue
			}
			if rec.SessionID == "" {
				continue
			}
			if _, ok := sessions[rec.SessionID]; !ok {
				t.Fatalf("observation/score references unknown session %q", rec.SessionID)
			}
		}
		if len(sessions) != 2 {
			t.Fatalf("distinct sessions = %d, want 2", len(sessions))
		}
		if _, okA := sessions["sess-a"]; !okA {
			t.Fatalf("session sess-a missing: %v", sessions)
		}
		if _, okB := sessions["sess-b"]; !okB {
			t.Fatalf("session sess-b missing: %v", sessions)
		}
		// The two sessions must project to distinct trace ids, and every
		// observation/score must reference one of those trace ids.
		traceIDs := map[string]bool{}
		for _, rec := range export.Records {
			if rec.Type == "trace" {
				traceIDs[rec.ID] = true
			}
		}
		if len(traceIDs) != 2 {
			t.Fatalf("distinct trace ids = %d, want 2", len(traceIDs))
		}
		for _, rec := range export.Records {
			if rec.Type == "observation" || rec.Type == "score" {
				if !traceIDs[rec.TraceID] {
					t.Fatalf("%s record references unknown trace id %q", rec.Type, rec.TraceID)
				}
			}
		}
	})

	t.Run("FailClosedUnknownShape", func(t *testing.T) {
		// A record kind the projection does not recognize must be refused, not
		// silently dropped or estimated.
		unknown := `{"schema":"fak-trajectory-audit/1","kind":"summary","sessions":1}
{"schema":"fak-trajectory-audit/1","kind":"mystery_kind","source":"claude"}`
		_, err := ExportLangfuse(strings.NewReader(unknown))
		if !errors.Is(err, ErrLangfuseExportUnknownShape) {
			t.Fatalf("error = %v, want ErrLangfuseExportUnknownShape", err)
		}
		if err == nil || !strings.Contains(err.Error(), "mystery_kind") {
			t.Fatalf("error should name the offending kind: %v", err)
		}
	})

	t.Run("FailClosedWrongSchema", func(t *testing.T) {
		// A record whose schema is not fak-trajectory-audit/1 is refused.
		wrongSchema := `{"schema":"fak-trajectory-audit/2","kind":"session","session_id":"x"}`
		_, err := ExportLangfuse(strings.NewReader(wrongSchema))
		if !errors.Is(err, ErrLangfuseExportUnknownShape) {
			t.Fatalf("error = %v, want ErrLangfuseExportUnknownShape", err)
		}
	})

	t.Run("DeterministicIDs", func(t *testing.T) {
		first, err := ExportLangfuse(strings.NewReader(langfuseTestSessionJSONL))
		if err != nil {
			t.Fatalf("first export: %v", err)
		}
		second, err := ExportLangfuse(strings.NewReader(langfuseTestSessionJSONL))
		if err != nil {
			t.Fatalf("second export: %v", err)
		}
		if len(first.Records) != len(second.Records) {
			t.Fatalf("record counts differ: %d vs %d", len(first.Records), len(second.Records))
		}
		for i := range first.Records {
			if first.Records[i].ID != second.Records[i].ID {
				t.Fatalf("record %d id differs across runs: %q vs %q", i, first.Records[i].ID, second.Records[i].ID)
			}
		}
	})
}
