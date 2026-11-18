package trajectory

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
)

// LangfuseExportSchema versions the Langfuse-shaped projection emitted over
// fak-trajectory-audit/1 JSONL. It is a distinct artifact from the audit schema
// it reads: the audit JSONL is the system of record, and this projection is a
// content-free, deterministic re-shaping of its session rows into the flat
// parent-pointer trace/observation tree that a self-hosted Langfuse sidecar
// ingests. No Langfuse UI, ClickHouse, or OTLP transport is ported; the shape
// mirrors Langfuse's ingestion records (trace, observation, score, session
// grouping) so the export can be POSTed to /api/public/ingestion unchanged.
//
// Upstream pin: langfuse/langfuse @ 2335676165d8f4f66ae276f7f187b44e43b26277
// (web/src/server/otel/otelRequestBody.ts, processOtelIngestion.ts).
const LangfuseExportSchema = "fak-trajectory-langfuse-export/1"

// langfuseExportTimestamp is the deterministic timestamp stamped on every
// projected record. The audit JSONL is a content-free rollup with no per-record
// wall clock, so a fixed RFC3339 value keeps the export byte-stable across runs
// while remaining a valid Langfuse timestamp.
const langfuseExportTimestamp = "1970-01-01T00:00:00Z"

// langfuseExportUnknownShape is the fail-closed refusal token for an audit JSONL
// record whose kind the projection does not recognize. Per PYTHON_PARITY
// discipline an unknown shape is refused, never estimated or dropped silently.
const langfuseExportUnknownShape = "LANGFUSE_EXPORT_SHAPE_REFUSED"

// ErrLangfuseExportUnknownShape is returned when the input JSONL carries a
// record kind the Langfuse projection does not know how to shape.
var ErrLangfuseExportUnknownShape = errors.New("trajectory langfuse export: " + langfuseExportUnknownShape)

// LangfuseExportRefusal names one record the projection refused.
type LangfuseExportRefusal struct {
	Line int    `json:"line"`
	Kind string `json:"kind"`
}

// LangfuseExportRecord is one Langfuse-shaped row. The Type field is the
// Langfuse ingestion discriminator ("trace", "observation", "score", or the
// projection's own "langfuse_export_receipt"); the remaining fields are the
// union of the per-type payloads, zero-valued where a type does not use them.
// JSON omits empty fields, so each row serializes to exactly its type's shape.
//
// Observation records are the one case where the wire "type" field carries the
// observation type ("agent"/"tool"/"generation") rather than the discriminator
// "observation": Langfuse's ingestion API keys observations by their type.
// MarshalJSON renders that; the Go-level Type field stays "observation" so
// in-process consumers can switch on it.
type LangfuseExportRecord struct {
	Type string `json:"type"`
	ID   string `json:"-"`

	// Trace / observation / score grouping.
	TraceID         string `json:"traceId,omitempty"`
	ParentID        string `json:"parentObservationId,omitempty"`
	SessionID       string `json:"sessionId,omitempty"`
	ObservationType string // observation type only ("agent"/"tool"/"generation"); not serialized
	Name            string `json:"name,omitempty"`
	Timestamp       string `json:"timestamp,omitempty"`
	Level           string `json:"level,omitempty"`
	StatusMessage   string `json:"statusMessage,omitempty"`
	Model           string `json:"model,omitempty"`
	ModelParameters any    `json:"modelParameters,omitempty"`
	Usage           any    `json:"usage,omitempty"`
	Input           any    `json:"input,omitempty"`
	Output          any    `json:"output,omitempty"`
	Metadata        any    `json:"metadata,omitempty"`
	CompletionStart string `json:"completionStartTime,omitempty"`

	// Score payload. ScoreName is the score's display name; it serializes under
	// the same "name" key as Name (scores have no other name), so MarshalJSON
	// routes it there and leaves Name empty for score records.
	ScoreName string   `json:"-"`
	Value     *float64 `json:"value,omitempty"`
	Comment   string   `json:"comment,omitempty"`
}

// MarshalJSON renders the record in Langfuse ingestion shape. The "type" field
// is the observation type for observation records and the record discriminator
// for every other record. A score's display name serializes under "name" (a
// score has no other name); all other fields are emitted only when set, in
// deterministic key order so the export is byte-stable across runs.
func (r LangfuseExportRecord) MarshalJSON() ([]byte, error) {
	wireType := r.Type
	if r.Type == "observation" {
		wireType = r.ObservationType
	}
	fields := map[string]any{"type": wireType}
	set := func(key string, value any) {
		if value == nil {
			return
		}
		if s, ok := value.(string); ok && s == "" {
			return
		}
		fields[key] = value
	}
	set("id", r.ID)
	set("traceId", r.TraceID)
	set("parentObservationId", r.ParentID)
	set("sessionId", r.SessionID)
	name := r.Name
	if r.Type == "score" {
		name = r.ScoreName
	}
	set("name", name)
	set("timestamp", r.Timestamp)
	set("level", r.Level)
	set("statusMessage", r.StatusMessage)
	set("model", r.Model)
	set("modelParameters", r.ModelParameters)
	set("usage", r.Usage)
	set("input", r.Input)
	set("output", r.Output)
	set("metadata", r.Metadata)
	set("completionStartTime", r.CompletionStart)
	set("value", r.Value)
	set("comment", r.Comment)

	keys := make([]string, 0, len(fields))
	for key := range fields {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	var builder strings.Builder
	builder.WriteByte('{')
	for i, key := range keys {
		if i > 0 {
			builder.WriteByte(',')
		}
		keyJSON, err := json.Marshal(key)
		if err != nil {
			return nil, err
		}
		valueJSON, err := json.Marshal(fields[key])
		if err != nil {
			return nil, err
		}
		builder.Write(keyJSON)
		builder.WriteByte(':')
		builder.Write(valueJSON)
	}
	builder.WriteByte('}')
	return []byte(builder.String()), nil
}

// LangfuseExportReceipt is the fidelity receipt appended as the final record of
// every export. It mirrors the audit's receipt pattern: it records what was
// read, what was shaped, and a digest of the input so a consumer can verify the
// projection was computed over the exact bytes it claims.
type LangfuseExportReceipt struct {
	Schema         string `json:"schema"`
	Kind           string `json:"kind"`
	InputRecords   int    `json:"input_records"`
	Sessions       int    `json:"sessions"`
	Traces         int    `json:"traces"`
	Observations   int    `json:"observations"`
	Scores         int    `json:"scores"`
	RefusedRecords int    `json:"refused_records"`
	SourceDigest   string `json:"source_digest"`
}

// LangfuseExport is the complete projection: the Langfuse-shaped records in
// deterministic order, followed by the receipt.
type LangfuseExport struct {
	Records []LangfuseExportRecord
	Receipt LangfuseExportReceipt
}

// ExportLangfuse reads fak-trajectory-audit/1 JSONL and projects it into a
// Langfuse-shaped trace/observation tree. Each "session" row becomes one trace
// (grouped into a Langfuse session by its session_id) with a root "agent"
// observation, one "tool" observation per tool in the session's tool
// distribution, and one "generation" observation per model. Two deterministic
// scores (tool-error fraction and accounted tokens) are emitted per session.
//
// The projection is fail-closed: a record whose kind is not one the projection
// recognizes returns ErrLangfuseExportUnknownShape (wrapped with the offending
// line and kind) and no records. A record whose schema is not
// fak-trajectory-audit/1 is likewise refused.
func ExportLangfuse(r io.Reader) (LangfuseExport, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return LangfuseExport{}, fmt.Errorf("trajectory langfuse export: read input: %w", err)
	}
	digest := "sha256:" + sha256hex(data)

	receipt := LangfuseExportReceipt{Schema: LangfuseExportSchema, Kind: "langfuse_export_receipt", SourceDigest: digest}
	var records []LangfuseExportRecord

	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 64*1024), 32*1024*1024)
	line := 0
	for scanner.Scan() {
		line++
		raw := bytes.TrimSpace(scanner.Bytes())
		if len(raw) == 0 {
			continue
		}
		receipt.InputRecords++

		var envelope struct {
			Schema string `json:"schema"`
			Kind   string `json:"kind"`
		}
		if err := json.Unmarshal(raw, &envelope); err != nil {
			return LangfuseExport{}, fmt.Errorf("trajectory langfuse export: line %d: decode envelope: %w", line, err)
		}
		if envelope.Schema != AuditSchema {
			return LangfuseExport{}, fmt.Errorf("trajectory langfuse export: line %d: schema %q is not %s: %w", line, envelope.Schema, AuditSchema, ErrLangfuseExportUnknownShape)
		}

		switch envelope.Kind {
		case "session":
			var row AuditTranscriptRow
			if err := json.Unmarshal(raw, &row); err != nil {
				return LangfuseExport{}, fmt.Errorf("trajectory langfuse export: line %d: decode session row: %w", line, err)
			}
			sessionRecords := langfuseSessionRecords(row)
			records = append(records, sessionRecords...)
			receipt.Sessions++
			for _, rec := range sessionRecords {
				switch rec.Type {
				case "trace":
					receipt.Traces++
				case "observation":
					receipt.Observations++
				case "score":
					receipt.Scores++
				}
			}
		case "summary", "source_denominator", "bottleneck", "baseline_delta",
			"event_schema", "schema_drift", "refusal", "mutation_churn",
			"tool_error_family", "corpus", "langfuse_export_receipt":
			// Known audit kinds that carry no per-session trace content. They are
			// read and accounted in InputRecords but contribute no Langfuse rows.
		default:
			return LangfuseExport{}, fmt.Errorf("%w: line %d kind %q", ErrLangfuseExportUnknownShape, line, envelope.Kind)
		}
	}
	if err := scanner.Err(); err != nil {
		return LangfuseExport{}, fmt.Errorf("trajectory langfuse export: scan input: %w", err)
	}

	records = append(records, langfuseReceiptRecord(receipt))
	return LangfuseExport{Records: records, Receipt: receipt}, nil
}

// langfuseSessionRecords projects one audit session row into its trace,
// observations, and scores, in deterministic order.
func langfuseSessionRecords(row AuditTranscriptRow) []LangfuseExportRecord {
	traceID := langfuseDeterministicID("trace", row.Source, row.TranscriptID)
	sessionID := row.TranscriptID
	if sessionID == "" {
		sessionID = langfuseDeterministicID("session", row.Source, row.SourcePath)
	}
	agentID := langfuseDeterministicID("agent", traceID)

	var out []LangfuseExportRecord
	out = append(out, LangfuseExportRecord{
		Type:      "trace",
		ID:        traceID,
		SessionID: sessionID,
		Name:      row.Source,
		Timestamp: langfuseExportTimestamp,
		Metadata: map[string]any{
			"source":      row.Source,
			"source_path": row.SourcePath,
			"tokens": map[string]int64{
				"input":        row.Tokens.InputTokens,
				"output":       row.Tokens.OutputTokens,
				"cache_create": row.Tokens.CacheCreateTokens,
				"cache_read":   row.Tokens.CacheReadTokens,
			},
		},
	})
	out = append(out, LangfuseExportRecord{
		Type:            "observation",
		ObservationType: "agent",
		ID:              agentID,
		TraceID:         traceID,
		SessionID:       sessionID,
		Name:            "fak.agent",
		Timestamp:       langfuseExportTimestamp,
		Level:           "DEFAULT",
		Metadata: map[string]any{
			"tool_calls":  row.ToolCalls,
			"tool_errors": row.ToolErrors,
		},
	})
	out = append(out, langfuseToolObservations(traceID, sessionID, agentID, row.ToolDistribution)...)
	out = append(out, langfuseGenerationObservations(traceID, sessionID, agentID, row)...)
	out = append(out, langfuseSessionScores(traceID, sessionID, row)...)
	return out
}

// langfuseToolObservations emits one "tool" observation per tool in the
// session's tool distribution, sorted by tool name for determinism.
func langfuseToolObservations(traceID, sessionID, parentID string, dist []AuditDistributionRow) []LangfuseExportRecord {
	if len(dist) == 0 {
		return nil
	}
	sorted := append([]AuditDistributionRow(nil), dist...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })
	out := make([]LangfuseExportRecord, 0, len(sorted))
	for _, tool := range sorted {
		out = append(out, LangfuseExportRecord{
			Type:            "observation",
			ObservationType: "tool",
			ID:              langfuseDeterministicID("tool", traceID, tool.Name),
			TraceID:         traceID,
			ParentID:        parentID,
			SessionID:       sessionID,
			Name:            tool.Name,
			Timestamp:       langfuseExportTimestamp,
			Level:           "DEFAULT",
			Metadata: map[string]any{
				"calls": tool.Calls,
				"bytes": tool.Bytes,
			},
		})
	}
	return out
}

// langfuseGenerationObservations emits one "generation" observation per model
// the session used, sorted by model name. The session's exact token buckets are
// carried on each generation's usage so a consumer can reconstruct the
// accounted total without a cross-vendor price.
func langfuseGenerationObservations(traceID, sessionID, parentID string, row AuditTranscriptRow) []LangfuseExportRecord {
	if len(row.Models) == 0 {
		return nil
	}
	models := append([]string(nil), row.Models...)
	sort.Strings(models)
	usage := map[string]int64{
		"input":        row.Tokens.InputTokens,
		"output":       row.Tokens.OutputTokens,
		"cache_create": row.Tokens.CacheCreateTokens,
		"cache_read":   row.Tokens.CacheReadTokens,
	}
	out := make([]LangfuseExportRecord, 0, len(models))
	for _, model := range models {
		out = append(out, LangfuseExportRecord{
			Type:            "observation",
			ObservationType: "generation",
			ID:              langfuseDeterministicID("generation", traceID, model),
			TraceID:         traceID,
			ParentID:        parentID,
			SessionID:       sessionID,
			Name:            "generate_content " + model,
			Timestamp:       langfuseExportTimestamp,
			Level:           "DEFAULT",
			Model:           model,
			Usage:           usage,
		})
	}
	return out
}

// langfuseSessionScores emits the deterministic per-session scores: the tool
// error fraction (only when the session made at least one tool call) and the
// exact accounted token total.
func langfuseSessionScores(traceID, sessionID string, row AuditTranscriptRow) []LangfuseExportRecord {
	var out []LangfuseExportRecord
	if row.ToolCalls > 0 {
		fraction := float64(row.ToolErrors) / float64(row.ToolCalls)
		out = append(out, LangfuseExportRecord{
			Type:      "score",
			ID:        langfuseDeterministicID("score", traceID, "tool_error_fraction"),
			TraceID:   traceID,
			SessionID: sessionID,
			ScoreName: "fak.trajectory.tool_error_fraction",
			Value:     &fraction,
			Comment:   fmt.Sprintf("%d tool errors over %d tool calls", row.ToolErrors, row.ToolCalls),
		})
	}
	tokens := float64(row.Tokens.accountedTotal())
	out = append(out, LangfuseExportRecord{
		Type:      "score",
		ID:        langfuseDeterministicID("score", traceID, "accounted_tokens"),
		TraceID:   traceID,
		SessionID: sessionID,
		ScoreName: "fak.trajectory.accounted_tokens",
		Value:     &tokens,
		Comment:   "exact accounted token total (input + cache_create + cache_read + output)",
	})
	return out
}

// langfuseReceiptRecord renders the receipt as the export's final record.
func langfuseReceiptRecord(receipt LangfuseExportReceipt) LangfuseExportRecord {
	return LangfuseExportRecord{
		Type: "langfuse_export_receipt",
		Metadata: map[string]any{
			"schema":          receipt.Schema,
			"kind":            receipt.Kind,
			"input_records":   receipt.InputRecords,
			"sessions":        receipt.Sessions,
			"traces":          receipt.Traces,
			"observations":    receipt.Observations,
			"scores":          receipt.Scores,
			"refused_records": receipt.RefusedRecords,
			"source_digest":   receipt.SourceDigest,
		},
	}
}

// langfuseDeterministicID derives a stable, content-free identifier from a
// label and the given parts, so the same session always projects to the same
// Langfuse ids across runs.
func langfuseDeterministicID(label string, parts ...string) string {
	var builder strings.Builder
	builder.WriteString(label)
	for _, part := range parts {
		builder.WriteByte('\x00')
		builder.WriteString(part)
	}
	return label + "-" + sha256hex([]byte(builder.String()))[:16]
}

func sha256hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
