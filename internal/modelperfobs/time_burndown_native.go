package modelperfobs

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

// readTimeBurndownObservations accepts the proxy observation wire and the native
// gateway ledger wire. Native rows retain the server-minted identity used by
// /v1/fak/perf/phases; a proxy's observation ID is not that identity.
func readTimeBurndownObservations(r io.Reader) ([]Observation, error) {
	var rows []Observation
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 64*1024), 16*1024*1024)
	for s.Scan() {
		var header struct {
			Schema string `json:"schema"`
		}
		if err := json.Unmarshal(s.Bytes(), &header); err != nil {
			return nil, fmt.Errorf("decode observation %d: %w", len(rows)+1, err)
		}
		var row Observation
		switch header.Schema {
		case Schema:
			if err := json.Unmarshal(s.Bytes(), &row); err != nil {
				return nil, fmt.Errorf("decode observation %d: %w", len(rows)+1, err)
			}
		case perfledger.Schema:
			var native perfledger.Record
			if err := json.Unmarshal(s.Bytes(), &native); err != nil {
				return nil, fmt.Errorf("decode native observation %d: %w", len(rows)+1, err)
			}
			var err error
			row, err = nativeBurndownObservation(native)
			if err != nil {
				return nil, fmt.Errorf("native observation %d: %w", len(rows)+1, err)
			}
		default:
			return nil, fmt.Errorf("observation %d: schema %q, want %q or %q", len(rows)+1, header.Schema, Schema, perfledger.Schema)
		}
		rows = append(rows, row)
	}
	return rows, s.Err()
}

func nativeBurndownObservation(r perfledger.Record) (Observation, error) {
	if r.PromptTokens < 0 || r.CompletionTokens < 0 || r.CachedTokens < 0 || r.PromptTokens > int(^uint(0)>>1)-r.CachedTokens {
		return Observation{}, fmt.Errorf("%w: native token counts are invalid", ErrInvalidTimeBurndown)
	}
	status := r.Status
	// The native success constructor predates Status and leaves it zero. A
	// failure with no observed HTTP status must not become a successful sample.
	if status == 0 && r.Error == "" {
		status = 200
	}
	// Early native failures may have no completion and therefore no model.
	// Preserve that unknown identity for failure accounting; successful rows
	// still require all identities before joining native phase receipts.
	successful := r.Error == "" && status >= 200 && status < 400
	if strings.TrimSpace(r.RequestID) == "" || strings.TrimSpace(r.Backend) == "" || (successful && strings.TrimSpace(r.Model) == "") {
		return Observation{}, fmt.Errorf("%w: native record requires request/backend identities and successful records require a model", ErrInvalidTimeBurndown)
	}
	// Warmup reuses native IsProbe exclusion (including its tiny-cached-prompt
	// heuristic); it does not assign a cold/warm experimental cache state.
	row := Observation{
		Schema: Schema, RequestID: r.RequestID, Model: r.Model, Backend: r.Backend,
		Status: status, Error: r.Error, DurationMS: r.E2EMS, TTFTMS: r.TTFTMS,
		CacheState: "unknown", CacheStateSource: "unknown", Warmup: r.IsProbe(),
	}
	// Native PromptTokens counts uncached work, not the full logical prompt.
	// CacheRegime does not prove a declared cold/warm experimental arm. Native
	// TTFT may be planner-derived for a buffered completion, so Streaming stays
	// false: the API wall remains unpartitioned without first-output evidence.
	if !r.UsageUnreported {
		row.PromptTokens = int64(r.PromptTokens)
		row.CompletionTokens = int64(r.CompletionTokens)
	}
	return row, nil
}
