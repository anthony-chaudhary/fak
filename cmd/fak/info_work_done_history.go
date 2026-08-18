package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	guardInfoWorkHistorySchema      = "fak.info.work-done-history/1"
	guardInfoWorkHistoryMaxRecords  = 256
	guardInfoWorkHistoryMinDuration = time.Second
)

type guardInfoWorkHistoryRecord struct {
	Schema     string                 `json:"schema"`
	RecordedAt string                 `json:"recorded_at"`
	WorkloadID string                 `json:"workload_id"`
	RunID      string                 `json:"run_id"`
	Query      guardInfoWorkDoneQuery `json:"query"`
}

type guardInfoWorkHistoryExport struct {
	Schema     string                         `json:"schema"`
	Records    []guardInfoWorkHistoryRecord   `json:"records"`
	Comparison guardInfoWorkHistoryComparison `json:"comparison"`
}

type guardInfoWorkHistoryComparison struct {
	Status          string  `json:"status"`
	Attribution     string  `json:"attribution"`
	TokenDelta      float64 `json:"token_delta,omitempty"`
	CallDelta       int64   `json:"call_delta,omitempty"`
	PriorRecordedAt string  `json:"prior_recorded_at,omitempty"`
	Reason          string  `json:"reason,omitempty"`
}

func guardInfoHistoryIdentity(kind, raw string) string {
	if raw == "" {
		return ""
	}
	sum := sha256.Sum256([]byte("fak-work-history/" + kind + "\x00" + raw))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func guardInfoWorkHistoryRecordFromQuery(q guardInfoWorkDoneQuery, workloadKey, runKey string, at time.Time) guardInfoWorkHistoryRecord {
	return guardInfoWorkHistoryRecord{
		Schema: guardInfoWorkHistorySchema, RecordedAt: at.UTC().Format(time.RFC3339Nano),
		WorkloadID: guardInfoHistoryIdentity("workload", workloadKey), RunID: guardInfoHistoryIdentity("run", runKey), Query: q,
	}
}

func guardInfoAppendWorkHistory(path string, record guardInfoWorkHistoryRecord) error {
	if path == "" {
		return nil
	}
	records, err := guardInfoReadWorkHistory(path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	records = append(records, record)
	if len(records) > guardInfoWorkHistoryMaxRecords {
		records = records[len(records)-guardInfoWorkHistoryMaxRecords:]
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	for _, row := range records {
		if err := enc.Encode(row); err != nil {
			_ = f.Close()
			return err
		}
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func guardInfoReadWorkHistory(path string) ([]guardInfoWorkHistoryRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var records []guardInfoWorkHistoryRecord
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 64<<10), 4<<20)
	for s.Scan() {
		var row guardInfoWorkHistoryRecord
		if err := json.Unmarshal(s.Bytes(), &row); err != nil {
			return nil, err
		}
		if row.Schema == guardInfoWorkHistorySchema {
			records = append(records, row)
		}
	}
	return records, s.Err()
}

func guardInfoCompareWorkHistory(current guardInfoWorkHistoryRecord, records []guardInfoWorkHistoryRecord) guardInfoWorkHistoryComparison {
	if current.WorkloadID == "" {
		return guardInfoWorkHistoryComparison{Status: "unavailable", Attribution: "unavailable", Reason: "workload_identity_not_provided"}
	}
	var latestAny, prior *guardInfoWorkHistoryRecord
	for i := range records {
		r := &records[i]
		if r.RecordedAt >= current.RecordedAt {
			continue
		}
		if latestAny == nil || r.RecordedAt > latestAny.RecordedAt {
			latestAny = r
		}
		if r.WorkloadID == current.WorkloadID && (prior == nil || r.RecordedAt > prior.RecordedAt) {
			prior = r
		}
	}
	if prior == nil {
		if latestAny != nil && latestAny.WorkloadID != current.WorkloadID {
			return guardInfoWorkHistoryComparison{Status: "incompatible", Attribution: "workload_changed", Reason: "no_prior_window_for_workload"}
		}
		return guardInfoWorkHistoryComparison{Status: "unavailable", Attribution: "sparse_history", Reason: "no_prior_window"}
	}
	if !guardInfoWorkDoneBaselineCompatible(prior.Query.WorkDone.Baseline, current.Query.WorkDone.Baseline) {
		return guardInfoWorkHistoryComparison{Status: "incompatible", Attribution: "baseline_changed", PriorRecordedAt: prior.RecordedAt, Reason: "baseline_identity_or_fingerprint_changed"}
	}
	if prior.Query.Window.Reset || current.Query.Window.Reset {
		return guardInfoWorkHistoryComparison{Status: "unavailable", Attribution: "reset", PriorRecordedAt: prior.RecordedAt, Reason: "window_reset"}
	}
	pm, cm := prior.Query.WorkDone.Metrics, current.Query.WorkDone.Metrics
	if !pm.InputTokensAvoided.Available || !cm.InputTokensAvoided.Available || !pm.ModelCallsAvoided.Available || !cm.ModelCallsAvoided.Available {
		return guardInfoWorkHistoryComparison{Status: "unavailable", Attribution: "evidence_unavailable", PriorRecordedAt: prior.RecordedAt, Reason: "required_metric_unavailable"}
	}
	if current.Query.Window.DurationNanos < int64(guardInfoWorkHistoryMinDuration) || prior.Query.Window.DurationNanos < int64(guardInfoWorkHistoryMinDuration) {
		return guardInfoWorkHistoryComparison{Status: "unavailable", Attribution: "quality_gate", PriorRecordedAt: prior.RecordedAt, Reason: "window_shorter_than_1s"}
	}
	out := guardInfoWorkHistoryComparison{PriorRecordedAt: prior.RecordedAt, TokenDelta: cm.InputTokensAvoided.Value - pm.InputTokensAvoided.Value, CallDelta: int64(cm.ModelCallsAvoided.Value - pm.ModelCallsAvoided.Value)}
	switch {
	case out.TokenDelta > 0 || out.CallDelta > 0:
		out.Status = "improved"
	case out.TokenDelta < 0 || out.CallDelta < 0:
		out.Status = "regressed"
	default:
		out.Status = "steady"
	}
	out.Attribution = "fak_mechanism_change"
	if sameWorkSourceEffects(prior.Query.WorkDone.Sources, current.Query.WorkDone.Sources) {
		out.Attribution = "total_changed_same_source_mix"
	}
	return out
}

func sameWorkSourceEffects(a, b []guardInfoWorkDoneSource) bool {
	type effect struct {
		tokens float64
		calls  uint64
	}
	fold := func(rows []guardInfoWorkDoneSource) map[string]effect {
		m := map[string]effect{}
		for _, r := range rows {
			m[r.ID] = effect{r.InputTokenEquiv, r.ModelCallsAvoided}
		}
		return m
	}
	am, bm := fold(a), fold(b)
	if len(am) != len(bm) {
		return false
	}
	for k, v := range am {
		if bm[k] != v {
			return false
		}
	}
	return true
}

func guardInfoComparedHistoryRecords(current guardInfoWorkHistoryRecord, records []guardInfoWorkHistoryRecord) []guardInfoWorkHistoryRecord {
	var prior *guardInfoWorkHistoryRecord
	for i := range records {
		r := &records[i]
		if r.RecordedAt >= current.RecordedAt || r.WorkloadID != current.WorkloadID {
			continue
		}
		if prior == nil || r.RecordedAt > prior.RecordedAt {
			prior = r
		}
	}
	if prior == nil {
		return []guardInfoWorkHistoryRecord{current}
	}
	return []guardInfoWorkHistoryRecord{*prior, current}
}

func guardInfoWorkHistoryRows(c guardInfoWorkHistoryComparison) []string {
	if c.Status == "unavailable" || c.Status == "incompatible" {
		return []string{fmt.Sprintf(" history %s · %s · %s", c.Status, strings.ReplaceAll(c.Attribution, "_", " "), strings.ReplaceAll(c.Reason, "_", " "))}
	}
	return []string{fmt.Sprintf(" history %s · tokens %+.0f · calls %+.0f · %s", c.Status, c.TokenDelta, float64(c.CallDelta), strings.ReplaceAll(c.Attribution, "_", " "))}
}

func (c *claudeMacDebugClient) decorateWorkHistory(v *guardInfoVars) {
	if c == nil || c.workHistoryPath == "" || c.workloadKey == "" {
		return
	}
	records, err := guardInfoReadWorkHistory(c.workHistoryPath)
	if err != nil && !os.IsNotExist(err) {
		return
	}
	now := time.Now().UTC()
	q := guardInfoSessionWorkDoneQuery(*v, now)
	current := guardInfoWorkHistoryRecordFromQuery(q, c.workloadKey, c.runKey, now)
	comparison := guardInfoCompareWorkHistory(current, records)
	v.WorkHistory = &comparison
}
