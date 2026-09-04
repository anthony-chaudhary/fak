// Package observability provides pure Go telemetry monitoring, SQLite header analysis,
// and automated alarm evaluation for tokens, latency, and database growth.
//
// Invariant: telemetry alarms enforce fail-closed evaluation and deterministic thresholds.
// Invariant: database inspection parses SQLite headers directly without CGO dependencies or locks.
// Invariant: evaluations execute without network I/O, background routines, or mutable global state.
package observability

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
)

// TelemetryAlarmType identifies the specific category of telemetry anomaly being evaluated.
type TelemetryAlarmType string

const (
	// AlarmPromptDoubling signals that prompt token volume has doubled the baseline or breached hard limits.
	AlarmPromptDoubling TelemetryAlarmType = "PROMPT_DOUBLING"
	// AlarmLatencySpike signals that turn execution latency exceeded threshold duration or median multipliers.
	AlarmLatencySpike TelemetryAlarmType = "LATENCY_SPIKE"
	// AlarmDatabaseBloat signals that SQLite storage footprint or freelist fragmentation exceeded acceptable bounds.
	AlarmDatabaseBloat TelemetryAlarmType = "DATABASE_BLOAT"
)

// TelemetryAlarmSeverity represents the operational criticality classification of an alarm.
type TelemetryAlarmSeverity string

const (
	// SeverityOK indicates normal operation with all telemetry metrics within defined bounds.
	SeverityOK TelemetryAlarmSeverity = "OK"
	// SeverityWarn indicates an anomalous telemetry threshold breach requiring investigation.
	SeverityWarn TelemetryAlarmSeverity = "WARN"
)

// Default threshold constants for telemetry alarms (#11147).
const (
	// DefaultHardPromptCapTokens defines the ceiling token count before triggering an alarm.
	DefaultHardPromptCapTokens = 30000 // >30k tokens
	// DefaultPromptDoublingFactor defines the maximum relative growth multiple permitted against baseline token count.
	DefaultPromptDoublingFactor = 2.0 // >2x baseline
	// DefaultLatencySpikeThresholdSec defines the maximum wall-clock turn duration in seconds before flagging latency.
	DefaultLatencySpikeThresholdSec = 15.0 // >15s turn latency
	// DefaultLatencySpikeMultiplier defines the factor over median latency that triggers a latency spike alarm.
	DefaultLatencySpikeMultiplier = 2.5 // >2.5x median latency
	// DefaultMaxDatabaseBytes defines the size threshold in bytes above which database bloat is flagged.
	DefaultMaxDatabaseBytes int64 = 1 * 1024 * 1024 * 1024 // >1GB DB bloat
	// DefaultFreelistRatioThreshold defines the maximum tolerable ratio of freelist pages to total database pages.
	DefaultFreelistRatioThreshold = 0.25 // >25% large freelist
)

// TelemetryAlarm captures the evaluation result and diagnostic findings of a single telemetry check.
type TelemetryAlarm struct {
	Type      TelemetryAlarmType     `json:"type"`
	Severity  TelemetryAlarmSeverity `json:"severity"`
	Triggered bool                   `json:"triggered"`
	Message   string                 `json:"message"`
	Detail    string                 `json:"detail,omitempty"`
}

// TelemetryHealthReport aggregates multi-dimensional prompt, latency, and database telemetry evaluation results.
type TelemetryHealthReport struct {
	OK             bool             `json:"ok"`
	PromptAlarm    TelemetryAlarm   `json:"prompt_alarm"`
	LatencyAlarm   TelemetryAlarm   `json:"latency_alarm"`
	DatabaseAlarm  TelemetryAlarm   `json:"database_alarm"`
	Alarms         []TelemetryAlarm `json:"alarms"`
	PromptTokens   int              `json:"prompt_tokens"`
	BaselinePrompt int              `json:"baseline_prompt"`
	CurrentLatency float64          `json:"current_latency_sec"`
	MedianLatency  float64          `json:"median_latency_sec"`
	DBPath         string           `json:"db_path,omitempty"`
	DBBytes        int64            `json:"db_bytes,omitempty"`
	FreelistPages  int64            `json:"freelist_pages,omitempty"`
	PageCount      int64            `json:"page_count,omitempty"`
	PageSize       int64            `json:"page_size,omitempty"`
	DBError        string           `json:"db_error,omitempty"`
	Findings       int              `json:"findings"`
}

// CheckPromptTokenAlarm evaluates turn prompt tokens against doubling factor and hard capacity thresholds.
// Invariant: evaluations are deterministic and purely arithmetic based on input token counts.
func CheckPromptTokenAlarm(turnTokens int, baselineTokens int) TelemetryAlarm {
	triggered := turnTokens > DefaultHardPromptCapTokens || (baselineTokens > 0 && float64(turnTokens) > DefaultPromptDoublingFactor*float64(baselineTokens))
	if triggered {
		var detail string
		if turnTokens > DefaultHardPromptCapTokens && (baselineTokens > 0 && float64(turnTokens) > DefaultPromptDoublingFactor*float64(baselineTokens)) {
			detail = fmt.Sprintf("turn prompt tokens (%d) exceeds %d hard cap and %.1fx baseline (%d)", turnTokens, DefaultHardPromptCapTokens, DefaultPromptDoublingFactor, baselineTokens)
		} else if turnTokens > DefaultHardPromptCapTokens {
			detail = fmt.Sprintf("turn prompt tokens (%d) exceeds hard cap of %d", turnTokens, DefaultHardPromptCapTokens)
		} else {
			detail = fmt.Sprintf("turn prompt tokens (%d) exceeds %.1fx baseline (%d)", turnTokens, DefaultPromptDoublingFactor, baselineTokens)
		}
		return TelemetryAlarm{
			Type:      AlarmPromptDoubling,
			Severity:  SeverityWarn,
			Triggered: true,
			Message:   "prompt token doubling or threshold breach detected",
			Detail:    detail,
		}
	}
	return TelemetryAlarm{
		Type:      AlarmPromptDoubling,
		Severity:  SeverityOK,
		Triggered: false,
		Message:   "prompt token count within normal limits",
		Detail:    fmt.Sprintf("turn: %d, baseline: %d", turnTokens, baselineTokens),
	}
}

// CheckLatencyAlarm evaluates turn latency against duration caps and median multipliers.
// Invariant: non-positive median latencies evaluate safely without division-by-zero or panics.
func CheckLatencyAlarm(currentLatencySec float64, medianLatencySec float64, consecutiveSpikes int) TelemetryAlarm {
	triggered := currentLatencySec > DefaultLatencySpikeThresholdSec || (medianLatencySec > 0 && currentLatencySec > DefaultLatencySpikeMultiplier*medianLatencySec)
	if triggered {
		var detail string
		if currentLatencySec > DefaultLatencySpikeThresholdSec && (medianLatencySec > 0 && currentLatencySec > DefaultLatencySpikeMultiplier*medianLatencySec) {
			detail = fmt.Sprintf("current latency %.2fs exceeds %.1fs cap and %.1fx median (%.2fs), consecutive spikes: %d", currentLatencySec, DefaultLatencySpikeThresholdSec, DefaultLatencySpikeMultiplier, medianLatencySec, consecutiveSpikes)
		} else if currentLatencySec > DefaultLatencySpikeThresholdSec {
			detail = fmt.Sprintf("current latency %.2fs exceeds hard cap of %.1fs, consecutive spikes: %d", currentLatencySec, DefaultLatencySpikeThresholdSec, consecutiveSpikes)
		} else {
			detail = fmt.Sprintf("current latency %.2fs exceeds %.1fx median (%.2fs), consecutive spikes: %d", currentLatencySec, DefaultLatencySpikeMultiplier, medianLatencySec, consecutiveSpikes)
		}
		return TelemetryAlarm{
			Type:      AlarmLatencySpike,
			Severity:  SeverityWarn,
			Triggered: true,
			Message:   "turn latency spike detected",
			Detail:    detail,
		}
	}
	return TelemetryAlarm{
		Type:      AlarmLatencySpike,
		Severity:  SeverityOK,
		Triggered: false,
		Message:   "turn latency within normal limits",
		Detail:    fmt.Sprintf("current: %.2fs, median: %.2fs, consecutive spikes: %d", currentLatencySec, medianLatencySec, consecutiveSpikes),
	}
}

// CheckDatabaseBloatAlarm evaluates SQLite storage footprint and freelist page fragmentation.
// Invariant: zero page counts evaluate without division-by-zero and yield zero freelist ratios.
func CheckDatabaseBloatAlarm(dbBytes int64, freelistPages int64, pageCount int64, pageSize int64) TelemetryAlarm {
	freelistRatio := 0.0
	if pageCount > 0 {
		freelistRatio = float64(freelistPages) / float64(pageCount)
	}
	triggered := dbBytes > DefaultMaxDatabaseBytes || (pageCount > 0 && freelistRatio > DefaultFreelistRatioThreshold)
	if triggered {
		var detail string
		if dbBytes > DefaultMaxDatabaseBytes && (pageCount > 0 && freelistRatio > DefaultFreelistRatioThreshold) {
			detail = fmt.Sprintf("db size (%d bytes) exceeds 1GB and freelist ratio (%.2f%%) exceeds %.0f%%", dbBytes, freelistRatio*100, DefaultFreelistRatioThreshold*100)
		} else if dbBytes > DefaultMaxDatabaseBytes {
			detail = fmt.Sprintf("db size (%d bytes) exceeds 1GB threshold", dbBytes)
		} else {
			detail = fmt.Sprintf("freelist ratio (%.2f%%, %d/%d pages) exceeds %.0f%% threshold", freelistRatio*100, freelistPages, pageCount, DefaultFreelistRatioThreshold*100)
		}
		return TelemetryAlarm{
			Type:      AlarmDatabaseBloat,
			Severity:  SeverityWarn,
			Triggered: true,
			Message:   "database bloat or fragmentation detected",
			Detail:    detail,
		}
	}
	return TelemetryAlarm{
		Type:      AlarmDatabaseBloat,
		Severity:  SeverityOK,
		Triggered: false,
		Message:   "database size and freelist ratio within normal limits",
		Detail:    fmt.Sprintf("size: %d bytes, freelist ratio: %.2f%% (%d/%d pages, page size %d)", dbBytes, freelistRatio*100, freelistPages, pageCount, pageSize),
	}
}

// InspectSQLiteFileHeader reads file stat and parses the 100-byte SQLite header in pure Go without CGO.
// Invariant: inspects SQLite file metadata directly from header bytes without locking or side effects.
func InspectSQLiteFileHeader(dbPath string) (dbBytes, freelistPages, pageCount, pageSize int64, err error) {
	if dbPath == "" {
		return 0, 0, 0, 0, errors.New("empty database path")
	}
	info, err := os.Stat(dbPath)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	if info.IsDir() {
		return 0, 0, 0, 0, fmt.Errorf("path is a directory: %s", dbPath)
	}
	dbBytes = info.Size()
	if dbBytes < 100 {
		return dbBytes, 0, 0, 0, fmt.Errorf("file size (%d bytes) too small for SQLite header (min 100 bytes)", dbBytes)
	}

	f, err := os.Open(dbPath)
	if err != nil {
		return dbBytes, 0, 0, 0, err
	}
	defer f.Close()

	var header [100]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		return dbBytes, 0, 0, 0, fmt.Errorf("failed to read SQLite header: %w", err)
	}

	const sqliteMagic = "SQLite format 3\x00"
	if string(header[:16]) != sqliteMagic {
		return dbBytes, 0, 0, 0, fmt.Errorf("invalid SQLite header: magic mismatch %q", string(header[:16]))
	}

	rawPageSize := binary.BigEndian.Uint16(header[16:18])
	if rawPageSize == 1 {
		pageSize = 65536
	} else {
		pageSize = int64(rawPageSize)
	}
	if pageSize <= 0 {
		return dbBytes, 0, 0, 0, fmt.Errorf("invalid SQLite page size: %d", pageSize)
	}

	freelistPages = int64(binary.BigEndian.Uint32(header[36:40]))
	pageCount = dbBytes / pageSize

	return dbBytes, freelistPages, pageCount, pageSize, nil
}

// CheckLatencySpikeAlarm evaluates turn latency against spike thresholds and median multipliers.
// It delegates to CheckLatencyAlarm to verify whether current turn latency exceeds safe bounds.
func CheckLatencySpikeAlarm(currentLatencySec float64, medianLatencySec float64, consecutiveSpikes int) TelemetryAlarm {
	return CheckLatencyAlarm(currentLatencySec, medianLatencySec, consecutiveSpikes)
}

// EvaluateHealth evaluates prompt, latency, and database telemetry against alarm thresholds.
// It delegates to EvaluateTelemetryHealth to produce an aggregated TelemetryHealthReport.
func EvaluateHealth(promptTokens, baselinePrompt int, latencies []float64, dbPath string) TelemetryHealthReport {
	return EvaluateTelemetryHealth(promptTokens, baselinePrompt, latencies, dbPath)
}

// EvaluateTelemetryHealth evaluates prompt, latency, and database telemetry against alarm thresholds.
// Invariant: telemetry alarms enforce fail-closed evaluation and deterministic thresholds.
// Invariant: an empty latency sequence evaluates to a normal latency alarm rather than panicking.
// Invariant: an unreachable database path results in a triggered warning alarm to fail safe.
func EvaluateTelemetryHealth(promptTokens, baselinePrompt int, latencies []float64, dbPath string) TelemetryHealthReport {
	promptAlarm := CheckPromptTokenAlarm(promptTokens, baselinePrompt)

	var currentLatency, median float64
	consecutiveSpikes := 0
	if len(latencies) > 0 {
		currentLatency = latencies[len(latencies)-1]
		sorted := make([]float64, len(latencies))
		copy(sorted, latencies)
		sort.Float64s(sorted)
		n := len(sorted)
		if n%2 == 1 {
			median = sorted[n/2]
		} else {
			median = (sorted[n/2-1] + sorted[n/2]) / 2.0
		}

		for i := len(latencies) - 1; i >= 0; i-- {
			lat := latencies[i]
			isSpike := lat > DefaultLatencySpikeThresholdSec || (median > 0 && lat > DefaultLatencySpikeMultiplier*median)
			if isSpike {
				consecutiveSpikes++
			} else {
				break
			}
		}
	}
	latencyAlarm := CheckLatencyAlarm(currentLatency, median, consecutiveSpikes)

	var dbBytes, freelistPages, pageCount, pageSize int64
	var dbAlarm TelemetryAlarm
	var dbErrStr string
	if dbPath != "" {
		var err error
		dbBytes, freelistPages, pageCount, pageSize, err = InspectSQLiteFileHeader(dbPath)
		if err != nil {
			dbErrStr = err.Error()
			dbAlarm = TelemetryAlarm{
				Type:      AlarmDatabaseBloat,
				Severity:  SeverityWarn,
				Triggered: true,
				Message:   "database inspection failed",
				Detail:    err.Error(),
			}
		} else {
			dbAlarm = CheckDatabaseBloatAlarm(dbBytes, freelistPages, pageCount, pageSize)
		}
	} else {
		dbAlarm = CheckDatabaseBloatAlarm(0, 0, 0, 0)
	}

	alarms := []TelemetryAlarm{promptAlarm, latencyAlarm, dbAlarm}
	findings := 0
	for _, a := range alarms {
		if a.Severity == SeverityWarn {
			findings++
		}
	}

	return TelemetryHealthReport{
		OK:             findings == 0,
		PromptAlarm:    promptAlarm,
		LatencyAlarm:   latencyAlarm,
		DatabaseAlarm:  dbAlarm,
		Alarms:         alarms,
		PromptTokens:   promptTokens,
		BaselinePrompt: baselinePrompt,
		CurrentLatency: currentLatency,
		MedianLatency:  median,
		DBPath:         dbPath,
		DBBytes:        dbBytes,
		FreelistPages:  freelistPages,
		PageCount:      pageCount,
		PageSize:       pageSize,
		DBError:        dbErrStr,
		Findings:       findings,
	}
}
