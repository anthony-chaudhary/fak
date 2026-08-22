package toolcallcontrol

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

// ResultBudgetMetricsSchema identifies the stable published snapshot contract.
const ResultBudgetMetricsSchema = "fak-tool-result-budget-metrics/1"

// ResultBudgetArtifact pins the exact policy artifact behind a metric.
type ResultBudgetArtifact struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	SHA256  string `json:"sha256"`
}

// ResultBudgetBranchCost records measured work performed by a response branch.
type ResultBudgetBranchCost struct {
	Latency         time.Duration `json:"-"`
	InputTokens     int64         `json:"input_tokens"`
	OutputTokens    int64         `json:"output_tokens"`
	ModelRoundTrips int64         `json:"model_round_trips"`
}

// ResultBudgetMetricEvent is one completed result-budget decision observation.
type ResultBudgetMetricEvent struct {
	Tool             string
	Contract         string
	Reason           string
	Mode             string
	Policy           ResultBudgetArtifact
	RequestedItems   int64
	EffectiveItems   int64
	ActualItems      int64
	ResponseBytes    int64
	ToolLatency      time.Duration
	Branch           ResultBudgetBranchCost
	OverrideApplied  bool
	ContinuationUsed bool
	ImmediateRefetch bool
}

// ResultBudgetBranchMetrics is the accumulated measured cost of response branches.
type ResultBudgetBranchMetrics struct {
	LatencyNanos    int64 `json:"latency_ns"`
	InputTokens     int64 `json:"input_tokens"`
	OutputTokens    int64 `json:"output_tokens"`
	ModelRoundTrips int64 `json:"model_round_trips"`
}

// ResultBudgetMetricBucket aggregates events sharing every required breakdown axis.
type ResultBudgetMetricBucket struct {
	Tool               string                    `json:"tool"`
	Contract           string                    `json:"contract"`
	Reason             string                    `json:"reason"`
	Mode               string                    `json:"mode"`
	Policy             ResultBudgetArtifact      `json:"policy"`
	Decisions          uint64                    `json:"decisions"`
	RequestedItems     int64                     `json:"requested_items"`
	EffectiveItems     int64                     `json:"effective_items"`
	ActualItems        int64                     `json:"actual_items"`
	ResponseBytes      int64                     `json:"response_bytes"`
	ToolLatencyNanos   int64                     `json:"tool_latency_ns"`
	Branch             ResultBudgetBranchMetrics `json:"branch_cost"`
	Overrides          uint64                    `json:"overrides"`
	Continuations      uint64                    `json:"continuations"`
	ImmediateRefetches uint64                    `json:"immediate_refetches"`
}

// ResultBudgetMetricsSnapshot is a deterministic, JSON-publishable metrics view.
type ResultBudgetMetricsSnapshot struct {
	Schema  string                     `json:"schema"`
	Buckets []ResultBudgetMetricBucket `json:"buckets"`
}

type resultBudgetMetricKey struct {
	Tool     string
	Contract string
	Reason   string
	Mode     string
	Policy   ResultBudgetArtifact
}

// ResultBudgetMetrics safely accumulates observations from concurrent tool calls.
type ResultBudgetMetrics struct {
	mu      sync.Mutex
	buckets map[resultBudgetMetricKey]ResultBudgetMetricBucket
}

// Record validates and adds one completed decision without partially updating a bucket.
func (metrics *ResultBudgetMetrics) Record(event ResultBudgetMetricEvent) error {
	if metrics == nil {
		return fmt.Errorf("result-budget metrics recorder is nil")
	}
	normalizeResultBudgetEvent(&event)
	if err := validateResultBudgetEvent(event); err != nil {
		return err
	}
	key := resultBudgetMetricKey{
		Tool: event.Tool, Contract: event.Contract, Reason: event.Reason,
		Mode: event.Mode, Policy: event.Policy,
	}

	metrics.mu.Lock()
	defer metrics.mu.Unlock()
	if metrics.buckets == nil {
		metrics.buckets = make(map[resultBudgetMetricKey]ResultBudgetMetricBucket)
	}
	next := metrics.buckets[key]
	if next.Decisions == 0 {
		next.Tool = key.Tool
		next.Contract = key.Contract
		next.Reason = key.Reason
		next.Mode = key.Mode
		next.Policy = key.Policy
	}
	if next.Decisions == math.MaxUint64 {
		return fmt.Errorf("result-budget decision count overflow")
	}
	var err error
	if next.RequestedItems, err = addResultBudgetMetric("requested items", next.RequestedItems, event.RequestedItems); err != nil {
		return err
	}
	if next.EffectiveItems, err = addResultBudgetMetric("effective items", next.EffectiveItems, event.EffectiveItems); err != nil {
		return err
	}
	if next.ActualItems, err = addResultBudgetMetric("actual items", next.ActualItems, event.ActualItems); err != nil {
		return err
	}
	if next.ResponseBytes, err = addResultBudgetMetric("response bytes", next.ResponseBytes, event.ResponseBytes); err != nil {
		return err
	}
	if next.ToolLatencyNanos, err = addResultBudgetMetric("tool latency", next.ToolLatencyNanos, int64(event.ToolLatency)); err != nil {
		return err
	}
	if next.Branch.LatencyNanos, err = addResultBudgetMetric("branch latency", next.Branch.LatencyNanos, int64(event.Branch.Latency)); err != nil {
		return err
	}
	if next.Branch.InputTokens, err = addResultBudgetMetric("branch input tokens", next.Branch.InputTokens, event.Branch.InputTokens); err != nil {
		return err
	}
	if next.Branch.OutputTokens, err = addResultBudgetMetric("branch output tokens", next.Branch.OutputTokens, event.Branch.OutputTokens); err != nil {
		return err
	}
	if next.Branch.ModelRoundTrips, err = addResultBudgetMetric("branch model round trips", next.Branch.ModelRoundTrips, event.Branch.ModelRoundTrips); err != nil {
		return err
	}
	next.Decisions++
	if event.OverrideApplied {
		next.Overrides++
	}
	if event.ContinuationUsed {
		next.Continuations++
	}
	if event.ImmediateRefetch {
		next.ImmediateRefetches++
	}
	metrics.buckets[key] = next
	return nil
}

// Snapshot returns a stable copy ordered by every breakdown axis.
func (metrics *ResultBudgetMetrics) Snapshot() ResultBudgetMetricsSnapshot {
	snapshot := ResultBudgetMetricsSnapshot{Schema: ResultBudgetMetricsSchema}
	if metrics == nil {
		return snapshot
	}
	metrics.mu.Lock()
	defer metrics.mu.Unlock()
	snapshot.Buckets = make([]ResultBudgetMetricBucket, 0, len(metrics.buckets))
	for _, bucket := range metrics.buckets {
		snapshot.Buckets = append(snapshot.Buckets, bucket)
	}
	sort.Slice(snapshot.Buckets, func(i, j int) bool {
		left, right := snapshot.Buckets[i], snapshot.Buckets[j]
		return resultBudgetMetricSortKey(left) < resultBudgetMetricSortKey(right)
	})
	return snapshot
}

func normalizeResultBudgetEvent(event *ResultBudgetMetricEvent) {
	event.Tool = strings.TrimSpace(event.Tool)
	event.Contract = strings.TrimSpace(event.Contract)
	event.Reason = strings.TrimSpace(event.Reason)
	event.Mode = strings.TrimSpace(event.Mode)
	event.Policy.Name = strings.TrimSpace(event.Policy.Name)
	event.Policy.Version = strings.TrimSpace(event.Policy.Version)
	event.Policy.SHA256 = strings.TrimSpace(event.Policy.SHA256)
}

func validateResultBudgetEvent(event ResultBudgetMetricEvent) error {
	for name, value := range map[string]string{
		"tool": event.Tool, "contract": event.Contract, "reason": event.Reason,
		"mode": event.Mode, "policy name": event.Policy.Name,
		"policy version": event.Policy.Version, "policy digest": event.Policy.SHA256,
	} {
		if value == "" {
			return fmt.Errorf("result-budget metric %s is required", name)
		}
	}
	for name, value := range map[string]int64{
		"requested items": event.RequestedItems, "effective items": event.EffectiveItems,
		"actual items": event.ActualItems, "response bytes": event.ResponseBytes,
		"tool latency": int64(event.ToolLatency), "branch latency": int64(event.Branch.Latency),
		"branch input tokens": event.Branch.InputTokens, "branch output tokens": event.Branch.OutputTokens,
		"branch model round trips": event.Branch.ModelRoundTrips,
	} {
		if value < 0 {
			return fmt.Errorf("result-budget metric %s must be non-negative", name)
		}
	}
	return nil
}

func addResultBudgetMetric(name string, current, observed int64) (int64, error) {
	if observed > math.MaxInt64-current {
		return 0, fmt.Errorf("result-budget metric %s overflow", name)
	}
	return current + observed, nil
}

func resultBudgetMetricSortKey(bucket ResultBudgetMetricBucket) string {
	return strings.Join([]string{
		bucket.Tool, bucket.Contract, bucket.Reason, bucket.Mode,
		bucket.Policy.Name, bucket.Policy.Version, bucket.Policy.SHA256,
	}, "\x00")
}
