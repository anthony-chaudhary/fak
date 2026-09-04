package nativeperf

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const defaultReceiptMetricStaleAfter = 5 * time.Minute

var receiptMetricPhases = [...]string{"queue", "prefill", "decode", "kernel"}
var receiptMetricByteKinds = [...]string{"kv", "transfer"}
var receiptMetricSignals = [...]string{"queue", "prefill", "decode", "kv", "transfer", "kernel"}

type receiptMetricKey struct {
	engine      string
	backend     string
	forwardPath string
}

type receiptMetricTotals struct {
	requests    uint64
	phases      [len(receiptMetricPhases)]float64
	bytes       [len(receiptMetricByteKinds)]uint64
	phaseWall   [len(phaseOrder)]float64
	phaseActive [len(phaseOrder)]float64
	phaseWait   [len(phaseOrder)]float64
}

type receiptMetricLatest struct {
	at          time.Time
	supported   [len(receiptMetricSignals)]bool
	identityKey receiptMetricKey
	model       string
	planner     string
	owner       string
}

// ReceiptMetrics projects authoritative native inference receipts into a bounded
// Prometheus surface. Request, artifact, raw model, and machine identity never
// become labels; runtime identity is reduced to closed buckets.
type ReceiptMetrics struct {
	mu          sync.RWMutex
	staleAfter  time.Duration
	totals      map[receiptMetricKey]receiptMetricTotals
	latest      receiptMetricLatest
	unsupported uint64
}

func NewReceiptMetrics(staleAfter time.Duration) *ReceiptMetrics {
	if staleAfter <= 0 {
		staleAfter = defaultReceiptMetricStaleAfter
	}
	return &ReceiptMetrics{staleAfter: staleAfter, totals: make(map[receiptMetricKey]receiptMetricTotals)}
}

// Observe records one completed fak-native request. Invalid, fallback-active, or
// non-native receipts are counted as unsupported and never enter native totals.
func (m *ReceiptMetrics) Observe(receipt *model.NativeInferenceReceipt, observedAt time.Time) bool {
	if m == nil {
		return false
	}
	if receipt == nil || receipt.FallbackActive || boundedEngine(receipt.Engine) == "other" {
		m.ObserveUnsupported(observedAt)
		return false
	}
	if observedAt.IsZero() {
		observedAt = time.Now()
	}

	key := receiptMetricKey{
		engine:      boundedEngine(receipt.Engine),
		backend:     boundedBackend(receipt.Backend),
		forwardPath: boundedForwardPath(receipt.ForwardPath),
	}
	var phases [len(receiptMetricPhases)]float64
	phases[0] = finiteNonnegative(receipt.TTFTSeconds - receipt.PrefillSeconds)
	phases[1] = finiteNonnegative(receipt.PrefillSeconds)
	phases[2] = finiteNonnegative(receipt.DecodeSeconds)
	var byteTotals [len(receiptMetricByteKinds)]uint64
	var supported [len(receiptMetricSignals)]bool
	supported[0], supported[1], supported[2] = true, true, true

	if forward := receipt.Qwen35MetalForwardSequence; forward != nil && forward.Available {
		phases[3] = finiteNonnegative(forward.GPUMilliseconds / 1000)
		byteTotals[1] += saturatingAdd(forward.HostUploadBytes, forward.HostReadbackBytes)
		supported[5] = forward.TimingAvailable
		supported[4] = true
	}
	if state := receipt.Qwen35MetalStateIdentity; state != nil && state.Available {
		kvBytes := saturatingAdd(state.GDNStateD2HBytes, state.GDNStateH2DBytes)
		byteTotals[0] += kvBytes
		byteTotals[1] += kvBytes
		supported[3], supported[4] = true, true
	}
	if upload := receipt.CUDAImmutableWeightUploads; upload != nil {
		byteTotals[1] += upload.Delta.TransferBytes
		supported[4] = true
	}

	m.mu.Lock()
	t := m.totals[key]
	t.requests++
	for i := range t.phases {
		t.phases[i] += phases[i]
	}
	for i := range t.bytes {
		t.bytes[i] += byteTotals[i]
	}
	m.totals[key] = t
	m.latest = receiptMetricLatest{
		at:          observedAt,
		supported:   supported,
		identityKey: key,
		model:       boundedModel(receipt.Model),
		planner:     boundedPlanner(receipt.Planner),
		owner:       boundedOwner(receipt.Owner),
	}
	m.mu.Unlock()
	return true
}

// ObservePhases records reconciled exclusive phase accounting. Inclusive child
// timings remain available on the receipt but never enter counters, so nested or
// asynchronous work cannot be double-counted.
func (m *ReceiptMetrics) ObservePhases(receipt PhaseReceipt, observedAt time.Time) bool {
	if m == nil {
		return false
	}
	if err := receipt.Validate(time.Nanosecond); err != nil {
		m.ObserveUnsupported(observedAt)
		return false
	}
	if observedAt.IsZero() {
		observedAt = time.Now()
	}
	key := receiptMetricKey{engine: boundedEngine(receipt.Engine), backend: boundedBackend(receipt.Backend), forwardPath: boundedForwardPath(receipt.ForwardPath)}
	m.mu.Lock()
	totals := m.totals[key]
	for _, accounting := range receipt.Phases {
		index := phaseIndex(accounting.Phase)
		if index < 0 {
			continue
		}
		totals.phaseWall[index] += accounting.Exclusive.Wall.Seconds()
		totals.phaseActive[index] += accounting.Exclusive.Active.Seconds()
		totals.phaseWait[index] += accounting.Exclusive.Wait.Seconds()
	}
	m.totals[key] = totals
	m.latest.at = observedAt
	m.mu.Unlock()
	return true
}

func phaseIndex(want Phase) int {
	for index, phase := range phaseOrder {
		if phase == want {
			return index
		}
	}
	return -1
}

func (m *ReceiptMetrics) ObserveUnsupported(observedAt time.Time) {
	if m == nil {
		return
	}
	if observedAt.IsZero() {
		observedAt = time.Now()
	}
	m.mu.Lock()
	m.unsupported++
	m.latest = receiptMetricLatest{at: observedAt}
	m.mu.Unlock()
}

// Reset clears all accumulated receipt metrics. It is intended for the same
// explicit test/operational reset seams as the gateway's existing metrics state.
func (m *ReceiptMetrics) Reset() {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.totals = make(map[receiptMetricKey]receiptMetricTotals)
	m.latest = receiptMetricLatest{}
	m.unsupported = 0
	m.mu.Unlock()
}

func (m *ReceiptMetrics) Prometheus(now time.Time) string {
	if m == nil {
		return ""
	}
	if now.IsZero() {
		now = time.Now()
	}
	m.mu.RLock()
	totals := make(map[receiptMetricKey]receiptMetricTotals, len(m.totals))
	keys := make([]receiptMetricKey, 0, len(m.totals))
	for k, v := range m.totals {
		totals[k] = v
		keys = append(keys, k)
	}
	latest, unsupported, staleAfter := m.latest, m.unsupported, m.staleAfter
	m.mu.RUnlock()
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].engine != keys[j].engine {
			return keys[i].engine < keys[j].engine
		}
		if keys[i].backend != keys[j].backend {
			return keys[i].backend < keys[j].backend
		}
		return keys[i].forwardPath < keys[j].forwardPath
	})

	var b strings.Builder
	writeReceiptHelpType(&b, "fak_native_runtime_info", "Latest supported fak-native runtime identity reduced to bounded labels.", "gauge")
	if !latest.at.IsZero() {
		fmt.Fprintf(&b, "fak_native_runtime_info%s,model=%q,planner=%q,owner=%q} 1\n", strings.TrimSuffix(receiptLabels(latest.identityKey), "}"), latest.model, latest.planner, latest.owner)
	}
	writeReceiptHelpType(&b, "fak_native_receipt_requests_total", "Completed supported fak-native execution receipts.", "counter")
	for _, k := range keys {
		t := totals[k]
		fmt.Fprintf(&b, "fak_native_receipt_requests_total%s %d\n", receiptLabels(k), t.requests)
	}
	writeReceiptHelpType(&b, "fak_native_receipt_unsupported_total", "Native receipt observations excluded from native totals.", "counter")
	fmt.Fprintf(&b, "fak_native_receipt_unsupported_total %d\n", unsupported)
	writeReceiptHelpType(&b, "fak_native_receipt_phase_seconds_total", "Cumulative duration projected from native execution receipts.", "counter")
	for _, k := range keys {
		t := totals[k]
		for i, phase := range receiptMetricPhases {
			fmt.Fprintf(&b, "fak_native_receipt_phase_seconds_total%s,phase=%q} %s\n", strings.TrimSuffix(receiptLabels(k), "}"), phase, formatMetricFloat(t.phases[i]))
		}
	}
	writeReceiptHelpType(&b, "fak_native_phase_seconds_total", "Reconciled exclusive fak-native phase time by bounded phase and wall/active/wait kind.", "counter")
	for _, k := range keys {
		t := totals[k]
		for i, phase := range phaseOrder {
			labels := strings.TrimSuffix(receiptLabels(k), "}")
			for _, item := range []struct {
				kind string
				val  float64
			}{
				{"wall", t.phaseWall[i]},
				{"active", t.phaseActive[i]},
				{"wait", t.phaseWait[i]},
			} {
				fmt.Fprintf(&b, "fak_native_phase_seconds_total%s,phase=%q,kind=%q} %s\n", labels, phase, item.kind, formatMetricFloat(item.val))
			}
		}
	}
	writeReceiptHelpType(&b, "fak_native_receipt_bytes_total", "Cumulative bytes projected from native execution receipts.", "counter")
	for _, k := range keys {
		t := totals[k]
		for i, kind := range receiptMetricByteKinds {
			fmt.Fprintf(&b, "fak_native_receipt_bytes_total%s,kind=%q} %d\n", strings.TrimSuffix(receiptLabels(k), "}"), kind, t.bytes[i])
		}
	}
	writeReceiptHelpType(&b, "fak_native_receipt_signal_supported", "Whether the latest receipt carried authoritative support for a signal.", "gauge")
	for i, signal := range receiptMetricSignals {
		value := 0
		if latest.supported[i] {
			value = 1
		}
		fmt.Fprintf(&b, "fak_native_receipt_signal_supported{signal=%q} %d\n", signal, value)
	}
	age, stale := 0.0, 1
	if !latest.at.IsZero() {
		age = finiteNonnegative(now.Sub(latest.at).Seconds())
		if now.Sub(latest.at) <= staleAfter {
			stale = 0
		}
	}
	writeReceiptHelpType(&b, "fak_native_receipt_latest_age_seconds", "Age of the latest native receipt observation.", "gauge")
	fmt.Fprintf(&b, "fak_native_receipt_latest_age_seconds %s\n", formatMetricFloat(age))
	writeReceiptHelpType(&b, "fak_native_receipt_latest_stale", "Whether native receipt data is absent or older than its freshness bound.", "gauge")
	fmt.Fprintf(&b, "fak_native_receipt_latest_stale %d\n", stale)
	return b.String()
}

func writeReceiptHelpType(b *strings.Builder, name, help, metricType string) {
	fmt.Fprintf(b, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, metricType)
}

func receiptLabels(k receiptMetricKey) string {
	return fmt.Sprintf("{engine=%q,backend=%q,forward_path=%q}", k.engine, k.backend, k.forwardPath)
}

func boundedModel(v string) string {
	v = strings.ToLower(strings.TrimSpace(v))
	switch {
	case strings.Contains(v, "qwen3.8") || strings.Contains(v, "qwen38"):
		return "qwen3.8"
	case strings.Contains(v, "qwen3.5") || strings.Contains(v, "qwen35"):
		return "qwen3.5"
	case strings.Contains(v, "glm-4.7") || strings.Contains(v, "glm47"):
		return "glm-4.7"
	case strings.Contains(v, "synthetic"):
		return "synthetic"
	default:
		return "other"
	}
}

func boundedPlanner(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "inkernel":
		return "inkernel"
	default:
		return "other"
	}
}

func boundedOwner(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "fak":
		return "fak"
	default:
		return "other"
	}
}

func boundedEngine(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "inkernel", "native", "native-sched":
		return "inkernel"
	default:
		return "other"
	}
}
func boundedBackend(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "cuda":
		return "cuda"
	case "metal":
		return "metal"
	case "cpu":
		return "cpu"
	case "synthetic", "mock":
		return "synthetic"
	default:
		return "other"
	}
}
func boundedForwardPath(v string) string {
	v = strings.ToLower(strings.TrimSpace(v))
	switch {
	case strings.Contains(v, "qwen") && strings.Contains(v, "metal"):
		return "qwen_metal"
	case strings.Contains(v, "qwen") && strings.Contains(v, "cuda"):
		return "qwen_cuda"
	case v == "synthetic" || v == "mock":
		return "synthetic"
	case v == "":
		return "unknown"
	default:
		return "other"
	}
}
func finiteNonnegative(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
		return 0
	}
	return v
}
func saturatingAdd(a, b uint64) uint64 {
	if ^uint64(0)-a < b {
		return ^uint64(0)
	}
	return a + b
}
func formatMetricFloat(v float64) string { return fmt.Sprintf("%.9g", finiteNonnegative(v)) }
