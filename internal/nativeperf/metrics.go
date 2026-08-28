package nativeperf

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
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
	requests uint64
	phases   [len(receiptMetricPhases)]float64
	bytes    [len(receiptMetricByteKinds)]uint64
}

type receiptMetricLatest struct {
	at        time.Time
	supported [len(receiptMetricSignals)]bool
}

// ReceiptMetrics projects authoritative native inference receipts into a bounded
// Prometheus surface. It deliberately excludes model, request, artifact, and
// machine identity from labels; unrecognized execution identities collapse into
// closed "other" buckets.
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
func (m *ReceiptMetrics) Observe(receipt *agent.NativeInferenceReceipt, observedAt time.Time) bool {
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
	m.latest = receiptMetricLatest{at: observedAt, supported: supported}
	m.mu.Unlock()
	return true
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
	b.WriteString("# HELP fak_native_receipt_requests_total Completed supported fak-native execution receipts.\n")
	b.WriteString("# TYPE fak_native_receipt_requests_total counter\n")
	for _, k := range keys {
		t := totals[k]
		fmt.Fprintf(&b, "fak_native_receipt_requests_total%s %d\n", receiptLabels(k), t.requests)
	}
	b.WriteString("# HELP fak_native_receipt_unsupported_total Native receipt observations excluded from native totals.\n")
	b.WriteString("# TYPE fak_native_receipt_unsupported_total counter\n")
	fmt.Fprintf(&b, "fak_native_receipt_unsupported_total %d\n", unsupported)
	b.WriteString("# HELP fak_native_receipt_phase_seconds_total Cumulative duration projected from native execution receipts.\n")
	b.WriteString("# TYPE fak_native_receipt_phase_seconds_total counter\n")
	for _, k := range keys {
		t := totals[k]
		for i, phase := range receiptMetricPhases {
			fmt.Fprintf(&b, "fak_native_receipt_phase_seconds_total%s,phase=%q} %s\n", strings.TrimSuffix(receiptLabels(k), "}"), phase, formatMetricFloat(t.phases[i]))
		}
	}
	b.WriteString("# HELP fak_native_receipt_bytes_total Cumulative bytes projected from native execution receipts.\n")
	b.WriteString("# TYPE fak_native_receipt_bytes_total counter\n")
	for _, k := range keys {
		t := totals[k]
		for i, kind := range receiptMetricByteKinds {
			fmt.Fprintf(&b, "fak_native_receipt_bytes_total%s,kind=%q} %d\n", strings.TrimSuffix(receiptLabels(k), "}"), kind, t.bytes[i])
		}
	}
	b.WriteString("# HELP fak_native_receipt_signal_supported Whether the latest receipt carried authoritative support for a signal.\n")
	b.WriteString("# TYPE fak_native_receipt_signal_supported gauge\n")
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
	b.WriteString("# HELP fak_native_receipt_latest_age_seconds Age of the latest native receipt observation.\n")
	b.WriteString("# TYPE fak_native_receipt_latest_age_seconds gauge\n")
	fmt.Fprintf(&b, "fak_native_receipt_latest_age_seconds %s\n", formatMetricFloat(age))
	b.WriteString("# HELP fak_native_receipt_latest_stale Whether native receipt data is absent or older than its freshness bound.\n")
	b.WriteString("# TYPE fak_native_receipt_latest_stale gauge\n")
	fmt.Fprintf(&b, "fak_native_receipt_latest_stale %d\n", stale)
	return b.String()
}

func receiptLabels(k receiptMetricKey) string {
	return fmt.Sprintf("{engine=%q,backend=%q,forward_path=%q}", k.engine, k.backend, k.forwardPath)
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
