package gateway

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

// ServingMetricsEmitter returns rows in the normalized fak_serving_* schema.
// Implementations are deliberately pull-only from the renderer's point of view:
// scrape emitters can refresh from a vLLM/SGLang endpoint on their own cadence, while
// native emitters can be fed by the step loop as it runs.
type ServingMetricsEmitter interface {
	SnapshotServingMetrics() []ServingMetricRow
}

// ServingMetricLabels identify the worker that authored a serving signal. The
// worker and engine labels are required by issue #43; model is kept beside them
// because vLLM/SGLang expose per-model rows and dashboards commonly aggregate on it.
type ServingMetricLabels struct {
	Worker string
	Engine string
	Model  string
}

// ServingGauge is a Prometheus gauge value plus presence bit. A zero value may be
// a real sample, so Set distinguishes "0 was observed" from "not provided".
type ServingGauge struct {
	Value float64
	Set   bool
}

// ServingGaugeValue builds a present gauge.
func ServingGaugeValue(v float64) ServingGauge {
	return ServingGauge{Value: v, Set: true}
}

// ServingBucket is one histogram bucket. LE is the Prometheus "le" label value.
type ServingBucket struct {
	LE    string
	Value float64
}

// ServingHistogram carries a Prometheus histogram in normalized form. Bucket values
// are cumulative, matching Prometheus text exposition.
type ServingHistogram struct {
	Buckets []ServingBucket
	Sum     ServingGauge
	Count   ServingGauge
}

// Present reports whether the histogram has any samples to render.
func (h ServingHistogram) Present() bool {
	return len(h.Buckets) > 0 || h.Sum.Set || h.Count.Set
}

// ServingMetricRow is one worker's normalized serving telemetry row.
type ServingMetricRow struct {
	Labels             ServingMetricLabels
	TTFT               ServingHistogram
	TPOT               ServingHistogram
	ITL                ServingHistogram
	Goodput            ServingGauge
	Running            ServingGauge
	Waiting            ServingGauge
	KVUtilization      ServingGauge
	PrefixCacheHitRate ServingGauge
}

func (r ServingMetricRow) present() bool {
	return r.TTFT.Present() || r.TPOT.Present() || r.ITL.Present() ||
		r.Goodput.Set || r.Running.Set || r.Waiting.Set ||
		r.KVUtilization.Set || r.PrefixCacheHitRate.Set
}

// SetServingMetricsEmitters replaces the extra serving-metric emitters rendered
// onto /metrics. Passing none detaches every scrape/native external emitter.
func (s *Server) SetServingMetricsEmitters(emitters ...ServingMetricsEmitter) {
	if s == nil || s.metrics == nil {
		return
	}
	s.metrics.setServingMetricsEmitters(emitters...)
}

func (m *gatewayMetrics) setServingMetricsEmitters(emitters ...ServingMetricsEmitter) {
	if m == nil {
		return
	}
	copied := append([]ServingMetricsEmitter(nil), emitters...)
	m.servingMu.Lock()
	m.servingEmitters = copied
	m.servingMu.Unlock()
}

func (m *gatewayMetrics) servingEmitterRows() []ServingMetricRow {
	if m == nil {
		return nil
	}
	m.servingMu.Lock()
	emitters := append([]ServingMetricsEmitter(nil), m.servingEmitters...)
	m.servingMu.Unlock()
	var rows []ServingMetricRow
	for _, emitter := range emitters {
		if emitter == nil {
			continue
		}
		for _, row := range emitter.SnapshotServingMetrics() {
			if row.present() {
				rows = append(rows, row)
			}
		}
	}
	return rows
}

func (s *Server) writeServingMetrics(b *strings.Builder, inf inferenceSnapshot) {
	if s == nil || b == nil || s.metrics == nil {
		return
	}
	var rows []ServingMetricRow
	if row, ok := s.nativeServingMetricRow(inf); ok {
		rows = append(rows, row)
	}
	rows = append(rows, s.metrics.servingEmitterRows()...)
	rows = mergeServingMetricRows(rows)
	if len(rows) == 0 {
		return
	}

	writeServingHistogramFamily(b, "fak_serving_time_to_first_token_seconds",
		"Time to first token, normalized onto the fak serving schema. Name aligns with vLLM time_to_first_token_seconds; labels identify worker, engine, and model.",
		rows, func(r ServingMetricRow) ServingHistogram { return r.TTFT })
	writeServingHistogramFamily(b, "fak_serving_time_per_output_token_seconds",
		"Time per output token, normalized onto the fak serving schema. Name aligns with vLLM time_per_output_token_seconds; labels identify worker, engine, and model.",
		rows, func(r ServingMetricRow) ServingHistogram { return r.TPOT })
	writeServingHistogramFamily(b, "fak_serving_inter_token_latency_seconds",
		"Inter-token latency, normalized onto the fak serving schema for engines that publish ITL separately from TPOT.",
		rows, func(r ServingMetricRow) ServingHistogram { return r.ITL })
	writeServingGaugeFamily(b, "fak_serving_goodput_requests_per_second",
		"Serving goodput in successful requests per second for the worker. Scrape emitters may derive this from upstream success-counter deltas; native emitters feed the same schema directly.",
		rows, func(r ServingMetricRow) ServingGauge { return r.Goodput })
	writeServingGaugeFamily(b, "fak_serving_num_requests_running",
		"Requests currently running on the serving worker. Name aligns with vLLM num_requests_running.",
		rows, func(r ServingMetricRow) ServingGauge { return r.Running })
	writeServingGaugeFamily(b, "fak_serving_num_requests_waiting",
		"Requests waiting in the serving worker queue. Name aligns with vLLM num_requests_waiting.",
		rows, func(r ServingMetricRow) ServingGauge { return r.Waiting })
	writeServingGaugeFamily(b, "fak_serving_gpu_cache_usage_perc",
		"KV/GPU cache utilization for the serving worker. Name aligns with vLLM gpu_cache_usage_perc.",
		rows, func(r ServingMetricRow) ServingGauge { return r.KVUtilization })
	writeServingGaugeFamily(b, "fak_serving_gpu_prefix_cache_hit_rate",
		"Prefix-cache hit rate for the serving worker. Name aligns with vLLM gpu_prefix_cache_hit_rate.",
		rows, func(r ServingMetricRow) ServingGauge { return r.PrefixCacheHitRate })
}

func (s *Server) nativeServingMetricRow(inf inferenceSnapshot) (ServingMetricRow, bool) {
	row := ServingMetricRow{
		Labels: ServingMetricLabels{
			Worker: "local",
			Engine: s.engineID,
			Model:  s.model,
		},
	}
	var ok bool
	if inf.ttftHist.count > 0 {
		row.TTFT = servingHistogramFromLatencySnapshot(inf.ttftHist)
		ok = true
	}
	if inf.tpotHist.count > 0 {
		row.TPOT = servingHistogramFromLatencySnapshot(inf.tpotHist)
		row.ITL = row.TPOT
		ok = true
	}
	turns := inferenceTurnCount(inf)
	if inf.decodeSecs > 0 && turns > 0 {
		row.Goodput = ServingGaugeValue(float64(turns) / inf.decodeSecs)
		ok = true
	}

	s.admissionMu.RLock()
	admission := s.admissionCtl
	s.admissionMu.RUnlock()
	if admission != nil {
		st := admission.Stats()
		row.Running = ServingGaugeValue(float64(st.Running))
		row.Waiting = ServingGaugeValue(float64(st.Waiting))
		ok = true
	}

	if reporter, okReporter := s.planner.(agent.KVMemoryReporter); okReporter {
		if util, okUtil := nativeKVUtilization(reporter.KVMemoryStats()); okUtil {
			row.KVUtilization = ServingGaugeValue(util)
			ok = true
		}
	}
	if ok {
		if kv := cacheobs.Default.Snapshot(); kv.Turns > 0 {
			row.PrefixCacheHitRate = ServingGaugeValue(kv.ReuseRatio)
		} else if turns > 0 {
			row.PrefixCacheHitRate = ServingGaugeValue(float64(inf.cachedHits) / float64(turns))
		}
	}
	return row, ok && row.present()
}

func nativeKVUtilization(st agent.KVMemoryStats) (float64, bool) {
	if !st.CapacityKnown {
		return 0, false
	}
	denom := st.FitBudgetBytes
	if denom <= 0 {
		denom = st.CapacityTotalBytes
	}
	if denom <= 0 {
		return 0, false
	}
	util := float64(st.ResidentBytes) / float64(denom)
	if util < 0 {
		util = 0
	}
	return util, true
}

func inferenceTurnCount(inf inferenceSnapshot) uint64 {
	var turns uint64
	for _, n := range inf.reqs {
		turns += n
	}
	return turns
}

func mergeServingMetricRows(rows []ServingMetricRow) []ServingMetricRow {
	type keyed struct {
		key string
		row ServingMetricRow
	}
	merged := make(map[string]*ServingMetricRow)
	order := make([]keyed, 0, len(rows))
	for _, row := range rows {
		row.Labels = normalizeServingLabels(row.Labels)
		if !row.present() {
			continue
		}
		key := servingLabelKey(row.Labels)
		dst := merged[key]
		if dst == nil {
			cp := row
			merged[key] = &cp
			order = append(order, keyed{key: key, row: cp})
			continue
		}
		mergeServingMetricRow(dst, row)
	}
	out := make([]ServingMetricRow, 0, len(order))
	for _, item := range order {
		out = append(out, *merged[item.key])
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := servingLabelKey(out[i].Labels), servingLabelKey(out[j].Labels)
		return a < b
	})
	return out
}

func mergeServingMetricRow(dst *ServingMetricRow, src ServingMetricRow) {
	if src.TTFT.Present() {
		dst.TTFT = src.TTFT
	}
	if src.TPOT.Present() {
		dst.TPOT = src.TPOT
	}
	if src.ITL.Present() {
		dst.ITL = src.ITL
	}
	if src.Goodput.Set {
		dst.Goodput = src.Goodput
	}
	if src.Running.Set {
		dst.Running = src.Running
	}
	if src.Waiting.Set {
		dst.Waiting = src.Waiting
	}
	if src.KVUtilization.Set {
		dst.KVUtilization = src.KVUtilization
	}
	if src.PrefixCacheHitRate.Set {
		dst.PrefixCacheHitRate = src.PrefixCacheHitRate
	}
}

func writeServingHistogramFamily(b *strings.Builder, name, help string, rows []ServingMetricRow, pick func(ServingMetricRow) ServingHistogram) {
	writeHelpType(b, name, help, "histogram")
	for _, row := range rows {
		h := pick(row)
		if !h.Present() {
			continue
		}
		labels := servingLabels(row.Labels)
		for _, bucket := range sortedServingBuckets(h.Buckets) {
			fmt.Fprintf(b, "%s_bucket{%s,le=\"%s\"} %s\n", name, labels, promQuote(bucket.LE), promFloat(bucket.Value))
		}
		if h.Sum.Set {
			fmt.Fprintf(b, "%s_sum{%s} %s\n", name, labels, promFloat(h.Sum.Value))
		}
		if h.Count.Set {
			fmt.Fprintf(b, "%s_count{%s} %s\n", name, labels, promFloat(h.Count.Value))
		}
	}
}

func writeServingGaugeFamily(b *strings.Builder, name, help string, rows []ServingMetricRow, pick func(ServingMetricRow) ServingGauge) {
	writeHelpType(b, name, help, "gauge")
	for _, row := range rows {
		g := pick(row)
		if !g.Set {
			continue
		}
		fmt.Fprintf(b, "%s{%s} %s\n", name, servingLabels(row.Labels), promFloat(g.Value))
	}
}

func servingHistogramFromLatencySnapshot(s latencySnapshot) ServingHistogram {
	var h ServingHistogram
	for i, le := range gatewayLatencyBuckets {
		var v uint64
		if i < len(s.buckets) {
			v = s.buckets[i]
		}
		h.Buckets = append(h.Buckets, ServingBucket{LE: promFloat(le), Value: float64(v)})
	}
	h.Buckets = append(h.Buckets, ServingBucket{LE: "+Inf", Value: float64(s.count)})
	h.Sum = ServingGaugeValue(s.sum)
	h.Count = ServingGaugeValue(float64(s.count))
	return h
}

func (h *ServingHistogram) observeSeconds(seconds float64) {
	if seconds < 0 {
		return
	}
	for _, le := range gatewayLatencyBuckets {
		v := 0.0
		if seconds <= le {
			v = 1
		}
		h.addBucket(promFloat(le), v)
	}
	h.addBucket("+Inf", 1)
	h.addSum(seconds)
	h.addCount(1)
}

func (h *ServingHistogram) addBucket(le string, v float64) {
	for i := range h.Buckets {
		if h.Buckets[i].LE == le {
			h.Buckets[i].Value += v
			return
		}
	}
	h.Buckets = append(h.Buckets, ServingBucket{LE: le, Value: v})
}

func (h *ServingHistogram) addSum(v float64) {
	h.Sum = ServingGaugeValue(h.Sum.Value + v)
}

func (h *ServingHistogram) addCount(v float64) {
	h.Count = ServingGaugeValue(h.Count.Value + v)
}

func sortedServingBuckets(in []ServingBucket) []ServingBucket {
	out := append([]ServingBucket(nil), in...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].LE == "+Inf" {
			return false
		}
		if out[j].LE == "+Inf" {
			return true
		}
		fi, errI := strconv.ParseFloat(out[i].LE, 64)
		fj, errJ := strconv.ParseFloat(out[j].LE, 64)
		if errI == nil && errJ == nil && fi != fj {
			return fi < fj
		}
		return out[i].LE < out[j].LE
	})
	return out
}

func servingLabels(labels ServingMetricLabels) string {
	labels = normalizeServingLabels(labels)
	return fmt.Sprintf("worker=\"%s\",engine=\"%s\",model=\"%s\"",
		promQuote(labels.Worker), promQuote(labels.Engine), promQuote(labels.Model))
}

func normalizeServingLabels(labels ServingMetricLabels) ServingMetricLabels {
	labels.Worker = servingLabelOr(labels.Worker, "unknown")
	labels.Engine = servingLabelOr(labels.Engine, "unknown")
	labels.Model = servingLabelOr(labels.Model, "unknown")
	return labels
}

func servingLabelOr(v, fallback string) string {
	if v = strings.TrimSpace(v); v != "" {
		return v
	}
	return fallback
}

func servingLabelKey(labels ServingMetricLabels) string {
	labels = normalizeServingLabels(labels)
	return labels.Worker + "\x00" + labels.Engine + "\x00" + labels.Model
}

// ServingScrapeEmitter relabels a ridden vLLM/SGLang Prometheus scrape into the
// fak_serving_* schema. It does not fork engine internals: callers point it at the
// worker's existing /metrics endpoint, then attach the emitter to the gateway.
type ServingScrapeEmitter struct {
	labels ServingMetricLabels

	mu              sync.Mutex
	row             ServingMetricRow
	previousSuccess float64
	previousAt      time.Time
	havePrevious    bool
}

var defaultServingMetricsHTTPClient = &http.Client{Timeout: 10 * time.Second}

func NewServingScrapeEmitter(labels ServingMetricLabels) *ServingScrapeEmitter {
	return &ServingScrapeEmitter{labels: labels}
}

func (e *ServingScrapeEmitter) Scrape(ctx context.Context, endpoint string) error {
	return e.ScrapeWithClient(ctx, defaultServingMetricsHTTPClient, endpoint)
}

func (e *ServingScrapeEmitter) ScrapeWithClient(ctx context.Context, client *http.Client, endpoint string) error {
	if client == nil {
		client = defaultServingMetricsHTTPClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("gateway: serving metrics scrape %s returned HTTP %d", endpoint, resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	return e.IngestPrometheusAt(string(body), time.Now())
}

func (e *ServingScrapeEmitter) IngestPrometheus(text string) error {
	return e.IngestPrometheusAt(text, time.Now())
}

func (e *ServingScrapeEmitter) IngestPrometheusAt(text string, at time.Time) error {
	if e == nil {
		return nil
	}
	row := ServingMetricRow{Labels: e.labels}
	var successTotal float64
	var haveSuccess bool
	sc := bufio.NewScanner(strings.NewReader(text))
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		sample, ok := parsePromSample(sc.Text())
		if !ok {
			continue
		}
		if row.Labels.Model == "" {
			row.Labels.Model = firstNonEmpty(sample.labels["model_name"], sample.labels["model"])
		}
		applyServingPromSample(&row, sample, &successTotal, &haveSuccess)
	}
	if err := sc.Err(); err != nil {
		return err
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if haveSuccess && e.havePrevious && at.After(e.previousAt) && successTotal >= e.previousSuccess {
		row.Goodput = ServingGaugeValue((successTotal - e.previousSuccess) / at.Sub(e.previousAt).Seconds())
	}
	if haveSuccess {
		e.previousSuccess = successTotal
		e.previousAt = at
		e.havePrevious = true
	}
	e.row = row
	return nil
}

func (e *ServingScrapeEmitter) SnapshotServingMetrics() []ServingMetricRow {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.row.present() {
		return nil
	}
	return []ServingMetricRow{e.row}
}

// NativeServingMetricsEmitter is the native step-loop seam for the same schema.
// The continuous-batching loop can feed these hooks directly when it owns exact
// TTFT/TPOT/queue/KV-util observations.
type NativeServingMetricsEmitter struct {
	mu  sync.Mutex
	row ServingMetricRow
}

func NewNativeServingMetricsEmitter(labels ServingMetricLabels) *NativeServingMetricsEmitter {
	return &NativeServingMetricsEmitter{row: ServingMetricRow{Labels: labels}}
}

func (e *NativeServingMetricsEmitter) ObserveTTFT(d time.Duration) {
	if e == nil || d < 0 {
		return
	}
	e.mu.Lock()
	e.row.TTFT.observeSeconds(d.Seconds())
	e.mu.Unlock()
}

func (e *NativeServingMetricsEmitter) ObserveTPOT(d time.Duration) {
	if e == nil || d < 0 {
		return
	}
	e.mu.Lock()
	e.row.TPOT.observeSeconds(d.Seconds())
	e.row.ITL.observeSeconds(d.Seconds())
	e.mu.Unlock()
}

func (e *NativeServingMetricsEmitter) ObserveITL(d time.Duration) {
	if e == nil || d < 0 {
		return
	}
	e.mu.Lock()
	e.row.ITL.observeSeconds(d.Seconds())
	e.mu.Unlock()
}

func (e *NativeServingMetricsEmitter) SetGoodputRequestsPerSecond(v float64) {
	if e == nil {
		return
	}
	e.mu.Lock()
	e.row.Goodput = ServingGaugeValue(v)
	e.mu.Unlock()
}

func (e *NativeServingMetricsEmitter) SetQueue(running, waiting int) {
	if e == nil {
		return
	}
	e.mu.Lock()
	e.row.Running = ServingGaugeValue(float64(running))
	e.row.Waiting = ServingGaugeValue(float64(waiting))
	e.mu.Unlock()
}

func (e *NativeServingMetricsEmitter) SetKVUtilization(v float64) {
	if e == nil {
		return
	}
	e.mu.Lock()
	e.row.KVUtilization = ServingGaugeValue(v)
	e.mu.Unlock()
}

func (e *NativeServingMetricsEmitter) SetPrefixCacheHitRate(v float64) {
	if e == nil {
		return
	}
	e.mu.Lock()
	e.row.PrefixCacheHitRate = ServingGaugeValue(v)
	e.mu.Unlock()
}

func (e *NativeServingMetricsEmitter) SnapshotServingMetrics() []ServingMetricRow {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.row.present() {
		return nil
	}
	return []ServingMetricRow{e.row}
}

type promSample struct {
	name   string
	labels map[string]string
	value  float64
}

func parsePromSample(line string) (promSample, bool) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return promSample{}, false
	}
	nameEnd := strings.IndexAny(line, "{ \t")
	if nameEnd < 0 {
		return promSample{}, false
	}
	s := promSample{name: line[:nameEnd], labels: map[string]string{}}
	rest := strings.TrimSpace(line[nameEnd:])
	if strings.HasPrefix(rest, "{") {
		close := findPromLabelClose(rest)
		if close < 0 {
			return promSample{}, false
		}
		labels, ok := parsePromLabels(rest[1:close])
		if !ok {
			return promSample{}, false
		}
		s.labels = labels
		rest = strings.TrimSpace(rest[close+1:])
	}
	fields := strings.Fields(rest)
	if len(fields) == 0 {
		return promSample{}, false
	}
	v, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return promSample{}, false
	}
	s.value = v
	return s, true
}

func findPromLabelClose(s string) int {
	escaped := false
	quoted := false
	for i, r := range s {
		if i == 0 {
			continue
		}
		switch {
		case escaped:
			escaped = false
		case r == '\\':
			escaped = true
		case r == '"':
			quoted = !quoted
		case r == '}' && !quoted:
			return i
		}
	}
	return -1
}

func parsePromLabels(s string) (map[string]string, bool) {
	labels := map[string]string{}
	for strings.TrimSpace(s) != "" {
		s = strings.TrimLeft(s, " \t,")
		eq := strings.IndexByte(s, '=')
		if eq <= 0 {
			return nil, false
		}
		key := strings.TrimSpace(s[:eq])
		s = strings.TrimLeft(s[eq+1:], " \t")
		if !strings.HasPrefix(s, `"`) {
			return nil, false
		}
		value, n, ok := parsePromQuoted(s)
		if !ok {
			return nil, false
		}
		labels[key] = value
		s = s[n:]
	}
	return labels, true
}

func parsePromQuoted(s string) (string, int, bool) {
	var b strings.Builder
	escaped := false
	for i, r := range s {
		if i == 0 {
			continue
		}
		if escaped {
			switch r {
			case 'n':
				b.WriteByte('\n')
			case '\\', '"':
				b.WriteRune(r)
			default:
				b.WriteRune(r)
			}
			escaped = false
			continue
		}
		if r == '\\' {
			escaped = true
			continue
		}
		if r == '"' {
			return b.String(), i + 1, true
		}
		b.WriteRune(r)
	}
	return "", 0, false
}

func applyServingPromSample(row *ServingMetricRow, sample promSample, successTotal *float64, haveSuccess *bool) {
	base, suffix := promMetricBase(sample.name)
	switch servingPromMetricKind(base) {
	case "ttft":
		applyPromHistogramSample(&row.TTFT, suffix, sample)
	case "tpot":
		applyPromHistogramSample(&row.TPOT, suffix, sample)
	case "itl":
		applyPromHistogramSample(&row.ITL, suffix, sample)
	case "running":
		row.Running = ServingGaugeValue(sample.value)
	case "waiting":
		row.Waiting = ServingGaugeValue(sample.value)
	case "kv_util":
		row.KVUtilization = ServingGaugeValue(sample.value)
	case "prefix_hit":
		row.PrefixCacheHitRate = ServingGaugeValue(sample.value)
	case "goodput":
		row.Goodput = ServingGaugeValue(sample.value)
	case "success_total":
		*successTotal += sample.value
		*haveSuccess = true
	}
}

func applyPromHistogramSample(h *ServingHistogram, suffix string, sample promSample) {
	switch suffix {
	case "_bucket":
		le := sample.labels["le"]
		if le != "" {
			h.addBucket(le, sample.value)
		}
	case "_sum":
		h.addSum(sample.value)
	case "_count":
		h.addCount(sample.value)
	}
}

func promMetricBase(name string) (base, suffix string) {
	for _, suffix := range []string{"_bucket", "_sum", "_count"} {
		if strings.HasSuffix(name, suffix) {
			return strings.TrimSuffix(name, suffix), suffix
		}
	}
	return name, ""
}

func servingPromMetricKind(base string) string {
	switch {
	case strings.HasSuffix(base, "time_to_first_token_seconds"):
		return "ttft"
	case strings.HasSuffix(base, "time_per_output_token_seconds"):
		return "tpot"
	case strings.HasSuffix(base, "inter_token_latency_seconds") || strings.HasSuffix(base, "itl_seconds"):
		return "itl"
	case strings.HasSuffix(base, "num_requests_running") ||
		strings.HasSuffix(base, "num_running_reqs") ||
		strings.HasSuffix(base, "num_running_requests"):
		return "running"
	case strings.HasSuffix(base, "num_requests_waiting") ||
		strings.HasSuffix(base, "num_queue_reqs") ||
		strings.HasSuffix(base, "num_waiting_reqs") ||
		strings.HasSuffix(base, "num_queued_requests"):
		return "waiting"
	case strings.HasSuffix(base, "gpu_cache_usage_perc") ||
		strings.HasSuffix(base, "kv_cache_usage_perc") ||
		strings.HasSuffix(base, "kv_cache_utilization"):
		return "kv_util"
	case strings.HasSuffix(base, "gpu_prefix_cache_hit_rate") ||
		strings.HasSuffix(base, "prefix_cache_hit_rate"):
		return "prefix_hit"
	case strings.HasSuffix(base, "goodput_requests_per_second") ||
		strings.HasSuffix(base, "request_goodput_requests_per_second"):
		return "goodput"
	case strings.HasSuffix(base, "request_success_total") ||
		strings.HasSuffix(base, "requests_success_total"):
		return "success_total"
	}
	return ""
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
