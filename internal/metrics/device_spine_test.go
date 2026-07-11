package metrics

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
)

// fakeDevice is a raw sample source backed by a map: a key present in the map
// is supported, an absent key is unsupported (Metric returns ok=false). This
// stands in for a real backend so the spine is testable without hardware.
type fakeDevice struct {
	id string
	m  map[string]float64
}

func (f fakeDevice) ID() string { return f.id }
func (f fakeDevice) Metric(key string) (float64, bool) {
	v, ok := f.m[key]
	return v, ok
}

// fakeProbe is a plug-in backend whose Detect/error/panic behaviour is
// configurable so the registry's skip-not-fatal contract can be exercised.
type fakeProbe struct {
	backend  string
	detect   bool
	err      error
	panicNow bool
	devices  []Device
}

func (p fakeProbe) Backend() string { return p.backend }
func (p fakeProbe) Detect() bool    { return p.detect }
func (p fakeProbe) Devices() ([]Device, error) {
	if p.panicNow {
		panic("probe boom")
	}
	if p.err != nil {
		return nil, p.err
	}
	return p.devices, nil
}

func metricKeySet() map[string]bool {
	keys := make(map[string]bool, len(metricTable))
	for _, m := range metricTable {
		keys[m.Key] = true
	}
	return keys
}

// TestCollectNullOnError proves the declarative table fills supported metrics
// and leaves unsupported ones nil — never erroring on a partial device.
func TestCollectNullOnError(t *testing.T) {
	dev := fakeDevice{id: "gpu0", m: map[string]float64{
		"tokens_per_second": 42,
		"memory_used_bytes": 1 << 30,
	}}
	probe := fakeProbe{backend: "nvml", detect: true, devices: []Device{dev}}

	snap := Collect([]Probe{probe})
	if len(snap) != 1 {
		t.Fatalf("want 1 device, got %d", len(snap))
	}
	d := snap[0]
	if d.Backend != "nvml" || d.DeviceID != "gpu0" {
		t.Fatalf("identity not filled: %+v", d)
	}
	if d.TokensPerSecond == nil || *d.TokensPerSecond != 42 {
		t.Fatalf("tokens_per_second not filled: %+v", d.TokensPerSecond)
	}
	if d.MemoryUsedBytes == nil || *d.MemoryUsedBytes != float64(1<<30) {
		t.Fatalf("memory_used_bytes not filled: %+v", d.MemoryUsedBytes)
	}
	// Unsupported metrics stay nil (null-on-error), not zero.
	if d.PowerWatts != nil || d.QueueDepth != nil || d.TTFTSeconds != nil {
		t.Fatalf("unsupported metrics should be nil, got %+v", d)
	}
}

// TestCollectSkipsFailingProbes proves one failing backend (Detect false /
// error / panic) is skipped and never aborts the healthy probes.
func TestCollectSkipsFailingProbes(t *testing.T) {
	healthy := fakeProbe{backend: "nvml", detect: true, devices: []Device{
		fakeDevice{id: "gpu0", m: map[string]float64{"power_watts": 100}},
	}}
	undetected := fakeProbe{backend: "amdsmi", detect: false, devices: []Device{
		fakeDevice{id: "gpuX", m: map[string]float64{"power_watts": 1}},
	}}
	erroring := fakeProbe{backend: "oneapi", detect: true, err: errors.New("driver missing")}
	panicking := fakeProbe{backend: "neuron", detect: true, panicNow: true}

	snap := Collect([]Probe{undetected, erroring, panicking, healthy})
	if len(snap) != 1 {
		t.Fatalf("want only the healthy device, got %d: %+v", len(snap), snap)
	}
	if snap[0].Backend != "nvml" || snap[0].PowerWatts == nil || *snap[0].PowerWatts != 100 {
		t.Fatalf("healthy device not collected: %+v", snap[0])
	}
}

// TestSeedFromFrontHoldsLastGood proves the double-buffer seed-back-from-front
// keeps last-good values when a metric goes unread, and carries a vanished
// device forward whole.
func TestSeedFromFrontHoldsLastGood(t *testing.T) {
	buf := NewBuffer()

	full := fakeProbe{backend: "nvml", detect: true, devices: []Device{
		fakeDevice{id: "gpu0", m: map[string]float64{"power_watts": 100, "utilization_ratio": 0.5}},
		fakeDevice{id: "gpu1", m: map[string]float64{"power_watts": 80}},
	}}
	first := buf.Poll([]Probe{full})
	if len(first) != 2 {
		t.Fatalf("first poll want 2 devices, got %d", len(first))
	}

	// Second poll: gpu0 no longer reports utilization; gpu1 vanishes entirely.
	partial := fakeProbe{backend: "nvml", detect: true, devices: []Device{
		fakeDevice{id: "gpu0", m: map[string]float64{"power_watts": 110}},
	}}
	second := buf.Poll([]Probe{partial})

	byID := map[string]DeviceMetrics{}
	for _, d := range second {
		byID[d.DeviceID] = d
	}
	gpu0, ok := byID["gpu0"]
	if !ok {
		t.Fatalf("gpu0 missing from second poll: %+v", second)
	}
	if gpu0.PowerWatts == nil || *gpu0.PowerWatts != 110 {
		t.Fatalf("gpu0 power should refresh to 110, got %+v", gpu0.PowerWatts)
	}
	if gpu0.UtilizationRatio == nil || *gpu0.UtilizationRatio != 0.5 {
		t.Fatalf("gpu0 utilization should hold last-good 0.5, got %+v", gpu0.UtilizationRatio)
	}
	gpu1, ok := byID["gpu1"]
	if !ok {
		t.Fatalf("gpu1 should be carried forward across a failed poll: %+v", second)
	}
	if gpu1.PowerWatts == nil || *gpu1.PowerWatts != 80 {
		t.Fatalf("gpu1 should hold last-good power 80, got %+v", gpu1.PowerWatts)
	}
	// Snapshot() returns the published buffer.
	if !reflect.DeepEqual(buf.Snapshot(), second) {
		t.Fatalf("Snapshot should equal last published poll")
	}
}

// TestBufferConcurrentReadersNoTearing runs one writer and many readers; every
// observed snapshot must be internally consistent (identity always present),
// which holds because published snapshots are immutable. Run under -race for
// the full guarantee.
func TestBufferConcurrentReadersNoTearing(t *testing.T) {
	buf := NewBuffer()
	probe := fakeProbe{backend: "nvml", detect: true, devices: []Device{
		fakeDevice{id: "gpu0", m: map[string]float64{"power_watts": 100}},
	}}
	buf.Poll([]Probe{probe})

	var wg sync.WaitGroup
	stop := make(chan struct{})
	for r := 0; r < 8; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					for _, d := range buf.Snapshot() {
						if d.Backend == "" || d.DeviceID == "" {
							t.Errorf("torn snapshot: %+v", d)
							return
						}
					}
				}
			}
		}()
	}
	for i := 0; i < 500; i++ {
		buf.Poll([]Probe{probe})
	}
	close(stop)
	wg.Wait()
}

// TestJSONPromSameCollectionPath is the load-bearing contract: JSON and
// Prometheus are stateless readers over ONE snapshot, so their present-set of
// metrics is identical — proving `fak metrics --json` and `/metrics` reuse the
// same collection path.
func TestJSONPromSameCollectionPath(t *testing.T) {
	probe := fakeProbe{backend: "nvml", detect: true, devices: []Device{
		fakeDevice{id: "gpu0", m: map[string]float64{
			"tokens_per_second": 42,
			"utilization_ratio": 0.9,
		}},
		fakeDevice{id: "gpu1", m: map[string]float64{
			"power_watts": 75,
		}},
	}}
	snap := Collect([]Probe{probe})

	js, err := RenderJSON(snap)
	if err != nil {
		t.Fatalf("RenderJSON: %v", err)
	}
	pm, err := RenderProm(snap)
	if err != nil {
		t.Fatalf("RenderProm: %v", err)
	}

	keys := metricKeySet()

	// JSON present-set: union of metric keys across all device rows.
	var rows []map[string]json.RawMessage
	if err := json.Unmarshal(js, &rows); err != nil {
		t.Fatalf("unmarshal json: %v", err)
	}
	jsonSet := map[string]bool{}
	for _, row := range rows {
		for k := range row {
			if keys[k] {
				jsonSet[k] = true
			}
		}
	}

	// Prom present-set: every fak_device_<metric> family, excluding the info metric.
	promSet := map[string]bool{}
	for _, line := range strings.Split(string(pm), "\n") {
		if !strings.HasPrefix(line, "# TYPE fak_device_") {
			continue
		}
		name := strings.Fields(strings.TrimPrefix(line, "# TYPE "))[0]
		suffix := strings.TrimPrefix(name, "fak_device_")
		if suffix == "info" {
			continue
		}
		promSet[suffix] = true
	}

	if !reflect.DeepEqual(jsonSet, promSet) {
		t.Fatalf("present-set mismatch: json=%v prom=%v", jsonSet, promSet)
	}
	// Sanity: the set is exactly the three metrics we supplied.
	want := map[string]bool{"tokens_per_second": true, "utilization_ratio": true, "power_watts": true}
	if !reflect.DeepEqual(jsonSet, want) {
		t.Fatalf("present-set = %v, want %v", jsonSet, want)
	}
}

// TestRenderPromInfoMetricAndDeterminism proves the info-metric-with-labels is
// emitted and that rendering is deterministic (stateless reader).
func TestRenderPromInfoMetricAndDeterminism(t *testing.T) {
	snap := []DeviceMetrics{
		{Backend: "nvml", DeviceID: "gpu0", PowerWatts: f(120)},
		{Backend: "engine", DeviceID: "vllm0", Remote: true, Peer: "10.0.0.2", QueueDepth: f(3)},
	}
	out, err := RenderProm(snap)
	if err != nil {
		t.Fatalf("RenderProm: %v", err)
	}
	text := string(out)
	if !strings.Contains(text, "# TYPE fak_device_info gauge") {
		t.Fatalf("info metric family missing:\n%s", text)
	}
	if !strings.Contains(text, `fak_device_info{backend="nvml",device="gpu0"} 1`) {
		t.Fatalf("info sample missing:\n%s", text)
	}
	// Federated row carries remote/peer labels on its samples.
	if !strings.Contains(text, `peer="10.0.0.2"`) || !strings.Contains(text, `remote="true"`) {
		t.Fatalf("federation labels missing:\n%s", text)
	}
	out2, err := RenderProm(snap)
	if err != nil {
		t.Fatalf("RenderProm second: %v", err)
	}
	if text != string(out2) {
		t.Fatalf("render not deterministic")
	}
}

// TestRenderJSONOmitsUnread proves nil metrics are omitted from JSON while
// identity is always present, and a nil snapshot renders as [].
func TestRenderJSONOmitsUnread(t *testing.T) {
	js, err := RenderJSON([]DeviceMetrics{{Backend: "nvml", DeviceID: "gpu0", PowerWatts: f(90)}})
	if err != nil {
		t.Fatalf("RenderJSON: %v", err)
	}
	s := string(js)
	if !strings.Contains(s, `"backend":"nvml"`) || !strings.Contains(s, `"power_watts":90`) {
		t.Fatalf("expected identity + power in %s", s)
	}
	if strings.Contains(s, "queue_depth") || strings.Contains(s, "kv_cache_bytes") {
		t.Fatalf("nil metrics should be omitted: %s", s)
	}
	empty, err := RenderJSON(nil)
	if err != nil {
		t.Fatalf("RenderJSON nil: %v", err)
	}
	if string(empty) != "[]" {
		t.Fatalf("nil snapshot should render [], got %s", empty)
	}
}

func f(v float64) *float64 { return &v }
