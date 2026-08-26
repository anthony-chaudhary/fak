package metrics

// device_spine.go — the cross-vendor device-telemetry spine (issue #3237,
// epic #3236, ZML inspiration). The reusable mechanism is "one normalized
// struct, many probes in, many renderers out":
//
//   - DeviceMetrics: ONE normalized struct, every metric field optional
//     (nil = unsupported/unread). NVML-style C calls and sysfs reads fill the
//     same fields.
//   - metricTable: a declarative {key, query} table drives collection —
//     collection is data, and an unread metric is a uniform nil (null-on-error),
//     never a collection failure.
//   - Probe: a plug-in backend with a cheap Detect() gate before the expensive
//     Devices() collect. One backend erroring or panicking is caught and
//     skipped, never fatal.
//   - Buffer: a lock-free single-writer / many-reader double buffer. Readers
//     never tear; a poll seeds each device's unread fields from the last-good
//     front snapshot ("seed-back-from-front"), holding last-good across a
//     failed poll.
//   - RenderJSON / RenderProm: stateless readers over one shared snapshot. The
//     Prometheus surface is driven by the SAME descriptor table (per-metric
//     help + type) and emits a fak_device_info{…} 1 info metric. Both render
//     from the identical snapshot object, so `fak metrics --json` and `/metrics`
//     share the one collection path.
//
// Generation: gen/next (near-term foundation, kept gated/dogfooded). This is
// the spine only — a library seam with a fake-probe contract test. It is not
// wired into any live /metrics handler or `fak metrics` CLI yet.
//   - Promotion evidence (what moves it toward "now"): wire ONE real local
//     probe (GPU via existing device code) behind Detect(), expose the snapshot
//     on the gateway /metrics handler + a `fak metrics --json` verb, and show a
//     dogfood readout naming devices on a real host.
//   - Demotion / retirement evidence: if the fleet never grows a second backend
//     and per-device telemetry stays single-vendor, this normalization layer is
//     dead weight — retire it and inline the one probe.
//   - Invalidating assumption: every backend's telemetry is assumed to fit one
//     flat all-optional float64 struct. A backend whose native shape is a nested
//     or non-scalar reading (histograms, per-request tables) breaks the flat
//     model and forces a tagged-union redesign.

import (
	"encoding/json"
	"sync/atomic"
)

// DeviceMetrics is the ONE normalized device-telemetry struct shared across
// every backend. Identity fields (Backend, DeviceID) are always set; every
// metric is an optional *float64 where nil means the backend did not support
// or did not read it this poll (the null-on-error contract survives to the
// wire — omitempty drops nil metrics from JSON and Prom alike).
type DeviceMetrics struct {
	Backend  string `json:"backend"`
	DeviceID string `json:"device"`
	Remote   bool   `json:"remote,omitempty"` // federation seam: scraped from a peer
	Peer     string `json:"peer,omitempty"`   // federation seam: peer origin when Remote

	TokensPerSecond  *float64 `json:"tokens_per_second,omitempty"`
	QueueDepth       *float64 `json:"queue_depth,omitempty"`
	KVCacheBytes     *float64 `json:"kv_cache_bytes,omitempty"`
	TTFTSeconds      *float64 `json:"ttft_seconds,omitempty"`
	InFlight         *float64 `json:"in_flight,omitempty"`
	UtilizationRatio *float64 `json:"utilization_ratio,omitempty"`
	MemoryUsedBytes  *float64 `json:"memory_used_bytes,omitempty"`
	PowerWatts       *float64 `json:"power_watts,omitempty"`
}

// Device is one detected device's raw sample source. A Probe hands the
// collector a Device; the collection table reads each normalized metric by
// key. Metric returns ok=false when this device/backend does not support the
// metric — the uniform null-on-error contract (an unsupported metric is nil,
// never a collection failure). Backend specifics (C calls vs sysfs reads) live
// inside the Device implementation, so the collection table stays uniform.
type Device interface {
	ID() string
	Metric(key string) (float64, bool)
}

// Probe is one plug-in backend (nvml/amdsmi/oneapi/neuron/engine/…). Detect is
// the cheap presence check run before the expensive Devices() collect; a probe
// whose Detect is false — or whose Devices() errors or panics — is skipped,
// never fatal (Collect isolates each probe).
type Probe interface {
	Backend() string
	Detect() bool
	Devices() ([]Device, error)
}

// metricDesc is one row of the declarative collection + exposition table: a
// normalized metric's key (Device.Metric key, JSON tag base, and Prometheus
// family suffix), its help/type for the descriptor-table exporter, and the
// get/set accessors that bind the key to its DeviceMetrics field.
type metricDesc struct {
	Key  string
	Help string
	Type OpenMetricType
	get  func(DeviceMetrics) (float64, bool)
	set  func(*DeviceMetrics, float64)
}

// metricTable is the single source of truth mapping every normalized metric to
// its field, help text, and Prometheus type. Collection, JSON present-set, and
// the Prometheus descriptor table are all driven by this one table — add a
// metric here and every surface picks it up.
var metricTable = []metricDesc{
	newMetricDesc("tokens_per_second", "Decode throughput in tokens per second.", func(d *DeviceMetrics) **float64 { return &d.TokensPerSecond }),
	newMetricDesc("queue_depth", "Requests waiting in the engine queue.", func(d *DeviceMetrics) **float64 { return &d.QueueDepth }),
	newMetricDesc("kv_cache_bytes", "KV-cache memory in use, in bytes.", func(d *DeviceMetrics) **float64 { return &d.KVCacheBytes }),
	newMetricDesc("ttft_seconds", "Time to first token, in seconds.", func(d *DeviceMetrics) **float64 { return &d.TTFTSeconds }),
	newMetricDesc("in_flight", "Requests currently being served.", func(d *DeviceMetrics) **float64 { return &d.InFlight }),
	newMetricDesc("utilization_ratio", "Device compute utilization as a 0..1 ratio.", func(d *DeviceMetrics) **float64 { return &d.UtilizationRatio }),
	newMetricDesc("memory_used_bytes", "Device memory in use, in bytes.", func(d *DeviceMetrics) **float64 { return &d.MemoryUsedBytes }),
	newMetricDesc("power_watts", "Device power draw, in watts.", func(d *DeviceMetrics) **float64 { return &d.PowerWatts }),
}

func newMetricDesc(key, help string, field func(*DeviceMetrics) **float64) metricDesc {
	return metricDesc{
		Key: key, Help: help, Type: OpenMetricGauge,
		get: func(d DeviceMetrics) (float64, bool) { return deref(*field(&d)) },
		set: func(d *DeviceMetrics, v float64) { *field(d) = &v },
	}
}
func deref(p *float64) (float64, bool) {
	if p == nil {
		return 0, false
	}
	return *p, true
}

// Collect runs every probe and returns one normalized snapshot. Each probe is
// isolated: a probe that fails Detect, errors, or panics contributes nothing
// and never aborts the others (the plug-in registry's skip-not-fatal rule).
func Collect(probes []Probe) []DeviceMetrics {
	out := make([]DeviceMetrics, 0, len(probes))
	for _, p := range probes {
		out = append(out, collectProbe(p)...)
	}
	return out
}

// collectProbe runs one probe under a recover so a panicking backend is
// skipped, not fatal. The named return lets the deferred recover blank the
// partial slice for this probe only.
func collectProbe(p Probe) (devs []DeviceMetrics) {
	defer func() {
		if r := recover(); r != nil {
			devs = nil
		}
	}()
	if p == nil || !p.Detect() {
		return nil
	}
	ds, err := p.Devices()
	if err != nil {
		return nil
	}
	out := make([]DeviceMetrics, 0, len(ds))
	for _, d := range ds {
		if d == nil {
			continue
		}
		out = append(out, collectDevice(p.Backend(), d))
	}
	return out
}

// collectDevice drives the declarative table: for each metric key, read it from
// the device and fill the field on success; a miss leaves the field nil. This
// is the Go form of ZML's `inline for (table) |m| @field = m.query(dev) catch
// null` — collection is data, null-on-error is uniform.
func collectDevice(backend string, d Device) DeviceMetrics {
	dm := DeviceMetrics{Backend: backend, DeviceID: d.ID()}
	for _, m := range metricTable {
		if v, ok := d.Metric(m.Key); ok {
			m.set(&dm, v)
		}
	}
	return dm
}

// Buffer is a lock-free single-writer / many-reader double buffer over a device
// snapshot. Poll publishes a fresh snapshot with one atomic pointer store;
// Snapshot readers load the current pointer and never see a torn slice, because
// a published snapshot is never mutated in place.
type Buffer struct {
	front atomic.Pointer[[]DeviceMetrics]
}

// NewBuffer returns an empty double buffer whose Snapshot is nil until the
// first Poll.
func NewBuffer() *Buffer { return &Buffer{} }

// Snapshot returns the current published snapshot. The returned slice is
// read-only — callers must not mutate it (renderers here never do).
func (b *Buffer) Snapshot() []DeviceMetrics {
	p := b.front.Load()
	if p == nil {
		return nil
	}
	return *p
}

// Poll collects from probes, seeds each device's unread fields from the
// last-good front snapshot, atomically publishes the merged snapshot, and
// returns it. It is the single-writer half of the double buffer — call it from
// one goroutine.
func (b *Buffer) Poll(probes []Probe) []DeviceMetrics {
	merged := seedFromFront(b.Snapshot(), Collect(probes))
	published := append([]DeviceMetrics(nil), merged...)
	b.front.Store(&published)
	return published
}

// seedFromFront implements ZML's seed-back-from-front: a device present in the
// previous snapshot whose metric is unread (nil) in the fresh poll keeps its
// last-good value, and a device that vanished from the fresh poll entirely is
// carried forward whole. Ordering is deterministic: fresh devices in poll
// order, then carried-forward devices in prior order.
func seedFromFront(prev, fresh []DeviceMetrics) []DeviceMetrics {
	index := make(map[string]DeviceMetrics, len(prev))
	for _, d := range prev {
		index[deviceKey(d)] = d
	}
	seen := make(map[string]struct{}, len(fresh))
	out := make([]DeviceMetrics, 0, len(fresh)+len(prev))
	for _, f := range fresh {
		k := deviceKey(f)
		seen[k] = struct{}{}
		if old, ok := index[k]; ok {
			f = mergeLastGood(old, f)
		}
		out = append(out, f)
	}
	for _, d := range prev {
		if _, ok := seen[deviceKey(d)]; !ok {
			out = append(out, d)
		}
	}
	return out
}

// mergeLastGood fills any metric unread in fresh with the last-good value from
// old, leaving identity fields as fresh's.
func mergeLastGood(old, fresh DeviceMetrics) DeviceMetrics {
	for _, m := range metricTable {
		if _, ok := m.get(fresh); ok {
			continue
		}
		if v, ok := m.get(old); ok {
			m.set(&fresh, v)
		}
	}
	return fresh
}

func deviceKey(d DeviceMetrics) string { return d.Backend + "\x00" + d.DeviceID }

// RenderJSON renders the snapshot as JSON. Unread (nil) metrics are omitted, so
// the normalized-optional contract survives to the wire. A nil snapshot renders
// as an empty array, not null.
func RenderJSON(snapshot []DeviceMetrics) ([]byte, error) {
	if snapshot == nil {
		snapshot = []DeviceMetrics{}
	}
	return json.Marshal(snapshot)
}

// RenderProm renders the snapshot as OpenMetrics/Prometheus text via the shared
// descriptor table. It is a stateless reader over the same snapshot RenderJSON
// consumes — the one-collection-path guarantee (`fak metrics --json` and
// `/metrics` read the identical snapshot).
func RenderProm(snapshot []DeviceMetrics) ([]byte, error) {
	return RenderOpenMetricsText(DeviceFamilies(snapshot))
}

// DeviceFamilies projects a snapshot into OpenMetrics families using the shared
// descriptor table: one gauge family per metric (labelled {backend,device},
// plus remote/peer for federated rows), and a fak_device_info{…} 1 info metric
// carrying device identity. A metric with no present samples emits no family,
// so the Prometheus present-set matches JSON's.
func DeviceFamilies(snapshot []DeviceMetrics) []OpenMetricFamily {
	families := make([]OpenMetricFamily, 0, len(metricTable)+1)
	for _, m := range metricTable {
		var samples []OpenMetricSample
		for _, d := range snapshot {
			if v, ok := m.get(d); ok {
				samples = append(samples, OpenMetricSample{Labels: deviceLabels(d), Value: v})
			}
		}
		if len(samples) == 0 {
			continue
		}
		families = append(families, OpenMetricFamily{
			Name:    "fak_device_" + m.Key,
			Help:    m.Help,
			Type:    m.Type,
			Samples: samples,
		})
	}
	var info []OpenMetricSample
	for _, d := range snapshot {
		info = append(info, OpenMetricSample{Labels: deviceInfoLabels(d), Value: 1})
	}
	if len(info) > 0 {
		families = append(families, OpenMetricFamily{
			Name:    "fak_device_info",
			Help:    "Device identity labels; constant 1 (info metric).",
			Type:    OpenMetricGauge,
			Samples: info,
		})
	}
	return families
}

func deviceLabels(d DeviceMetrics) []OpenMetricLabel {
	labels := []OpenMetricLabel{
		{Name: "backend", Value: d.Backend},
		{Name: "device", Value: d.DeviceID},
	}
	if d.Remote {
		labels = append(labels, OpenMetricLabel{Name: "remote", Value: "true"})
		if d.Peer != "" {
			labels = append(labels, OpenMetricLabel{Name: "peer", Value: d.Peer})
		}
	}
	return labels
}

func deviceInfoLabels(d DeviceMetrics) []OpenMetricLabel {
	return deviceLabels(d)
}
