// Package inferencetopology models decode-step resource constraints from explicit inputs.
package inferencetopology

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/url"
	"os"
	"time"
)

const Schema = "fak-inference-topology/1"

var ErrInvalid = errors.New("invalid inference topology manifest")

type Manifest struct {
	Schema    string     `json:"schema"`
	ID        string     `json:"id"`
	AsOf      string     `json:"as_of"`
	Sources   []Source   `json:"sources"`
	Claims    []Claim    `json:"claims"`
	Scenarios []Scenario `json:"scenarios"`
}

type Source struct {
	ID       string `json:"id"`
	URL      string `json:"url"`
	Title    string `json:"title"`
	Accessed string `json:"accessed"`
	ReviewBy string `json:"review_by"`
	Revision string `json:"revision,omitempty"`
	Kind     string `json:"kind"`
}

type Claim struct {
	ID          string   `json:"id"`
	Text        string   `json:"text"`
	Status      string   `json:"status"`
	SourceIDs   []string `json:"source_ids"`
	ReviewBy    string   `json:"review_by"`
	Limitations string   `json:"limitations,omitempty"`
}

type Scenario struct {
	ID          string     `json:"id"`
	Label       string     `json:"label"`
	Mode        string     `json:"mode"`
	SourceIDs   []string   `json:"source_ids"`
	Assumptions []string   `json:"assumptions"`
	Model       Model      `json:"model"`
	Hardware    Hardware   `json:"hardware"`
	PerRank     *Resources `json:"per_rank,omitempty"`
}

type Model struct {
	DenseParameters               float64 `json:"dense_parameters"`
	MoELayers                     int     `json:"moe_layers"`
	ExpertsPerLayer               int     `json:"experts_per_layer"`
	ActiveExpertsPerToken         int     `json:"active_experts_per_token"`
	ParametersPerExpert           float64 `json:"parameters_per_expert"`
	BytesPerWeight                float64 `json:"bytes_per_weight"`
	KVBytesPerToken               float64 `json:"kv_bytes_per_token"`
	BatchSize                     int     `json:"batch_size"`
	ContextTokens                 int     `json:"context_tokens"`
	HiddenSize                    int     `json:"hidden_size"`
	Layers                        int     `json:"layers"`
	BytesPerActivation            float64 `json:"bytes_per_activation"`
	AdditionalFLOPsPerStep        float64 `json:"additional_flops_per_step"`
	ActivationTrafficBytesPerStep float64 `json:"activation_traffic_bytes_per_step"`
}

type Hardware struct {
	GPUs                               int     `json:"gpus"`
	TPRanks                            int     `json:"tp_ranks"`
	KVShards                           int     `json:"kv_shards"`
	MemoryBytesPerGPU                  float64 `json:"memory_bytes_per_gpu"`
	HBMBytesPerSecondPerGPU            float64 `json:"hbm_bytes_per_second_per_gpu"`
	FLOPsPerSecondPerGPU               float64 `json:"flops_per_second_per_gpu"`
	LinkBytesPerSecondPerRankOneWay    float64 `json:"link_bytes_per_second_per_rank_one_way"`
	RingAlphaSeconds                   float64 `json:"ring_alpha_seconds"`
	AllReducesPerStep                  int     `json:"allreduces_per_step"`
	CollectivePayloadBytes             float64 `json:"collective_payload_bytes,omitempty"`
	ReservedBytesPerGPU                float64 `json:"reserved_bytes_per_gpu"`
	ExtraNetworkBytesPerRankPerStep    float64 `json:"extra_network_bytes_per_rank_per_step"`
	ExtraNetworkEventsPerStep          int     `json:"extra_network_events_per_step"`
	ExtraNetworkLatencySecondsPerEvent float64 `json:"extra_network_latency_seconds_per_event"`
}

// Resources describes a peak rank, not an average rank or an aggregate memory pool.
type Resources struct {
	ResidentWeightBytes           float64 `json:"resident_weight_bytes"`
	WeightStreamBytesPerStep      float64 `json:"weight_stream_bytes_per_step"`
	KVResidentBytes               float64 `json:"kv_resident_bytes"`
	KVReadBytesPerStep            float64 `json:"kv_read_bytes_per_step"`
	KVWriteBytesPerStep           float64 `json:"kv_write_bytes_per_step"`
	ActivationTrafficBytesPerStep float64 `json:"activation_traffic_bytes_per_step"`
	FLOPsPerStep                  float64 `json:"flops_per_step"`
}

type Result struct {
	ID                               string    `json:"id"`
	Label                            string    `json:"label"`
	Mode                             string    `json:"mode"`
	Evidence                         string    `json:"evidence"`
	SourceIDs                        []string  `json:"source_ids"`
	ExpectedDistinctExpertsPerLayer  float64   `json:"expected_distinct_experts_per_layer"`
	UniformFormatLogicalWeightBytes  float64   `json:"uniform_format_logical_weight_bytes"`
	PerRank                          Resources `json:"per_rank"`
	PerRankTotalResidentBytes        float64   `json:"per_rank_total_resident_bytes"`
	Fits                             bool      `json:"fits"`
	HBMBytesPerStep                  float64   `json:"hbm_bytes_per_step"`
	ArithmeticIntensityFLOPsPerByte  *float64  `json:"arithmetic_intensity_flops_per_byte,omitempty"`
	MachineBalanceFLOPsPerByte       float64   `json:"machine_balance_flops_per_byte"`
	KVReadEqualsWeightContextTokens  *float64  `json:"kv_read_equals_weight_context_tokens,omitempty"`
	DenseWeightComputeCrossoverBatch *float64  `json:"dense_weight_compute_crossover_batch,omitempty"`
	HBMSeconds                       float64   `json:"hbm_seconds"`
	ComputeSeconds                   float64   `json:"compute_seconds"`
	RingSeconds                      float64   `json:"ring_seconds"`
	ExtraNetworkSeconds              float64   `json:"extra_network_seconds"`
	IdealOverlapLowerBoundSeconds    float64   `json:"ideal_overlap_lower_bound_seconds"`
	SerialResourceSumSeconds         float64   `json:"serial_resource_sum_seconds"`
	DominantModeledResource          string    `json:"dominant_modeled_resource"`
	Warnings                         []string  `json:"warnings"`
}

type Analysis struct {
	Schema      string   `json:"schema"`
	ManifestID  string   `json:"manifest_id"`
	AsOf        string   `json:"as_of"`
	Evidence    string   `json:"evidence"`
	Limitations []string `json:"limitations"`
	Scenarios   []Result `json:"scenarios"`
}

func Load(path string) (Manifest, error) {
	f, err := os.Open(path)
	if err != nil {
		return Manifest{}, err
	}
	defer f.Close()
	return Decode(f)
}

func Decode(r io.Reader) (Manifest, error) {
	var m Manifest
	raw, err := io.ReadAll(io.LimitReader(r, (16<<20)+1))
	if err != nil {
		return m, fmt.Errorf("%w: read: %v", ErrInvalid, err)
	}
	if len(raw) > 16<<20 {
		return m, invalid("manifest exceeds 16 MiB")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&m); err != nil {
		return m, fmt.Errorf("%w: decode: %v", ErrInvalid, err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return m, fmt.Errorf("%w: trailing JSON", ErrInvalid)
	}
	return m, Validate(m)
}

func date(v string) bool         { _, err := time.Parse("2006-01-02", v); return err == nil }
func nonnegative(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) && v >= 0 }
func invalid(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalid, fmt.Sprintf(format, args...))
}

func Validate(m Manifest) error {
	if m.Schema != Schema || m.ID == "" || !date(m.AsOf) || len(m.Scenarios) == 0 || len(m.Sources) == 0 {
		return invalid("schema, id, as_of, sources and scenarios required")
	}
	sources := map[string]bool{}
	for _, s := range m.Sources {
		u, e := url.Parse(s.URL)
		if s.ID == "" || sources[s.ID] || s.Title == "" || e != nil || u.Host == "" || (u.Scheme != "https" && u.Scheme != "http") || !date(s.Accessed) || !date(s.ReviewBy) || s.Accessed > m.AsOf || s.ReviewBy < s.Accessed {
			return invalid("source %q identity, URL or dates", s.ID)
		}
		switch s.Kind {
		case "primary", "vendor", "oss", "secondary", "social", "assumption":
		default:
			return invalid("source %q kind", s.ID)
		}
		sources[s.ID] = true
	}
	refs := func(ids []string) bool {
		if len(ids) == 0 {
			return false
		}
		seen := map[string]bool{}
		for _, id := range ids {
			if !sources[id] || seen[id] {
				return false
			}
			seen[id] = true
		}
		return true
	}
	claims := map[string]bool{}
	for _, c := range m.Claims {
		if c.ID == "" || claims[c.ID] || c.Text == "" || !date(c.ReviewBy) || !refs(c.SourceIDs) {
			return invalid("claim %q identity, references or dates", c.ID)
		}
		switch c.Status {
		case "supported", "disputed", "open", "derived":
		default:
			return invalid("claim %q status", c.ID)
		}
		claims[c.ID] = true
	}
	scenarios := map[string]bool{}
	for _, s := range m.Scenarios {
		if s.ID == "" || scenarios[s.ID] || s.Label == "" || len(s.Assumptions) == 0 || !refs(s.SourceIDs) {
			return invalid("scenario %q identity, assumptions or references", s.ID)
		}
		scenarios[s.ID] = true
		if err := validateScenario(s); err != nil {
			return invalid("scenario %q: %v", s.ID, err)
		}
	}
	return nil
}

func validateScenario(s Scenario) error {
	m, h := s.Model, s.Hardware
	if s.Mode != "tensor_parallel" && s.Mode != "explicit_per_rank" {
		return invalid("mode must be tensor_parallel or explicit_per_rank")
	}
	if h.GPUs < 1 || h.TPRanks < 1 || h.TPRanks > h.GPUs || h.GPUs%h.TPRanks != 0 || h.KVShards < 1 || h.KVShards > h.TPRanks || h.TPRanks%h.KVShards != 0 {
		return invalid("incompatible GPUs, TP ranks or KV shards")
	}
	if s.Mode == "tensor_parallel" && (h.GPUs != h.TPRanks || s.PerRank != nil) {
		return invalid("tensor_parallel requires one group (gpus == tp_ranks) and no per_rank override")
	}
	if s.Mode == "tensor_parallel" && m.HiddenSize%h.TPRanks != 0 {
		return invalid("hidden_size must divide across TP ranks; irregular layouts need explicit_per_rank")
	}
	if s.Mode == "explicit_per_rank" && s.PerRank == nil {
		return invalid("explicit_per_rank requires peak-rank resources")
	}
	if m.BatchSize < 1 || m.ContextTokens < 0 || m.HiddenSize < 1 || m.Layers < 1 || m.MoELayers < 0 || m.MoELayers > m.Layers || m.ExpertsPerLayer < 0 || m.ActiveExpertsPerToken < 0 || m.ActiveExpertsPerToken > m.ExpertsPerLayer || h.AllReducesPerStep < 0 || h.ExtraNetworkEventsPerStep < 0 {
		return invalid("counts outside permitted range")
	}
	if m.MoELayers > 0 && (m.ExpertsPerLayer == 0 || m.ActiveExpertsPerToken == 0 || m.ParametersPerExpert <= 0) {
		return invalid("MoE dimensions incomplete")
	}
	if m.MoELayers == 0 && (m.ExpertsPerLayer != 0 || m.ActiveExpertsPerToken != 0 || m.ParametersPerExpert != 0) {
		return invalid("nonzero experts require moe_layers")
	}
	values := []float64{m.DenseParameters, m.ParametersPerExpert, m.BytesPerWeight, m.KVBytesPerToken, m.BytesPerActivation, m.AdditionalFLOPsPerStep, m.ActivationTrafficBytesPerStep, h.MemoryBytesPerGPU, h.HBMBytesPerSecondPerGPU, h.FLOPsPerSecondPerGPU, h.LinkBytesPerSecondPerRankOneWay, h.RingAlphaSeconds, h.CollectivePayloadBytes, h.ReservedBytesPerGPU, h.ExtraNetworkBytesPerRankPerStep, h.ExtraNetworkLatencySecondsPerEvent}
	if s.PerRank != nil {
		p := s.PerRank
		values = append(values, p.ResidentWeightBytes, p.WeightStreamBytesPerStep, p.KVResidentBytes, p.KVReadBytesPerStep, p.KVWriteBytesPerStep, p.ActivationTrafficBytesPerStep, p.FLOPsPerStep)
	}
	for _, v := range values {
		if !nonnegative(v) {
			return invalid("numeric values must be finite and nonnegative")
		}
	}
	if m.BytesPerWeight <= 0 || m.BytesPerActivation <= 0 || h.MemoryBytesPerGPU <= 0 || h.HBMBytesPerSecondPerGPU <= 0 || h.FLOPsPerSecondPerGPU <= 0 || (m.DenseParameters == 0 && m.MoELayers == 0) {
		return invalid("model and hardware capacities must be positive")
	}
	if ((h.TPRanks > 1 && h.AllReducesPerStep > 0) || h.ExtraNetworkBytesPerRankPerStep > 0) && h.LinkBytesPerSecondPerRankOneWay <= 0 {
		return invalid("communication requires a positive one-way bottleneck link rate")
	}
	return nil
}

func Analyze(m Manifest) (Analysis, error) {
	if err := Validate(m); err != nil {
		return Analysis{}, err
	}
	a := Analysis{Schema: "fak-inference-topology-analysis/1", ManifestID: m.ID, AsOf: m.AsOf, Evidence: "ANALYTICAL_NOT_MEASURED", Limitations: []string{
		"Rates are caller-supplied capacities, not measured utilization; these are conditional resource estimates, not benchmark performance.",
		"Tensor-parallel mode assumes perfectly balanced weight and compute shards; KV sharding is explicit. Extra GPUs are never pooled into tensor-parallel capacity.",
		"MoE weight traffic assumes independent uniform top-k token routing and ideal batch reuse; skew, expert placement, padding and hot-rank imbalance require explicit_per_rank inputs.",
		"Ring cost is an illustrative ring with per-phase alpha and a per-rank one-way bottleneck rate, not an NVSwitch, NCCL, PCIe or RDMA guarantee. Full-duplex send+receive totals must not be entered as one-way bandwidth.",
		"The maximum of HBM, compute, ring and extra-network times is an ideal-overlap lower bound under the supplied traffic/rates. The serial sum is not an upper bound on real execution. Ring phases and extra-network events are internally serialized in their respective estimates, but the lower bound permits overlap between the two paths. Layer dependencies, shared resources, contention and launch overhead can increase time.",
		"Explicit per-rank resources should describe jointly realizable work on a physical critical rank. Maxima taken from different ranks can supply separate resource bounds, but their serial sum does not describe any one actual rank.",
		"Capacity excludes workspaces, activations, graph buffers, fragmentation and other processes unless reserved_bytes_per_gpu includes them. An infeasible scenario has no executable latency prediction.",
		"Additional attention/non-matmul FLOPs and activation traffic must be supplied. This decode-step model does not predict prefill, tail latency, queue saturation, training, or useful service throughput.",
	}}
	for _, s := range m.Scenarios {
		r := analyze(s)
		values := []float64{r.ExpectedDistinctExpertsPerLayer, r.UniformFormatLogicalWeightBytes, r.PerRankTotalResidentBytes, r.HBMBytesPerStep, r.MachineBalanceFLOPsPerByte, r.HBMSeconds, r.ComputeSeconds, r.RingSeconds, r.ExtraNetworkSeconds, r.IdealOverlapLowerBoundSeconds, r.SerialResourceSumSeconds}
		if r.ArithmeticIntensityFLOPsPerByte != nil {
			values = append(values, *r.ArithmeticIntensityFLOPsPerByte)
		}
		if r.KVReadEqualsWeightContextTokens != nil {
			values = append(values, *r.KVReadEqualsWeightContextTokens)
		}
		if r.DenseWeightComputeCrossoverBatch != nil {
			values = append(values, *r.DenseWeightComputeCrossoverBatch)
		}
		for _, v := range values {
			if !nonnegative(v) {
				return Analysis{}, invalid("scenario %q arithmetic overflow", s.ID)
			}
		}
		a.Scenarios = append(a.Scenarios, r)
	}
	return a, nil
}

func analyze(s Scenario) Result {
	m, h := s.Model, s.Hardware
	b, n := float64(m.BatchSize), float64(h.TPRanks)
	u := 0.0
	if m.MoELayers > 0 {
		e, k := float64(m.ExpertsPerLayer), float64(m.ActiveExpertsPerToken)
		if k == e {
			u = e
		} else {
			u = -e * math.Expm1(b*math.Log1p(-k/e))
		}
	}
	expertParams := float64(m.MoELayers) * m.ParametersPerExpert
	weights := (m.DenseParameters + float64(m.ExpertsPerLayer)*expertParams) * m.BytesPerWeight
	kv := b * float64(m.ContextTokens) * m.KVBytesPerToken
	p := Resources{ResidentWeightBytes: weights / n, WeightStreamBytesPerStep: (m.DenseParameters + u*expertParams) * m.BytesPerWeight / n, KVResidentBytes: (kv + b*m.KVBytesPerToken) / float64(h.KVShards), KVReadBytesPerStep: kv / float64(h.KVShards), KVWriteBytesPerStep: b * m.KVBytesPerToken / float64(h.KVShards), ActivationTrafficBytesPerStep: m.ActivationTrafficBytesPerStep / n, FLOPsPerStep: (2*b*(m.DenseParameters+float64(m.ActiveExpertsPerToken)*expertParams) + m.AdditionalFLOPsPerStep) / n}
	if s.PerRank != nil {
		p = *s.PerRank
	}
	r := Result{ID: s.ID, Label: s.Label, Mode: s.Mode, Evidence: "ANALYTICAL_NOT_MEASURED", SourceIDs: s.SourceIDs, ExpectedDistinctExpertsPerLayer: u, UniformFormatLogicalWeightBytes: weights, PerRank: p, Warnings: []string{}}
	r.PerRankTotalResidentBytes = p.ResidentWeightBytes + p.KVResidentBytes + h.ReservedBytesPerGPU
	r.Fits = r.PerRankTotalResidentBytes <= h.MemoryBytesPerGPU
	r.HBMBytesPerStep = p.WeightStreamBytesPerStep + p.KVReadBytesPerStep + p.KVWriteBytesPerStep + p.ActivationTrafficBytesPerStep
	r.HBMSeconds = r.HBMBytesPerStep / h.HBMBytesPerSecondPerGPU
	if r.HBMBytesPerStep > 0 {
		v := p.FLOPsPerStep / r.HBMBytesPerStep
		r.ArithmeticIntensityFLOPsPerByte = &v
	} else {
		r.Warnings = append(r.Warnings, "arithmetic intensity undefined: no HBM bytes supplied; validate on-chip reuse and omitted traffic")
	}
	r.MachineBalanceFLOPsPerByte = h.FLOPsPerSecondPerGPU / h.HBMBytesPerSecondPerGPU
	if s.Mode == "tensor_parallel" {
		if m.KVBytesPerToken > 0 {
			v := p.WeightStreamBytesPerStep / (b * m.KVBytesPerToken / float64(h.KVShards))
			r.KVReadEqualsWeightContextTokens = &v
		}
		if m.MoELayers == 0 {
			v := m.BytesPerWeight * r.MachineBalanceFLOPsPerByte / 2
			r.DenseWeightComputeCrossoverBatch = &v
		}
	}
	r.ComputeSeconds = p.FLOPsPerStep / h.FLOPsPerSecondPerGPU
	payload := h.CollectivePayloadBytes
	if payload == 0 {
		payload = b * float64(m.HiddenSize) * m.BytesPerActivation
	}
	if h.TPRanks > 1 && h.AllReducesPerStep > 0 {
		r.RingSeconds = float64(h.AllReducesPerStep) * (2*(n-1)*h.RingAlphaSeconds + 2*(n-1)/n*payload/h.LinkBytesPerSecondPerRankOneWay)
	}
	r.ExtraNetworkSeconds = float64(h.ExtraNetworkEventsPerStep) * h.ExtraNetworkLatencySecondsPerEvent
	if h.ExtraNetworkBytesPerRankPerStep > 0 {
		r.ExtraNetworkSeconds += h.ExtraNetworkBytesPerRankPerStep / h.LinkBytesPerSecondPerRankOneWay
	}
	r.DominantModeledResource = "hbm"
	r.IdealOverlapLowerBoundSeconds = r.HBMSeconds
	for _, x := range []struct {
		name    string
		seconds float64
	}{{"compute", r.ComputeSeconds}, {"ring", r.RingSeconds}, {"extra_network", r.ExtraNetworkSeconds}} {
		if x.seconds > r.IdealOverlapLowerBoundSeconds {
			r.IdealOverlapLowerBoundSeconds = x.seconds
			r.DominantModeledResource = x.name
		}
	}
	r.SerialResourceSumSeconds = r.HBMSeconds + r.ComputeSeconds + r.RingSeconds + r.ExtraNetworkSeconds
	if !r.Fits {
		r.Warnings = append(r.Warnings, "CAPACITY_INFEASIBLE: resident peak rank exceeds memory; timing numbers are arithmetic resource estimates only")
	}
	if m.AdditionalFLOPsPerStep == 0 {
		r.Warnings = append(r.Warnings, "attention and non-matmul compute not added unless included in explicit per-rank FLOPs")
	}
	if h.TPRanks > 1 && h.AllReducesPerStep == 0 {
		r.Warnings = append(r.Warnings, "no TP all-reduces supplied; communication may be omitted")
	}
	if s.Mode == "explicit_per_rank" {
		r.Warnings = append(r.Warnings, "explicit peak-rank inputs replace derived resource partitioning; validate placement and imbalance independently")
	}
	return r
}

type Inventory struct {
	Schema            string   `json:"schema"`
	ManifestID        string   `json:"manifest_id"`
	At                string   `json:"at"`
	SourceCount       int      `json:"source_count"`
	ClaimCount        int      `json:"claim_count"`
	ScenarioCount     int      `json:"scenario_count"`
	StaleSources      []string `json:"stale_sources"`
	StaleClaims       []string `json:"stale_claims"`
	AffectedScenarios []string `json:"affected_scenarios"`
	OpenClaims        []string `json:"open_claims"`
	Notice            string   `json:"notice"`
}

func Reinventory(m Manifest, at string) (Inventory, error) {
	if err := Validate(m); err != nil {
		return Inventory{}, err
	}
	if !date(at) || at < m.AsOf {
		return Inventory{}, invalid("inventory date must be valid and at or after as_of")
	}
	i := Inventory{Schema: "fak-inference-topology-inventory/1", ManifestID: m.ID, At: at, SourceCount: len(m.Sources), ClaimCount: len(m.Claims), ScenarioCount: len(m.Scenarios), StaleSources: []string{}, StaleClaims: []string{}, AffectedScenarios: []string{}, OpenClaims: []string{}, Notice: "Review dates mark evidence due for rechecking; inventory does not fetch sources or assert current truth."}
	stale := map[string]bool{}
	for _, s := range m.Sources {
		if s.ReviewBy < at {
			i.StaleSources = append(i.StaleSources, s.ID)
			stale[s.ID] = true
		}
	}
	for _, c := range m.Claims {
		due := c.ReviewBy < at
		for _, id := range c.SourceIDs {
			due = due || stale[id]
		}
		if due {
			i.StaleClaims = append(i.StaleClaims, c.ID)
		}
		if c.Status == "open" || c.Status == "disputed" {
			i.OpenClaims = append(i.OpenClaims, c.ID)
		}
	}
	for _, s := range m.Scenarios {
		for _, id := range s.SourceIDs {
			if stale[id] {
				i.AffectedScenarios = append(i.AffectedScenarios, s.ID)
				break
			}
		}
	}
	return i, nil
}
