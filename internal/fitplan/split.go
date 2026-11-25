package fitplan

import (
	"math"
	"sort"
)

// split.go prices how a model is spread over N identical discrete devices and what
// runs the weights that spill to host RAM / NVMe, so a planner can rank "a second
// GPU" against "more host RAM" on the same header facts and calibration Plan uses.
//
// Split modes (one decode step at batch b, priced serially where the engine is serial):
//
//	single  one device; the dense-core-first placement Plan already offers.
//	layer   pipeline / layer split: every device holds 1/N of the layers (dense, KV,
//	        experts); a token visits the stages in turn, so step = sum of stages plus
//	        N-1 activation hops. It buys capacity (more experts resident), not speed.
//	tensor  every device holds 1/N of every layer; step = slowest rank plus two ring
//	        all-reduces per layer (2(N-1) latency steps each). KV shards over at most
//	        num_kv_heads ranks; a compressed (MLA-class) KV is replicated.
//	expert  dense core and KV on device 0, routed experts across all devices; step =
//	        dense + slowest expert rank + two all-to-alls per layer.
//
// Offload modes for weights below the devices:
//
//	cpu     host-resident weights are computed in place at host-RAM bandwidth (NVMe
//	        pages are read at NVMe rate), plus a host<->device crossing per layer that
//	        carries host experts.
//	stream  host-resident weights are copied over the host link to the GPU each step,
//	        at min(link, host RAM) bandwidth; experts are selected after the router, so
//	        the copy is serial with compute.
//
// The unique routed-expert bytes a batch reads are 1-(1-k/E)^b of the routed weights
// (uniform routing), split by residency. Everything here is MODELED: a ranking aid, not
// a measurement, and every generic interconnect default is labelled in the row.

// SplitMode names how the weights are spread over the devices.
type SplitMode string

const (
	SplitSingle SplitMode = "single"
	SplitLayer  SplitMode = "layer"
	SplitTensor SplitMode = "tensor"
	SplitExpert SplitMode = "expert"
)

// OffloadMode names what runs the weights below the devices.
type OffloadMode string

const (
	OffloadCPU    OffloadMode = "cpu"
	OffloadStream OffloadMode = "stream"
)

// Binding resources a split row's step time is set by.
const (
	BindDeviceBandwidth = "device_bw"
	BindDeviceCompute   = "device_compute"
	BindHostBandwidth   = "host_bw"
	BindHostCompute     = "host_compute"
	BindHostLink        = "host_link"
	BindP2P             = "p2p_link"
	BindNVMe            = "nvme"
	BindSync            = "sync_latency"
	BindOverhead        = "step_overhead" // the calibrated fixed per-step cost
)

// Split refusal reasons (row-level; the report's Status says whether any row fits).
const (
	SplitRefusalKVExceedsDevice = "kv_exceeds_device"
	SplitRefusalNotDiscrete     = "multi_device_needs_discrete_topology"
	SplitRefusalNoRoutedExperts = "expert_split_needs_routed_experts"
)

// Generic interconnect rates, used until Interconnect states them: a PCIe 4.0 x16 host
// link at ~0.8 of 32 GB/s, a PCIe-class peer hop (no NVLink), and a host sync crossing.
const (
	genericHostLinkBytesPerSec = 25.6e9
	genericP2PLatencySeconds   = 10e-6
	genericHostSyncSeconds     = 20e-6
	activationBytesPerElem     = 2 // f16/bf16 activations crossing a link
)

// Interconnect is the multi-device and offload facts a split is priced against. Every
// device is assumed identical to the one Hardware describes, and Hardware.PoolUsableBytes
// is PER DEVICE.
type Interconnect struct {
	Devices int `json:"devices"`
	// HostLinkBytesPerSec is one device's host link (PCIe), effective; <=0 generic.
	HostLinkBytesPerSec float64 `json:"host_link_bytes_per_s,omitempty"`
	// P2PBytesPerSec is the device-to-device rate; <=0 routes peer traffic over the host link.
	P2PBytesPerSec float64 `json:"p2p_bytes_per_s,omitempty"`
	// P2PLatencySeconds is one peer hop / ring step; <=0 generic (PCIe class).
	P2PLatencySeconds float64 `json:"p2p_latency_s,omitempty"`
	// HostSyncSeconds is one host<->device crossing; <=0 generic.
	HostSyncSeconds float64 `json:"host_sync_s,omitempty"`
	// HostFLOPsPerSec is the CPU's effective compute for offload=cpu; <=0 prices host
	// compute as bandwidth-only.
	HostFLOPsPerSec float64 `json:"host_flops_per_s,omitempty"`
	// Executable is the split modes the serving engine can actually run on this host;
	// nil means ServeExecutableSplits(). Chosen is picked among these only.
	Executable []SplitMode `json:"executable,omitempty"`
}

// ServeExecutableSplits is what `fak serve` can run in one process today: a single
// device, and expert parallel over NCCL (`--expert-parallel N`, a -tags cuda,nccl build).
// Layer split runs only across worker processes (modelengine.NewPipelineEngine) and
// tensor parallel is a collective gate with no sharded serve forward, so both are
// priced (what the hardware could do) but never chosen.
//
//enumlint:exempt the serving engine can run only single and expert splits; layer and tensor are priced for what the hardware could do and are deliberately absent here.
func ServeExecutableSplits() []SplitMode { return []SplitMode{SplitSingle, SplitExpert} }

// SplitPhase is one serial component of a split row's decode step.
type SplitPhase struct {
	Name    string  `json:"name"`
	Seconds float64 `json:"seconds"`
	Binding string  `json:"binding"`
}

// SplitRow is one priced (split mode, offload mode) configuration.
type SplitRow struct {
	Mode    SplitMode   `json:"mode"`
	Offload OffloadMode `json:"offload,omitempty"` // absent when nothing spills below the devices
	Devices int         `json:"devices"`
	Batch   int         `json:"batch"`
	// DeviceBytes is the weights+KV+scratch each device holds.
	DeviceBytes         []int64      `json:"device_bytes,omitempty"`
	HostBytes           int64        `json:"host_bytes,omitempty"`
	NVMeBytes           int64        `json:"nvme_bytes,omitempty"`
	DecodeTokSPerSeq    float64      `json:"decode_tok_s_per_seq,omitempty"`
	DecodeTokSAggregate float64      `json:"decode_tok_s_aggregate,omitempty"`
	Bound               Bound        `json:"bound,omitempty"`   // memory | compute | link
	Binding             string       `json:"binding,omitempty"` // the dominant phase's resource
	Phases              []SplitPhase `json:"phases,omitempty"`
	GenericRates        []string     `json:"generic_rates,omitempty"` // rates priced at a generic default
	// Executable reports the serving engine can run this mode here (Interconnect.Executable).
	Executable bool   `json:"executable"`
	Refusal    string `json:"refusal,omitempty"`
}

// SplitReport ranks every split row, fastest aggregate decode first. Chosen is the
// fastest unrefused EXECUTABLE row (ties keep the simpler mode); Best is the fastest
// unrefused row of any mode, present only when it is not executable, so the gap an
// engine feature would close is visible.
type SplitReport struct {
	Status       Status       `json:"status"`
	Chosen       *SplitRow    `json:"chosen,omitempty"`
	Best         *SplitRow    `json:"best,omitempty"`
	Rows         []SplitRow   `json:"rows"`
	Interconnect Interconnect `json:"interconnect"`
	Tier         Tier         `json:"tier"`
	Omitted      []Omission   `json:"omitted,omitempty"`
}

// splitRates are the effective rates a split is priced at. generic names the rates that
// fell back to a default; a row records only the generic rates its phases actually used.
type splitRates struct {
	devBW, devF, hostBW, hostF, nvmeBW, link, p2p, p2pLat, sync float64
	p2pViaLink                                                  bool
	generic                                                     map[string]bool
}

// Generic-rate tags a row may carry in GenericRates.
const (
	rateHostBW     = "host_bw"
	rateNVMe       = "nvme"
	rateHostLink   = "host_link"
	rateP2PLatency = "p2p_latency"
	rateHostSync   = "host_sync"
)

func (p *planner) splitRates(ic Interconnect) splitRates {
	r := splitRates{generic: map[string]bool{}}
	r.devBW, r.devF = p.effectiveRates()
	for _, t := range p.placementTiers() {
		switch t.pool {
		case PlacementPoolHostRAM:
			r.hostBW, r.generic[rateHostBW] = t.bw, t.generic
		case PlacementPoolNVMe:
			r.nvmeBW, r.generic[rateNVMe] = t.bw, t.generic
		default:
			// Device-pool tiers are priced by effectiveRates above, not as offload tiers.
		}
	}
	if positiveFinite(ic.HostFLOPsPerSec) {
		r.hostF = ic.HostFLOPsPerSec
	}
	r.link = ic.HostLinkBytesPerSec
	if !positiveFinite(r.link) {
		r.link, r.generic[rateHostLink] = genericHostLinkBytesPerSec, true
	}
	r.p2p = ic.P2PBytesPerSec
	if !positiveFinite(r.p2p) {
		r.p2p, r.p2pViaLink = r.link, true
	}
	r.p2pLat = ic.P2PLatencySeconds
	if !positiveFinite(r.p2pLat) {
		r.p2pLat, r.generic[rateP2PLatency] = genericP2PLatencySeconds, true
	}
	r.sync = ic.HostSyncSeconds
	if !positiveFinite(r.sync) {
		r.sync, r.generic[rateHostSync] = genericHostSyncSeconds, true
	}
	return r
}

// splitGeom is the per-step demand a split distributes.
type splitGeom struct {
	dense, routed     int64   // resident weight bytes per segment
	denseStream       float64 // dense bytes read per step (batch-independent)
	uniqueRouted      float64 // unique routed bytes a batch-b step reads, all pools
	weightFLOPs       float64 // weight FLOPs of one step (2 x active params x b)
	attnFLOPs         float64 // attention FLOPs of one step; they run where the KV lives
	denseFLOPShare    float64 // fraction of weight FLOPs in the dense core
	kvTotal           int64   // KV + session state for every stream at the priced context
	kvReadPerStep     float64 // KV bytes one step reads (all batched streams)
	scratch           int64
	kvHeads, layers   int
	compressedKV      bool
	activationPerStep float64 // one hidden-state crossing for the batch, bytes
	batch             int     // sequences one step advances
	streams           int     // concurrent streams; > batch when decode is serial
}

// splitGeometry returns the demand, or an omission when the model's per-token stream or
// MoE active set is unknown: the same gate Plan applies before stating a rate.
func (p *planner) splitGeometry(c Candidate) (splitGeom, Omission) {
	m := p.in.Model
	g := splitGeom{batch: 1, streams: max(p.wl.Concurrency, 1)}
	if p.wl.BatchedDecode {
		g.batch = g.streams
	}
	if m.ActiveSetUnknown {
		return g, OmitMoEActiveSetUnknown
	}
	if m.StreamBytesPerToken <= 0 || !p.rhoOK {
		return g, OmitModelStreamUnknown
	}
	b := float64(g.batch)
	g.dense, g.routed = p.segmentBytes()
	g.denseStream = p.rho * float64(max(m.StreamBytesPerToken-max(m.ActiveRoutedBytesPerToken, 0), 0))
	activeRoutedPerTok := 0.0
	switch {
	case m.ActiveRoutedBytesPerToken > 0:
		activeRoutedPerTok = p.rho * float64(m.ActiveRoutedBytesPerToken)
	case g.routed > 0 && m.NumExperts > 0 && m.ExpertsUsed > 0:
		activeRoutedPerTok = float64(g.routed) * math.Min(float64(m.ExpertsUsed)/float64(m.NumExperts), 1)
	}
	if g.routed > 0 {
		if activeRoutedPerTok <= 0 {
			return g, OmitMoEActiveSetUnknown
		}
		// A batch reads the union of its tokens' experts under uniform routing. With expert
		// counts known this is Model.StreamBytes' rule, max(active, routed x (1-(1-k/E)^b)),
		// so every batch agrees with Plan; otherwise the declared active share stands in for k/E.
		if m.NumExperts > 0 && m.ExpertsUsed > 0 {
			u := uniqueFraction(float64(m.ExpertsUsed)/float64(m.NumExperts), g.batch)
			g.uniqueRouted = math.Max(activeRoutedPerTok, float64(g.routed)*u)
		} else {
			g.uniqueRouted = float64(g.routed) * uniqueFraction(activeRoutedPerTok/float64(g.routed), g.batch)
		}
	}
	if tot := g.denseStream + activeRoutedPerTok; tot > 0 {
		g.denseFLOPShare = g.denseStream / tot
	} else {
		g.denseFLOPShare = 1
	}
	lkv := min(p.wl.PromptTokens, c.ContextTokens)
	g.weightFLOPs = b * 2 * nonneg(m.ActiveParamsPerToken)
	g.attnFLOPs = b * 4 * p.attnWidth() * nonneg(int64(lkv))
	g.kvTotal = mulSat(perStreamBytes(c, c.ContextTokens), int64(g.streams))
	g.kvReadPerStep = b * float64(p.decodeKVRead(c, lkv))
	g.scratch = scratchBytes(c)
	g.kvHeads = c.Context.KV.NumKVHeads
	g.compressedKV = len(c.Context.CompressedKV) > 0
	g.layers = m.AttnLayers
	if g.layers <= 0 {
		g.layers = max(c.Context.KV.NumLayers, c.Context.Scratch.NumLayers)
	}
	hidden := c.Context.Scratch.HiddenSize
	if hidden <= 0 {
		hidden = m.NumHeads * m.HeadDim
	}
	g.activationPerStep = b * float64(max(hidden, 0)) * activationBytesPerElem
	return g, ""
}

// uniqueFraction is the expected share of routed experts a batch of b tokens touches when
// each token selects a fraction f uniformly: 1-(1-f)^b.
func uniqueFraction(f float64, b int) float64 {
	f = math.Min(math.Max(f, 0), 1)
	return 1 - math.Pow(1-f, float64(max(b, 1)))
}

// kvShards is how many ranks the KV divides over under tensor split.
func (g splitGeom) kvShards(n int) int {
	if g.compressedKV {
		return 1
	}
	if g.kvHeads > 0 {
		return max(min(n, g.kvHeads), 1)
	}
	return n
}

// MaxSplitDevices bounds Interconnect.Devices; a larger count is not priced (status unknown,
// split_devices_out_of_range), never silently clamped.
const MaxSplitDevices = 64

// Split omissions.
const (
	OmitSplitDevicesOutOfRange Omission = "split_devices_out_of_range"
	// OmitNoExecutableSplit: rows fit, but none is in Interconnect.Executable; Best names
	// the fastest fitting row.
	OmitNoExecutableSplit Omission = "no_executable_split"
)

// PlanSplit prices every split and offload mode for in.Hardware replicated
// ic.Devices times, at the reference candidate's context and in.Workload's batch.
func PlanSplit(in Input, ic Interconnect) SplitReport {
	p := newPlanner(in)
	n := max(ic.Devices, 1)
	ic.Devices = n
	rep := SplitReport{Status: StatusUnknown, Rows: []SplitRow{}, Interconnect: ic, Tier: in.Calibration.Tier}
	switch {
	case n > MaxSplitDevices:
		rep.Omitted = append(rep.Omitted, OmitSplitDevicesOutOfRange)
		return rep
	case !in.Calibration.valid():
		rep.Omitted = append(rep.Omitted, OmitCalibrationInvalid)
		return rep
	case p.avail <= 0:
		rep.Omitted = append(rep.Omitted, OmitPoolCapacityUnknown)
		return rep
	case len(in.Candidates) == 0:
		rep.Omitted = append(rep.Omitted, OmitContextUnknown)
		return rep
	}
	c := in.Candidates[referenceIndex(in.Candidates)]
	if !kvSizable(c.Context) || c.ContextTokens <= 0 {
		rep.Omitted = append(rep.Omitted, OmitKVGeometryUnknown)
		return rep
	}
	g, omit := p.splitGeometry(c)
	if omit != "" {
		rep.Omitted = append(rep.Omitted, omit)
		return rep
	}
	if addSat(g.dense, g.routed) <= 0 {
		rep.Omitted = append(rep.Omitted, OmitModelStreamUnknown)
		return rep
	}
	r := p.splitRates(ic)
	if !positiveFinite(r.devBW) {
		rep.Omitted = append(rep.Omitted, OmitPeakBandwidthUnknown)
		return rep
	}
	if in.Hardware.Topology == TopologyDiscrete && in.Hardware.HostPoolUsableBytes <= 0 {
		rep.Omitted = append(rep.Omitted, OmitHostPoolCapacityUnknown)
	}
	modes := []SplitMode{SplitSingle}
	if n >= 2 {
		modes = append(modes, SplitLayer, SplitTensor, SplitExpert)
	}
	exec := ic.Executable
	if exec == nil {
		exec = ServeExecutableSplits()
	}
	for _, mode := range modes {
		for _, off := range []OffloadMode{OffloadCPU, OffloadStream} {
			row := p.priceSplit(mode, off, n, g, r)
			for _, e := range exec {
				row.Executable = row.Executable || e == mode
			}
			if row.Offload == "" && off == OffloadStream {
				continue // nothing spills: the cpu/stream distinction is moot
			}
			rep.Rows = append(rep.Rows, row)
		}
	}
	order := map[SplitMode]int{SplitSingle: 0, SplitLayer: 1, SplitExpert: 2, SplitTensor: 3}
	sort.SliceStable(rep.Rows, func(i, j int) bool {
		a, b := rep.Rows[i], rep.Rows[j]
		if (a.Refusal == "") != (b.Refusal == "") {
			return a.Refusal == ""
		}
		if a.DecodeTokSAggregate != b.DecodeTokSAggregate {
			return a.DecodeTokSAggregate > b.DecodeTokSAggregate
		}
		return order[a.Mode] < order[b.Mode]
	})
	for i := range rep.Rows {
		row := rep.Rows[i]
		if row.Refusal != "" || row.DecodeTokSAggregate <= 0 {
			continue
		}
		if rep.Best == nil {
			b := row
			rep.Best = &b
		}
		if row.Executable {
			ch := row
			rep.Chosen = &ch
			break
		}
	}
	switch {
	case rep.Chosen != nil:
		rep.Status = StatusFits
		if rep.Chosen.HostBytes+rep.Chosen.NVMeBytes > 0 {
			rep.Status = StatusTiered
		}
		if rep.Best != nil && rep.Best.Mode == rep.Chosen.Mode && rep.Best.Offload == rep.Chosen.Offload {
			rep.Best = nil
		}
	case rep.Best != nil:
		rep.Omitted = append(rep.Omitted, OmitNoExecutableSplit) // stays StatusUnknown
	case len(rep.Rows) > 0:
		rep.Status = StatusRefused
	}
	// A rate priced on a generic default is neither speed-of-light nor calibrated evidence.
	top := rep.Chosen
	if top == nil {
		top = rep.Best
	}
	if top != nil && len(top.GenericRates) > 0 {
		rep.Tier = TierEmpirical
	}
	return rep
}

// priceSplit places and prices one (mode, offload) row.
func (p *planner) priceSplit(mode SplitMode, off OffloadMode, n int, g splitGeom, r splitRates) SplitRow {
	row := SplitRow{Mode: mode, Devices: n, Batch: g.batch}
	if mode == SplitSingle {
		n = 1
		row.Devices = 1
	}
	if n > 1 && p.in.Hardware.Topology != TopologyDiscrete {
		row.Refusal = SplitRefusalNotDiscrete
		return row
	}
	if mode == SplitExpert && g.routed <= 0 {
		row.Refusal = SplitRefusalNoRoutedExperts
		return row
	}
	used := map[string]bool{}
	use := func(tags ...string) {
		for _, t := range tags {
			if r.generic[t] {
				used[t] = true
			}
		}
	}
	devLeft := make([]int64, n)
	kvOn := make([]int64, n)
	kvRead := make([]float64, n)    // KV bytes each device reads per step
	attnShare := make([]float64, n) // fraction of attention FLOPs each device runs
	for i := range devLeft {
		devLeft[i] = p.avail
	}
	// KV and scratch first: they must be device-resident (the engine does not page KV).
	switch mode {
	case SplitSingle, SplitExpert:
		kvOn[0], kvRead[0], attnShare[0] = g.kvTotal, g.kvReadPerStep, 1
	case SplitLayer:
		for i := range kvOn {
			kvOn[i], kvRead[i], attnShare[i] = g.kvTotal/int64(n), g.kvReadPerStep/float64(n), 1/float64(n)
		}
	case SplitTensor:
		// a replicated KV is read in full by every rank holding a copy; heads still split
		sh := g.kvShards(n)
		for i := range kvOn {
			kvOn[i], kvRead[i], attnShare[i] = g.kvTotal/int64(sh), g.kvReadPerStep/float64(sh), 1/float64(n)
		}
	}
	for i := range devLeft {
		devLeft[i] = subSat(subSat(devLeft[i], g.scratch), kvOn[i])
		if devLeft[i] < 0 {
			row.Refusal = SplitRefusalKVExceedsDevice
			return row
		}
	}
	hostLeft := max(p.in.Hardware.HostPoolUsableBytes, 0)
	if p.in.Hardware.Topology != TopologyDiscrete {
		hostLeft = 0 // a shared pool has no separate host tier
	}
	// place spreads want evenly over devs, offers what a full rank could not hold to any
	// rank in devs with room left, then spills to host RAM and finally NVMe.
	place := func(want int64, on []int64, devs []int) (host, nvme int64) {
		per := want / int64(len(devs))
		rem := want - per*int64(len(devs))
		var over int64
		for k, d := range devs {
			w := per
			if k == 0 {
				w += rem
			}
			take := min(w, max(devLeft[d], 0))
			on[d] += take
			devLeft[d] -= take
			over += w - take
		}
		for _, d := range devs {
			if over <= 0 {
				break
			}
			take := min(over, max(devLeft[d], 0))
			on[d] += take
			devLeft[d] -= take
			over -= take
		}
		host = min(over, hostLeft)
		hostLeft -= host
		return host, over - host
	}
	all := make([]int, n)
	for i := range all {
		all[i] = i
	}
	// Dense core: device 0 only for single/expert, evenly for layer/tensor.
	denseOn := make([]int64, n)
	var denseHost, denseNVMe int64
	switch mode {
	case SplitSingle, SplitExpert:
		denseHost, denseNVMe = place(g.dense, denseOn, []int{0})
	default:
		denseHost, denseNVMe = place(g.dense, denseOn, all)
	}
	routedOn := make([]int64, n)
	routedHost, routedNVMe := place(g.routed, routedOn, all)

	row.DeviceBytes = make([]int64, n)
	for i := 0; i < n; i++ {
		row.DeviceBytes[i] = addSat(addSat(addSat(denseOn[i], routedOn[i]), kvOn[i]), g.scratch)
	}
	row.HostBytes = addSat(denseHost, routedHost)
	row.NVMeBytes = addSat(denseNVMe, routedNVMe)
	if row.HostBytes+row.NVMeBytes > 0 {
		row.Offload = off
	}

	share := func(on int64, total int64) float64 {
		if total <= 0 {
			return 0
		}
		return float64(on) / float64(total)
	}
	denseF, routedF := g.weightFLOPs*g.denseFLOPShare, g.weightFLOPs*(1-g.denseFLOPShare)
	devDenseB, devRoutedB := make([]float64, n), make([]float64, n)
	devDenseF, devRoutedF := make([]float64, n), make([]float64, n)
	for i := 0; i < n; i++ {
		ds, rs := share(denseOn[i], g.dense), share(routedOn[i], g.routed)
		devDenseB[i] = g.denseStream*ds + kvRead[i]
		devDenseF[i] = denseF*ds + g.attnFLOPs*attnShare[i]
		devRoutedB[i] = g.uniqueRouted * rs
		devRoutedF[i] = routedF * rs
	}
	// Weights below the devices; attention never runs there (the KV is device-resident).
	lowDenseB := g.denseStream * share(denseHost, g.dense)
	lowRoutedB := g.uniqueRouted * share(routedHost, g.routed)
	nvmeDenseB := g.denseStream * share(denseNVMe, g.dense)
	nvmeRoutedB := g.uniqueRouted * share(routedNVMe, g.routed)
	lowDenseF := denseF * share(denseHost+denseNVMe, g.dense)
	lowRoutedF := routedF * share(routedHost+routedNVMe, g.routed)

	roof := func(bytes, flops float64) (float64, string) {
		tb := bytes / r.devBW
		tf := 0.0
		if r.devF > 0 {
			tf = flops / r.devF
		}
		if tf > tb {
			return tf, BindDeviceCompute
		}
		return tb, BindDeviceBandwidth
	}
	var ph []SplitPhase
	add := func(name string, s float64, bind string) {
		if s > 0 {
			ph = append(ph, SplitPhase{Name: name, Seconds: s, Binding: bind})
		}
	}
	peer := func(name string, s float64) {
		add(name, s, BindP2P)
		use(rateP2PLatency)
		if r.p2pViaLink {
			use(rateHostLink)
		}
	}
	switch mode {
	case SplitSingle:
		s, b := roof(devDenseB[0]+devRoutedB[0], devDenseF[0]+devRoutedF[0])
		add("device0", s, b)
	case SplitLayer:
		for i := 0; i < n; i++ {
			s, b := roof(devDenseB[i]+devRoutedB[i], devDenseF[i]+devRoutedF[i])
			add("stage", s, b)
		}
		peer("stage_hops", float64(n-1)*(r.p2pLat+g.activationPerStep/r.p2p))
	case SplitTensor:
		worst, wb := 0.0, BindDeviceBandwidth
		for i := 0; i < n; i++ {
			if s, b := roof(devDenseB[i]+devRoutedB[i], devDenseF[i]+devRoutedF[i]); s > worst {
				worst, wb = s, b
			}
		}
		add("ranks", worst, wb)
		steps := 2 * float64(n-1)
		perAR := steps*r.p2pLat + steps/float64(n)*g.activationPerStep/r.p2p
		peer("all_reduce", 2*float64(g.layers)*perAR)
	case SplitExpert:
		// device 0 runs the dense core and attention; the routed experts then run on every
		// rank at once.
		s, b := roof(devDenseB[0], devDenseF[0])
		add("dense_device0", s, b)
		worst, wb := 0.0, BindDeviceBandwidth
		for i := 0; i < n; i++ {
			if s, b := roof(devRoutedB[i], devRoutedF[i]); s > worst {
				worst, wb = s, b
			}
		}
		add("expert_ranks", worst, wb)
		peer("all_to_all", 2*float64(g.layers)*(r.p2pLat+g.activationPerStep/r.p2p))
	}
	if lowDenseB+lowRoutedB+nvmeDenseB+nvmeRoutedB > 0 {
		// Each rank pulls its own share over its own link: the dense spill is sharded only
		// under tensor split, the routed spill under tensor and expert split. A layer split
		// streams stage by stage, one link at a time.
		denseLinks, routedLinks := 1.0, 1.0
		if mode == SplitTensor {
			denseLinks = float64(n)
		}
		if mode == SplitTensor || mode == SplitExpert {
			routedLinks = float64(n)
		}
		switch off {
		case OffloadCPU:
			th, tn := 0.0, 0.0
			if hb := lowDenseB + lowRoutedB; hb > 0 && positiveFinite(r.hostBW) {
				th = hb / r.hostBW
				use(rateHostBW)
			}
			if nb := nvmeDenseB + nvmeRoutedB; nb > 0 && positiveFinite(r.nvmeBW) {
				tn = nb / r.nvmeBW
				use(rateNVMe)
			}
			tf := 0.0
			if r.hostF > 0 {
				tf = (lowDenseF + lowRoutedF) / r.hostF
			}
			if tf > th+tn {
				add("host_compute", tf, BindHostCompute)
			} else {
				add("host_weights", th, BindHostBandwidth)
				add("nvme_weights", tn, BindNVMe)
			}
			// The fill spreads host experts over every MoE layer, so each layer crosses to
			// the host and back; a contiguous dense spill crosses once each way.
			crossings := 2.0
			if routedHost+routedNVMe > 0 {
				crossings = 2 * float64(g.layers)
			}
			add("host_sync", crossings*r.sync, BindSync)
			use(rateHostSync)
		case OffloadStream:
			stream := func(name string, bytes, links, tierBW float64, tierTag, tierBind string) {
				if bytes <= 0 {
					return
				}
				rate := math.Min(r.link*links, tierBW)
				if !positiveFinite(rate) {
					return
				}
				bind := BindHostLink
				if tierBW < r.link*links {
					bind = tierBind
					use(tierTag)
				} else {
					use(rateHostLink)
				}
				add(name, bytes/rate, bind)
			}
			stream("stream_host_dense", lowDenseB, denseLinks, r.hostBW, rateHostBW, BindHostBandwidth)
			stream("stream_host_experts", lowRoutedB, routedLinks, r.hostBW, rateHostBW, BindHostBandwidth)
			stream("stream_nvme_dense", nvmeDenseB, denseLinks, r.nvmeBW, rateNVMe, BindNVMe)
			stream("stream_nvme_experts", nvmeRoutedB, routedLinks, r.nvmeBW, rateNVMe, BindNVMe)
			if r.devF > 0 {
				add("streamed_compute", lowDenseF/(r.devF*denseLinks)+lowRoutedF/(r.devF*routedLinks), BindDeviceCompute)
			}
		}
	}
	add("overhead", p.in.Calibration.OverheadSeconds, BindOverhead)
	row.Phases = ph
	for _, t := range []string{rateHostBW, rateNVMe, rateHostLink, rateP2PLatency, rateHostSync} {
		if used[t] {
			row.GenericRates = append(row.GenericRates, t)
		}
	}
	total, dom := 0.0, SplitPhase{}
	for _, q := range ph {
		total += q.Seconds
		// The fixed per-step overhead is not a resource the split can move; it never binds.
		if q.Binding != BindOverhead && q.Seconds > dom.Seconds {
			dom = q
		}
	}
	if total > 0 {
		row.DecodeTokSAggregate = finiteOrZero(float64(g.batch) / total)
		// Serial decode (BatchedDecode off) time-slices one stream's steps across all
		// streams, as Plan prices it.
		row.DecodeTokSPerSeq = finiteOrZero(row.DecodeTokSAggregate / float64(g.streams))
		if g.batch == g.streams {
			row.DecodeTokSPerSeq = finiteOrZero(1 / total)
		}
		row.Binding = dom.Binding
		row.Bound = boundOf(dom.Binding)
	}
	return row
}

func boundOf(binding string) Bound {
	switch binding {
	case BindDeviceCompute, BindHostCompute:
		return BoundCompute
	case BindHostLink, BindP2P, BindNVMe, BindSync:
		return BoundLink
	default:
		return BoundMemory
	}
}
