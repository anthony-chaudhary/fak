// Package placementtax compares a candidate compute placement with an explicit,
// quality-matched reference without collapsing latency, throughput, money, energy,
// or capacity into one score.
package placementtax

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// Provenance says whether a modeled component came from an estimate or a
// measurement. ProvenanceMixed is output-only and appears when an aggregate contains
// both kinds of evidence.
type Provenance string

const (
	ProvenanceEstimated Provenance = "estimated"
	ProvenanceMeasured  Provenance = "measured"
	ProvenanceMixed     Provenance = "mixed"
)

// ModeledValue distinguishes an observed or estimated zero from an unmodeled
// dimension. Unmodeled values must leave Value at zero.
type ModeledValue struct {
	Value   float64
	Modeled bool
}

// Capacity is the minimum resource envelope checked before any placement metric is
// reported. ComputeUnits is deliberately abstract: callers choose a stable unit
// appropriate to their planner, while MemoryBytes has an unambiguous physical unit.
type Capacity struct {
	MemoryBytes  uint64
	ComputeUnits float64
}

// QualityEnvelope holds the workload axes that must remain fixed while comparing
// placements. Target is a caller-defined acceptance condition such as an exact model
// digest, a maximum perplexity delta, or a task-quality threshold.
type QualityEnvelope struct {
	ModelID        string
	Precision      string
	SequenceLength int
	BatchSize      int
	Target         string
}

// Workload is the matched unit of useful work. Units divided by a placement's
// steady-state cycle time produces throughput in Unit per second.
type Workload struct {
	ID      string
	Units   float64
	Unit    string
	Quality QualityEnvelope
}

// Domain is a locality or failure domain. Examples include one unified-memory host,
// one NUMA island, or one rack.
type Domain struct {
	ID string
}

// Node is one placement target within a domain.
type Node struct {
	ID       string
	DomainID string
	Capacity Capacity
}

// Link is one directed, finite-capacity communication path. Its alpha-beta transfer
// time is Messages*Latency + Bytes/BandwidthBytesPerSecond.
type Link struct {
	ID                      string
	FromNode                string
	ToNode                  string
	Latency                 time.Duration
	BandwidthBytesPerSecond float64
	MonetaryUSDPerByte      ModeledValue
	EnergyJoulesPerByte     ModeledValue
}

// Topology is the placement substrate shared by the candidate and reference.
type Topology struct {
	Domains []Domain
	Nodes   []Node
	Links   []Link
}

// Allocation places a resource demand on one node.
type Allocation struct {
	NodeID string
	Demand Capacity
}

// ComponentCost is a component's contribution to request latency and to the
// steady-state cycle time used for throughput. Monetary and energy values remain
// independent and optional.
type ComponentCost struct {
	Latency      time.Duration
	Cycle        time.Duration
	MonetaryUSD  ModeledValue
	EnergyJoules ModeledValue
	Provenance   Provenance
}

// Overlap records communication time hidden behind useful work. Latency and Cycle
// are separate because a pipeline can hide communication in steady state without
// hiding it from an individual request.
type Overlap struct {
	Latency time.Duration
	Cycle   time.Duration
}

// Transfer is one communication event on an explicit direct link.
type Transfer struct {
	LinkID     string
	Messages   uint64
	Bytes      uint64
	Overlap    Overlap
	Provenance Provenance
}

// Placement is an explicit candidate or reference plan. Communication is derived
// from Transfers and Links; the other components are supplied by a measurement or a
// planner estimate.
type Placement struct {
	Name                  string
	Allocations           []Allocation
	UsefulCompute         ComponentCost
	Transfers             []Transfer
	Synchronization       ComponentCost
	ImbalanceStraggler    ComponentCost
	DataMovement          ComponentCost
	OrchestrationRecovery ComponentCost
}

// Comparison names the one workload envelope and the two placements being compared.
// Sharing Workload and Topology by construction prevents an unmatched reference from
// silently licensing ratios.
type Comparison struct {
	Workload  Workload
	Topology  Topology
	Candidate Placement
	Reference Placement
}

// ComponentKind is the closed component-ledger vocabulary.
type ComponentKind string

const (
	ComponentUsefulCompute         ComponentKind = "useful_compute"
	ComponentExposedCommunication  ComponentKind = "exposed_communication"
	ComponentSynchronization       ComponentKind = "synchronization"
	ComponentImbalanceStraggler    ComponentKind = "imbalance_straggler"
	ComponentDataMovement          ComponentKind = "replication_migration_remote_memory_paging"
	ComponentOrchestrationRecovery ComponentKind = "orchestration_recovery"
)

// LedgerEntry is one non-overlapping contribution to placement metrics.
type LedgerEntry struct {
	Kind ComponentKind
	Cost ComponentCost
}

// CommunicationRecord makes overlap auditable. Hidden time is excluded from the
// component ledger, while monetary and energy costs still charge the full transfer.
type CommunicationRecord struct {
	LinkID         string
	FromNode       string
	ToNode         string
	Messages       uint64
	Bytes          uint64
	RawLatency     time.Duration
	HiddenLatency  time.Duration
	ExposedLatency time.Duration
	RawCycle       time.Duration
	HiddenCycle    time.Duration
	ExposedCycle   time.Duration
	MonetaryUSD    ModeledValue
	EnergyJoules   ModeledValue
	Provenance     Provenance
}

// CapacityCheck is the feasibility result for one allocated node.
type CapacityCheck struct {
	NodeID    string
	Demand    Capacity
	Available Capacity
	Feasible  bool
}

// Feasibility is evaluated before component metrics.
type Feasibility struct {
	Feasible bool
	Checks   []CapacityCheck
	Reasons  []string
}

// Metrics preserves the separate placement outputs. Throughput is Workload.Units
// divided by Cycle and is labeled with ThroughputUnit.
type Metrics struct {
	Latency        time.Duration
	Cycle          time.Duration
	Throughput     float64
	ThroughputUnit string
	MonetaryUSD    ModeledValue
	EnergyJoules   ModeledValue
}

// Evaluation is one placement's feasibility and, only when feasible, its ledger and
// metrics.
type Evaluation struct {
	Placement     string
	Feasibility   Feasibility
	Ledger        []LedgerEntry
	Communication []CommunicationRecord
	Metrics       *Metrics
}

// Delta is candidate minus reference for every modeled dimension. A negative latency,
// monetary, or energy delta favors the candidate; a positive throughput delta does.
type Delta struct {
	Latency      time.Duration
	Throughput   float64
	MonetaryUSD  ModeledValue
	EnergyJoules ModeledValue
}

// RelativeMetric reports a penalty ratio and placement efficiency for one dimension.
// Availability is false when the dimension is unmodeled or either side is zero.
type RelativeMetric struct {
	PenaltyRatio        float64
	PlacementEfficiency float64
	Available           bool
}

// RelativeMetrics keeps each ratio independent. For lower-is-better dimensions,
// penalty is candidate/reference and efficiency is reference/candidate. Throughput
// uses the inverse directions.
type RelativeMetrics struct {
	Latency      RelativeMetric
	Throughput   RelativeMetric
	MonetaryUSD  RelativeMetric
	EnergyJoules RelativeMetric
}

// Report is the complete comparison. Delta and Relative are nil unless both
// placements are feasible, so an impossible candidate or reference never yields a
// ratio.
type Report struct {
	Workload  Workload
	Candidate Evaluation
	Reference Evaluation
	Delta     *Delta
	Relative  *RelativeMetrics
}

// Analyze validates the matched comparison, checks capacity feasibility, builds the
// component ledgers, and computes deltas and ratios only when both placements are
// feasible.
func Analyze(c Comparison) (Report, error) {
	if err := validateWorkload(c.Workload); err != nil {
		return Report{}, err
	}
	nodes, links, err := validateTopology(c.Topology)
	if err != nil {
		return Report{}, err
	}
	if err := validatePlacement("candidate", c.Candidate, nodes, links); err != nil {
		return Report{}, err
	}
	if err := validatePlacement("reference", c.Reference, nodes, links); err != nil {
		return Report{}, err
	}
	if c.Candidate.Name == c.Reference.Name {
		return Report{}, fmt.Errorf("candidate and reference placements must have distinct names")
	}

	candidate, err := evaluate(c.Workload, c.Candidate, nodes, links)
	if err != nil {
		return Report{}, fmt.Errorf("candidate %q: %w", c.Candidate.Name, err)
	}
	reference, err := evaluate(c.Workload, c.Reference, nodes, links)
	if err != nil {
		return Report{}, fmt.Errorf("reference %q: %w", c.Reference.Name, err)
	}

	report := Report{
		Workload:  c.Workload,
		Candidate: candidate,
		Reference: reference,
	}
	if !candidate.Feasibility.Feasible || !reference.Feasibility.Feasible {
		return report, nil
	}

	cm, rm := candidate.Metrics, reference.Metrics
	report.Delta = &Delta{
		Latency:      cm.Latency - rm.Latency,
		Throughput:   cm.Throughput - rm.Throughput,
		MonetaryUSD:  subtractModeled(cm.MonetaryUSD, rm.MonetaryUSD),
		EnergyJoules: subtractModeled(cm.EnergyJoules, rm.EnergyJoules),
	}
	report.Relative = &RelativeMetrics{
		Latency:      lowerIsBetter(float64(cm.Latency), float64(rm.Latency)),
		Throughput:   higherIsBetter(cm.Throughput, rm.Throughput),
		MonetaryUSD:  relativeModeled(cm.MonetaryUSD, rm.MonetaryUSD, false),
		EnergyJoules: relativeModeled(cm.EnergyJoules, rm.EnergyJoules, false),
	}
	return report, nil
}

func validateWorkload(w Workload) error {
	if err := nonblankID("workload id", w.ID); err != nil {
		return err
	}
	if !positiveFinite(w.Units) {
		return fmt.Errorf("workload units must be finite and greater than zero")
	}
	if err := nonblankID("workload unit", w.Unit); err != nil {
		return err
	}
	if err := nonblankID("quality model id", w.Quality.ModelID); err != nil {
		return err
	}
	if err := nonblankID("quality precision", w.Quality.Precision); err != nil {
		return err
	}
	if w.Quality.SequenceLength <= 0 {
		return fmt.Errorf("quality sequence length must be greater than zero")
	}
	if w.Quality.BatchSize <= 0 {
		return fmt.Errorf("quality batch size must be greater than zero")
	}
	if err := nonblankID("quality target", w.Quality.Target); err != nil {
		return err
	}
	return nil
}

func validateTopology(t Topology) (map[string]Node, map[string]Link, error) {
	if len(t.Domains) == 0 {
		return nil, nil, fmt.Errorf("topology must declare at least one domain")
	}
	domains := make(map[string]bool, len(t.Domains))
	for i, d := range t.Domains {
		if err := nonblankID(fmt.Sprintf("domain[%d] id", i), d.ID); err != nil {
			return nil, nil, err
		}
		if domains[d.ID] {
			return nil, nil, fmt.Errorf("duplicate domain id %q", d.ID)
		}
		domains[d.ID] = true
	}

	if len(t.Nodes) == 0 {
		return nil, nil, fmt.Errorf("topology must declare at least one node")
	}
	nodes := make(map[string]Node, len(t.Nodes))
	for i, n := range t.Nodes {
		if err := nonblankID(fmt.Sprintf("node[%d] id", i), n.ID); err != nil {
			return nil, nil, err
		}
		if nodes[n.ID].ID != "" {
			return nil, nil, fmt.Errorf("duplicate node id %q", n.ID)
		}
		if !domains[n.DomainID] {
			return nil, nil, fmt.Errorf("node %q names unknown domain %q", n.ID, n.DomainID)
		}
		if !nonnegativeFinite(n.Capacity.ComputeUnits) {
			return nil, nil, fmt.Errorf("node %q compute capacity must be finite and non-negative", n.ID)
		}
		if n.Capacity.MemoryBytes == 0 && n.Capacity.ComputeUnits == 0 {
			return nil, nil, fmt.Errorf("node %q must expose memory or compute capacity", n.ID)
		}
		nodes[n.ID] = n
	}

	links := make(map[string]Link, len(t.Links))
	for i, l := range t.Links {
		if err := nonblankID(fmt.Sprintf("link[%d] id", i), l.ID); err != nil {
			return nil, nil, err
		}
		if links[l.ID].ID != "" {
			return nil, nil, fmt.Errorf("duplicate link id %q", l.ID)
		}
		if nodes[l.FromNode].ID == "" {
			return nil, nil, fmt.Errorf("link %q names unknown source node %q", l.ID, l.FromNode)
		}
		if nodes[l.ToNode].ID == "" {
			return nil, nil, fmt.Errorf("link %q names unknown destination node %q", l.ID, l.ToNode)
		}
		if l.FromNode == l.ToNode {
			return nil, nil, fmt.Errorf("link %q must connect distinct nodes", l.ID)
		}
		if l.Latency < 0 {
			return nil, nil, fmt.Errorf("link %q latency must be non-negative", l.ID)
		}
		if !positiveFinite(l.BandwidthBytesPerSecond) {
			return nil, nil, fmt.Errorf("link %q bandwidth must be finite and greater than zero", l.ID)
		}
		if err := validateModeledValue("link "+l.ID+" monetary USD per byte", l.MonetaryUSDPerByte); err != nil {
			return nil, nil, err
		}
		if err := validateModeledValue("link "+l.ID+" energy joules per byte", l.EnergyJoulesPerByte); err != nil {
			return nil, nil, err
		}
		links[l.ID] = l
	}
	return nodes, links, nil
}

func validatePlacement(role string, p Placement, nodes map[string]Node, links map[string]Link) error {
	if err := nonblankID(role+" placement name", p.Name); err != nil {
		return err
	}
	if len(p.Allocations) == 0 {
		return fmt.Errorf("%s placement %q must declare at least one allocation", role, p.Name)
	}
	allocated := make(map[string]bool, len(p.Allocations))
	hasDemand := false
	for i, a := range p.Allocations {
		if nodes[a.NodeID].ID == "" {
			return fmt.Errorf("%s placement %q allocation[%d] names unknown node %q", role, p.Name, i, a.NodeID)
		}
		if allocated[a.NodeID] {
			return fmt.Errorf("%s placement %q allocates node %q more than once", role, p.Name, a.NodeID)
		}
		if !nonnegativeFinite(a.Demand.ComputeUnits) {
			return fmt.Errorf("%s placement %q node %q compute demand must be finite and non-negative", role, p.Name, a.NodeID)
		}
		if a.Demand.MemoryBytes > 0 || a.Demand.ComputeUnits > 0 {
			hasDemand = true
		}
		allocated[a.NodeID] = true
	}
	if !hasDemand {
		return fmt.Errorf("%s placement %q must demand memory or compute capacity", role, p.Name)
	}

	if err := validateComponent(role, p.Name, string(ComponentUsefulCompute), p.UsefulCompute, true); err != nil {
		return err
	}
	for _, item := range []struct {
		kind ComponentKind
		cost ComponentCost
	}{
		{ComponentSynchronization, p.Synchronization},
		{ComponentImbalanceStraggler, p.ImbalanceStraggler},
		{ComponentDataMovement, p.DataMovement},
		{ComponentOrchestrationRecovery, p.OrchestrationRecovery},
	} {
		if err := validateComponent(role, p.Name, string(item.kind), item.cost, false); err != nil {
			return err
		}
	}

	for i, tr := range p.Transfers {
		link, ok := links[tr.LinkID]
		if !ok {
			return fmt.Errorf("%s placement %q transfer[%d] names unknown link %q", role, p.Name, i, tr.LinkID)
		}
		if !allocated[link.FromNode] || !allocated[link.ToNode] {
			return fmt.Errorf("%s placement %q transfer[%d] link %q endpoints must both be allocated", role, p.Name, i, tr.LinkID)
		}
		if tr.Messages == 0 && tr.Bytes == 0 {
			return fmt.Errorf("%s placement %q transfer[%d] must contain a message or bytes", role, p.Name, i)
		}
		if tr.Overlap.Latency < 0 || tr.Overlap.Cycle < 0 {
			return fmt.Errorf("%s placement %q transfer[%d] overlap must be non-negative", role, p.Name, i)
		}
		if !validInputProvenance(tr.Provenance) {
			return fmt.Errorf("%s placement %q transfer[%d] provenance must be %q or %q", role, p.Name, i, ProvenanceEstimated, ProvenanceMeasured)
		}
	}
	return nil
}

func validateComponent(role, placement, name string, c ComponentCost, required bool) error {
	if c.Latency < 0 || c.Cycle < 0 {
		return fmt.Errorf("%s placement %q component %q durations must be non-negative", role, placement, name)
	}
	if required && (c.Latency <= 0 || c.Cycle <= 0) {
		return fmt.Errorf("%s placement %q useful compute latency and cycle must be greater than zero", role, placement)
	}
	if err := validateModeledValue(role+" placement "+placement+" component "+name+" monetary USD", c.MonetaryUSD); err != nil {
		return err
	}
	if err := validateModeledValue(role+" placement "+placement+" component "+name+" energy joules", c.EnergyJoules); err != nil {
		return err
	}
	active := required || c.Latency > 0 || c.Cycle > 0 || c.MonetaryUSD.Value > 0 || c.EnergyJoules.Value > 0
	if active && !validInputProvenance(c.Provenance) {
		return fmt.Errorf("%s placement %q component %q provenance must be %q or %q", role, placement, name, ProvenanceEstimated, ProvenanceMeasured)
	}
	return nil
}

func evaluate(w Workload, p Placement, nodes map[string]Node, links map[string]Link) (Evaluation, error) {
	e := Evaluation{
		Placement: p.Name,
		Feasibility: Feasibility{
			Feasible: true,
			Checks:   make([]CapacityCheck, 0, len(p.Allocations)),
		},
	}
	for _, a := range p.Allocations {
		available := nodes[a.NodeID].Capacity
		ok := a.Demand.MemoryBytes <= available.MemoryBytes &&
			a.Demand.ComputeUnits <= available.ComputeUnits
		e.Feasibility.Checks = append(e.Feasibility.Checks, CapacityCheck{
			NodeID:    a.NodeID,
			Demand:    a.Demand,
			Available: available,
			Feasible:  ok,
		})
		if !ok {
			e.Feasibility.Feasible = false
			e.Feasibility.Reasons = append(e.Feasibility.Reasons, fmt.Sprintf(
				"node %q demand exceeds capacity: memory %d/%d bytes, compute %.6g/%.6g units",
				a.NodeID, a.Demand.MemoryBytes, available.MemoryBytes,
				a.Demand.ComputeUnits, available.ComputeUnits))
		}
	}
	if !e.Feasibility.Feasible {
		return e, nil
	}

	communication, communicationCost, err := evaluateCommunication(p.Transfers, links)
	if err != nil {
		return Evaluation{}, err
	}
	e.Communication = communication
	e.Ledger = []LedgerEntry{
		{Kind: ComponentUsefulCompute, Cost: normalizeEmptyCost(p.UsefulCompute)},
		{Kind: ComponentExposedCommunication, Cost: communicationCost},
		{Kind: ComponentSynchronization, Cost: normalizeEmptyCost(p.Synchronization)},
		{Kind: ComponentImbalanceStraggler, Cost: normalizeEmptyCost(p.ImbalanceStraggler)},
		{Kind: ComponentDataMovement, Cost: normalizeEmptyCost(p.DataMovement)},
		{Kind: ComponentOrchestrationRecovery, Cost: normalizeEmptyCost(p.OrchestrationRecovery)},
	}

	var latency, cycle time.Duration
	costs := make([]ModeledValue, 0, len(e.Ledger))
	energies := make([]ModeledValue, 0, len(e.Ledger))
	for _, entry := range e.Ledger {
		latency, err = addDuration(latency, entry.Cost.Latency)
		if err != nil {
			return Evaluation{}, fmt.Errorf("latency ledger overflow: %w", err)
		}
		cycle, err = addDuration(cycle, entry.Cost.Cycle)
		if err != nil {
			return Evaluation{}, fmt.Errorf("cycle ledger overflow: %w", err)
		}
		costs = append(costs, entry.Cost.MonetaryUSD)
		energies = append(energies, entry.Cost.EnergyJoules)
	}
	if latency <= 0 || cycle <= 0 {
		return Evaluation{}, fmt.Errorf("component ledger must produce positive latency and cycle")
	}
	e.Metrics = &Metrics{
		Latency:        latency,
		Cycle:          cycle,
		Throughput:     w.Units / cycle.Seconds(),
		ThroughputUnit: w.Unit + "/s",
		MonetaryUSD:    sumModeled(costs),
		EnergyJoules:   sumModeled(energies),
	}
	return e, nil
}

func evaluateCommunication(transfers []Transfer, links map[string]Link) ([]CommunicationRecord, ComponentCost, error) {
	if len(transfers) == 0 {
		return nil, emptyCost(), nil
	}
	records := make([]CommunicationRecord, 0, len(transfers))
	var exposedLatency, exposedCycle time.Duration
	costs := make([]ModeledValue, 0, len(transfers))
	energies := make([]ModeledValue, 0, len(transfers))
	var provenance Provenance

	for _, tr := range transfers {
		link := links[tr.LinkID]
		raw, err := transferDuration(link, tr)
		if err != nil {
			return nil, ComponentCost{}, fmt.Errorf("transfer on link %q: %w", tr.LinkID, err)
		}
		hiddenLatency := minDuration(raw, tr.Overlap.Latency)
		hiddenCycle := minDuration(raw, tr.Overlap.Cycle)
		rec := CommunicationRecord{
			LinkID:         link.ID,
			FromNode:       link.FromNode,
			ToNode:         link.ToNode,
			Messages:       tr.Messages,
			Bytes:          tr.Bytes,
			RawLatency:     raw,
			HiddenLatency:  hiddenLatency,
			ExposedLatency: raw - hiddenLatency,
			RawCycle:       raw,
			HiddenCycle:    hiddenCycle,
			ExposedCycle:   raw - hiddenCycle,
			MonetaryUSD:    multiplyModeled(link.MonetaryUSDPerByte, float64(tr.Bytes)),
			EnergyJoules:   multiplyModeled(link.EnergyJoulesPerByte, float64(tr.Bytes)),
			Provenance:     tr.Provenance,
		}
		exposedLatency, err = addDuration(exposedLatency, rec.ExposedLatency)
		if err != nil {
			return nil, ComponentCost{}, fmt.Errorf("exposed communication latency overflow: %w", err)
		}
		exposedCycle, err = addDuration(exposedCycle, rec.ExposedCycle)
		if err != nil {
			return nil, ComponentCost{}, fmt.Errorf("exposed communication cycle overflow: %w", err)
		}
		costs = append(costs, rec.MonetaryUSD)
		energies = append(energies, rec.EnergyJoules)
		provenance = mergeProvenance(provenance, tr.Provenance)
		records = append(records, rec)
	}
	return records, ComponentCost{
		Latency:      exposedLatency,
		Cycle:        exposedCycle,
		MonetaryUSD:  sumModeled(costs),
		EnergyJoules: sumModeled(energies),
		Provenance:   provenance,
	}, nil
}

func transferDuration(link Link, tr Transfer) (time.Duration, error) {
	seconds := float64(tr.Messages)*link.Latency.Seconds() +
		float64(tr.Bytes)/link.BandwidthBytesPerSecond
	nanos := seconds * float64(time.Second)
	if math.IsNaN(nanos) || math.IsInf(nanos, 0) || nanos > float64(math.MaxInt64) {
		return 0, fmt.Errorf("alpha-beta duration is not representable")
	}
	return time.Duration(math.Round(nanos)), nil
}

func emptyCost() ComponentCost {
	return ComponentCost{
		MonetaryUSD:  ModeledValue{Modeled: true},
		EnergyJoules: ModeledValue{Modeled: true},
	}
}

func normalizeEmptyCost(c ComponentCost) ComponentCost {
	if c.Latency == 0 && c.Cycle == 0 &&
		!c.MonetaryUSD.Modeled && !c.EnergyJoules.Modeled &&
		c.Provenance == "" {
		return emptyCost()
	}
	return c
}

func sumModeled(values []ModeledValue) ModeledValue {
	var total float64
	for _, v := range values {
		if !v.Modeled {
			return ModeledValue{}
		}
		total += v.Value
	}
	return ModeledValue{Value: total, Modeled: true}
}

func subtractModeled(candidate, reference ModeledValue) ModeledValue {
	if !candidate.Modeled || !reference.Modeled {
		return ModeledValue{}
	}
	return ModeledValue{Value: candidate.Value - reference.Value, Modeled: true}
}

func multiplyModeled(rate ModeledValue, multiplier float64) ModeledValue {
	if !rate.Modeled {
		if multiplier == 0 {
			return ModeledValue{Modeled: true}
		}
		return ModeledValue{}
	}
	return ModeledValue{Value: rate.Value * multiplier, Modeled: true}
}

func relativeModeled(candidate, reference ModeledValue, higherBetter bool) RelativeMetric {
	if !candidate.Modeled || !reference.Modeled {
		return RelativeMetric{}
	}
	if higherBetter {
		return higherIsBetter(candidate.Value, reference.Value)
	}
	return lowerIsBetter(candidate.Value, reference.Value)
}

func lowerIsBetter(candidate, reference float64) RelativeMetric {
	if candidate <= 0 || reference <= 0 {
		return RelativeMetric{}
	}
	return RelativeMetric{
		PenaltyRatio:        candidate / reference,
		PlacementEfficiency: reference / candidate,
		Available:           true,
	}
}

func higherIsBetter(candidate, reference float64) RelativeMetric {
	if candidate <= 0 || reference <= 0 {
		return RelativeMetric{}
	}
	return RelativeMetric{
		PenaltyRatio:        reference / candidate,
		PlacementEfficiency: candidate / reference,
		Available:           true,
	}
}

func validateModeledValue(name string, value ModeledValue) error {
	if !value.Modeled && value.Value != 0 {
		return fmt.Errorf("%s has a value but is not marked modeled", name)
	}
	if value.Modeled && !nonnegativeFinite(value.Value) {
		return fmt.Errorf("%s must be finite and non-negative", name)
	}
	return nil
}

func validInputProvenance(p Provenance) bool {
	return p == ProvenanceEstimated || p == ProvenanceMeasured
}

func mergeProvenance(a, b Provenance) Provenance {
	if a == "" {
		return b
	}
	if a == b {
		return a
	}
	return ProvenanceMixed
}

func nonblankID(name, value string) error {
	if strings.TrimSpace(value) == "" {
		return fmt.Errorf("%s must not be blank", name)
	}
	if value != strings.TrimSpace(value) {
		return fmt.Errorf("%s %q must not have surrounding whitespace", name, value)
	}
	return nil
}

func positiveFinite(v float64) bool {
	return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0)
}

func nonnegativeFinite(v float64) bool {
	return v >= 0 && !math.IsNaN(v) && !math.IsInf(v, 0)
}

func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}

func addDuration(a, b time.Duration) (time.Duration, error) {
	if b > 0 && a > time.Duration(math.MaxInt64)-b {
		return 0, fmt.Errorf("duration exceeds time.Duration")
	}
	return a + b, nil
}
