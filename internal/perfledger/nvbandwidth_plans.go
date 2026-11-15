// Measurement-plan model for native memory-transport benchmarks, inspired by
// NVIDIA/nvbandwidth at 82fc4e8c6afa0babb8687793678f615b3b8d793e (Apache-2.0;
// see THIRD_PARTY_NOTICES.md). INSPIRE-ONLY: the named copy-plan taxonomy
// (copy initiator x access type x direction), the per-pair bandwidth matrix,
// and the moment-in-time receipt discipline are re-expressed as a pure-Go
// plan model. No CUDA kernels are ported and no physical number is claimed
// from software runs: the values carried here are plan-shape state, not
// measurements.
package perfledger

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"time"
)

// ErrUnknownCopyPlan is returned when a plan name or plan field is outside the
// closed nvbandwidth-inspired taxonomy.
var ErrUnknownCopyPlan = errors.New("perfledger: unknown copy plan")

// ErrInvalidMatrixCell is returned when a bandwidth sample cannot be recorded
// in a transport matrix.
var ErrInvalidMatrixCell = errors.New("perfledger: invalid matrix cell")

// Direction is one axis of the copy-plan taxonomy: which two memory domains a
// copy moves between.
type Direction string

const (
	DirectionD2D Direction = "device_to_device"
	DirectionH2D Direction = "host_to_device"
	DirectionD2H Direction = "device_to_host"
	DirectionH2H Direction = "host_to_host"
)

// allDirections is the closed direction set, in canonical order.
var allDirections = []Direction{DirectionD2D, DirectionH2D, DirectionD2H, DirectionH2H}

// valid reports whether d is a known direction.
func (d Direction) valid() bool {
	for _, known := range allDirections {
		if d == known {
			return true
		}
	}
	return false
}

// Initiator is the copy engine a plan nominates. The names mirror the
// nvbandwidth taxonomy (copy engine, SM, TMA) as plan labels only; nothing in
// this package executes a copy.
type Initiator string

const (
	InitiatorCE  Initiator = "ce"
	InitiatorSM  Initiator = "sm"
	InitiatorTMA Initiator = "tma"
)

// allInitiators is the closed initiator set, in canonical order.
var allInitiators = []Initiator{InitiatorCE, InitiatorSM, InitiatorTMA}

// valid reports whether i is a known initiator.
func (i Initiator) valid() bool {
	for _, known := range allInitiators {
		if i == known {
			return true
		}
	}
	return false
}

// AccessType is the memory-access side of the copy: the source is read or the
// destination is written as the measured operation.
type AccessType string

const (
	AccessRead  AccessType = "read"
	AccessWrite AccessType = "write"
)

// allAccessTypes is the closed access-type set, in canonical order.
var allAccessTypes = []AccessType{AccessRead, AccessWrite}

// valid reports whether a is a known access type.
func (a AccessType) valid() bool {
	for _, known := range allAccessTypes {
		if a == known {
			return true
		}
	}
	return false
}

// CopyPlan is one named memory-transport measurement plan: a direction, the
// copy initiator it nominates, and the access type it measures. A plan is a
// description of what a native benchmark would run, not a result.
type CopyPlan struct {
	Direction Direction
	Initiator Initiator
	Access    AccessType
}

// NewCopyPlan validates a plan against the closed taxonomy and returns it.
// Unknown fields fail closed with ErrUnknownCopyPlan.
func NewCopyPlan(dir Direction, init Initiator, access AccessType) (CopyPlan, error) {
	if !dir.valid() {
		return CopyPlan{}, fmt.Errorf("%w: direction %q", ErrUnknownCopyPlan, dir)
	}
	if !init.valid() {
		return CopyPlan{}, fmt.Errorf("%w: initiator %q", ErrUnknownCopyPlan, init)
	}
	if !access.valid() {
		return CopyPlan{}, fmt.Errorf("%w: access %q", ErrUnknownCopyPlan, access)
	}
	return CopyPlan{Direction: dir, Initiator: init, Access: access}, nil
}

// PlanName renders the canonical plan name: "<direction>_copy_<initiator>_<access>".
func (p CopyPlan) PlanName() string {
	return string(p.Direction) + "_copy_" + string(p.Initiator) + "_" + string(p.Access)
}

// KnownPlan parses a canonical plan name back into its plan. The direction is
// matched by prefix against the closed direction set (directions contain
// underscores, so the name is not split on "_"), then the remainder must be
// exactly "<initiator>_<access>". Unknown names fail closed with
// ErrUnknownCopyPlan.
func KnownPlan(name string) (CopyPlan, error) {
	for _, dir := range allDirections {
		prefix := string(dir) + "_copy_"
		if !strings.HasPrefix(name, prefix) {
			continue
		}
		parts := strings.Split(name[len(prefix):], "_")
		if len(parts) != 2 {
			return CopyPlan{}, fmt.Errorf("%w: %q", ErrUnknownCopyPlan, name)
		}
		return NewCopyPlan(dir, Initiator(parts[0]), AccessType(parts[1]))
	}
	return CopyPlan{}, fmt.Errorf("%w: %q", ErrUnknownCopyPlan, name)
}

// NamedCopyPlans returns the full closed plan set: every direction x
// initiator x access combination, in canonical order.
func NamedCopyPlans() []CopyPlan {
	plans := make([]CopyPlan, 0, len(allDirections)*len(allInitiators)*len(allAccessTypes))
	for _, dir := range allDirections {
		for _, init := range allInitiators {
			for _, access := range allAccessTypes {
				plans = append(plans, CopyPlan{Direction: dir, Initiator: init, Access: access})
			}
		}
	}
	return plans
}

// MatrixStats is the per-matrix summary a receipt carries: the aggregate
// bandwidth_description over every observed pair.
type MatrixStats struct {
	Sum     float64
	Max     float64
	Min     float64
	Average float64
}

// TransportMatrix is the per-pair bandwidth matrix for one plan:
// BW[src][dst] is the last recorded sample for that ordered device pair.
// Device 0 is the host, matching the nvbandwidth convention.
type TransportMatrix struct {
	Devices int
	BW      [][]float64
	// Samples keeps the run-to-run history per pair so the stability metric
	// (worst coefficient of variation) is computable from the matrix alone.
	Samples [][][]float64
}

// NewTransportMatrix returns an empty matrix for devices devices.
func NewTransportMatrix(devices int) (*TransportMatrix, error) {
	if devices < 1 {
		return nil, fmt.Errorf("%w: devices %d", ErrInvalidMatrixCell, devices)
	}
	bw := make([][]float64, devices)
	samples := make([][][]float64, devices)
	for i := range bw {
		bw[i] = make([]float64, devices)
		samples[i] = make([][]float64, devices)
	}
	return &TransportMatrix{Devices: devices, BW: bw, Samples: samples}, nil
}

// Set records one bandwidth sample (GB/s) for the ordered pair (src, dst).
// Out-of-range devices and non-finite or negative samples fail closed.
func (m *TransportMatrix) Set(src, dst int, bw float64) error {
	if m == nil {
		return fmt.Errorf("%w: nil matrix", ErrInvalidMatrixCell)
	}
	if src < 0 || src >= m.Devices || dst < 0 || dst >= m.Devices {
		return fmt.Errorf("%w: pair (%d,%d) outside %d devices", ErrInvalidMatrixCell, src, dst, m.Devices)
	}
	if math.IsNaN(bw) || math.IsInf(bw, 0) || bw < 0 {
		return fmt.Errorf("%w: bandwidth %v", ErrInvalidMatrixCell, bw)
	}
	m.BW[src][dst] = bw
	m.Samples[src][dst] = append(m.Samples[src][dst], bw)
	return nil
}

// Pair returns the last recorded sample for the ordered pair (src, dst) and
// whether it was observed.
func (m *TransportMatrix) Pair(src, dst int) (float64, bool) {
	if m == nil || src < 0 || src >= m.Devices || dst < 0 || dst >= m.Devices {
		return 0, false
	}
	return m.BW[src][dst], len(m.Samples[src][dst]) > 0
}

// Stats summarizes every observed pair in the matrix. An empty matrix yields
// the zero stats.
func (m *TransportMatrix) Stats() MatrixStats {
	var s MatrixStats
	if m == nil {
		return s
	}
	n := 0
	for i := 0; i < m.Devices; i++ {
		for j := 0; j < m.Devices; j++ {
			if len(m.Samples[i][j]) == 0 {
				continue
			}
			v := m.BW[i][j]
			s.Sum += v
			if n == 0 || v > s.Max {
				s.Max = v
			}
			if n == 0 || v < s.Min {
				s.Min = v
			}
			n++
		}
	}
	if n > 0 {
		s.Average = s.Sum / float64(n)
	}
	return s
}

// CoefficientOfVariation is the run-to-run dispersion of a sample set:
// standard deviation over the mean. Fewer than two samples or a zero mean
// yields NaN: dispersion is unmeasured, not zero.
func CoefficientOfVariation(samples []float64) float64 {
	if len(samples) < 2 {
		return math.NaN()
	}
	var sum float64
	for _, v := range samples {
		sum += v
	}
	mean := sum / float64(len(samples))
	if mean == 0 {
		return math.NaN()
	}
	var sq float64
	for _, v := range samples {
		d := v - mean
		sq += d * d
	}
	return math.Sqrt(sq/float64(len(samples))) / mean
}

// WorstCV is the worst (largest) run-to-run coefficient of variation across
// every observed pair: the stability bound a receipt stamps so a bandwidth
// regression stays distinguishable from noise. NaN when no pair has enough
// samples.
func (m *TransportMatrix) WorstCV() float64 {
	if m == nil {
		return math.NaN()
	}
	worst := math.NaN()
	for i := 0; i < m.Devices; i++ {
		for j := 0; j < m.Devices; j++ {
			cv := CoefficientOfVariation(m.Samples[i][j])
			if math.IsNaN(cv) {
				continue
			}
			if math.IsNaN(worst) || cv > worst {
				worst = cv
			}
		}
	}
	return worst
}

// ReceiptStatus is the per-plan outcome a receipt records.
type ReceiptStatus string

const (
	ReceiptPassed ReceiptStatus = "Passed"
	ReceiptError  ReceiptStatus = "Error"
	ReceiptWaived ReceiptStatus = "Waived"
)

// ReceiptEntry is one plan's result inside a receipt: its status, the
// moment-in-time it was recorded, and its matrix with the stability bound.
type ReceiptEntry struct {
	Plan      string        `json:"plan"`
	Status    ReceiptStatus `json:"status"`
	MomentMS  int64         `json:"moment_ms"`
	Matrix    *TransportMatrix
	Stats     MatrixStats `json:"stats"`
	WorstCV   float64     `json:"worst_cv,omitempty"`
}

// Receipt is the self-describing moment-in-time record for one benchmark
// run: the tool identity, the host, and one entry per plan recorded.
type Receipt struct {
	ToolVersion string         `json:"tool_version"`
	GitVersion  string         `json:"git_version,omitempty"`
	Hostname    string         `json:"hostname"`
	MomentMS    int64          `json:"moment_ms"`
	Entries     []ReceiptEntry `json:"entries"`
}

// NewReceipt returns an empty receipt stamped at now.
func NewReceipt(toolVersion, hostname string, now time.Time) *Receipt {
	return &Receipt{
		ToolVersion: toolVersion,
		Hostname:    hostname,
		MomentMS:    now.UnixMilli(),
	}
}

// Record appends one plan's result to the receipt, stamped at now. Unknown
// plans and nil matrices fail closed; the receipt is left unchanged.
func (r *Receipt) Record(plan CopyPlan, m *TransportMatrix, status ReceiptStatus, now time.Time) error {
	if r == nil {
		return fmt.Errorf("%w: nil receipt", ErrUnknownCopyPlan)
	}
	if _, err := KnownPlan(plan.PlanName()); err != nil {
		return err
	}
	if m == nil {
		return fmt.Errorf("%w: nil matrix for plan %q", ErrInvalidMatrixCell, plan.PlanName())
	}
	r.Entries = append(r.Entries, ReceiptEntry{
		Plan:     plan.PlanName(),
		Status:   status,
		MomentMS: now.UnixMilli(),
		Matrix:   m,
		Stats:    m.Stats(),
		WorstCV:  m.WorstCV(),
	})
	return nil
}

// Validate checks the receipt's moment-in-time discipline: the run moment and
// every entry moment must be set, and every entry must name a known plan.
func (r *Receipt) Validate() error {
	if r == nil {
		return fmt.Errorf("%w: nil receipt", ErrUnknownCopyPlan)
	}
	if r.MomentMS == 0 {
		return fmt.Errorf("%w: receipt moment not stamped", ErrUnknownCopyPlan)
	}
	for _, e := range r.Entries {
		if e.MomentMS == 0 {
			return fmt.Errorf("%w: entry %q moment not stamped", ErrUnknownCopyPlan, e.Plan)
		}
		if _, err := KnownPlan(e.Plan); err != nil {
			return err
		}
	}
	return nil
}
