// Package microfleeteconomics deterministically accounts for the net physical
// cost of accepted work from high-cardinality model fleets.
package microfleeteconomics

import (
	"errors"
	"fmt"
	"math"
)

// Rates converts physical usage into integer micro-USD. Callers choose and
// record the rates; this package does not claim measured hardware performance.
type Rates struct {
	ComputeMicroUSDPerJoule       uint64 `json:"compute_micro_usd_per_joule"`
	MachineMicroUSDPerMillisecond uint64 `json:"machine_micro_usd_per_millisecond"`
	NetworkMicroUSDPerByte        uint64 `json:"network_micro_usd_per_byte"`
	StorageMicroUSDPerByteSecond  uint64 `json:"storage_micro_usd_per_byte_second"`
}

// PhysicalCost keeps resource quantities visible instead of hiding them in a
// token-only price. DirectMicroUSD covers externally priced costs that have no
// physical rate in the envelope.
type PhysicalCost struct {
	ComputeJoules      uint64 `json:"compute_joules"`
	MachineMillis      uint64 `json:"machine_millis"`
	NetworkBytes       uint64 `json:"network_bytes"`
	StorageByteSeconds uint64 `json:"storage_byte_seconds"`
	DirectMicroUSD     uint64 `json:"direct_micro_usd"`
}

// Ledger names every cost required for a net-true fan-out comparison.
type Ledger struct {
	UsefulWork        PhysicalCost `json:"useful_work"`
	Branching         PhysicalCost `json:"branching"`
	CacheConstruction PhysicalCost `json:"cache_construction"`
	CacheHits         PhysicalCost `json:"cache_hits"`
	QueueDelay        PhysicalCost `json:"queue_delay"`
	Verification      PhysicalCost `json:"verification"`
	FanIn             PhysicalCost `json:"fan_in"`
	Cancellation      PhysicalCost `json:"cancellation"`
	Failures          PhysicalCost `json:"failures"`
	Retries           PhysicalCost `json:"retries"`
	Recovery          PhysicalCost `json:"recovery"`
	Stragglers        PhysicalCost `json:"stragglers"`
}

// Operations makes the event denominator auditable alongside its costs.
type Operations struct {
	Branches           uint64 `json:"branches"`
	CacheConstructions uint64 `json:"cache_constructions"`
	CacheHits          uint64 `json:"cache_hits"`
	Verifications      uint64 `json:"verifications"`
	Cancellations      uint64 `json:"cancellations"`
	Failures           uint64 `json:"failures"`
	Retries            uint64 `json:"retries"`
	Recoveries         uint64 `json:"recoveries"`
	Stragglers         uint64 `json:"stragglers"`
}

// Receipt is one observed or synthetic accounting envelope. Quality is an
// integer score chosen by the experiment so comparisons do not depend on
// floating-point rounding.
type Receipt struct {
	Name         string     `json:"name"`
	Task         string     `json:"task"`
	Width        uint64     `json:"width"`
	Attempted    uint64     `json:"attempted"`
	Accepted     uint64     `json:"accepted"`
	QualityMilli uint64     `json:"quality_milli"`
	Operations   Operations `json:"operations"`
	Costs        Ledger     `json:"costs"`
}

// Result reports total physical usage and the accepted-work denominator. The
// exact cost ratio is TotalMicroUSD / Accepted; no attempted-work denominator
// is substituted.
type Result struct {
	Name                       string       `json:"name"`
	Width                      uint64       `json:"width"`
	Attempted                  uint64       `json:"attempted"`
	Accepted                   uint64       `json:"accepted"`
	Total                      PhysicalCost `json:"total"`
	TotalMicroUSD              uint64       `json:"total_micro_usd"`
	CostPerAcceptedNumerator   uint64       `json:"cost_per_accepted_numerator"`
	CostPerAcceptedDenominator uint64       `json:"cost_per_accepted_denominator"`
}

// Invariant: microfleet economics accounting is fail-closed and bounded.
// Any zero-accepted receipt, uncounted branch execution, or integer overflow
// fails closed with a typed refusal error rather than returning an invalid ratio.
var (
	// ErrNoAcceptedWork indicates that a receipt claimed zero accepted units of work.
	ErrNoAcceptedWork = errors.New("accepted work must be greater than zero")

	// ErrOverflow indicates that an accumulator or price computation exceeded uint64 bounds.
	ErrOverflow = errors.New("cost accounting overflow")
)

// Evaluate validates a receipt, sums every ledger category, and prices it.
func Evaluate(r Receipt, rates Rates) (Result, error) {
	if r.Name == "" || r.Task == "" {
		return Result{}, errors.New("name and task are required")
	}
	if r.Width == 0 || r.Attempted == 0 {
		return Result{}, errors.New("width and attempted work must be greater than zero")
	}
	if r.Accepted == 0 {
		return Result{}, ErrNoAcceptedWork
	}
	if r.Accepted > r.Attempted {
		return Result{}, errors.New("accepted work exceeds attempted work")
	}
	if r.Operations.Branches != r.Attempted {
		return Result{}, errors.New("branch count must equal attempted work")
	}

	total, err := sumLedger(r.Costs)
	if err != nil {
		return Result{}, err
	}
	priced, err := price(total, rates)
	if err != nil {
		return Result{}, err
	}
	return Result{
		Name: r.Name, Width: r.Width, Attempted: r.Attempted, Accepted: r.Accepted,
		Total: total, TotalMicroUSD: priced,
		CostPerAcceptedNumerator: priced, CostPerAcceptedDenominator: r.Accepted,
	}, nil
}

// Compare returns -1 when a is cheaper per accepted result, +1 when b is
// cheaper, and 0 on a tie. It refuses unmatched task or quality envelopes.
func Compare(a, b Receipt, rates Rates) (int, error) {
	if a.Task != b.Task || a.QualityMilli != b.QualityMilli {
		return 0, errors.New("receipts must have equal task and quality envelopes")
	}
	ra, err := Evaluate(a, rates)
	if err != nil {
		return 0, fmt.Errorf("evaluate %s: %w", a.Name, err)
	}
	rb, err := Evaluate(b, rates)
	if err != nil {
		return 0, fmt.Errorf("evaluate %s: %w", b.Name, err)
	}
	left, ok := mul(ra.TotalMicroUSD, rb.Accepted)
	if !ok {
		return 0, ErrOverflow
	}
	right, ok := mul(rb.TotalMicroUSD, ra.Accepted)
	if !ok {
		return 0, ErrOverflow
	}
	if left < right {
		return -1, nil
	}
	if left > right {
		return 1, nil
	}
	return 0, nil
}

func sumLedger(l Ledger) (PhysicalCost, error) {
	items := []PhysicalCost{l.UsefulWork, l.Branching, l.CacheConstruction, l.CacheHits, l.QueueDelay, l.Verification, l.FanIn, l.Cancellation, l.Failures, l.Retries, l.Recovery, l.Stragglers}
	var total PhysicalCost
	for _, item := range items {
		var ok bool
		if total.ComputeJoules, ok = add(total.ComputeJoules, item.ComputeJoules); !ok {
			return PhysicalCost{}, ErrOverflow
		}
		if total.MachineMillis, ok = add(total.MachineMillis, item.MachineMillis); !ok {
			return PhysicalCost{}, ErrOverflow
		}
		if total.NetworkBytes, ok = add(total.NetworkBytes, item.NetworkBytes); !ok {
			return PhysicalCost{}, ErrOverflow
		}
		if total.StorageByteSeconds, ok = add(total.StorageByteSeconds, item.StorageByteSeconds); !ok {
			return PhysicalCost{}, ErrOverflow
		}
		if total.DirectMicroUSD, ok = add(total.DirectMicroUSD, item.DirectMicroUSD); !ok {
			return PhysicalCost{}, ErrOverflow
		}
	}
	return total, nil
}

func price(c PhysicalCost, r Rates) (uint64, error) {
	pairs := [][2]uint64{{c.ComputeJoules, r.ComputeMicroUSDPerJoule}, {c.MachineMillis, r.MachineMicroUSDPerMillisecond}, {c.NetworkBytes, r.NetworkMicroUSDPerByte}, {c.StorageByteSeconds, r.StorageMicroUSDPerByteSecond}}
	total := c.DirectMicroUSD
	for _, pair := range pairs {
		v, ok := mul(pair[0], pair[1])
		if !ok {
			return 0, ErrOverflow
		}
		total, ok = add(total, v)
		if !ok {
			return 0, ErrOverflow
		}
	}
	return total, nil
}

func add(a, b uint64) (uint64, bool) {
	if math.MaxUint64-a < b {
		return 0, false
	}
	return a + b, true
}
func mul(a, b uint64) (uint64, bool) {
	if a != 0 && b > math.MaxUint64/a {
		return 0, false
	}
	return a * b, true
}
