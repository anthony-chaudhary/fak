package pricebook

// CostState is the closed vocabulary describing how a cost figure was
// established. It is the figure's lineage: it fixes whether the amount may
// be used, so "unknown" can never be mistaken for "known zero".
type CostState string

const (
	// CostKnown means the figure was computed from fully known inputs.
	// Amount carries it; a known zero is legal (for example a declared
	// zero count against any rate, or a declared zero rate).
	CostKnown CostState = "known"

	// CostUnknown means at least one required input was unknown, so no
	// figure can be established. Amount must be zero. A consumer must
	// treat the cost as unknown; what to do about an unknown cost is a
	// decision this package does not make.
	CostUnknown CostState = "unknown"
)

// CostStates is the closed, ordered cost-state vocabulary.
var CostStates = [...]CostState{CostKnown, CostUnknown}

// Valid reports whether s is a member of the closed cost-state vocabulary.
func (s CostState) Valid() bool {
	for _, known := range CostStates {
		if s == known {
			return true
		}
	}
	return false
}

// Cost is a typed monetary figure: a state plus, when known, its amount in
// the price book's currency. It is an analytical quantity, not a
// settlement amount.
type Cost struct {
	State  CostState
	Amount float64
}

// KnownCost returns a Cost with State CostKnown carrying amount.
func KnownCost(amount float64) Cost {
	return Cost{State: CostKnown, Amount: amount}
}

// ZeroCost returns the known-zero Cost.
func ZeroCost() Cost {
	return Cost{State: CostKnown, Amount: 0}
}

// UnknownCost returns the Cost for a figure that cannot be established from
// the supplied inputs.
func UnknownCost() Cost {
	return Cost{State: CostUnknown}
}

// IsKnown reports whether the figure was established from fully known
// inputs.
func (c Cost) IsKnown() bool {
	return c.State == CostKnown
}

// Add returns the sum of two figures. An unknown operand leaves the sum
// unknown: unknown is never treated as known.
func (c Cost) Add(other Cost) Cost {
	if c.State != CostKnown || other.State != CostKnown {
		return UnknownCost()
	}
	return KnownCost(c.Amount + other.Amount)
}
