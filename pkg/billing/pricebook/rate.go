package pricebook

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// RateStatus is the closed vocabulary describing how a token rate was
// established on a price card. It is the rate's lineage: it fixes whether
// a valuation may use the value, so "unpriced" can never be mistaken for
// "priced at zero".
type RateStatus string

const (
	// RatePriced means the operator declared an explicit rate. Value
	// carries it; a declared zero is legal and distinct from unpriced.
	RatePriced RateStatus = "priced"

	// RateUnpriced means no rate is declared for this line. Value must
	// be zero. A valuation that needs this rate yields an unknown cost;
	// it must never substitute zero.
	RateUnpriced RateStatus = "unpriced"
)

// RateStatuses is the closed, ordered rate-status vocabulary.
var RateStatuses = [...]RateStatus{RatePriced, RateUnpriced}

// Valid reports whether s is a member of the closed rate-status
// vocabulary.
func (s RateStatus) Valid() bool {
	for _, known := range RateStatuses {
		if s == known {
			return true
		}
	}
	return false
}

// Rate is one token-rate entry on a price card: a status plus, when
// priced, its value in the card's currency per the card's units.
//
// The wire form keeps the distinction structural: a priced rate encodes
// as a JSON number and an unpriced rate as JSON null, so decoding can
// never turn an absent rate into a priced zero.
type Rate struct {
	Status RateStatus
	Value  float64
}

// PricedRate returns a Rate with Status RatePriced carrying value.
func PricedRate(value float64) Rate {
	return Rate{Status: RatePriced, Value: value}
}

// UnpricedRate returns the Rate for a line with no declared rate.
func UnpricedRate() Rate {
	return Rate{Status: RateUnpriced}
}

// IsUnpriced reports whether the rate is unpriced.
func (r Rate) IsUnpriced() bool {
	return r.Status == RateUnpriced
}

// Valid checks the closed contract: a known status and a finite,
// non-negative value consistent with it.
func (r Rate) Valid() error {
	if !r.Status.Valid() {
		return fmt.Errorf("%w: %q", ErrInvalidRate, r.Status)
	}
	switch r.Status {
	case RatePriced:
		if math.IsNaN(r.Value) || math.IsInf(r.Value, 0) {
			return fmt.Errorf("%w: non-finite value", ErrInvalidRate)
		}
		if r.Value < 0 {
			return fmt.Errorf("%w: negative value", ErrInvalidRate)
		}
	case RateUnpriced:
		if r.Value != 0 {
			return fmt.Errorf("%w: unpriced rate carries a value", ErrInvalidRate)
		}
	default:
		// Unreachable once Status.Valid() passed; guard the switch so a
		// future vocabulary addition cannot silently skip the rules.
		return fmt.Errorf("%w: unhandled status %q", ErrInvalidPriceBook, r.Status)
	}
	return nil
}

// MarshalJSON encodes a priced rate as its number and an unpriced rate as
// null. It refuses to encode a rate that fails Valid.
func (r Rate) MarshalJSON() ([]byte, error) {
	if err := r.Valid(); err != nil {
		return nil, err
	}
	if r.Status != RatePriced {
		return []byte("null"), nil
	}
	return strconv.AppendFloat(nil, r.Value, 'f', -1, 64), nil
}

// UnmarshalJSON decodes a JSON number into a priced rate and JSON null
// into an unpriced rate. Anything else is refused.
func (r *Rate) UnmarshalJSON(data []byte) error {
	s := strings.TrimSpace(string(data))
	if s == "null" {
		*r = UnpricedRate()
		return nil
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return fmt.Errorf("%w: %q is not a rate", ErrInvalidRate, s)
	}
	*r = PricedRate(v)
	return nil
}
