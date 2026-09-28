package naivecontrol

import (
	"fmt"
	"strconv"
)

// MetricState is the closed vocabulary every recorded measure carries on the wire.
type MetricState string

const (
	// StateMeasured means Value holds a real observation, including an observed 0.
	StateMeasured MetricState = "MEASURED"
	// StateMissing means the field could not be read. Reason says why.
	StateMissing MetricState = "MISSING_MEASUREMENT"
)

// Unknown is how every surface renders a metric that is not MEASURED.
const Unknown = "UNKNOWN"

// Number is the value domain of a Metric.
type Number interface{ int64 | float64 }

// Metric is one measure that is either MEASURED (with a value, which may be 0) or
// MISSING_MEASUREMENT (with a reason and no value).
//
// The zero Metric, and a field absent from a decoded row, read as missing: a row
// written by an older recorder, or one that simply omitted a field, can never
// surface a 0 it did not observe.
type Metric[T Number] struct {
	State  MetricState `json:"state"`
	Value  *T          `json:"value,omitempty"`
	Reason string      `json:"reason,omitempty"`
}

// Measured builds a MEASURED metric.
func Measured[T Number](v T) Metric[T] {
	return Metric[T]{State: StateMeasured, Value: &v}
}

// Missing builds a MISSING_MEASUREMENT metric. An empty reason is replaced, so an
// unread field always says it was unread.
func Missing[T Number](reason string) Metric[T] {
	if reason == "" {
		reason = "not recorded"
	}
	return Metric[T]{State: StateMissing, Reason: reason}
}

// Get returns the value and true only for a well-formed MEASURED metric. Anything
// else, including a MEASURED state with no value, is not a measurement.
func (m Metric[T]) Get() (T, bool) {
	if m.State == StateMeasured && m.Value != nil {
		return *m.Value, true
	}
	var zero T
	return zero, false
}

// IsMeasured reports whether Get would succeed.
func (m Metric[T]) IsMeasured() bool {
	_, ok := m.Get()
	return ok
}

// WhyMissing is the recorded reason for an unmeasured metric, or "" when measured.
func (m Metric[T]) WhyMissing() string {
	if m.IsMeasured() {
		return ""
	}
	switch {
	case m.Reason != "":
		return m.Reason
	case m.State == "":
		return "field absent from the recorded row"
	case m.State == StateMeasured:
		return "recorded MEASURED with no value"
	}
	return "not recorded"
}

// String renders the value, or UNKNOWN. It never renders an unmeasured metric as 0.
func (m Metric[T]) String() string {
	v, ok := m.Get()
	if !ok {
		return Unknown
	}
	switch x := any(v).(type) {
	case int64:
		return strconv.FormatInt(x, 10)
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	}
	return fmt.Sprint(v)
}

// validate rejects a metric whose state contradicts its payload. An absent state is
// legal (it reads as missing); a negative value is not a count, time, or price.
func (m Metric[T]) validate(field string) error {
	switch m.State {
	case "":
		if m.Value != nil {
			return fmt.Errorf("%s: value without a state", field)
		}
	case StateMissing:
		if m.Value != nil {
			return fmt.Errorf("%s: MISSING_MEASUREMENT carries a value", field)
		}
	case StateMeasured:
		if m.Value == nil {
			return fmt.Errorf("%s: MEASURED without a value", field)
		}
		if *m.Value < 0 {
			return fmt.Errorf("%s: negative value %v", field, *m.Value)
		}
	default:
		return fmt.Errorf("%s: unknown state %q", field, m.State)
	}
	return nil
}
