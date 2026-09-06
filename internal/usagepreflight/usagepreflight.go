package usagepreflight

import (
	"context"
	"errors"
	"fmt"
)

// Policy controls behavior when known remaining quota is at or below Reserve.
type Policy string

const (
	// PolicyConfirm halts and requires operator confirmation when reserve is reached.
	PolicyConfirm Policy = "confirm"
	// PolicyAuto automatically selects an alternative seat when reserve is reached.
	PolicyAuto Policy = "auto"
	// PolicyFailClosed unconditionally refuses requests when reserve is reached.
	PolicyFailClosed Policy = "fail-closed"
)

// Action is the structured outcome emitted for every enabled preflight.
type Action string

const (
	// ActionProceed allows the request to continue on the requested seat.
	ActionProceed Action = "proceed"
	// ActionSwitch redirects the request to an alternate seat.
	ActionSwitch Action = "switch-seat"
	// ActionRefuse blocks the request from proceeding.
	ActionRefuse Action = "refuse"
)

var (
	// ErrConfirmationRequired indicates spend was refused pending confirmation.
	ErrConfirmationRequired = errors.New("usage preflight requires confirmation")
	// ErrReserveReached indicates spend was refused because reserve quota was reached.
	ErrReserveReached = errors.New("usage preflight reserve reached")
	// ErrNoAlternateSeat indicates an auto-switch failed because no candidate seat was found.
	ErrNoAlternateSeat = errors.New("usage preflight found no alternate seat")
)

// Reading is a provider's latest usage observation. Remaining and Limit use the
// same provider-defined unit. Known=false means unavailable or unreadable and
// must fail open.
type Reading struct {
	Remaining int64
	Limit     int64
	Known     bool
}

// Reader obtains a remaining-quota reading. Implementations may use an existing
// response cache or a provider usage endpoint. Reader errors fail open.
type Reader interface {
	Remaining(context.Context, string) (Reading, error)
}

// Selector chooses an eligible seat other than current. It must not mutate
// reactive cooldown or seatpark backoff state.
type Selector interface {
	Alternate(context.Context, string) (string, bool)
}

// Record makes preflight saves and refusals measurable.
type Record struct {
	Seat           string `json:"seat"`
	SelectedSeat   string `json:"selected_seat"`
	Action         Action `json:"action"`
	Policy         Policy `json:"policy"`
	Remaining      int64  `json:"remaining,omitempty"`
	Limit          int64  `json:"limit,omitempty"`
	ReservePercent int    `json:"reserve_percent"`
	UsageKnown     bool   `json:"usage_known"`
	UsageReadError bool   `json:"usage_read_error,omitempty"`
}

type Recorder interface{ RecordUsagePreflight(context.Context, Record) }

// Config is default-off. ReservePercent is inclusive: exactly the reserve
// boundary is treated as reserve reached.
type Config struct {
	Enabled        bool
	ReservePercent int
	Policy         Policy
}

// Hook is the outbound admission seam. Call invokes send exactly once for the
// admitted seat and never invokes it for a refused or replaced original seat.
type Hook struct {
	Config   Config
	Reader   Reader
	Selector Selector
	Recorder Recorder
}

func (h Hook) Call(ctx context.Context, seat string, send func(context.Context, string) error) error {
	selected, rec, err := h.Decide(ctx, seat)
	if h.Config.Enabled && h.Recorder != nil {
		h.Recorder.RecordUsagePreflight(ctx, rec)
	}
	if err != nil {
		return err
	}
	return send(ctx, selected)
}

// Decide evaluates quota against reserve and returns the chosen seat, a structured
// record, and any admission refusal.
func (h Hook) Decide(ctx context.Context, seat string) (string, Record, error) {
	rec := Record{Seat: seat, SelectedSeat: seat, Action: ActionProceed, Policy: h.Config.Policy, ReservePercent: h.Config.ReservePercent}
	if !h.Config.Enabled {
		return seat, rec, nil
	}
	if h.Reader == nil {
		return seat, rec, nil
	}

	reading, err := h.Reader.Remaining(ctx, seat)
	if err != nil || !reading.Known || reading.Limit <= 0 {
		rec.UsageReadError = err != nil
		return seat, rec, nil
	}
	rec.UsageKnown, rec.Remaining, rec.Limit = true, reading.Remaining, reading.Limit
	reserve := h.Config.ReservePercent
	if reserve < 0 {
		reserve = 0
	}
	if reserve > 100 {
		reserve = 100
	}
	// Multiplication avoids rounding ambiguity at the inclusive boundary.
	if reading.Remaining*100 > reading.Limit*int64(reserve) {
		return seat, rec, nil
	}

	switch h.Config.Policy {
	case PolicyAuto:
		if h.Selector != nil {
			if alternate, ok := h.Selector.Alternate(ctx, seat); ok && alternate != "" && alternate != seat {
				altReading, err := h.Reader.Remaining(ctx, alternate)
				if err == nil && altReading.Known && altReading.Limit > 0 {
					if altReading.Remaining*100 <= altReading.Limit*int64(reserve) {
						rec.Action = ActionRefuse
						return seat, rec, ErrNoAlternateSeat
					}
				}
				rec.Action, rec.SelectedSeat = ActionSwitch, alternate
				return alternate, rec, nil
			}
		}
		rec.Action = ActionRefuse
		return seat, rec, ErrNoAlternateSeat
	case PolicyConfirm:
		rec.Action = ActionRefuse
		return seat, rec, ErrConfirmationRequired
	case PolicyFailClosed:
		rec.Action = ActionRefuse
		return seat, rec, ErrReserveReached
	default:
		rec.Action = ActionRefuse
		return seat, rec, fmt.Errorf("unknown usage preflight policy %q", h.Config.Policy)
	}
}
