package resume

// HeadroomSeat is the minimum live-seat evidence needed to size a watchdog tick.
type HeadroomSeat struct {
	Available      bool `json:"available"`
	Throttled      bool `json:"throttled"`
	ActiveSessions int  `json:"active_sessions"`
}

type WatchdogCap struct {
	Cap          int `json:"cap"`
	Floor        int `json:"floor"`
	Ceiling      int `json:"ceiling"`
	SeatCap      int `json:"seat_cap"`
	HealthySeats int `json:"healthy_seats"`
	Headroom     int `json:"headroom"`
}

// DeriveWatchdogCap scales recovery launches to currently healthy seat headroom.
// With no headroom it deliberately retains floor: this preserves today's bounded
// retry behavior while the per-launch resolver continues to refuse blocked seats.
func DeriveWatchdogCap(seats []HeadroomSeat, floor, ceiling, seatCap int) WatchdogCap {
	if floor < 1 {
		floor = 1
	}
	if ceiling < floor {
		ceiling = floor
	}
	if seatCap < 1 {
		seatCap = 1
	}
	out := WatchdogCap{Floor: floor, Ceiling: ceiling, SeatCap: seatCap}
	for _, seat := range seats {
		if !seat.Available || seat.Throttled {
			continue
		}
		out.HealthySeats++
		active := seat.ActiveSessions
		if active < 0 {
			active = 0
		}
		if active < seatCap {
			out.Headroom += seatCap - active
		}
	}
	out.Cap = out.Headroom
	if out.Cap < floor {
		out.Cap = floor
	}
	if out.Cap > ceiling {
		out.Cap = ceiling
	}
	return out
}
