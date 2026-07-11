package resume

import "testing"

func TestDeriveWatchdogCap(t *testing.T) {
	tests := []struct {
		name                                                        string
		seats                                                       []HeadroomSeat
		floor, ceiling, seatCap, wantCap, wantHealthy, wantHeadroom int
	}{
		{"scales to headroom", []HeadroomSeat{{true, false, 1}, {true, false, 2}}, 4, 20, 5, 7, 2, 7},
		{"clamps ceiling", []HeadroomSeat{{true, false, 0}, {true, false, 0}}, 4, 6, 5, 6, 2, 10},
		{"ignores blocked seats", []HeadroomSeat{{false, false, 0}, {true, true, 0}}, 4, 20, 5, 4, 0, 0},
		{"floor when exhausted", []HeadroomSeat{{true, false, 5}}, 4, 20, 5, 4, 1, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DeriveWatchdogCap(tt.seats, tt.floor, tt.ceiling, tt.seatCap)
			if got.Cap != tt.wantCap || got.HealthySeats != tt.wantHealthy || got.Headroom != tt.wantHeadroom {
				t.Fatalf("got %+v", got)
			}
		})
	}
}
