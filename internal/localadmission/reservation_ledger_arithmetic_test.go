package localadmission

import (
	"bytes"
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms
func TestReservationLedgerArithmetic(t *testing.T) {
	for _, tt := range []struct {
		name      string
		held      []int64
		deadFirst bool
		wantError string
		wantTotal int64
		wantAdmit bool
	}{
		{"overflow", []int64{math.MaxInt64, math.MaxInt64}, false, "held-byte total overflows int64", 0, false},
		{"negative", []int64{-1, 20}, false, "negative held bytes", 0, false},
		{"exact_limit", []int64{math.MaxInt64 - 1, 1}, false, "", math.MaxInt64, false},
		{"ordinary", []int64{30, 20}, false, "", 50, true},
		{"dead_invalid_owner", []int64{-1, 20}, true, "", 20, true},
		{"dead_overflow_owner", []int64{math.MaxInt64, 20}, true, "", 20, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			store := NewReservationStore(t.TempDir())
			store.alive = func(pid int) bool { return !tt.deadFirst || pid != 1 }
			ledger := reservationLedger{Schema: reservationSchema}
			for i, held := range tt.held {
				ledger.Reservations = append(ledger.Reservations, Reservation{
					ID: string(rune('a' + i)), OwnerPID: i + 1,
					StartupPeakBytes: held, SteadyBytes: held, HeldBytes: held, Phase: "startup",
				})
			}
			if err := store.writeLedger(ledger); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(store.dir, "reservations.json")
			before, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			decision, err := store.Reserve(ctx, reservationRequest(99, 1, 1, 100, PressureNormal))
			if tt.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantError) || decision.Admit || decision.Reservation != nil {
					t.Fatalf("ledger admission admit=%v reservationPresent=%v err=%v; want arithmetic refusal", decision.Admit, decision.Reservation != nil, err)
				}
				if total, err := store.TotalReservedBytes(ctx); err == nil || total != 0 || !strings.Contains(err.Error(), tt.wantError) {
					t.Fatalf("invalid ledger total=%d err=%v", total, err)
				}
				after, err := os.ReadFile(path)
				if err != nil || !bytes.Equal(before, after) {
					t.Fatalf("refusal mutated live ledger: err=%v", err)
				}
				// Explicit release stays available to repair the bad live entry.
				if err := store.Release(ctx, ledger.Reservations[0].ID); err != nil {
					t.Fatal(err)
				}
				if total, err := store.TotalReservedBytes(ctx); err != nil || total != tt.held[1] {
					t.Fatalf("release recovery total=%d err=%v", total, err)
				}
				return
			}
			if err != nil || decision.Admit != tt.wantAdmit || decision.ReservedBytes != tt.wantTotal {
				t.Fatalf("valid ledger decision=%+v err=%v", decision, err)
			}
			wantTotal := tt.wantTotal
			if tt.wantAdmit {
				wantTotal++
			} else if decision.Reason != "aggregate_capacity" {
				t.Fatalf("exact-limit refusal reason=%q", decision.Reason)
			}
			if total, err := store.TotalReservedBytes(ctx); err != nil || total != wantTotal {
				t.Fatalf("valid total=%d want=%d err=%v", total, wantTotal, err)
			}
		})
	}
}
