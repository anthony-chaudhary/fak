package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/localadmission"
)

func TestServeReservationObservation(t *testing.T) {
	for _, name := range []string{"malformed_known_host", "malformed_unknown_host", "malformed_live_holder", "empty", "recovery"} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "reservations.json")
			store := localadmission.NewReservationStore(dir)
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			facts := serveHostFacts{TotalBytes: 100, ReservedBytes: 42}
			if name == "malformed_unknown_host" {
				facts.TotalBytes = 0
			}
			if name == "malformed_live_holder" {
				facts.GPULeaseHeld = true
				facts.GPULeaseHolderVerdict = string(gpulease.HolderProgressLiveProgressing)
				facts.GPULeaseHolderDetail = "fixture progress"
				facts.GPULeaseHolderPID = os.Getpid()
			}
			if name != "empty" {
				if err := os.WriteFile(path, []byte("{"), 0600); err != nil {
					t.Fatal(err)
				}
				_, expectedErr := store.TotalReservedBytes(ctx)
				if expectedErr == nil {
					t.Fatal("malformed ledger unexpectedly readable")
				}
				facts.observeReservations(ctx, store)
				row := serveReservationsRow(facts)
				if facts.ReservedBytesError != expectedErr.Error() || facts.ReservedBytes != 0 || row == nil || row.Status != sevWarn || row.Tier != "Marginal" || !strings.Contains(row.Finding, expectedErr.Error()) || !strings.Contains(row.Finding, "unknown") || strings.Contains(row.Finding, "zero active") {
					t.Fatalf("unreadable ledger facts=%+v row=%+v; want explicit unknown with exact error", facts, row)
				}
				if facts.GPULeaseHeld && (!strings.Contains(row.Finding, facts.GPULeaseHolderVerdict) || !strings.Contains(row.Finding, facts.GPULeaseHolderDetail) || !strings.Contains(row.Remediation, "progressing")) {
					t.Fatalf("known holder evidence lost: %+v", row)
				}
				if got, err := os.ReadFile(path); err != nil || string(got) != "{" {
					t.Fatalf("probe changed malformed ledger: %q %v", got, err)
				}
				if name != "recovery" {
					return
				}
				data, err := json.Marshal(struct {
					Schema       string                       `json:"schema"`
					Reservations []localadmission.Reservation `json:"reservations"`
				}{"fak-local-memory-reservations/2", []localadmission.Reservation{{ID: "live", OwnerPID: os.Getpid(), HeldBytes: 7}}})
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, data, 0600); err != nil {
					t.Fatal(err)
				}
				facts.observeReservations(ctx, store)
				row = serveReservationsRow(facts)
				if facts.ReservedBytesError != "" || facts.ReservedBytes != 7 || row == nil || row.Status != sevWarn || strings.Contains(row.Finding, "unknown") {
					t.Fatalf("healthy recovery facts=%+v row=%+v", facts, row)
				}
				if err := store.Release(ctx, "live"); err != nil {
					t.Fatal(err)
				}
			}
			facts.observeReservations(ctx, store)
			row := serveReservationsRow(facts)
			if facts.ReservedBytesError != "" || facts.ReservedBytes != 0 || row == nil || row.Status != sevOK || !strings.Contains(row.Finding, "zero active local memory reservations") {
				t.Fatalf("measured zero facts=%+v row=%+v", facts, row)
			}
		})
	}
}
