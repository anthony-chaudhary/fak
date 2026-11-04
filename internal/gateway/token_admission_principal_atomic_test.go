package gateway

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ratelimit"
)

// This is a deterministic ownership witness, not a scheduler-dependent reproduction:
// the public Admit must retain the book lock until the selected gate has a reservation.
// Otherwise an idle sweep can detach that gate in the selection-to-reservation gap.
// fak-test:runtime fast est=100ms lane=default
func TestPrincipalAdmissionOwnsBookThroughReservation(t *testing.T) {
	const window = time.Minute
	clock := time.Unix(1_700_000_000, 0)
	book := NewPrincipalTokenRates(TokenRatePolicy{
		Window: window,
		Caps:   ratelimit.TokenCaps{MaxTotalTokens: 100, MaxConcurrent: 1},
	}, 1)
	book.now = func() time.Time { return clock }

	seed, err := book.Admit("tenant-a", ratelimit.NewTokenUsage(1, 0, 0))
	if err != nil || seed == nil {
		t.Fatalf("seed Admit = (%v, %v), want reservation", seed, err)
	}
	t.Cleanup(seed.Release)
	seed.Settle(ratelimit.TokenUsage{})
	gate := seed.gate
	clock = clock.Add(window)
	if !gate.idle() {
		t.Fatal("fixture must begin with an expired, idle named gate eligible for sweeping")
	}

	// Install only after the gate exists. The clock runs inside gate.Admit, before
	// reservation insertion. With no other goroutines, TryLock identifies ownership
	// without scheduling luck. On the parent it succeeds; release it immediately.
	probeCalls := 0
	bookHeld := false
	gate.mu.Lock()
	gate.now = func() time.Time {
		if probeCalls == 0 {
			if book.mu.TryLock() {
				book.mu.Unlock()
			} else {
				bookHeld = true
			}
		}
		probeCalls++
		return clock
	}
	gate.mu.Unlock()
	held, err := book.Admit(" tenant-a ", ratelimit.NewTokenUsage(60, 0, 0))
	if err != nil || held == nil {
		t.Fatalf("existing principal Admit = (%v, %v), want reservation", held, err)
	}
	t.Cleanup(held.Release)
	if probeCalls != 1 || !bookHeld {
		t.Errorf("reservation probe calls=%d book-held=%v, want 1 and true", probeCalls, bookHeld)
	}
	if !book.mu.TryLock() {
		t.Fatal("Admit returned without releasing the book lock")
	}
	retained := book.book[keyForPrincipal("tenant-a")]
	book.mu.Unlock()
	if held.gate != gate || retained != gate {
		t.Fatal("reservation must belong to the named gate retained in the book")
	}
	if got := book.SnapshotFor("tenant-a"); got.InFlight != 1 || got.Reserved.TotalTokens() != 60 || got.Settled.TotalTokens() != 0 {
		t.Fatalf("reserved snapshot = %+v, want one 60-token reservation and no settled usage", got)
	}

	// Even after another window passes, an active reservation prevents eviction.
	// A fresh principal must use overflow, and tenant-a must retain its own cap.
	clock = clock.Add(window)
	overflow, err := book.Admit("tenant-b", ratelimit.NewTokenUsage(10, 0, 0))
	if err != nil || overflow == nil {
		t.Fatalf("overflow Admit = (%v, %v), want reservation", overflow, err)
	}
	t.Cleanup(overflow.Release)
	if overflow.gate == gate || overflow.gate.subject != "shared overflow principal" {
		t.Fatal("full book must send a fresh principal to the separate overflow gate")
	}
	if named, total := book.AllotmentCounts(); named != 1 || total != 2 {
		t.Fatalf("held-window counts = (%d, %d), want (1, 2)", named, total)
	}
	if extra, err := book.Admit("tenant-a", ratelimit.NewTokenUsage(1, 0, 0)); err == nil || extra != nil {
		extra.Release()
		t.Fatalf("second concurrent Admit = (%v, %v), want refusal", extra, err)
	}
	held.Settle(ratelimit.NewTokenUsage(60, 0, 0))
	held.Release() // No double charge after settlement.
	if got := book.SnapshotFor("tenant-a"); got.InFlight != 0 || got.Reserved.TotalTokens() != 0 || got.Settled.TotalTokens() != 60 {
		t.Fatalf("settled snapshot = %+v, want 60 settled and no reservation", got)
	}
	extra, err := book.Admit("tenant-a", ratelimit.NewTokenUsage(41, 0, 0))
	var admissionErr *AdmissionError
	if extra != nil || !errors.As(err, &admissionErr) || admissionErr.Verdict != VerdictShed || !strings.Contains(admissionErr.Reason, ratelimit.TokenCapTotalTokens) {
		extra.Release()
		t.Fatalf("spent-window Admit = (%v, %v), want typed total-token shed", extra, err)
	}
	if !book.mu.TryLock() {
		t.Fatal("refused Admit returned without releasing the book lock")
	}
	book.mu.Unlock()

	// Once the reservation is settled and the window expires, ordinary safe
	// eviction still works; the permanent overflow bucket stays in the book.
	overflow.Settle(ratelimit.NewTokenUsage(10, 0, 0))
	clock = clock.Add(window)
	next, err := book.Admit("tenant-c", ratelimit.NewTokenUsage(100, 0, 0))
	if err != nil || next == nil {
		t.Fatalf("post-sweep Admit = (%v, %v), want fresh named reservation", next, err)
	}
	t.Cleanup(next.Release)
	if next.gate == gate || next.gate == overflow.gate || next.gate.subject != `principal "tenant-c"` {
		t.Fatal("expired idle slot must be reclaimed for a fresh named gate")
	}
	if named, total := book.AllotmentCounts(); named != 1 || total != 2 {
		t.Fatalf("post-sweep counts = (%d, %d), want (1, 2)", named, total)
	}
	if got := book.SnapshotFor("tenant-a"); got != (TokenRateSnapshot{}) {
		t.Fatalf("evicted tenant snapshot = %+v, want zero", got)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestPrincipalAdmissionNilBookStillUnmetered(t *testing.T) {
	var book *PrincipalTokenRates
	if reservation, err := book.Admit("tenant-a", ratelimit.NewTokenUsage(100, 0, 0)); reservation != nil || err != nil {
		t.Fatalf("nil book Admit = (%v, %v), want (nil, nil)", reservation, err)
	}
}
