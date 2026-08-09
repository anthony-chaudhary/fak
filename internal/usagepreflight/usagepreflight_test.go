package usagepreflight

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

type fakeReader struct {
	reading Reading
	err     error
	calls   int
}

func (f *fakeReader) Remaining(context.Context, string) (Reading, error) {
	f.calls++
	return f.reading, f.err
}

type fakeSelector struct {
	seat  string
	ok    bool
	calls int
}

func (f *fakeSelector) Alternate(context.Context, string) (string, bool) {
	f.calls++
	return f.seat, f.ok
}

type fakeRecorder struct{ records []Record }

func (f *fakeRecorder) RecordUsagePreflight(_ context.Context, r Record) {
	f.records = append(f.records, r)
}

func TestPolicyMatrix(t *testing.T) {
	tests := []struct {
		name       string
		reading    Reading
		readErr    error
		policy     Policy
		alternate  string
		wantAction Action
		wantSeat   string
		wantErr    error
		known      bool
	}{
		{"above reserve proceeds", Reading{Remaining: 21, Limit: 100, Known: true}, nil, PolicyAuto, "seat-b", ActionProceed, "seat-a", nil, true},
		{"reserve boundary auto switches", Reading{Remaining: 20, Limit: 100, Known: true}, nil, PolicyAuto, "seat-b", ActionSwitch, "seat-b", nil, true},
		{"depleted auto switches", Reading{Remaining: 0, Limit: 100, Known: true}, nil, PolicyAuto, "seat-b", ActionSwitch, "seat-b", nil, true},
		{"depleted confirm refuses", Reading{Remaining: 0, Limit: 100, Known: true}, nil, PolicyConfirm, "seat-b", ActionRefuse, "seat-a", ErrConfirmationRequired, true},
		{"boundary confirm refuses", Reading{Remaining: 20, Limit: 100, Known: true}, nil, PolicyConfirm, "seat-b", ActionRefuse, "seat-a", ErrConfirmationRequired, true},
		{"depleted fail closed", Reading{Remaining: 0, Limit: 100, Known: true}, nil, PolicyFailClosed, "seat-b", ActionRefuse, "seat-a", ErrReserveReached, true},
		{"boundary fail closed", Reading{Remaining: 20, Limit: 100, Known: true}, nil, PolicyFailClosed, "seat-b", ActionRefuse, "seat-a", ErrReserveReached, true},
		{"unknown fails open", Reading{}, nil, PolicyFailClosed, "seat-b", ActionProceed, "seat-a", nil, false},
		{"unreadable fails open", Reading{}, errors.New("usage unavailable"), PolicyFailClosed, "seat-b", ActionProceed, "seat-a", nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := &fakeReader{reading: tt.reading, err: tt.readErr}
			selector := &fakeSelector{seat: tt.alternate, ok: tt.alternate != ""}
			selected, got, err := (Hook{Config: Config{Enabled: true, ReservePercent: 20, Policy: tt.policy}, Reader: reader, Selector: selector}).Decide(context.Background(), "seat-a")
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("error = %v, want %v", err, tt.wantErr)
			}
			if selected != tt.wantSeat || got.Action != tt.wantAction || got.UsageKnown != tt.known {
				t.Fatalf("selected/record = %q/%+v", selected, got)
			}
		})
	}
}

func TestCallSwitchesBeforeSpendAndRecordsDecision(t *testing.T) {
	reader := &fakeReader{reading: Reading{Remaining: 5, Limit: 100, Known: true}}
	selector := &fakeSelector{seat: "seat-b", ok: true}
	recorder := &fakeRecorder{}
	var called []string
	hook := Hook{Config: Config{Enabled: true, ReservePercent: 10, Policy: PolicyAuto}, Reader: reader, Selector: selector, Recorder: recorder}
	if err := hook.Call(context.Background(), "seat-a", func(_ context.Context, seat string) error { called = append(called, seat); return nil }); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(called, []string{"seat-b"}) {
		t.Fatalf("provider calls = %v; original seat must never be called", called)
	}
	want := Record{Seat: "seat-a", SelectedSeat: "seat-b", Action: ActionSwitch, Policy: PolicyAuto, Remaining: 5, Limit: 100, ReservePercent: 10, UsageKnown: true}
	if !reflect.DeepEqual(recorder.records, []Record{want}) {
		t.Fatalf("records = %+v, want %+v", recorder.records, want)
	}
	if selector.calls != 1 {
		t.Fatalf("selector calls = %d", selector.calls)
	}
}

func TestDefaultOffIsTransparent(t *testing.T) {
	reader := &fakeReader{reading: Reading{Remaining: 0, Limit: 100, Known: true}}
	selector := &fakeSelector{seat: "seat-b", ok: true}
	recorder := &fakeRecorder{}
	calls := 0
	hook := Hook{Reader: reader, Selector: selector, Recorder: recorder}
	sentinel := errors.New("reactive path result")
	err := hook.Call(context.Background(), "seat-a", func(_ context.Context, seat string) error {
		calls++
		if seat != "seat-a" {
			t.Fatalf("seat = %q", seat)
		}
		return sentinel
	})
	if !errors.Is(err, sentinel) || calls != 1 || reader.calls != 0 || selector.calls != 0 || len(recorder.records) != 0 {
		t.Fatalf("default-off changed path: err=%v calls=%d reader=%d selector=%d records=%d", err, calls, reader.calls, selector.calls, len(recorder.records))
	}
}

func TestAutoWithoutAlternateRefusesWithoutCallingProvider(t *testing.T) {
	called := false
	hook := Hook{Config: Config{Enabled: true, ReservePercent: 10, Policy: PolicyAuto}, Reader: &fakeReader{reading: Reading{Remaining: 0, Limit: 100, Known: true}}, Selector: &fakeSelector{}}
	err := hook.Call(context.Background(), "seat-a", func(context.Context, string) error { called = true; return nil })
	if !errors.Is(err, ErrNoAlternateSeat) || called {
		t.Fatalf("err=%v called=%v", err, called)
	}
}

// Preflight selection only reads usage and selects a seat. It never calls the
// reactive cooldown/seatpark APIs, so it cannot double-count their backoff.
func TestSelectorIsOnlyBackoffInteraction(t *testing.T) {
	selector := &fakeSelector{seat: "seat-b", ok: true}
	hook := Hook{Config: Config{Enabled: true, ReservePercent: 50, Policy: PolicyAuto}, Reader: &fakeReader{reading: Reading{Remaining: 1, Limit: 100, Known: true}}, Selector: selector}
	if _, _, err := hook.Decide(context.Background(), "seat-a"); err != nil {
		t.Fatal(err)
	}
	if selector.calls != 1 {
		t.Fatalf("selector calls=%d; preflight must make exactly one non-mutating selection", selector.calls)
	}
}
