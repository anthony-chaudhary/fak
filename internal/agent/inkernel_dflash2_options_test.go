package agent

import (
	"context"
	"errors"
	"runtime"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type dflash2DiagnosticScript struct{ *dflash2ScriptRequest }

func (s *dflash2DiagnosticScript) PrepareTimed(context.Context, int) (*model.Qwen38DFlash2Round, model.DFlash2PrepareTiming, error) {
	return nil, model.DFlash2PrepareTiming{}, errors.New("script does not fabricate native rounds")
}
func (s *dflash2DiagnosticScript) CommittedPosition(context.Context) (int, error) { return 1, nil }

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2TypedOptionsRefuseBeforeAcquisition(t *testing.T) {
	_, target := dflash2ServingFixture(t)
	calls := 0
	poison := func(context.Context, []ggufload.ShardIdentity, string) (*ggufload.WeightSource, error) {
		calls++
		panic("unexpected I/O")
	}
	cases := []DFlash2TargetOptions{
		{Draft: DFlash2ResidentMLP},
		{Draft: 255},
		{Request: model.Qwen38DFlash2RequestOptions{TargetPanel: 255}},
		{Request: model.Qwen38DFlash2RequestOptions{TargetPanel: model.DFlash2TargetPanel4}},
		{Request: model.Qwen38DFlash2RequestOptions{TargetPanel: model.DFlash2TargetPanel8}},
	}
	for _, o := range cases {
		_, err := newDFlash2TargetWithOptions(context.Background(), dflash2ValidSpec(), target.backend, o, poison)
		if err == nil {
			t.Fatal("unsupported startup selection admitted")
		}
		if o.Draft == DFlash2ResidentMLP && !errors.Is(err, ErrDFlash2ResidentAdmissionUnavailable) {
			t.Fatal(err)
		}
	}
	if calls != 0 {
		t.Fatalf("early refusal opened %d sources", calls)
	}
	if err := (DFlash2TargetOptions{}).validate(nil); err != nil {
		t.Fatal("zero options changed default admission")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2DiagnosticLeaseBoundary(t *testing.T) {
	p, target := dflash2ServingFixture(t)
	r := &dflash2DiagnosticScript{&dflash2ScriptRequest{}}
	target.factory = func(_ context.Context, l *dflash2Lease) error { l.request = r; return nil }
	q, err := target.OpenDiagnosticRequest(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = q.Close() })
	if len(target.leases) != 1 {
		t.Fatal("request not published to target ledger")
	}
	if _, err = p.acquireDFlash2Domain(context.Background()); !errors.Is(err, ErrDFlash2Busy) {
		t.Fatalf("ordinary alias %v", err)
	}
	if err = target.Close(); !errors.Is(err, ErrDFlash2Busy) {
		t.Fatalf("target retired live request: %v", err)
	}
	if err = q.Bootstrap(context.Background(), []int{0}); err != nil || r.bootstraps != 1 {
		t.Fatal("diagnostic dispatch missing")
	}
	if _, err = q.CommittedPosition(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err = q.Close(); err != nil {
		t.Fatal(err)
	}
	if err = q.Close(); err != nil || r.closes != 1 || len(target.leases) != 0 {
		t.Fatal("retirement was not once-only")
	}
	if err = q.Bootstrap(context.Background(), []int{0}); err == nil {
		t.Fatal("retired wrapper usable")
	}
	release, err := p.acquireDFlash2Domain(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	release()
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2DiagnosticIncompleteCloseRetainsGate(t *testing.T) {
	for _, kind := range []string{"error", "panic", "nil", "legacy-nil", "goexit"} {
		t.Run(kind, func(t *testing.T) {
			if kind == "nil" {
				t.Setenv("GODEBUG", "panicnil=0")
			}
			if kind == "legacy-nil" {
				t.Setenv("GODEBUG", "panicnil=1")
			}
			p, target := dflash2ServingFixture(t)
			cause := errors.New("incomplete diagnostic close")
			r := &dflash2DiagnosticScript{&dflash2ScriptRequest{closeFault: func() error {
				switch kind {
				case "error":
					return cause
				case "panic":
					panic(cause)
				case "nil", "legacy-nil":
					panic(nil)
				default:
					runtime.Goexit()
				}
				return nil
			}}}
			target.factory = func(_ context.Context, l *dflash2Lease) error { l.request = r; return nil }
			q, err := target.OpenDiagnosticRequest(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			done := make(chan struct{})
			returned := false
			go func() { defer close(done); defer func() { _ = recover() }(); _ = q.Close(); returned = true }()
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("close did not unwind")
			}
			if kind != "error" && returned {
				t.Fatal("interruption returned normally")
			}
			if q.Close() == nil || len(target.leases) != 1 || r.closes != 1 || len(target.domain.retained) == 0 {
				t.Fatal("incomplete owner lost")
			}
			if _, err = p.acquireDFlash2Domain(context.Background()); !errors.Is(err, ErrDFlash2Busy) {
				t.Fatalf("gate released after incomplete close: %v", err)
			}
		})
	}
}
