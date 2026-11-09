package servelaunch

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
)

// recordingProber answers per-port from a table and records every port it was
// asked about, so tests can assert exactly which ports the scan touched.
type recordingProber struct {
	free    map[int]bool
	err     map[int]error
	attempt []int
}

func (p *recordingProber) probe(port int) (bool, error) {
	p.attempt = append(p.attempt, port)
	if err, ok := p.err[port]; ok {
		return false, err
	}
	return p.free[port], nil
}

var errProbe = errors.New("probe failed")

// TestPortClaimAcceptsOnlyPositiveConfirmation proves the fail-closed core:
// conclusive-held and inconclusive ports are both skipped, and only the port
// the prober positively confirms free is claimed — with the scan stopping at
// the claim.
// fak-test:runtime fast est=1ms lane=default
func TestPortClaimAcceptsOnlyPositiveConfirmation(t *testing.T) {
	p := &recordingProber{
		free: map[int]bool{8000: false, 8002: true},
		err:  map[int]error{8001: errors.New("probe timed out")},
	}
	out, err := PortClaim(ClaimOptions{Start: 8000, Window: 4, Prober: p.probe})
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if out.Port != 8002 {
		t.Errorf("Port = %d, want 8002 (the only positive confirmation)", out.Port)
	}
	if want := []int{8000, 8001, 8002}; !reflect.DeepEqual(p.attempt, want) {
		t.Errorf("probed ports = %v, want %v (scan stops at the claim)", p.attempt, want)
	}
	if want := []int{8000}; !reflect.DeepEqual(out.Busy, want) {
		t.Errorf("Busy = %v, want %v", out.Busy, want)
	}
	if want := []int{8001}; !reflect.DeepEqual(out.Inconclusive, want) {
		t.Errorf("Inconclusive = %v, want %v", out.Inconclusive, want)
	}
}

// TestPortClaimInconclusiveTreatedAsBusy proves the fail-closed treatment of
// probe failure: when every probe errors, no port is claimed, and the error
// carries BOTH typed sentinels — ErrNoFreePort for the drained window and
// ErrPortProbeInconclusive because the probes could not confirm anything.
// fak-test:runtime fast est=1ms lane=default
func TestPortClaimInconclusiveTreatedAsBusy(t *testing.T) {
	p := &recordingProber{err: map[int]error{9000: errProbe, 9001: errProbe, 9002: errProbe}}
	out, err := PortClaim(ClaimOptions{Start: 9000, Window: 3, Prober: p.probe})
	if out == nil || out.Port != 0 {
		t.Fatalf("out = %+v, want no claimed port", out)
	}
	if !errors.Is(err, ErrNoFreePort) {
		t.Errorf("err = %v, want errors.Is(err, ErrNoFreePort)", err)
	}
	if !errors.Is(err, ErrPortProbeInconclusive) {
		t.Errorf("err = %v, want errors.Is(err, ErrPortProbeInconclusive)", err)
	}
	if want := []int{9000, 9001, 9002}; !reflect.DeepEqual(out.Inconclusive, want) {
		t.Errorf("Inconclusive = %v, want %v (every errored probe recorded, none accepted)", out.Inconclusive, want)
	}
	if len(out.Busy) != 0 {
		t.Errorf("Busy = %v, want empty (an error is not a conclusive hold)", out.Busy)
	}
}

// TestPortClaimBoundedWindow proves the scan never wanders past its window:
// only the window's ports are probed, a positive answer outside it is never
// seen, and a zero window falls back to DefaultWindow.
// fak-test:runtime fast est=1ms lane=default
func TestPortClaimBoundedWindow(t *testing.T) {
	t.Run("explicit window bounds the scan", func(t *testing.T) {
		p := &recordingProber{free: map[int]bool{99999: true}} // outside any window below
		out, err := PortClaim(ClaimOptions{Start: 1000, Window: 5, Prober: p.probe})
		if out == nil || out.Port != 0 {
			t.Fatalf("out = %+v, want no claimed port", out)
		}
		if !errors.Is(err, ErrNoFreePort) {
			t.Errorf("err = %v, want errors.Is(err, ErrNoFreePort)", err)
		}
		if errors.Is(err, ErrPortProbeInconclusive) {
			t.Errorf("err = %v must not claim inconclusive probes when every probe answered conclusively", err)
		}
		want := []int{1000, 1001, 1002, 1003, 1004}
		if !reflect.DeepEqual(p.attempt, want) {
			t.Errorf("probed ports = %v, want %v (bounded window)", p.attempt, want)
		}
		if !reflect.DeepEqual(out.Busy, want) {
			t.Errorf("Busy = %v, want %v", out.Busy, want)
		}
	})

	t.Run("zero window takes the default", func(t *testing.T) {
		p := &recordingProber{}
		_, err := PortClaim(ClaimOptions{Start: 2000, Prober: p.probe})
		if !errors.Is(err, ErrNoFreePort) {
			t.Fatalf("err = %v, want errors.Is(err, ErrNoFreePort)", err)
		}
		if len(p.attempt) != DefaultWindow {
			t.Errorf("probed %d ports, want the default window of %d", len(p.attempt), DefaultWindow)
		}
	})
}

// TestPortClaimRefusesInvalidWindow proves invalid windows are refused up
// front with a config error that matches neither claim sentinel.
// fak-test:runtime fast est=1ms lane=default
func TestPortClaimRefusesInvalidWindow(t *testing.T) {
	cases := []struct {
		name string
		opts ClaimOptions
	}{
		{"start below range", ClaimOptions{Start: 0, Window: 3}},
		{"negative start", ClaimOptions{Start: -1, Window: 3}},
		{"window runs past 65535", ClaimOptions{Start: 65535, Window: 2}},
		{"default window runs past 65535", ClaimOptions{Start: 65530}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := PortClaim(tc.opts)
			if err == nil {
				t.Fatal("expected a config refusal")
			}
			if out != nil {
				t.Errorf("out = %+v, want nil on refusal", out)
			}
			for _, s := range []error{ErrNoFreePort, ErrPortProbeInconclusive} {
				if errors.Is(err, s) {
					t.Fatalf("config refusal matched claim sentinel %v", s)
				}
			}
		})
	}
}

// TestPortClaimProbeFreeDecision pins the one-port contract directly:
// positive confirmation accepts, a conclusive hold rejects without error,
// and anything the prober cannot say is the typed inconclusive sentinel.
// fak-test:runtime fast est=1ms lane=default
func TestPortClaimProbeFreeDecision(t *testing.T) {
	t.Run("positive confirmation accepts", func(t *testing.T) {
		free, err := ProbeFree(func(int) (bool, error) { return true, nil }, 7000)
		if !free || err != nil {
			t.Errorf("ProbeFree = (%v, %v), want (true, nil)", free, err)
		}
	})

	t.Run("conclusive hold rejects without error", func(t *testing.T) {
		free, err := ProbeFree(func(int) (bool, error) { return false, nil }, 7000)
		if free || err != nil {
			t.Errorf("ProbeFree = (%v, %v), want (false, nil)", free, err)
		}
	})

	t.Run("prober error is the typed inconclusive sentinel", func(t *testing.T) {
		boom := fmt.Errorf("socket: %w", errors.New("connection reset"))
		free, err := ProbeFree(func(int) (bool, error) { return false, boom }, 7000)
		if free {
			t.Error("ProbeFree accepted a port its prober errored on")
		}
		if !errors.Is(err, ErrPortProbeInconclusive) {
			t.Errorf("err = %v, want errors.Is(err, ErrPortProbeInconclusive)", err)
		}
		if !errors.Is(err, boom) {
			t.Errorf("err = %v, want the prober error preserved in the chain", err)
		}
	})

	t.Run("nil prober fails closed", func(t *testing.T) {
		free, err := ProbeFree(nil, 7000)
		if free || !errors.Is(err, ErrPortProbeInconclusive) {
			t.Errorf("ProbeFree(nil) = (%v, %v), want (false, ErrPortProbeInconclusive)", free, err)
		}
	})
}
