package main

import (
	"bytes"
	"context"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
)

// faultBus delegates to a real bus, then corrupts the result of chosen calls
// (by zero-based call index) so each self-check branch in run can be driven.
type faultBus struct {
	*a2achan.Bus
	flipSend, flipRecv map[int]bool
	launderRecv        map[int]bool
	dropFanout         bool
	sends, recvs       int
}

func flip(v abi.Verdict) abi.Verdict {
	if v.Kind == abi.VerdictAllow {
		return abi.Verdict{Kind: abi.VerdictDeny}
	}
	return abi.Verdict{Kind: abi.VerdictAllow}
}

func (f *faultBus) Send(ctx context.Context, from string, to a2achan.ChannelKey, body abi.Ref, caps ...abi.Capability) abi.Verdict {
	i := f.sends
	f.sends++
	v := f.Bus.Send(ctx, from, to, body, caps...)
	if f.flipSend[i] {
		return flip(v)
	}
	return v
}

func (f *faultBus) Recv(ctx context.Context, to a2achan.ChannelKey, caps ...abi.Capability) (a2achan.Message, abi.Verdict, error) {
	i := f.recvs
	f.recvs++
	m, v, err := f.Bus.Recv(ctx, to, caps...)
	if f.launderRecv[i] {
		m.Body.Taint = abi.TaintTrusted
	}
	if f.flipRecv[i] {
		v = flip(v)
	}
	return m, v, err
}

func (f *faultBus) Publish(ctx context.Context, from string, topic a2achan.ChannelKey, body abi.Ref, caps ...abi.Capability) (abi.Verdict, int) {
	v, n := f.Bus.Publish(ctx, from, topic, body, caps...)
	if f.dropFanout {
		n--
	}
	return v, n
}

func testCtx(t *testing.T) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func failCount(stderr string) int {
	n := 0
	for _, l := range strings.Split(stderr, "\n") {
		if strings.HasPrefix(l, "FAIL: ") {
			n++
		}
	}
	return n
}

// fak-test:runtime fast est=10ms lane=default
func TestRunRealBusPassesEveryLeg(t *testing.T) {
	bus := a2achan.NewBus()
	var stdout, stderr bytes.Buffer
	if code := run(testCtx(t), bus, &stdout, &stderr); code != exitOK {
		t.Fatalf("exit = %d, want %d; stderr:\n%s", code, exitOK, stderr.String())
	}
	if n := failCount(stderr.String()); n != 0 {
		t.Fatalf("FAIL lines = %d, want 0", n)
	}
	want := a2achan.Stats{Sent: 5, Received: 5, Denied: 3, Held: 0}
	if got := bus.Stats(); got != want {
		t.Fatalf("stats = %+v, want %+v", got, want)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestRunFailsClosedOnEachBrokenLeg(t *testing.T) {
	cases := []struct {
		name string
		bus  faultBus
	}{
		{"shared send refused", faultBus{flipSend: map[int]bool{0: true}}},
		{"p2p recv refused", faultBus{flipRecv: map[int]bool{0: true}}},
		{"taint laundered on delivery", faultBus{launderRecv: map[int]bool{0: true}}},
		{"private cross-agent send admitted", faultBus{flipSend: map[int]bool{1: true}}},
		{"quarantined send admitted", faultBus{flipSend: map[int]bool{2: true}}},
		{"uncapped send admitted", faultBus{flipSend: map[int]bool{3: true}}},
		{"session send refused", faultBus{flipSend: map[int]bool{4: true}}},
		{"session recv refused", faultBus{flipRecv: map[int]bool{1: true}}},
		{"window send refused", faultBus{flipSend: map[int]bool{5: true}}},
		{"window recv refused", faultBus{flipRecv: map[int]bool{2: true}}},
		{"publish fanout short", faultBus{dropFanout: true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fb := tc.bus
			fb.Bus = a2achan.NewBus()
			var stdout, stderr bytes.Buffer
			if code := run(testCtx(t), &fb, &stdout, &stderr); code != exitFailed {
				t.Fatalf("exit = %d, want %d", code, exitFailed)
			}
			if n := failCount(stderr.String()); n != 1 {
				t.Fatalf("FAIL lines = %d, want exactly 1; stderr:\n%s", n, stderr.String())
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestVerdictStrNamesKindAndReason(t *testing.T) {
	cases := []struct {
		v    abi.Verdict
		want string
	}{
		{abi.Verdict{Kind: abi.VerdictAllow}, "ALLOW"},
		{abi.Verdict{Kind: abi.VerdictDefer}, "DEFER"},
		{abi.Verdict{Kind: abi.VerdictKind(255)}, "?"},
		{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonNone}, "DENY (" + abi.ReasonName(abi.ReasonNone) + ")"},
		{abi.Verdict{Kind: abi.VerdictQuarantine, Reason: abi.ReasonNone}, "QUARANTINE(held) (" + abi.ReasonName(abi.ReasonNone) + ")"},
	}
	for _, tc := range cases {
		if got := verdictStr(tc.v); got != tc.want {
			t.Errorf("verdictStr(%v) = %q, want %q", tc.v.Kind, got, tc.want)
		}
	}
}
