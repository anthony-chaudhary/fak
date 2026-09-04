package cloudhandoff

import (
	"errors"
	"fmt"
	"testing"
)

func base() (Policy, Request) {
	return Policy{
			Eligible:        true,
			Consent:         ConsentPreapproved,
			Destinations:    []string{"vendor"},
			AllowedTriggers: []Trigger{TriggerUnsupported, TriggerDeadline, TriggerQuality, TriggerPressure, TriggerFault, TriggerUser},
		}, Request{
			OperationID:      "op-7",
			Trigger:          TriggerFault,
			Data:             []DataClass{{Name: "prompt"}},
			DestinationClass: "vendor",
			Consequence:      "provider billing",
			Alternatives:     []string{"retry local", "cancel"},
			Payload:          []byte("work"),
		}
}

func TestAuthorityTriggersAndAttribution(t *testing.T) {
	p, r := base()
	for _, trigger := range p.AllowedTriggers {
		b := New()
		r.Trigger = trigger
		sent := false
		receipt, err := b.Handoff(p, r, nil, func(pkg Package) error {
			sent = true
			if pkg.OperationID != r.OperationID || pkg.Schema != Schema {
				t.Fatal("operation identity lost")
			}
			return nil
		}, Attempt{Engine: "fak-native", Location: "local", Outcome: "failed"})
		if err != nil || !sent || !receipt.RemoteCompleted || receipt.LocalCompleted {
			t.Fatalf("trigger=%s receipt=%+v err=%v", trigger, receipt, err)
		}
		if len(b.Events()) != 3 || b.Events()[0].State != "proposed" || b.Events()[1].State != "approved" || b.Events()[2].State != "completed" {
			t.Fatalf("events=%+v", b.Events())
		}
	}
}

func TestAskNeverLocalBoundaryAndFailure(t *testing.T) {
	p, r := base()
	p.Consent = ConsentAsk
	b := New()
	sent := false
	receipt, err := b.Handoff(p, r, func(Event) bool { return false }, func(Package) error { sent = true; return nil }, Attempt{Engine: "fak-native", Location: "local", Outcome: "quality_reject"})
	if !errors.Is(err, ErrDenied) || sent || receipt.Terminal != "denied" {
		t.Fatalf("denial receipt=%+v err=%v sent=%v", receipt, err, sent)
	}
	p.Consent = ConsentNever
	if _, err = b.Handoff(p, r, nil, func(Package) error { sent = true; return nil }, Attempt{}); !errors.Is(err, ErrDenied) {
		t.Fatal(err)
	}
	p.Consent = ConsentPreapproved
	r.Data = []DataClass{{Name: "medical", LocalRequired: true}}
	before := sent
	if _, err = b.Handoff(p, r, nil, func(Package) error { sent = true; return nil }, Attempt{}); !errors.Is(err, ErrDenied) || sent != before {
		t.Fatal("local-required data escaped")
	}
	r.Data = []DataClass{{Name: "prompt"}}
	receipt, err = b.Handoff(p, r, nil, func(Package) error { return errors.New("offline") }, Attempt{Engine: "fak-native", Location: "local", Outcome: "runtime_fault"})
	if !errors.Is(err, ErrRemote) || receipt.Terminal != "failed" || receipt.RemoteCompleted || receipt.LocalCompleted {
		t.Fatalf("failure receipt=%+v err=%v", receipt, err)
	}
	events := b.Events()
	if events[len(events)-1].State != "failed" {
		t.Fatal("missing failed event")
	}
}

func TestEligibilityTriggerAndDestinationDeny(t *testing.T) {
	p, r := base()
	tests := []func(*Policy, *Request){
		func(p *Policy, _ *Request) { p.Eligible = false },
		func(p *Policy, r *Request) { r.Trigger = "unknown" },
		func(_ *Policy, r *Request) { r.DestinationClass = "other" },
	}
	for _, mutate := range tests {
		pp, rr := p, r
		mutate(&pp, &rr)
		called := false
		if _, err := New().Handoff(pp, rr, nil, func(Package) error { called = true; return nil }, Attempt{}); !errors.Is(err, ErrDenied) || called {
			t.Fatalf("err=%v called=%v", err, called)
		}
	}
}

func BenchmarkHandoffPreapproved(b *testing.B) {
	p, r := base()
	local := Attempt{Engine: "fak-native", Location: "local", Outcome: "failed"}
	send := func(pkg Package) error { return nil }

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		broker := New()
		receipt, err := broker.Handoff(p, r, nil, send, local)
		if err != nil || !receipt.RemoteCompleted {
			b.Fatalf("handoff failed: %v", err)
		}
	}
}

func BenchmarkHandoffDenialGuard(b *testing.B) {
	p, r := base()
	r.Data = []DataClass{{Name: "medical", LocalRequired: true}}
	local := Attempt{Engine: "fak-native", Location: "local", Outcome: "failed"}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		broker := New()
		receipt, err := broker.Handoff(p, r, nil, nil, local)
		if !errors.Is(err, ErrDenied) || receipt.Terminal != "denied" {
			b.Fatalf("expected denial, got receipt=%+v err=%v", receipt, err)
		}
	}
}

func BenchmarkHandoffPackaging(b *testing.B) {
	p, r := base()
	payload := make([]byte, 1024)
	r.Payload = payload
	local := Attempt{Engine: "fak-native", Location: "local", Outcome: "failed"}
	var receivedSize int
	send := func(pkg Package) error {
		receivedSize = len(pkg.Payload)
		return nil
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		broker := New()
		receipt, err := broker.Handoff(p, r, nil, send, local)
		if err != nil || !receipt.RemoteCompleted || receivedSize != 1024 {
			b.Fatalf("benchmark failed: %v", err)
		}
	}
}

func BenchmarkBrokerEventsCopy(b *testing.B) {
	broker := New()
	p, r := base()
	local := Attempt{Engine: "fak-native", Location: "local", Outcome: "failed"}
	send := func(pkg Package) error { return nil }
	for i := 0; i < 10; i++ {
		r.OperationID = fmt.Sprintf("op-%d", i)
		_, _ = broker.Handoff(p, r, nil, send, local)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		ev := broker.Events()
		if len(ev) == 0 {
			b.Fatal("missing events")
		}
	}
}
