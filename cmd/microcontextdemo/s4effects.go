package main

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/anthony-chaudhary/fak/internal/idempotency"
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

const effectsSchema = "fak-microcontext-effects/1"

type effectsReport struct {
	Schema               string   `json:"schema"`
	Verdict              string   `json:"verdict"`
	ObservedAt           string   `json:"observed_at"`
	Submitted            int      `json:"submitted"`
	DisjointCompleted    int      `json:"disjoint_completed"`
	ConflictsRefused     int      `json:"conflicts_refused"`
	AuthorityRefused     int      `json:"authority_refused"`
	Verified             int      `json:"verified"`
	ReplayedAfterRestart int      `json:"replayed_after_restart"`
	PhysicalApplies      int      `json:"physical_applies"`
	JournalEntries       int      `json:"journal_entries"`
	Claims               []string `json:"claims"`
	NonClaims            []string `json:"non_claims"`
}

func buildEffectsReport() (effectsReport, error) {
	root, e := os.MkdirTemp("", "fak-effects-")
	if e != nil {
		return effectsReport{}, e
	}
	defer os.RemoveAll(root)
	path := filepath.Join(root, "ledger")
	s, e := idempotency.Open(path, time.Hour)
	if e != nil {
		return effectsReport{}, e
	}
	c := microagent.NewEffectCoordinator(s)
	r := effectsReport{Schema: effectsSchema, Verdict: "PASS", ObservedAt: time.Now().UTC().Format(time.RFC3339), Submitted: 5, Claims: []string{"explicit tools and nonblocking resource ownership isolate parallel effects", "durable idempotency plus independent readback prevents duplicate retry/restart effects"}, NonClaims: []string{"fixture file effects are not arbitrary tool safety or distributed transaction evidence"}}
	var applies, disjoint, verified atomic.Int32
	read := func(_ context.Context, in microagent.EffectIntent, result string) error {
		b, e := os.ReadFile(filepath.Join(root, in.Resource))
		if e != nil {
			return e
		}
		if string(b) != result {
			return errors.New("readback mismatch")
		}
		return nil
	}
	apply := func(resource, value string) func() (string, error) {
		return func() (string, error) {
			applies.Add(1)
			return value, os.WriteFile(filepath.Join(root, resource), []byte(value), 0o600)
		}
	}
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			in := microagent.EffectIntent{ContextID: string(rune('a' + i)), Capability: "write", Resource: string(rune('x' + i)), Operation: "write", IdempotencyToken: string(rune('k' + i))}
			o, e := c.Run(context.Background(), in, []string{"write"}, apply(in.Resource, "ok"), read)
			if e == nil && o.Verified {
				disjoint.Add(1)
				verified.Add(1)
			}
		}(i)
	}
	wg.Wait()
	hold := make(chan struct{})
	started := make(chan struct{})
	in := microagent.EffectIntent{ContextID: "owner", Capability: "write", Resource: "z", Operation: "write", IdempotencyToken: "kz"}
	wg.Add(1)
	go func() {
		defer wg.Done()
		o, e := c.Run(context.Background(), in, []string{"write"}, func() (string, error) {
			applies.Add(1)
			close(started)
			<-hold
			return "z", os.WriteFile(filepath.Join(root, "z"), []byte("z"), 0o600)
		}, read)
		if e == nil && o.Verified {
			verified.Add(1)
		}
	}()
	<-started
	other := in
	other.ContextID = "conflict"
	if _, e = c.Run(context.Background(), other, []string{"write"}, apply("z", "bad"), read); errors.Is(e, microagent.ErrEffectConflict) {
		r.ConflictsRefused++
	}
	close(hold)
	wg.Wait()
	denied := in
	denied.ContextID = "denied"
	denied.Resource = "denied"
	if _, e = c.Run(context.Background(), denied, []string{"read"}, apply("denied", "bad"), read); errors.Is(e, microagent.ErrAuthorityRefused) {
		r.AuthorityRefused++
	}
	s2, _ := idempotency.Open(path, time.Hour)
	c2 := microagent.NewEffectCoordinator(s2)
	if o, e := c2.Run(context.Background(), in, []string{"write"}, apply("z", "bad"), read); e == nil && o.Replayed && o.Verified {
		r.ReplayedAfterRestart++
		verified.Add(1)
	}
	r.DisjointCompleted = int(disjoint.Load())
	r.Verified = int(verified.Load())
	r.PhysicalApplies = int(applies.Load())
	r.JournalEntries = len(c.Journal()) + len(c2.Journal())
	if e = verifyEffectsReport(r); e != nil {
		r.Verdict = "FAIL"
	}
	return r, e
}
func verifyEffectsReport(r effectsReport) error {
	if r.Schema != effectsSchema || r.Verdict != "PASS" || r.DisjointCompleted != 2 || r.ConflictsRefused != 1 || r.AuthorityRefused != 1 || r.ReplayedAfterRestart != 1 || r.PhysicalApplies != 3 || r.Verified != 4 {
		return errors.New("effect witness invariant failed")
	}
	return nil
}
func verifyEffectsArtifact(p string) error {
	b, e := os.ReadFile(p)
	if e != nil {
		return e
	}
	var r effectsReport
	if e = json.Unmarshal(b, &r); e != nil {
		return e
	}
	return verifyEffectsReport(r)
}
