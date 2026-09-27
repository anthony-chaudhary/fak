package leaseref

import (
	"testing"
	"time"
)

// TestReleaseFencedRefusesOmittedGenerationForPositiveLiveLease pins the legacy
// boundary: generation zero may identify only a generation-zero record. It cannot
// authorize deleting a positive live generation, even for the matching holder.
func TestReleaseFencedRefusesOmittedGenerationForPositiveLiveLease(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Unix(1000, 0)
	rec, av, err := s.AcquireFenced(ctx(), Record{ID: "lane", Holder: "me", TTLSeconds: 60}, now)
	if err != nil || !av.OK {
		t.Fatalf("AcquireFenced: %+v %v", av, err)
	}
	v, err := s.ReleaseFenced(ctx(), "lane", "me", 0, now.Add(time.Second))
	if err != nil {
		t.Fatalf("ReleaseFenced: %v", err)
	}
	if v.OK || v.Reason != ReasonStaleLease || v.Presented != 0 || v.Current != rec.Generation {
		t.Fatalf("verdict = %+v, want STALE_LEASE presented=0 current=%d", v, rec.Generation)
	}
	if got, ok, err := s.Get(ctx(), "lane"); err != nil || !ok || got.Generation != rec.Generation {
		t.Fatalf("refused release changed lease: got=%+v ok=%v err=%v", got, ok, err)
	}
}
