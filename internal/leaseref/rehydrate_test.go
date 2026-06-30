package leaseref

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/rehydrate"
)

func TestRehydrateFenceRungRefusesStaleColdReentry(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")

	t0 := time.Unix(1000, 0)
	aRec, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"internal/kernel/**"}, Holder: "A", TTLSeconds: 300}, t0)
	if err != nil || !v.OK {
		t.Fatalf("A acquire: ok=%v err=%v", v.OK, err)
	}
	tLate := t0.Add(10 * time.Minute)
	if _, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"internal/kernel/**"}, Holder: "B", TTLSeconds: 300}, tLate); err != nil || !v.OK {
		t.Fatalf("B transition: ok=%v err=%v", v.OK, err)
	}

	rung := NewRehydrateFenceRung(s, aRec, func() time.Time { return tLate.Add(time.Second) })
	if rung.FiresAt() != RehydrateFenceFiresAt || rung.FiresAt() != dormancy.Cold {
		t.Fatalf("lease fence fires at %s, want Cold", rung.FiresAt())
	}
	adm := rehydrate.NewGate(rung).Admit(context.Background(), dormancy.Cold)
	if adm.Admitted {
		t.Fatalf("stale lease re-entry admitted: %+v", adm)
	}
	if adm.RefusedBy != rehydrate.StaleLease {
		t.Fatalf("RefusedBy = %q, want STALE_LEASE", adm.RefusedBy)
	}
	if got := adm.RanReasons(); len(got) != 1 || got[0] != rehydrate.StaleLease {
		t.Fatalf("ran reasons = %v, want [STALE_LEASE]", got)
	}
	if !strings.Contains(adm.Detail, ReasonStaleLease) || !strings.Contains(adm.Detail, "halt and reacquire") {
		t.Fatalf("detail %q does not carry stale-lease halt policy", adm.Detail)
	}
}

func TestRehydrateFenceRungClearsCurrentLease(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")

	t0 := time.Unix(1000, 0)
	rec, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"internal/kernel/**"}, Holder: "A", TTLSeconds: 300}, t0)
	if err != nil || !v.OK {
		t.Fatalf("acquire: ok=%v err=%v", v.OK, err)
	}

	adm := rehydrate.NewGate(NewRehydrateFenceRung(s, rec, func() time.Time {
		return t0.Add(time.Minute)
	})).Admit(context.Background(), dormancy.Frozen)
	if !adm.Admitted {
		t.Fatalf("current lease was refused: %+v", adm)
	}
	if got := adm.RanReasons(); len(got) != 1 || got[0] != rehydrate.StaleLease {
		t.Fatalf("ran reasons = %v, want [STALE_LEASE]", got)
	}
}

func TestRehydrateFenceRungDoesNotRunBeforeCold(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")

	t0 := time.Unix(1000, 0)
	rec, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"internal/kernel/**"}, Holder: "A", TTLSeconds: 300}, t0)
	if err != nil || !v.OK {
		t.Fatalf("acquire: ok=%v err=%v", v.OK, err)
	}
	rung := NewRehydrateFenceRung(s, rec, func() time.Time {
		t.Fatal("fence clock should not be read before the rung fires")
		return time.Time{}
	})
	for _, h := range []dormancy.Horizon{dormancy.Warm, dormancy.Cool} {
		adm := rehydrate.NewGate(rung).Admit(context.Background(), h)
		if !adm.Admitted || len(adm.Ran) != 0 {
			t.Fatalf("%s re-entry should not run lease fence: %+v", h, adm)
		}
	}
}
