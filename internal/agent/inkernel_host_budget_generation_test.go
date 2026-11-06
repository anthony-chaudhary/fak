package agent

import (
	"context"
	"math"
	"sync"
	"testing"
)

// fak-test:runtime fast est=500ms lane=default
func TestHostBudgetGenerationRetainsSessionAndBeam(t *testing.T) {
	p := bareHostPlanner()
	const prompt, maxNew = 32, 8
	session, retained := hostDemand(t, p, prompt, maxNew)
	if retained != 0 {
		t.Fatal("fixture must disable retained cache")
	}
	p.SetHostMemoryBudget(1<<40, func() (int64, bool) { return 0, true })
	old := p.hostBudget
	_, sessionRelease, err := p.admitHostMemory(context.Background(), prompt, maxNew)
	if err != nil {
		t.Fatal(err)
	}
	beamRelease, err := p.admitBeamWorkspace(old)
	if err != nil {
		t.Fatal(err)
	}
	held := session + int64(32<<20)
	if got := p.HostMemoryBudgetStats().Reserved; got != held {
		t.Fatalf("held=%d want%d", got, held)
	}
	p.SetHostMemoryBudget(held+session/2, func() (int64, bool) { return 0, true })
	st := p.HostMemoryBudgetStats()
	if st.Reserved != held || st.AvailSigned != session/2 || st.Structural {
		t.Fatalf("replacement lost charges: %+v", st)
	}
	if _, release, err := p.admitHostMemory(context.Background(), prompt, maxNew); err == nil {
		release()
		t.Fatal("new session ignored old generation")
	}
	if _, err := p.admitBeamWorkspace(old); err == nil {
		t.Fatal("stale budget identity accepted")
	}
	p.SetHostMemoryBudget(0, nil)
	if st := p.HostMemoryBudgetStats(); st.Armed || st.Reserved != held {
		t.Fatalf("disarm lost debt: %+v", st)
	}
	p.SetHostMemoryBudget(held, func() (int64, bool) { return 0, true })
	if st := p.HostMemoryBudgetStats(); st.AvailSigned != 0 || st.Reserved != held {
		t.Fatalf("rearm lost debt: %+v", st)
	}
	sessionRelease()
	sessionRelease()
	if got := p.HostMemoryBudgetStats().Reserved; got != 32<<20 {
		t.Fatalf("session release affected beam: %d", got)
	}
	beamRelease()
	beamRelease()
	if got := p.HostMemoryBudgetStats().Reserved; got != 0 {
		t.Fatalf("beam release left %d", got)
	}
	_, release, err := p.admitHostMemory(context.Background(), prompt, maxNew)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	if got := p.HostMemoryBudgetStats().Reserved; got != session {
		t.Fatalf("fresh charge=%d", got)
	}
	sessionRelease()
	beamRelease()
	if got := p.HostMemoryBudgetStats().Reserved; got != session {
		t.Fatalf("old duplicate release freed fresh charge: %d", got)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestHostBudgetGenerationOverflowFailsClosed(t *testing.T) {
	held := int64(math.MaxInt64 - 1)
	b := &hostMemoryBudget{ceiling: math.MaxInt64, used: func() (int64, bool) { return 0, true }, sharedReserved: &held}
	if b.reserve(2) || b.reserved != 0 || held != math.MaxInt64-1 {
		t.Fatal("overflow mutated ledger")
	}
	if !b.reserve(1) || held != math.MaxInt64 {
		t.Fatal("last representable byte refused")
	}
	b.release(1)
	if held != math.MaxInt64-1 || b.reserved != 0 {
		t.Fatal("release changed unrelated debt")
	}
	if got := hostAvailableAfterReservations(1, math.MaxInt64, math.MaxInt64); got != math.MinInt64 {
		t.Fatalf("availability wrapped: %d", got)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestHostBudgetGenerationConcurrentBeamCharges(t *testing.T) {
	p := bareHostPlanner()
	p.SetHostMemoryBudget(3*(32<<20), func() (int64, bool) { return 0, true })
	b := p.hostBudget
	var wg sync.WaitGroup
	var mu sync.Mutex
	var releases []func()
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if release, err := p.admitBeamWorkspace(b); err == nil {
				mu.Lock()
				releases = append(releases, release)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if len(releases) != 3 || p.HostMemoryBudgetStats().Reserved != 3*(32<<20) {
		t.Fatal("concurrent admission exceeded configured headroom")
	}
	p.SetHostMemoryBudget(32<<20, func() (int64, bool) { return 0, true })
	for _, release := range releases {
		release()
		release()
	}
	if p.HostMemoryBudgetStats().Reserved != 0 {
		t.Fatal("release after replacement leaked")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestHostBudgetGenerationOverlappingReleaseAndUnknownProbe(t *testing.T) {
	p := bareHostPlanner()
	p.SetHostMemoryBudget(1<<40, func() (int64, bool) { return 0, false })
	old := p.hostBudget
	oldRelease, err := p.admitBeamWorkspace(old)
	if err != nil {
		t.Fatal(err)
	}
	// Preserve beam's historical unknown-usage policy, but never lose its charge.
	p.SetHostMemoryBudget(1<<40, func() (int64, bool) { return 0, true })
	current := p.hostBudget
	currentRelease, err := p.admitBeamWorkspace(current)
	if err != nil {
		t.Fatal(err)
	}
	if p.HostMemoryBudgetStats().Reserved != 2*(32<<20) {
		t.Fatal("overlapping generation total wrong")
	}
	oldRelease()
	oldRelease()
	if p.HostMemoryBudgetStats().Reserved != 32<<20 || current.reserved != 32<<20 || old.reserved != 0 {
		t.Fatal("old release changed current provenance")
	}
	p.SetHostMemoryBudget(0, nil)
	currentRelease()
	currentRelease()
	if st := p.HostMemoryBudgetStats(); st.Armed || st.Reserved != 0 {
		t.Fatalf("disarmed release leaked: %+v", st)
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestHostBudgetGenerationReplacementRacesRelease(t *testing.T) {
	p := bareHostPlanner()
	known := func() (int64, bool) { return 0, true }
	p.SetHostMemoryBudget(1<<40, known)
	old := p.hostBudget
	var releases []func()
	for i := 0; i < 32; i++ {
		release, err := p.admitBeamWorkspace(old)
		if err != nil {
			t.Fatal(err)
		}
		releases = append(releases, release)
	}
	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		for i := 0; i < 64; i++ {
			p.SetHostMemoryBudget(0, nil)
			p.SetHostMemoryBudget(1<<40, known)
		}
	}()
	for _, release := range releases {
		wg.Add(1)
		go func(release func()) {
			defer wg.Done()
			<-start
			release()
			release()
		}(release)
	}
	close(start)
	wg.Wait()
	if st := p.HostMemoryBudgetStats(); !st.Armed || st.Reserved != 0 {
		t.Fatalf("replacement/release lost accounting: %+v", st)
	}
	if old.reserved != 0 {
		t.Fatal("old generation charge remains")
	}
	// Old release callbacks must not subtract a later generation's new charge.
	release, err := p.admitBeamWorkspace(p.hostBudget)
	if err != nil {
		t.Fatal(err)
	}
	for _, oldRelease := range releases {
		oldRelease()
	}
	if got := p.HostMemoryBudgetStats().Reserved; got != 32<<20 {
		t.Fatalf("old callbacks changed new charge: %d", got)
	}
	release()
	if got := p.HostMemoryBudgetStats().Reserved; got != 0 {
		t.Fatalf("charge leaked: %d", got)
	}
}
