package fleetbus

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func testBus(t *testing.T) *DirBus {
	t.Helper()
	b, err := OpenDir(t.TempDir())
	if err != nil {
		t.Fatalf("OpenDir: %v", err)
	}
	return b
}

func TestDirBusRosterHonorsFreshness(t *testing.T) {
	b := testBus(t)
	live := testInstance(t, "serve-1", "box-a", "serve", testNow)
	stale := testInstance(t, "serve-2", "box-b", "serve", testNow.Add(-10*time.Minute))
	for _, inst := range []Instance{live, stale} {
		if err := b.Announce(inst); err != nil {
			t.Fatalf("Announce(%s): %v", inst.ID, err)
		}
	}

	roster, err := b.Instances(testNow, DefaultInstanceTTL)
	if err != nil {
		t.Fatalf("Instances: %v", err)
	}
	if len(roster) != 1 || roster[0].ID != "serve-1" {
		t.Fatalf("roster = %+v, want only the fresh serve-1", roster)
	}
	// A stale record is dropped from the roster but never deleted: reaping is an
	// operator act, not a side effect of reading.
	if _, err := os.Stat(filepath.Join(b.Root, instancesDir, "serve-2.json")); err != nil {
		t.Fatalf("reading the roster deleted a stale record: %v", err)
	}

	// Re-announcing refreshes rather than duplicating.
	if err := b.Announce(testInstance(t, "serve-2", "box-b", "serve", testNow)); err != nil {
		t.Fatalf("re-Announce: %v", err)
	}
	roster, _ = b.Instances(testNow, DefaultInstanceTTL)
	if len(roster) != 2 {
		t.Fatalf("roster = %+v, want 2 after the refresh", roster)
	}
	if roster[0].ID != "serve-1" || roster[1].ID != "serve-2" {
		t.Fatalf("roster order = %s,%s, want a stable id sort", roster[0].ID, roster[1].ID)
	}
}

func TestConcurrentAnnounceNeverTearsARecord(t *testing.T) {
	// Two writers wearing one identity is a real state — a restart overlapping a
	// shutdown, a copied config. A torn presence file would fail Validate and drop the
	// instance out of the roster, silently shrinking the denominator every report is
	// measured against, so the announce has to stay atomic under the collision.
	b := testBus(t)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			inst := testInstance(t, "serve-1", "box", "serve", testNow.Add(time.Duration(i)*time.Millisecond))
			if err := b.Announce(inst); err != nil {
				t.Errorf("Announce %d: %v", i, err)
			}
		}(i)
	}
	wg.Wait()

	roster, err := b.Instances(testNow, DefaultInstanceTTL)
	if err != nil {
		t.Fatalf("Instances: %v", err)
	}
	if len(roster) != 1 || roster[0].ID != "serve-1" || roster[0].Role != "serve" {
		t.Fatalf("roster = %+v, want one intact serve-1", roster)
	}
}

func TestDirBusRosterIgnoresGarbage(t *testing.T) {
	b := testBus(t)
	good := testInstance(t, "serve-1", "box", "serve", testNow)
	if err := b.Announce(good); err != nil {
		t.Fatalf("Announce: %v", err)
	}
	dir := filepath.Join(b.Root, instancesDir)
	for name, body := range map[string]string{
		"torn.json":    `{"schema":"fak.fleetbus.instan`,
		"future.json":  `{"schema":"fak.fleetbus.instance/v9","id":"x","seen_utc":"2026-08-05T12:00:00Z"}`,
		"notjson.txt":  "hello",
		"noclock.json": `{"schema":"fak.fleetbus.instance/v1","id":"y"}`,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	roster, err := b.Instances(testNow, DefaultInstanceTTL)
	if err != nil {
		t.Fatalf("Instances: %v", err)
	}
	if len(roster) != 1 || roster[0].ID != "serve-1" {
		t.Fatalf("roster = %+v, want only the one well-formed record", roster)
	}
}

func TestDirBusPublishAndRead(t *testing.T) {
	b := testBus(t)
	if got, err := b.Directives(); err != nil || len(got) != 0 {
		t.Fatalf("empty bus: Directives() = %v, %v; want none and no error", got, err)
	}

	first, _ := NewDirective("op-a", "steer", "go", Selector{All: true}, time.Minute, "", testNow)
	second, _ := NewDirective("op-b", "pause", "", Selector{Role: []string{"serve"}}, time.Minute, "seat drain", testNow.Add(time.Second))
	for _, d := range []Directive{first, second} {
		if err := b.Publish(d); err != nil {
			t.Fatalf("Publish(%s): %v", d.ID, err)
		}
	}

	got, err := b.Directives()
	if err != nil {
		t.Fatalf("Directives: %v", err)
	}
	if len(got) != 2 || got[0].ID != first.ID || got[1].ID != second.ID {
		t.Fatalf("Directives() = %+v, want [%s %s] in publish order", got, first.ID, second.ID)
	}
	if got[0].Payload != "go" || got[1].Reason != "seat drain" {
		t.Fatalf("round-trip lost a field: %+v", got)
	}

	// A malformed directive never reaches the log.
	bad := first
	bad.Selector = Selector{}
	if err := b.Publish(bad); err == nil {
		t.Fatal("Publish accepted a directive addressing nobody")
	}
	if got, _ := b.Directives(); len(got) != 2 {
		t.Fatalf("a refused publish still landed: %d directives", len(got))
	}
}

func TestDirBusReadsAcrossARotation(t *testing.T) {
	// A rotation must cost history, never a live directive: a drainer that missed
	// the sealed generation would leave the control point staring at a permanently
	// outstanding row.
	b := testBus(t)
	b.MaxLedgerBytes = 400
	var ids []string
	for i := 0; i < 6; i++ {
		d, _ := NewDirective("op", "steer", "go", Selector{All: true}, time.Minute, "", testNow.Add(time.Duration(i)*time.Second))
		if err := b.Publish(d); err != nil {
			t.Fatalf("Publish %d: %v", i, err)
		}
		ids = append(ids, d.ID)
	}
	if _, err := os.Stat(filepath.Join(b.Root, directivesLog+".1")); err != nil {
		t.Fatalf("the ledger never rotated under a %d-byte bound: %v", b.MaxLedgerBytes, err)
	}
	got, err := b.Directives()
	if err != nil {
		t.Fatalf("Directives: %v", err)
	}
	seen := map[string]bool{}
	for _, d := range got {
		seen[d.ID] = true
	}
	// The newest directives must all survive; the sealed generation supplies the rest.
	for _, id := range ids[len(ids)-3:] {
		if !seen[id] {
			t.Errorf("directive %s vanished across the rotation", id)
		}
	}
	if len(got) < 4 {
		t.Errorf("only %d directives survived; the sealed generation was not read", len(got))
	}
}

func TestClaimApplyStakesExactlyOnce(t *testing.T) {
	b := testBus(t)
	fresh, err := b.ClaimApply("serve-1", "d-abc")
	if err != nil || !fresh {
		t.Fatalf("first claim = %v, %v; want true, nil", fresh, err)
	}
	again, err := b.ClaimApply("serve-1", "d-abc")
	if err != nil || again {
		t.Fatalf("second claim = %v, %v; want false, nil", again, err)
	}
	// The claim is per (instance, directive): a peer must still get its turn, and
	// the same instance must still get a turn at a different directive.
	if peer, _ := b.ClaimApply("serve-2", "d-abc"); !peer {
		t.Error("one instance's claim blocked a peer's")
	}
	if other, _ := b.ClaimApply("serve-1", "d-def"); !other {
		t.Error("a claim on one directive blocked another")
	}
	if _, err := b.ClaimApply("../escape", "d-abc"); err == nil {
		t.Error("ClaimApply accepted an id that is not a bus token")
	}
}

func TestClaimApplyIsAtomicUnderConcurrency(t *testing.T) {
	b := testBus(t)
	const racers = 24
	var wg sync.WaitGroup
	won := make([]bool, racers)
	start := make(chan struct{})
	for i := 0; i < racers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			ok, err := b.ClaimApply("serve-1", "d-abc")
			if err != nil {
				t.Errorf("racer %d: %v", i, err)
			}
			won[i] = ok
		}(i)
	}
	close(start)
	wg.Wait()

	wins := 0
	for _, w := range won {
		if w {
			wins++
		}
	}
	if wins != 1 {
		t.Fatalf("%d racers won the claim, want exactly 1 — a double-apply is live", wins)
	}
}

func TestDirBusAcksRoundTrip(t *testing.T) {
	b := testBus(t)
	mk := func(inst string, status AckStatus, reason RefuseReason) Ack {
		return Ack{
			Schema: AckSchema, Directive: "d-abc", Instance: inst,
			Status: status, Reason: reason, Witness: "run=paused", Affected: 2,
			AckedUTC: utc(testNow),
		}
	}
	if err := b.Ack(mk("serve-1", AckApplied, "")); err != nil {
		t.Fatalf("Ack: %v", err)
	}
	if err := b.Ack(mk("serve-2", AckRefused, ApplyRefused)); err != nil {
		t.Fatalf("Ack: %v", err)
	}
	// An unusable ack is refused, not written.
	if err := b.Ack(mk("serve-3", AckRefused, "")); err == nil {
		t.Fatal("Ack accepted a refusal with no closed token")
	}

	got, err := b.Acks("d-abc")
	if err != nil {
		t.Fatalf("Acks: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("Acks() = %+v, want 2", got)
	}
	if got[0].Instance != "serve-1" || got[0].Affected != 2 || got[1].Reason != ApplyRefused {
		t.Fatalf("round-trip lost a field: %+v", got)
	}
	// Acks are partitioned per directive, so an unrelated directive reads empty.
	if other, _ := b.Acks("d-zzz"); len(other) != 0 {
		t.Fatalf("Acks(d-zzz) = %+v, want none", other)
	}
	if _, err := b.Acks("../escape"); err == nil {
		t.Error("Acks accepted an id that is not a bus token")
	}
}

func TestConcurrentAcksAllSurvive(t *testing.T) {
	// N instances answering one directive at once is the normal case, not the edge
	// case: a lost ack is an instance that silently reads as outstanding forever.
	b := testBus(t)
	const answerers = 16
	var wg sync.WaitGroup
	for i := 0; i < answerers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			err := b.Ack(Ack{
				Schema: AckSchema, Directive: "d-abc", Instance: instanceID(i),
				Status: AckApplied, Affected: 1, AckedUTC: utc(testNow),
			})
			if err != nil {
				t.Errorf("answerer %d: %v", i, err)
			}
		}(i)
	}
	wg.Wait()

	got, err := b.Acks("d-abc")
	if err != nil {
		t.Fatalf("Acks: %v", err)
	}
	if len(got) != answerers {
		t.Fatalf("recorded %d of %d concurrent acks", len(got), answerers)
	}
}

func instanceID(i int) string {
	return "serve-" + string(rune('a'+i%26)) + string(rune('a'+i/26))
}
