package session

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const quarantineTestSecret = "SECRET-DESCRIPTOR-CONTENT-4658"

// corruptRegistryFixtures maps each normalized cause class to registry bytes
// that provoke it. Every fixture embeds the sentinel secret so privacy tests
// can prove telemetry never echoes descriptor contents.
func corruptRegistryFixtures() map[RecoveryCause][]byte {
	return map[RecoveryCause][]byte{
		RecoveryCauseDecode:  []byte(`{"broken ` + quarantineTestSecret),
		RecoveryCauseVersion: []byte(`{"version":"bogus.v9","descriptors":[{"id":"` + quarantineTestSecret + `","trace":"t","run":0}]}`),
		RecoveryCauseBlankID: []byte(`{"version":"fak.session-descriptors.v1","descriptors":[{"id":"","trace":"` + quarantineTestSecret + `","run":0}]}`),
	}
}

// corruptListError writes fixture bytes to path and returns the classified
// corrupt error the store reports for them.
func corruptListError(t *testing.T, path string, fixture []byte) error {
	t.Helper()
	if err := os.WriteFile(path, fixture, 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := NewFileStore(path).List()
	if err == nil {
		t.Fatalf("List() accepted corrupt fixture %q", fixture)
	}
	if !IsCorruptDescriptorFile(err) {
		t.Fatalf("List() error %v is not a corrupt-descriptor error", err)
	}
	return err
}

func TestClassifyRecoveryCauseNormalizesCorruptionClasses(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.json")
	for want, fixture := range corruptRegistryFixtures() {
		err := corruptListError(t, path, fixture)
		if got := ClassifyRecoveryCause(err); got != want {
			t.Fatalf("ClassifyRecoveryCause(%v) = %q, want %q", err, got, want)
		}
	}
	if got := ClassifyRecoveryCause(errors.New("plain")); got != RecoveryCauseUnknown {
		t.Fatalf("ClassifyRecoveryCause(plain) = %q, want %q", got, RecoveryCauseUnknown)
	}
}

// TestRecoverCorruptRegistryRepeatedLoopWitness is the #4658 witness: a
// deterministic loop injects corrupt registries of every cause class, proves
// the store keeps working after each recovery, verifies counters and cause
// classes, and confirms the retention bound holds with descriptor contents
// absent from all telemetry.
func TestRecoverCorruptRegistryRepeatedLoopWitness(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.json")
	policy := QuarantineRetention{MaxCount: 3}
	fixtures := corruptRegistryFixtures()
	order := []RecoveryCause{RecoveryCauseDecode, RecoveryCauseVersion, RecoveryCauseBlankID}
	base := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)

	const rounds = 9
	var lastEvidence string
	for i := 0; i < rounds; i++ {
		cause := order[i%len(order)]
		err := corruptListError(t, path, fixtures[cause])
		now := base.Add(time.Duration(i) * time.Second)
		rec, recErr := RecoverCorruptRegistry(path, err, policy, now)
		if recErr != nil {
			t.Fatalf("round %d: RecoverCorruptRegistry() error = %v", i, recErr)
		}
		if rec.LedgerErr != nil || rec.ReapErr != nil {
			t.Fatalf("round %d: advisory errors ledger=%v reap=%v", i, rec.LedgerErr, rec.ReapErr)
		}
		if !rec.Event.Quarantined || rec.Event.Cause != cause {
			t.Fatalf("round %d: event = %+v, want quarantined with cause %q", i, rec.Event, cause)
		}
		if rec.Event.Bytes != int64(len(fixtures[cause])) {
			t.Fatalf("round %d: event bytes = %d, want %d", i, rec.Event.Bytes, len(fixtures[cause]))
		}
		if rec.Stats.Total != i+1 {
			t.Fatalf("round %d: total = %d, want %d", i, rec.Stats.Total, i+1)
		}
		lastEvidence = rec.Event.QuarantinePath

		// The registry keeps working after every recovery: the mainline
		// startup path (restore -> quarantine -> restore) sees a clean,
		// empty store, so initialize can continue.
		descs, listErr := NewFileStore(path).List()
		if listErr != nil || len(descs) != 0 {
			t.Fatalf("round %d: post-recovery List() = %v, %v; want empty, nil", i, descs, listErr)
		}

		count, cErr := QuarantineEvidenceCount(path)
		if cErr != nil {
			t.Fatal(cErr)
		}
		if count > policy.MaxCount {
			t.Fatalf("round %d: evidence count %d exceeds retention bound %d", i, count, policy.MaxCount)
		}
	}

	stats, ok, err := ReadRecoveryStats(path)
	if err != nil || !ok {
		t.Fatalf("ReadRecoveryStats() = %+v, %v, %v", stats, ok, err)
	}
	if stats.Total != rounds {
		t.Fatalf("total recoveries = %d, want %d", stats.Total, rounds)
	}
	perClass := rounds / len(order)
	for _, cause := range order {
		if got := stats.Causes[string(cause)]; got != perClass {
			t.Fatalf("cause %q count = %d, want %d", cause, got, perClass)
		}
	}
	if wantLast := base.Add((rounds - 1) * time.Second); !stats.LastAt.Equal(wantLast) {
		t.Fatalf("last recovery time = %v, want %v", stats.LastAt, wantLast)
	}
	if stats.QuarantineFailures != 0 {
		t.Fatalf("quarantine failures = %d, want 0", stats.QuarantineFailures)
	}

	// The newest evidence file is always preserved.
	if _, err := os.Stat(lastEvidence); err != nil {
		t.Fatalf("newest evidence missing: %v", err)
	}

	// Privacy: the ledger and the event/stats telemetry never echo
	// descriptor contents; only the quarantined evidence files may.
	ledger, err := os.ReadFile(RecoveryLedgerPath(path))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(ledger), quarantineTestSecret) {
		t.Fatalf("recovery ledger leaks descriptor contents: %s", ledger)
	}
	telemetry, err := json.Marshal(stats)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(telemetry), quarantineTestSecret) {
		t.Fatalf("recovery stats leak descriptor contents: %s", telemetry)
	}
}

func TestQuarantineCorruptRegistryTimestampCollisions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.json")
	now := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	var got []string
	for i := 0; i < 3; i++ {
		if err := os.WriteFile(path, []byte("{"), 0o600); err != nil {
			t.Fatal(err)
		}
		dst, err := QuarantineCorruptRegistry(path, now)
		if err != nil {
			t.Fatalf("collision %d: %v", i, err)
		}
		got = append(got, dst)
	}
	stamp := path + ".corrupt-" + now.UTC().Format(quarantineStampFormat)
	want := []string{stamp, stamp + "-1", stamp + "-2"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("collision paths = %v, want %v", got, want)
		}
		if _, err := os.Stat(want[i]); err != nil {
			t.Fatalf("evidence %s missing: %v", want[i], err)
		}
	}

	// Retention stays deterministic under collisions: with a bound of one,
	// the highest collision suffix (the newest capture) survives.
	removed, err := ReapQuarantine(path, QuarantineRetention{MaxCount: 1}, now)
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 2 {
		t.Fatalf("removed = %v, want two entries", removed)
	}
	if _, err := os.Stat(stamp + "-2"); err != nil {
		t.Fatalf("newest collision evidence removed: %v", err)
	}
}

func TestRecoverCorruptRegistryConcurrentRecoverers(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.json")
	corruptErr := corruptListError(t, path, corruptRegistryFixtures()[RecoveryCauseDecode])
	now := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)

	const workers = 8
	results := make([]RegistryRecovery, workers)
	errs := make([]error, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = RecoverCorruptRegistry(path, corruptErr, DefaultQuarantineRetention(), now)
		}(i)
	}
	wg.Wait()

	quarantined := 0
	for i := 0; i < workers; i++ {
		if errs[i] != nil {
			t.Fatalf("worker %d: hard error %v", i, errs[i])
		}
		if results[i].Event.Quarantined {
			quarantined++
		} else if !results[i].AlreadyRecovered {
			t.Fatalf("worker %d: neither quarantined nor deferred: %+v", i, results[i])
		}
	}
	if quarantined != 1 {
		t.Fatalf("quarantine winners = %d, want exactly 1", quarantined)
	}
	stats, ok, err := ReadRecoveryStats(path)
	if err != nil || !ok {
		t.Fatalf("ReadRecoveryStats() = %v, %v", ok, err)
	}
	if stats.Total != 1 {
		t.Fatalf("concurrent recovery double-counted: total = %d, want 1", stats.Total)
	}
}

func TestReapQuarantineCleanupFailureIsAdvisory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.json")
	if err := os.WriteFile(path, []byte("active"), 0o600); err != nil {
		t.Fatal(err)
	}
	base := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	var names []string
	for i := 0; i < 4; i++ {
		name := path + ".corrupt-" + base.Add(time.Duration(i)*time.Second).UTC().Format(quarantineStampFormat)
		if err := os.WriteFile(name, []byte("evidence"), 0o600); err != nil {
			t.Fatal(err)
		}
		names = append(names, name)
	}

	stuck := names[0] // oldest: over the bound, but removal will fail
	injected := errors.New("injected removal failure")
	quarantineRemove = func(p string) error {
		if p == stuck {
			return injected
		}
		return os.Remove(p)
	}
	defer func() { quarantineRemove = os.Remove }()

	removed, err := ReapQuarantine(path, QuarantineRetention{MaxCount: 1}, base.Add(time.Hour))
	if !errors.Is(err, injected) {
		t.Fatalf("ReapQuarantine() error = %v, want injected failure", err)
	}
	if len(removed) != 2 {
		t.Fatalf("removed = %v, want the two removable over-bound files", removed)
	}
	if _, statErr := os.Stat(names[3]); statErr != nil {
		t.Fatalf("newest evidence removed: %v", statErr)
	}
	if _, statErr := os.Stat(path); statErr != nil {
		t.Fatalf("active file removed by cleanup: %v", statErr)
	}

	// End to end, a cleanup failure stays advisory: recovery still succeeds.
	corruptErr := corruptListError(t, path, corruptRegistryFixtures()[RecoveryCauseDecode])
	rec, recErr := RecoverCorruptRegistry(path, corruptErr, QuarantineRetention{MaxCount: 1}, base.Add(2*time.Hour))
	if recErr != nil {
		t.Fatalf("RecoverCorruptRegistry() error = %v, want nil despite cleanup failure", recErr)
	}
	if !errors.Is(rec.ReapErr, injected) {
		t.Fatalf("ReapErr = %v, want injected failure surfaced as advisory", rec.ReapErr)
	}
	if !rec.Event.Quarantined {
		t.Fatalf("recovery event = %+v, want quarantined", rec.Event)
	}
}

func TestReapQuarantineBoundsAgeAndBytesAndNeverTouchesActive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.json")
	if err := os.WriteFile(path, []byte("active"), 0o600); err != nil {
		t.Fatal(err)
	}
	base := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	old := path + ".corrupt-" + base.Add(-48*time.Hour).UTC().Format(quarantineStampFormat)
	fresh := path + ".corrupt-" + base.UTC().Format(quarantineStampFormat)
	for _, name := range []string{old, fresh} {
		if err := os.WriteFile(name, []byte("0123456789"), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	// Age bound: the stale capture goes; the fresh one stays.
	removed, err := ReapQuarantine(path, QuarantineRetention{MaxAge: 24 * time.Hour}, base.Add(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 1 || removed[0] != old {
		t.Fatalf("age reap removed %v, want only %s", removed, old)
	}

	// Byte bound smaller than the sole remaining file: newest is preserved
	// anyway — the bound never deletes the last evidence.
	removed, err = ReapQuarantine(path, QuarantineRetention{MaxBytes: 4}, base.Add(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 0 {
		t.Fatalf("byte reap removed %v, want newest preserved", removed)
	}

	// Byte bound with two files: keeping both would exceed the bound, so the
	// older one goes.
	older := path + ".corrupt-" + base.Add(-time.Hour).UTC().Format(quarantineStampFormat)
	if err := os.WriteFile(older, []byte("0123456789"), 0o600); err != nil {
		t.Fatal(err)
	}
	removed, err = ReapQuarantine(path, QuarantineRetention{MaxBytes: 15}, base.Add(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 1 || removed[0] != older {
		t.Fatalf("byte reap removed %v, want only %s", removed, older)
	}

	// Off disables cleanup entirely.
	removed, err = ReapQuarantine(path, QuarantineRetention{Off: true, MaxCount: 1}, base)
	if err != nil || len(removed) != 0 {
		t.Fatalf("off reap = %v, %v; want no removals", removed, err)
	}

	// The active file is never cleanup's to take.
	if got, err := os.ReadFile(path); err != nil || string(got) != "active" {
		t.Fatalf("active file disturbed: %q, %v", got, err)
	}
}

func TestParseQuarantineRetention(t *testing.T) {
	def := DefaultQuarantineRetention()
	cases := []struct {
		in      string
		want    QuarantineRetention
		wantErr bool
	}{
		{"", def, false},
		{"off", QuarantineRetention{Off: true}, false},
		{"OFF", QuarantineRetention{Off: true}, false},
		{"count=2", QuarantineRetention{MaxCount: 2, MaxAge: def.MaxAge, MaxBytes: def.MaxBytes}, false},
		{"count=2,age=1h,bytes=100", QuarantineRetention{MaxCount: 2, MaxAge: time.Hour, MaxBytes: 100}, false},
		{"count=0,age=0s,bytes=0", QuarantineRetention{}, false},
		{"count=-1", def, true},
		{"age=later", def, true},
		{"pets=2", def, true},
		{"count", def, true},
	}
	for _, tc := range cases {
		got, err := ParseQuarantineRetention(tc.in)
		if (err != nil) != tc.wantErr {
			t.Fatalf("ParseQuarantineRetention(%q) error = %v, wantErr=%v", tc.in, err, tc.wantErr)
		}
		if got != tc.want {
			t.Fatalf("ParseQuarantineRetention(%q) = %+v, want %+v", tc.in, got, tc.want)
		}
	}
}

func TestReadRecoveryStatsToleratesMissingAndCorruptLedger(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.json")
	if _, ok, err := ReadRecoveryStats(path); ok || err != nil {
		t.Fatalf("missing ledger: ok=%v err=%v, want false, nil", ok, err)
	}
	if err := os.WriteFile(RecoveryLedgerPath(path), []byte("{broken"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := ReadRecoveryStats(path); ok || err != nil {
		t.Fatalf("corrupt ledger: ok=%v err=%v, want false, nil", ok, err)
	}
	// A corrupt ledger resets rather than blocking measurement.
	rec, err := recordRecovery(path, RecoveryEvent{At: time.Now().UTC(), Cause: RecoveryCauseDecode, Quarantined: true})
	if err != nil {
		t.Fatal(err)
	}
	if rec.Total != 1 {
		t.Fatalf("total after reset = %d, want 1", rec.Total)
	}
}

func TestQuarantineRetentionStampParsingIgnoresForeignSiblings(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.json")
	base := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	stamped := path + ".corrupt-" + base.UTC().Format(quarantineStampFormat)
	foreign := path + ".corrupt-manual-copy"
	for _, name := range []string{stamped, foreign} {
		if err := os.WriteFile(name, []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	list, err := listQuarantineEvidence(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 2 {
		t.Fatalf("evidence = %v, want both siblings tracked", list)
	}
	for _, ev := range list {
		if ev.path == foreign && !ev.observedAt().Equal(ev.mod) {
			t.Fatalf("foreign sibling %s should fall back to modtime", ev.path)
		}
		if ev.path == stamped && !ev.observedAt().Equal(base) {
			t.Fatalf("stamped sibling time = %v, want %v", ev.observedAt(), base)
		}
	}
}
