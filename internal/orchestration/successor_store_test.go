package orchestration

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestEffectSuccessorStoreAdmitPersistsBeforeReturning(t *testing.T) {
	now := time.Date(2026, time.August, 24, 12, 34, 56, 123456789, time.FixedZone("test", -7*60*60))
	store, err := openEffectSuccessorStore(t.TempDir(), time.Hour, func() time.Time { return now })
	if err != nil {
		t.Fatal(err)
	}
	proposal := validEffectSuccessorProposal()
	marker := filepath.Join(t.TempDir(), "must-not-exist")
	proposal.Effect.WriteSet = []string{marker}
	proposal.Lease.WriteSet = []string{marker}
	proposal.Capability.EnvelopeDigest, err = digestValue(normalizeEffectEnvelope(proposal.Effect))
	if err != nil {
		t.Fatal(err)
	}
	admission, outcome, err := store.Admit(proposal)
	if err != nil || outcome != EffectSuccessorStored {
		t.Fatalf("Admit() = %q, %v", outcome, err)
	}
	if got, want := admission.Receipt.AdmittedAt, now.UTC().Format(time.RFC3339Nano); got != want {
		t.Fatalf("Admit() admitted_at = %q, want %q", got, want)
	}
	if _, err := os.Stat(marker); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("Admit executed effect path: %v", err)
	}
	duplicate, outcome, err := store.Admit(proposal)
	if err != nil || outcome != EffectSuccessorDuplicate || !reflect.DeepEqual(duplicate, admission) {
		t.Fatalf("duplicate Admit() = %#v, %q, %v", duplicate, outcome, err)
	}
	got, outcome, err := store.Lookup(admission.Receipt.RunID, admission.Receipt.ObserverID, admission.Receipt.NodeID, admission.Receipt.ID)
	if err != nil || outcome != EffectSuccessorStored || !reflect.DeepEqual(got, admission.Receipt) {
		t.Fatalf("persisted Lookup() = %#v, %q, %v", got, outcome, err)
	}
}

func TestEffectSuccessorStoreLookupBindsRunChildAndSuccessor(t *testing.T) {
	store, err := OpenEffectSuccessorStore(t.TempDir(), time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	admission, _, err := store.Admit(validEffectSuccessorProposal())
	if err != nil {
		t.Fatal(err)
	}
	r := admission.Receipt
	for name, ids := range map[string][3]string{
		"run":       {"other-run", r.ObserverID, r.NodeID},
		"child":     {r.RunID, "other-child", r.NodeID},
		"successor": {r.RunID, r.ObserverID, "effect-other"},
	} {
		t.Run(name, func(t *testing.T) {
			if _, outcome, err := store.Lookup(ids[0], ids[1], ids[2], r.ID); outcome != EffectSuccessorMissing || err != nil {
				t.Fatalf("Lookup(mismatch) = %q, %v", outcome, err)
			}
		})
	}
	for name, ids := range map[string][3]string{
		"blank run":       {"", r.ObserverID, r.NodeID},
		"blank child":     {r.RunID, "", r.NodeID},
		"blank successor": {r.RunID, r.ObserverID, ""},
	} {
		t.Run(name, func(t *testing.T) {
			if _, outcome, err := store.Lookup(ids[0], ids[1], ids[2], r.ID); outcome != EffectSuccessorMalformed || err == nil {
				t.Fatalf("Lookup(blank identity) = %q, %v", outcome, err)
			}
		})
	}
}

func TestEffectSuccessorStoreConcurrentConflictingWritersDoNotOverwrite(t *testing.T) {
	store, err := OpenEffectSuccessorStore(t.TempDir(), time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	first := testEffectSuccessorReceipt(t)
	second := first
	second.LeaseID = "lease-competing"
	results := make(chan EffectSuccessorStoreOutcome, 2)
	errs := make(chan error, 2)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for _, receipt := range []EffectSuccessorReceipt{first, second} {
		wg.Add(1)
		go func(receipt EffectSuccessorReceipt) {
			defer wg.Done()
			<-start
			outcome, err := store.store(receipt)
			results <- outcome
			errs <- err
		}(receipt)
	}
	close(start)
	wg.Wait()
	close(results)
	close(errs)
	counts := map[EffectSuccessorStoreOutcome]int{}
	for outcome := range results {
		counts[outcome]++
	}
	if counts[EffectSuccessorStored] != 1 || counts[EffectSuccessorConflict] != 1 {
		t.Fatalf("outcomes = %#v, want one stored and one conflict", counts)
	}
	var failures int
	for err := range errs {
		if err != nil {
			failures++
		}
	}
	if failures != 1 {
		t.Fatalf("errors = %d, want conflict only", failures)
	}
	got, outcome, err := store.Lookup(first.RunID, first.ObserverID, first.NodeID, first.ID)
	if err != nil || outcome != EffectSuccessorStored {
		t.Fatalf("Lookup(winner) = %q, %v", outcome, err)
	}
	if !reflect.DeepEqual(got, first) && !reflect.DeepEqual(got, second) {
		t.Fatalf("winner was overwritten: %#v", got)
	}
}

func TestEffectSuccessorStorePersistsExactAdmissionAndReopens(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "admissions")
	now := time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)
	store, err := openEffectSuccessorStore(dir, time.Hour, func() time.Time { return now })
	if err != nil {
		t.Fatal(err)
	}
	receipt := testEffectSuccessorReceipt(t)
	receipt.AdmittedAt = now.Format(time.RFC3339Nano)
	outcome, err := store.store(receipt)
	if err != nil || outcome != EffectSuccessorStored {
		t.Fatalf("Store() = %q, %v", outcome, err)
	}

	reopened, err := openEffectSuccessorStore(dir, time.Hour, func() time.Time { return now })
	if err != nil {
		t.Fatal(err)
	}
	got, outcome, err := reopened.Lookup(receipt.RunID, receipt.ObserverID, receipt.NodeID, receipt.ID)
	if err != nil || outcome != EffectSuccessorStored {
		t.Fatalf("Lookup() = %#v, %q, %v", got, outcome, err)
	}
	if !reflect.DeepEqual(got, receipt) {
		t.Fatalf("Lookup() receipt = %#v, want %#v", got, receipt)
	}
	if runtime.GOOS != "windows" {
		if mode := mustStat(t, dir).Mode().Perm(); mode != 0o700 {
			t.Fatalf("directory mode = %o, want 700", mode)
		}
		if mode := mustStat(t, filepath.Join(dir, receipt.ID+".json")).Mode().Perm(); mode != 0o600 {
			t.Fatalf("file mode = %o, want 600", mode)
		}
	}
}

func TestEffectSuccessorStoreDuplicateIsIdempotentConflictCannotReplace(t *testing.T) {
	store, err := OpenEffectSuccessorStore(t.TempDir(), time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	receipt := testEffectSuccessorReceipt(t)
	if outcome, err := store.store(receipt); outcome != EffectSuccessorStored || err != nil {
		t.Fatalf("first Store() = %q, %v", outcome, err)
	}
	if outcome, err := store.store(receipt); outcome != EffectSuccessorDuplicate || err != nil {
		t.Fatalf("duplicate Store() = %q, %v", outcome, err)
	}

	conflict := receipt
	conflict.LeaseID = "lease-attacker"
	outcome, err := store.store(conflict)
	var storeErr *EffectSuccessorStoreError
	if outcome != EffectSuccessorConflict || !errors.As(err, &storeErr) || storeErr.Outcome != EffectSuccessorConflict {
		t.Fatalf("conflicting Store() = %q, %v", outcome, err)
	}
	got, outcome, err := store.Lookup(receipt.RunID, receipt.ObserverID, receipt.NodeID, receipt.ID)
	if err != nil || outcome != EffectSuccessorStored || !reflect.DeepEqual(got, receipt) {
		t.Fatalf("original after conflict = %#v, %q, %v", got, outcome, err)
	}
}

func TestEffectSuccessorStoreMissingStaleAndMalformedAreTypedNonSuccess(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)
	store, err := openEffectSuccessorStore(dir, time.Minute, func() time.Time { return now })
	if err != nil {
		t.Fatal(err)
	}
	receipt := testEffectSuccessorReceipt(t)
	if _, outcome, err := store.Lookup(receipt.RunID, receipt.ObserverID, receipt.NodeID, receipt.ID); outcome != EffectSuccessorMissing || err != nil {
		t.Fatalf("missing Lookup() = %q, %v", outcome, err)
	}
	receipt.AdmittedAt = now.Add(-2 * time.Minute).Format(time.RFC3339Nano)
	if _, err := store.store(receipt); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, receipt.ID+".json")
	if err := os.Chtimes(path, now, now); err != nil {
		t.Fatal(err)
	}
	if _, outcome, err := store.Lookup(receipt.RunID, receipt.ObserverID, receipt.NodeID, receipt.ID); outcome != EffectSuccessorStale || err != nil {
		t.Fatalf("stale Lookup() after fresh mtime = %q, %v", outcome, err)
	}

	future := testEffectSuccessorReceipt(t)
	future.ID = "effect-receipt-future"
	future.AdmittedAt = now.Add(time.Second).Format(time.RFC3339Nano)
	if _, err := store.store(future); err != nil {
		t.Fatal(err)
	}
	if _, outcome, err := store.Lookup(future.RunID, future.ObserverID, future.NodeID, future.ID); outcome != EffectSuccessorMalformed || err == nil {
		t.Fatalf("future Lookup() = %q, %v", outcome, err)
	}

	malformed := testEffectSuccessorReceipt(t)
	malformed.ID = "effect-receipt-malformed"
	malformedPath := filepath.Join(dir, malformed.ID+".json")
	if err := os.WriteFile(malformedPath, []byte(`{"schema":"fak-orchestration-effect-successor/1","id":"effect-receipt-malformed","private":"secret"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, outcome, err := store.Lookup(malformed.RunID, malformed.ObserverID, malformed.NodeID, malformed.ID); outcome != EffectSuccessorMalformed || err == nil {
		t.Fatalf("malformed Lookup() = %q, %v", outcome, err)
	}
}

func TestEffectSuccessorStoreRejectsMalformedAdmissionBeforeWrite(t *testing.T) {
	store, err := OpenEffectSuccessorStore(t.TempDir(), time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	proposed, err := ProposeEffectSuccessor(validEffectSuccessorProposal())
	if err != nil {
		t.Fatal(err)
	}
	if proposed.Receipt.AdmittedAt != "" {
		t.Fatalf("ProposeEffectSuccessor() admitted_at = %q, want empty", proposed.Receipt.AdmittedAt)
	}
	if outcome, err := store.store(proposed.Receipt); outcome != EffectSuccessorMalformed || err == nil {
		t.Fatalf("Store(raw proposal) = %q, %v", outcome, err)
	}

	receipt := testEffectSuccessorReceipt(t)
	receipt.Schema = "post-effect-invention/1"
	outcome, err := store.store(receipt)
	if outcome != EffectSuccessorMalformed || err == nil {
		t.Fatalf("Store(malformed) = %q, %v", outcome, err)
	}
}

func testEffectSuccessorReceipt(t *testing.T) EffectSuccessorReceipt {
	t.Helper()
	admission, err := ProposeEffectSuccessor(validEffectSuccessorProposal())
	if err != nil {
		t.Fatal(err)
	}
	admission.Receipt.AdmittedAt = time.Now().UTC().Format(time.RFC3339Nano)
	return admission.Receipt
}

func mustStat(t *testing.T, path string) os.FileInfo {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return info
}
