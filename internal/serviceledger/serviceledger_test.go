package serviceledger

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

func testIdentity() servicespec.Identity {
	return servicespec.Identity{Node: "node-a", Service: "guard"}
}

func readyEvent(atMS int64, uid string) Event {
	return Event{
		Type: EventReadiness, AtUnixMS: atMS, Source: SourceFak, SourceUID: uid,
		Identity: testIdentity(), Phase: servicespec.PhaseReady,
	}
}

func crashEvent(atMS int64, uid string) Event {
	return Event{
		Type: EventProcessExit, AtUnixMS: atMS, Source: SourceFak, SourceUID: uid,
		Identity: testIdentity(),
		Exit:     &servicespec.ExitRecord{Class: servicespec.ExitCrash, Code: 137, AtUnixMS: atMS},
	}
}

func TestAppendPersistsExactOnceAndReplays(t *testing.T) {
	dir := t.TempDir()
	led, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	ev, ok, err := led.Append(readyEvent(1000, "u1"))
	if err != nil || !ok {
		t.Fatalf("first append: ok=%v err=%v", ok, err)
	}
	if ev.Seq != 1 || ev.Schema != EventSchemaV1 || ev.Identity.Workload != "guard" {
		t.Fatalf("normalize/seq wrong: %+v", ev)
	}
	if _, ok, _ = led.Append(readyEvent(1000, "u1")); ok {
		t.Fatal("duplicate (source, uid) was ingested twice")
	}
	if _, ok, _ = led.Append(crashEvent(2000, "u2")); !ok {
		t.Fatal("distinct uid refused")
	}
	// Reopen: the exact-once index must survive a process restart.
	led2, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got := led2.Events(); len(got) != 2 || got[1].Seq != 2 {
		t.Fatalf("replay lost events: %+v", got)
	}
	if _, ok, _ = led2.Append(readyEvent(1000, "u1")); ok {
		t.Fatal("replayed ledger re-ingested a known uid")
	}
}

func TestContentDigestKeysIdempotentSyntheticReplay(t *testing.T) {
	led := Memory()
	e := readyEvent(1000, "")
	if _, ok, err := led.Append(e); err != nil || !ok {
		t.Fatalf("first: ok=%v err=%v", ok, err)
	}
	if _, ok, _ := led.Append(e); ok {
		t.Fatal("identical uid-less event ingested twice")
	}
	e2 := readyEvent(2000, "") // different content -> new row
	if _, ok, _ := led.Append(e2); !ok {
		t.Fatal("distinct uid-less event refused")
	}
}

func TestValidateRefusals(t *testing.T) {
	led := Memory()
	cases := []Event{
		{Type: "made-up", AtUnixMS: 1, Source: "x", Identity: testIdentity()},
		{Type: EventReadiness, AtUnixMS: 1, Source: "x", Identity: servicespec.Identity{Node: "n"}, Phase: servicespec.PhaseReady},
		{Type: EventReadiness, AtUnixMS: 0, Source: "x", Identity: testIdentity(), Phase: servicespec.PhaseReady},
		{Type: EventReadiness, AtUnixMS: 1, Source: "", Identity: testIdentity(), Phase: servicespec.PhaseReady},
		{Type: EventReadiness, AtUnixMS: 1, Source: "x", Identity: testIdentity(), Phase: "sideways"},
		{Type: EventDesiredChange, AtUnixMS: 1, Source: "x", Identity: testIdentity(), Desired: "running"},
		{Type: EventProcessExit, AtUnixMS: 1, Source: "x", Identity: testIdentity()},
		{Type: EventProcessExit, AtUnixMS: 1, Source: "x", Identity: testIdentity(), Exit: &servicespec.ExitRecord{Class: "shrug"}},
	}
	for i, c := range cases {
		if _, _, err := led.Append(c); err == nil {
			t.Errorf("case %d: invalid event was ledgered: %+v", i, c)
		}
	}
}

func TestRedactStripsSecretsAndPrivateIdentifiers(t *testing.T) {
	in := `run --lease-token=deadbeef --api_key: hunter2 Authorization: Bearer abc.def ` +
		`host db1.corp 10.1.2.3 192.168.0.9 172.31.4.4 public.example.com 8.8.8.8`
	got := Redact(in)
	for _, leaked := range []string{"deadbeef", "hunter2", "abc.def", "db1.corp", "10.1.2.3", "192.168.0.9", "172.31.4.4"} {
		if strings.Contains(got, leaked) {
			t.Errorf("redaction leaked %q in %q", leaked, got)
		}
	}
	for _, kept := range []string{"public.example.com", "8.8.8.8", "--lease-token="} {
		if !strings.Contains(got, kept) {
			t.Errorf("redaction over-stripped %q from %q", kept, got)
		}
	}
}

func TestLeaseTokenIsHashedNeverStored(t *testing.T) {
	dir := t.TempDir()
	led, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	e := readyEvent(1000, "u1")
	e.Type = EventLeaseFence
	e.Phase = ""
	e.Correlation.Generation = 2
	e.Correlation.LeaseTokenHash = HashLeaseToken("raw-secret-lease")
	if _, _, err := led.Append(e); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "events.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "raw-secret-lease") {
		t.Fatal("raw lease token reached disk")
	}
	if !strings.Contains(string(b), HashLeaseToken("raw-secret-lease")) {
		t.Fatal("lease token hash missing from ledger row")
	}
}

func TestStatusDetectsRestartStorm(t *testing.T) {
	led := Memory()
	for i := int64(0); i < 5; i++ {
		if _, _, err := led.Append(crashEvent(1000+i*1000, "")); err != nil {
			t.Fatal(err)
		}
	}
	sts := Status(led.Events(), StatusOptions{})
	if len(sts) != 1 {
		t.Fatalf("want 1 workload, got %d", len(sts))
	}
	st := sts[0]
	if st.RestartsInWindow != 5 || !st.RestartStorm {
		t.Fatalf("restart storm not detected: %+v", st)
	}
	if st.Phase != servicespec.PhaseFailed || st.LastExit == nil || st.LastExit.Class != servicespec.ExitCrash {
		t.Fatalf("rollup phase/exit wrong: %+v", st)
	}
	// Two crashes far apart are not a storm.
	calm := Memory()
	_, _, _ = calm.Append(crashEvent(1000, ""))
	_, _, _ = calm.Append(crashEvent(1000+2*servicespec.DefaultWindowMS, ""))
	if got := Status(calm.Events(), StatusOptions{}); got[0].RestartStorm {
		t.Fatalf("calm timeline flagged as a storm: %+v", got[0])
	}
}

func TestStatusDetectsStaleOwner(t *testing.T) {
	led := Memory()
	fence := Event{
		Type: EventLeaseFence, AtUnixMS: 1000, Source: SourceFak, Identity: testIdentity(),
		Correlation: Correlation{Generation: 3, ManagerInvocation: "inv-3"},
	}
	if _, _, err := led.Append(fence); err != nil {
		t.Fatal(err)
	}
	stale := readyEvent(2000, "old-owner-heartbeat")
	stale.Correlation = Correlation{Generation: 2, PID: 41, ManagerInvocation: "inv-2"}
	if _, _, err := led.Append(stale); err != nil {
		t.Fatal(err)
	}
	sts := Status(led.Events(), StatusOptions{})
	if len(sts) != 1 || len(sts[0].StaleOwners) != 1 {
		t.Fatalf("stale owner not detected: %+v", sts)
	}
	so := sts[0].StaleOwners[0]
	if so.Generation != 2 || so.FencedGeneration != 3 || so.PID != 41 || so.LastAtUnixMS != 2000 {
		t.Fatalf("stale owner fields wrong: %+v", so)
	}
	if sts[0].Generation != 3 {
		t.Fatalf("generation high-water mark wrong: %+v", sts[0])
	}
}

func TestOpenRefusesMalformedLedgerLine(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "events.jsonl"), []byte("{not json}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(dir); err == nil {
		t.Fatal("malformed ledger line was silently accepted")
	}
}

func TestOpenRepairsOnlyTruncatedFinalRecord(t *testing.T) {
	dir := t.TempDir()
	led, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok, err := led.Append(readyEvent(1000, "u1")); err != nil || !ok {
		t.Fatalf("append: ok=%v err=%v", ok, err)
	}
	path := filepath.Join(dir, "events.jsonl")
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	torn := []byte(`{"version":1,"event":`)
	if err := os.WriteFile(path, append(before, torn...), 0o600); err != nil {
		t.Fatal(err)
	}

	reopened, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got := reopened.Events(); len(got) != 1 || got[0].Seq != 1 {
		t.Fatalf("recovery changed acknowledged events: %+v", got)
	}
	receipt := reopened.Recovery()
	if receipt.CorruptionClass != "truncated_tail" || receipt.DiscardedTailBytes != int64(len(torn)) || receipt.RecoveredSequence != 1 {
		t.Fatalf("recovery receipt = %+v", receipt)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatalf("repair did not truncate exactly the torn tail: got %d bytes want %d", len(after), len(before))
	}
}

func TestOpenRefusesChecksumCorruption(t *testing.T) {
	dir := t.TempDir()
	led, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok, err := led.Append(readyEvent(1000, "u1")); err != nil || !ok {
		t.Fatalf("append: ok=%v err=%v", ok, err)
	}
	path := filepath.Join(dir, "events.jsonl")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data[strings.Index(string(data), "u1")] = 'x'
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(dir); err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("checksum corruption was not refused: %v", err)
	}
}
