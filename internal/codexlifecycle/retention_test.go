package codexlifecycle

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"
)

// retention_test.go — the #4765 bound: the native Codex rollout archive
// (~/.codex/sessions) must be boundable by age and bytes WITHOUT losing
// witnesses. These tests are the fixture-backed proof: a deterministic
// synthetic corpus shaped like the audited one (many sessions, a few huge
// rollouts dominating bytes) is planned under a cap, and the plan must
// (a) bring retained raw bytes under the cap, (b) never select an active
// writer or a protected witness, and (c) report honest receipts — a cap it
// cannot meet without destroying evidence is reported unmet, not forced.

var retNow = time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)

func day(n int) time.Time { return retNow.AddDate(0, 0, -n) }

func retPolicy() RetentionContract {
	return RetentionContract{
		Now:          retNow,
		ActiveWithin: 48 * time.Hour,
		WarmWithin:   30 * 24 * time.Hour,
		RawBytesCap:  100 << 20, // 100 MiB
	}
}

// fixtureCorpus mirrors the audited shape in miniature: an active writer, a
// pinned historical session, witness-referenced sessions, completed unpinned
// bulk, and one dominating rollout — 190 MiB raw against a 100 MiB cap.
func fixtureCorpus() []SessionRecord {
	mib := int64(1 << 20)
	return []SessionRecord{
		{ID: "s-live", Bytes: 10 * mib, ModTime: day(90), Live: true},
		{ID: "s-fresh", Bytes: 5 * mib, ModTime: day(1)},
		{ID: "s-pinned", Bytes: 30 * mib, ModTime: day(200), Protected: []ProtectReason{ProtectPin}},
		{ID: "s-goal", Bytes: 15 * mib, ModTime: day(120), Protected: []ProtectReason{ProtectActiveGoal}},
		{ID: "s-witness", Bytes: 10 * mib, ModTime: day(150), Protected: []ProtectReason{ProtectRefereeEvidence}},
		{ID: "s-issue", Bytes: 5 * mib, ModTime: day(60), Protected: []ProtectReason{ProtectUnresolvedIssue}},
		{ID: "s-warm-a", Bytes: 20 * mib, ModTime: day(10)},
		{ID: "s-warm-b", Bytes: 10 * mib, ModTime: day(20)},
		{ID: "s-old-huge", Bytes: 60 * mib, ModTime: day(180)},
		{ID: "s-old-mid", Bytes: 20 * mib, ModTime: day(90)},
		{ID: "s-old-small", Bytes: 5 * mib, ModTime: day(45)},
		{ID: "s-compacted", Bytes: 1 * mib, ModTime: day(300), Compacted: true},
	}
}

// TestRetentionManifestBoundHolds is the core #4765 bound: planning the fixture
// corpus under a 100 MiB cap must leave retained raw bytes at or under the
// cap, reclaim only unprotected non-active sessions oldest-first, and emit
// arithmetic-consistent receipts.
func TestRetentionManifestBoundHolds(t *testing.T) {
	plan, err := DecideRetention(fixtureCorpus(), retPolicy())
	if err != nil {
		t.Fatalf("DecideRetention: %v", err)
	}
	if !plan.CapSatisfied {
		t.Fatalf("cap should be satisfiable on this corpus; plan: %+v", plan)
	}
	if plan.AfterBytes > plan.RawBytesCap {
		t.Fatalf("bound violated: after=%d > cap=%d", plan.AfterBytes, plan.RawBytesCap)
	}
	if plan.BeforeBytes-plan.ReclaimedBytes != plan.AfterBytes {
		t.Fatalf("receipt arithmetic broken: before=%d reclaimed=%d after=%d",
			plan.BeforeBytes, plan.ReclaimedBytes, plan.AfterBytes)
	}
	byID := map[string]Decision{}
	for _, d := range plan.Decisions {
		byID[d.ID] = d
	}
	// Active writers and protected witnesses must never be selected.
	for _, id := range []string{"s-live", "s-fresh", "s-pinned", "s-goal", "s-witness", "s-issue"} {
		if byID[id].Expire {
			t.Fatalf("%s must never be expired (class=%s)", id, byID[id].Class)
		}
	}
	if byID["s-live"].Class != ClassActive || byID["s-fresh"].Class != ClassActive {
		t.Fatalf("live/fresh sessions must classify active: %+v %+v", byID["s-live"], byID["s-fresh"])
	}
	if byID["s-pinned"].Class != ClassWarmEvidence {
		t.Fatalf("pinned session must stay warm evidence, got %s", byID["s-pinned"].Class)
	}
	if byID["s-compacted"].Class != ClassCompacted || byID["s-compacted"].Expire {
		t.Fatalf("already-compacted aggregate must be retained as-is: %+v", byID["s-compacted"])
	}
	// Age expiry alone (older than WarmWithin, unprotected) marks the old bulk.
	for _, id := range []string{"s-old-huge", "s-old-mid", "s-old-small"} {
		d := byID[id]
		if d.Class != ClassExpired || !d.Expire {
			t.Fatalf("%s should expire by age, got %+v", id, d)
		}
	}
	// Oldest-first: the cap is already met by age expiry here (190-85=105 raw
	// retained > 100, so exactly one warm session — the OLDER s-warm-b — must
	// also be taken, and s-warm-a must survive).
	if !byID["s-warm-b"].Expire {
		t.Fatalf("cap requires expiring the oldest unprotected warm session s-warm-b: %+v", byID["s-warm-b"])
	}
	if byID["s-warm-a"].Expire {
		t.Fatalf("s-warm-a is newer than s-warm-b and the cap is met without it: %+v", byID["s-warm-a"])
	}
}

// TestRetentionNeverSacrificesWitnesses: when protected evidence alone
// exceeds the cap, the plan must NOT expire it; it reports the bound unmet
// with the protected overage, so the operator sees the truth instead of a
// fabricated success.
func TestRetentionNeverSacrificesWitnesses(t *testing.T) {
	mib := int64(1 << 20)
	pol := retPolicy()
	pol.RawBytesCap = 10 * mib
	sessions := []SessionRecord{
		{ID: "w1", Bytes: 20 * mib, ModTime: day(100), Protected: []ProtectReason{ProtectRefereeEvidence}},
		{ID: "w2", Bytes: 15 * mib, ModTime: day(200), Protected: []ProtectReason{ProtectPin}},
		{ID: "junk", Bytes: 5 * mib, ModTime: day(400)},
	}
	plan, err := DecideRetention(sessions, pol)
	if err != nil {
		t.Fatalf("DecideRetention: %v", err)
	}
	if plan.CapSatisfied {
		t.Fatalf("cap cannot be met without destroying witnesses; plan must say so: %+v", plan)
	}
	if plan.ProtectedOverCapBytes != 25*mib {
		t.Fatalf("protected overage receipt wrong: got %d want %d", plan.ProtectedOverCapBytes, 25*mib)
	}
	for _, d := range plan.Decisions {
		if len(d.Protected) > 0 && d.Expire {
			t.Fatalf("protected session %s selected for expiry", d.ID)
		}
	}
	// The unprotected junk still gets reclaimed.
	for _, d := range plan.Decisions {
		if d.ID == "junk" && !d.Expire {
			t.Fatalf("unprotected expired-age session must still be reclaimed: %+v", d)
		}
	}
}

// TestRetentionManifestDeterministicManifest: identical input yields an identical
// machine-readable manifest (stable ordering), and the manifest round-trips
// through JSON without loss of the retained-evidence receipt.
func TestRetentionManifestDeterministicManifest(t *testing.T) {
	a, err := DecideRetention(fixtureCorpus(), retPolicy())
	if err != nil {
		t.Fatalf("DecideRetention: %v", err)
	}
	b, err := DecideRetention(fixtureCorpus(), retPolicy())
	if err != nil {
		t.Fatalf("DecideRetention: %v", err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("plan is not deterministic:\n%+v\n%+v", a, b)
	}
	raw, err := json.Marshal(a)
	if err != nil {
		t.Fatalf("manifest must be machine-readable: %v", err)
	}
	var back RetentionManifest
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("manifest round-trip: %v", err)
	}
	if back.AfterBytes != a.AfterBytes || len(back.Decisions) != len(a.Decisions) {
		t.Fatalf("manifest lost receipt data: %+v vs %+v", back, a)
	}
}

// TestRetentionRejectsClocklessPolicy: determinism is a contract — a policy
// without an injected Now is refused rather than silently reading the clock.
func TestRetentionRejectsClocklessPolicy(t *testing.T) {
	pol := retPolicy()
	pol.Now = time.Time{}
	if _, err := DecideRetention(fixtureCorpus(), pol); err == nil {
		t.Fatal("clockless policy must be refused")
	}
}

// TestQuarantineBoundByAgeAndBytes: the #4765 grace store is itself bounded —
// entries past the grace age purge, and the byte bound keeps the newest
// entries, with receipts that add up.
func TestQuarantineBoundByAgeAndBytes(t *testing.T) {
	mib := int64(1 << 20)
	items := []QuarantineItem{
		{ID: "q-old", Bytes: 10 * mib, QuarantinedAt: day(40)},
		{ID: "q-big", Bytes: 30 * mib, QuarantinedAt: day(5)},
		{ID: "q-mid", Bytes: 15 * mib, QuarantinedAt: day(3)},
		{ID: "q-new", Bytes: 5 * mib, QuarantinedAt: day(1)},
	}
	rec, err := BoundQuarantine(items, QuarantineBound{
		Now:      retNow,
		MaxAge:   30 * 24 * time.Hour,
		MaxBytes: 20 * mib,
	})
	if err != nil {
		t.Fatalf("BoundQuarantine: %v", err)
	}
	purged := map[string]bool{}
	for _, p := range rec.Purge {
		purged[p.ID] = true
	}
	if !purged["q-old"] {
		t.Fatal("entry past grace age must purge")
	}
	if !purged["q-big"] {
		t.Fatal("byte bound must purge oldest-first until under MaxBytes")
	}
	if purged["q-new"] || purged["q-mid"] {
		t.Fatalf("newest entries within both bounds must be kept: %+v", rec.Purge)
	}
	if rec.AfterBytes != 20*mib || rec.AfterBytes > 20*mib {
		t.Fatalf("quarantine bound violated: after=%d", rec.AfterBytes)
	}
	if rec.BeforeBytes-rec.ReclaimedBytes != rec.AfterBytes {
		t.Fatalf("quarantine receipt arithmetic broken: %+v", rec)
	}
}
