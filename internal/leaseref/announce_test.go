package leaseref

import (
	"reflect"
	"strings"
	"testing"
)

// TestAnnounceRoundTrip is the witness the issue named (#2300): a record RENDERED to a
// comment body and PARSED back is byte-for-byte the same record. It exercises every
// field combination — a full acquire, a fenced renew, a minimal release with no tree/TTL
// — so the omitempty fields are covered in both the present and absent case.
func TestAnnounceRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		rec  AnnounceRecord
	}{
		{
			name: "full acquire",
			rec: AnnounceRecord{
				LeaseID:    "internal-foo",
				Holder:     "node-a/sess-1",
				Generation: 3,
				Tree:       []string{"internal/foo/**", "cmd/foo/**"},
				TTLSeconds: 3600,
				Action:     AnnounceAcquire,
			},
		},
		{
			name: "fenced renew",
			rec: AnnounceRecord{
				LeaseID:    "bar",
				Holder:     "nodeB/sess-2",
				Generation: 12,
				Tree:       []string{"internal/bar/**"},
				TTLSeconds: 900,
				Action:     AnnounceRenew,
			},
		},
		{
			name: "minimal release, no tree/ttl/generation",
			rec: AnnounceRecord{
				LeaseID: "baz",
				Holder:  "legacy-holder",
				Action:  AnnounceRelease,
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := RenderAnnounce(tc.rec)
			got, ok := ParseAnnounce(body)
			if !ok {
				t.Fatalf("ParseAnnounce returned ok=false for a body RenderAnnounce produced:\n%s", body)
			}
			if !reflect.DeepEqual(got, tc.rec) {
				t.Fatalf("round-trip mismatch\n want: %+v\n  got: %+v\n body:\n%s", tc.rec, got, body)
			}
		})
	}
}

// TestRenderAnnounceShape asserts the two structural contracts the acceptance calls out:
// the body is human-legible (a summary line naming the action + lease) AND carries
// exactly ONE fenced JSON line the parser reads back.
func TestRenderAnnounceShape(t *testing.T) {
	rec := AnnounceRecord{LeaseID: "foo", Holder: "h", Generation: 1, Tree: []string{"internal/foo/**"}, TTLSeconds: 60, Action: AnnounceAcquire}
	body := RenderAnnounce(rec)
	if !strings.Contains(body, "leaseref announce — acquire") {
		t.Errorf("body missing human summary line:\n%s", body)
	}
	if n := strings.Count(body, "```"); n != 2 {
		t.Errorf("want exactly one fenced block (2 fences), got %d:\n%s", n, body)
	}
	// The schema tag must ride inside the JSON, not the fence info string.
	if !strings.Contains(body, `"schema":"`+AnnounceSchema+`"`) {
		t.Errorf("body missing in-JSON schema tag:\n%s", body)
	}
	// Exactly one line carries the schema tag (the single JSON line).
	jsonLines := 0
	for _, line := range strings.Split(body, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "{") && strings.Contains(line, AnnounceSchema) {
			jsonLines++
		}
	}
	if jsonLines != 1 {
		t.Errorf("want exactly one JSON announce line, got %d:\n%s", jsonLines, body)
	}
}

// TestParseAnnounceIgnoresNonAnnounce confirms the parser is silent on comments that are
// not fak announces — an ordinary discussion comment, an empty body, and a JSON code
// block that is not a fak announce all yield ok=false.
func TestParseAnnounceIgnoresNonAnnounce(t *testing.T) {
	for _, body := range []string{
		"",
		"Looks good to me, shipping now.",
		"```json\n{\"some\":\"other\",\"json\":1}\n```",
		// right shape, wrong action (out of vocabulary) -> dropped
		"```json\n{\"schema\":\"" + AnnounceSchema + "\",\"lease_id\":\"x\",\"holder\":\"h\",\"action\":\"steal\"}\n```",
	} {
		if _, ok := ParseAnnounce(body); ok {
			t.Errorf("ParseAnnounce should ignore non-announce body:\n%q", body)
		}
	}
}

// TestFoldAnnouncements folds a chronological comment stream into the advisory held-set:
// the latest per lease id, with a release removing the lease from the view.
func TestFoldAnnouncements(t *testing.T) {
	bodies := []string{
		RenderAnnounce(AnnounceRecord{LeaseID: "a", Holder: "h1", Generation: 1, Tree: []string{"internal/a/**"}, TTLSeconds: 60, Action: AnnounceAcquire}),
		RenderAnnounce(AnnounceRecord{LeaseID: "b", Holder: "h2", Generation: 1, Tree: []string{"internal/b/**"}, TTLSeconds: 60, Action: AnnounceAcquire}),
		"a plain human comment that carries no announce",
		// a renews (generation bumps to 2) — the latest wins
		RenderAnnounce(AnnounceRecord{LeaseID: "a", Holder: "h1", Generation: 2, Tree: []string{"internal/a/**"}, TTLSeconds: 60, Action: AnnounceRenew}),
		// b is released — drops out of the held view
		RenderAnnounce(AnnounceRecord{LeaseID: "b", Holder: "h2", Action: AnnounceRelease}),
	}
	view := FoldAnnouncements(bodies)
	if len(view) != 1 {
		t.Fatalf("want 1 held lease after fold (b released), got %d: %+v", len(view), view)
	}
	if view[0].LeaseID != "a" || view[0].Generation != 2 || view[0].Action != AnnounceRenew {
		t.Fatalf("fold kept the wrong/stale record for lease a: %+v", view[0])
	}
}

// TestFoldAnnouncementsReacquireAfterRelease confirms a release is not permanent in the
// fold: a lease released then re-acquired later shows as held again (the view reflects
// the latest transition, not a tombstone).
func TestFoldAnnouncementsReacquireAfterRelease(t *testing.T) {
	bodies := []string{
		RenderAnnounce(AnnounceRecord{LeaseID: "c", Holder: "h", Action: AnnounceAcquire}),
		RenderAnnounce(AnnounceRecord{LeaseID: "c", Holder: "h", Action: AnnounceRelease}),
		RenderAnnounce(AnnounceRecord{LeaseID: "c", Holder: "h2", Generation: 5, Action: AnnounceAcquire}),
	}
	view := FoldAnnouncements(bodies)
	if len(view) != 1 || view[0].Holder != "h2" || view[0].Generation != 5 {
		t.Fatalf("re-acquire after release should show held by h2 gen 5, got: %+v", view)
	}
}

// TestAnnounceFromRecord confirms the one-vocabulary bridge maps a lock-lease Record onto
// the announce vocabulary field-for-field.
func TestAnnounceFromRecord(t *testing.T) {
	rec := Record{ID: "lane", Holder: "node/sess", Generation: 4, TreeGlobs: []string{"internal/lane/**"}, TTLSeconds: 1800}
	got := AnnounceFromRecord(rec, AnnounceAcquire)
	want := AnnounceRecord{LeaseID: "lane", Holder: "node/sess", Generation: 4, Tree: []string{"internal/lane/**"}, TTLSeconds: 1800, Action: AnnounceAcquire}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("AnnounceFromRecord mismatch\n want: %+v\n  got: %+v", want, got)
	}
}

func TestPublicSafeAnnounceHidesRawValuesAndRoundTrips(t *testing.T) {
	raw := AnnounceRecord{LeaseID: "lease-secret", Holder: "workstation/session-9", Generation: 7, Tree: []string{"internal/private/**", "cmd/fak/**"}, TTLSeconds: 90, Action: "acquire"}
	a, err := PublicSafeAnnounce(raw, []byte("shared-test-key"))
	if err != nil {
		t.Fatal(err)
	}
	b, err := PublicSafeAnnounce(raw, []byte("shared-test-key"))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("same shared key must match across nodes:\n%+v\n%+v", a, b)
	}
	if a.LeaseFingerprint == a.HolderFingerprint {
		t.Fatal("domain separation failed")
	}
	body := RenderAnnounce(a)
	for _, secret := range []string{raw.LeaseID, raw.Holder, raw.Tree[0], raw.Tree[1], "shared-test-key"} {
		if strings.Contains(body, secret) {
			t.Fatalf("public body leaked %q:\n%s", secret, body)
		}
	}
	if !strings.Contains(body, PublicAnnounceSchema) {
		t.Fatalf("missing public schema: %s", body)
	}
	got, ok := ParseAnnounce(body)
	if !ok || !reflect.DeepEqual(got, a) {
		t.Fatalf("round trip got (%+v, %v), want %+v", got, ok, a)
	}
}

func TestPublicSafeAnnounceFoldReleaseUsesFingerprint(t *testing.T) {
	raw := AnnounceRecord{LeaseID: "L", Holder: "node/session", Tree: []string{"internal/leaseref/**"}, TTLSeconds: 60, Action: "acquire"}
	acquire, _ := PublicSafeAnnounce(raw, []byte("key"))
	raw.Action = "release"
	release, _ := PublicSafeAnnounce(raw, []byte("key"))
	if got := FoldAnnouncements([]string{RenderAnnounce(acquire), RenderAnnounce(release)}); len(got) != 0 {
		t.Fatalf("held = %+v, want empty", got)
	}
}

func TestPublicSafeAnnounceRejectsEmptyKey(t *testing.T) {
	if _, err := PublicSafeAnnounce(AnnounceRecord{LeaseID: "L"}, nil); err == nil {
		t.Fatal("expected empty-key error")
	}
}
