package resume

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

// FoldIdentity is a total function that folds append-only pairing rows into the two
// join directions: last row per key wins in each, a row missing either endpoint is
// skipped in both, and nil/empty input yields empty (non-nil) maps.
func TestFoldIdentity(t *testing.T) {
	cases := []struct {
		name        string
		rows        []IdentityRow
		wantByUUID  map[string]string
		wantByTrace map[string]string
	}{
		{
			name:        "nil rows yield empty maps",
			rows:        nil,
			wantByUUID:  map[string]string{},
			wantByTrace: map[string]string{},
		},
		{
			name:        "empty rows yield empty maps",
			rows:        []IdentityRow{},
			wantByUUID:  map[string]string{},
			wantByTrace: map[string]string{},
		},
		{
			name:        "single pairing maps both directions",
			rows:        []IdentityRow{{UUID: "u1", Trace: "t1"}},
			wantByUUID:  map[string]string{"u1": "t1"},
			wantByTrace: map[string]string{"t1": "u1"},
		},
		{
			name: "last row wins on repeated uuid",
			rows: []IdentityRow{
				{UUID: "u1", Trace: "t1"},
				{UUID: "u1", Trace: "t2"},
			},
			wantByUUID: map[string]string{"u1": "t2"},
			// both traces are seen, each pointing at u1
			wantByTrace: map[string]string{"t1": "u1", "t2": "u1"},
		},
		{
			name: "last row wins on repeated trace",
			rows: []IdentityRow{
				{UUID: "u1", Trace: "t1"},
				{UUID: "u2", Trace: "t1"},
			},
			wantByUUID:  map[string]string{"u1": "t1", "u2": "t1"},
			wantByTrace: map[string]string{"t1": "u2"},
		},
		{
			name: "blank uuid row is skipped in both directions",
			rows: []IdentityRow{
				{UUID: "u1", Trace: "t1"},
				{UUID: "", Trace: "t9"},
			},
			wantByUUID:  map[string]string{"u1": "t1"},
			wantByTrace: map[string]string{"t1": "u1"},
		},
		{
			name: "blank trace row is skipped in both directions",
			rows: []IdentityRow{
				{UUID: "u1", Trace: "t1"},
				{UUID: "u9", Trace: ""},
			},
			wantByUUID:  map[string]string{"u1": "t1"},
			wantByTrace: map[string]string{"t1": "u1"},
		},
		{
			name: "both-blank row is skipped",
			rows: []IdentityRow{
				{UUID: "u1", Trace: "t1"},
				{UUID: "  ", Trace: "  "},
			},
			wantByUUID:  map[string]string{"u1": "t1"},
			wantByTrace: map[string]string{"t1": "u1"},
		},
		{
			name: "whitespace endpoints are trimmed before join",
			rows: []IdentityRow{
				{UUID: " u1 ", Trace: " t1 "},
			},
			wantByUUID:  map[string]string{"u1": "t1"},
			wantByTrace: map[string]string{"t1": "u1"},
		},
		{
			name: "a half row never clobbers a prior valid pairing",
			rows: []IdentityRow{
				{UUID: "u1", Trace: "t1"},
				{UUID: "u1", Trace: ""},
			},
			wantByUUID:  map[string]string{"u1": "t1"},
			wantByTrace: map[string]string{"t1": "u1"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotByUUID, gotByTrace := FoldIdentity(tc.rows)
			if gotByUUID == nil || gotByTrace == nil {
				t.Fatalf("FoldIdentity returned a nil map: byUUID=%v byTrace=%v", gotByUUID, gotByTrace)
			}
			assertMapEqual(t, "traceByUUID", gotByUUID, tc.wantByUUID)
			assertMapEqual(t, "uuidByTrace", gotByTrace, tc.wantByTrace)
		})
	}
}

// A forward-extended row (extra unknown fields) still decodes through the same
// jsonlledger.Parse the loader uses — the store can grow new columns without
// reddening this fold.
func TestFoldIdentityForwardExtended(t *testing.T) {
	content := `{"ts":"2026-07-10T00:00:00Z","uuid":"u1","trace":"t1","handle":"h","account":"a","via":"guard SessionStart"}
{"uuid":"u2","trace":"t2","future_field":{"nested":true},"another":42}
not-json-should-be-skipped

{"uuid":"","trace":"t3"}`
	rows := jsonlledger.Parse[IdentityRow](content, nil)
	byUUID, byTrace := FoldIdentity(rows)
	wantByUUID := map[string]string{"u1": "t1", "u2": "t2"}
	wantByTrace := map[string]string{"t1": "u1", "t2": "u2"}
	assertMapEqual(t, "traceByUUID", byUUID, wantByUUID)
	assertMapEqual(t, "uuidByTrace", byTrace, wantByTrace)
}

// IdentityLedgerPath names resume_identity.jsonl under the given regDir, and
// LoadIdentity folds it — a missing file yielding empty (non-nil) maps (fail-open),
// a present append-only file folding last-row-wins through the shared parser.
func TestLoadIdentity(t *testing.T) {
	dir := t.TempDir()

	if got, want := IdentityLedgerPath(dir), filepath.Join(dir, "resume_identity.jsonl"); got != want {
		t.Fatalf("IdentityLedgerPath = %q, want %q", got, want)
	}

	// Missing store: fail-open to empty (non-nil) maps, never a nil deref.
	byUUID, byTrace := LoadIdentity(dir)
	if byUUID == nil || byTrace == nil {
		t.Fatalf("LoadIdentity on missing store returned nil map: byUUID=%v byTrace=%v", byUUID, byTrace)
	}
	if len(byUUID) != 0 || len(byTrace) != 0 {
		t.Fatalf("LoadIdentity on missing store = %v/%v, want empty", byUUID, byTrace)
	}

	// Present store: append-only, last-row-wins re-pairing of u1.
	content := `{"uuid":"u1","trace":"t1","via":"guard SessionStart"}
{"uuid":"u2","trace":"t2"}
{"uuid":"u1","trace":"t3"}
`
	if err := os.WriteFile(IdentityLedgerPath(dir), []byte(content), 0o644); err != nil {
		t.Fatalf("write store: %v", err)
	}
	byUUID, byTrace = LoadIdentity(dir)
	assertMapEqual(t, "traceByUUID", byUUID, map[string]string{"u1": "t3", "u2": "t2"})
	assertMapEqual(t, "uuidByTrace", byTrace, map[string]string{"t1": "u1", "t2": "u2", "t3": "u1"})
}

func TestLoadIdentityRowsStrictReportsMalformedAuthorityAndProviderProvenance(t *testing.T) {
	dir := t.TempDir()
	content := `{"ts":"2026-08-25T10:00:00Z","uuid":"u1","trace":"t1","provider":"codex","source":"startup"}
not-json
`
	if err := os.WriteFile(IdentityLedgerPath(dir), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	rows, invalid, err := LoadIdentityRowsStrict(dir)
	if err != nil {
		t.Fatal(err)
	}
	if invalid != 1 || len(rows) != 1 || rows[0].Provider != "codex" || rows[0].Source != "startup" {
		t.Fatalf("rows=%+v invalid=%d", rows, invalid)
	}
}

// ResolveIdentity resolves a query id against the append-only rows in either direction,
// honoring last-row-wins, skipping half rows, surfacing the winning row's provenance, and
// reporting OK=false (never inventing a join) for a blank or unknown query.
func TestResolveIdentity(t *testing.T) {
	rows := []IdentityRow{
		{UUID: "u1", Trace: "t1", Handle: "old"},
		{UUID: "u2", Trace: "t2", Account: "worker-a", Via: "guard SessionStart"},
		{UUID: "u1", Trace: "t3", Handle: "new"}, // re-pairs u1: last row wins
		{UUID: "", Trace: "t9"},                  // half row: never a match
	}

	t.Run("uuid resolves forward to its newest trace", func(t *testing.T) {
		m := ResolveIdentity(rows, "u1")
		if !m.OK || m.Direction != "uuid->trace" || m.Paired != "t3" {
			t.Fatalf("resolve u1 = %+v, want ok uuid->trace t3", m)
		}
		if m.Row.Handle != "new" {
			t.Fatalf("expected the winning (last) row's provenance handle=new, got %q", m.Row.Handle)
		}
	})

	t.Run("trace resolves backward to its uuid with provenance", func(t *testing.T) {
		m := ResolveIdentity(rows, "t2")
		if !m.OK || m.Direction != "trace->uuid" || m.Paired != "u2" {
			t.Fatalf("resolve t2 = %+v, want ok trace->uuid u2", m)
		}
		if m.Row.Account != "worker-a" || m.Row.Via != "guard SessionStart" {
			t.Fatalf("provenance not carried: %+v", m.Row)
		}
	})

	t.Run("a superseded trace still resolves its uuid", func(t *testing.T) {
		// t1 was u1's first trace; even after u1 re-pairs to t3, the t1 row still joins to u1.
		m := ResolveIdentity(rows, "t1")
		if !m.OK || m.Direction != "trace->uuid" || m.Paired != "u1" {
			t.Fatalf("resolve t1 = %+v, want ok trace->uuid u1", m)
		}
	})

	t.Run("whitespace query is trimmed before resolving", func(t *testing.T) {
		if m := ResolveIdentity(rows, "  u2  "); !m.OK || m.Paired != "t2" {
			t.Fatalf("resolve padded u2 = %+v, want ok -> t2", m)
		}
	})

	t.Run("unknown query is a clean miss, never an invented join", func(t *testing.T) {
		if m := ResolveIdentity(rows, "nope"); m.OK {
			t.Fatalf("resolve unknown = %+v, want OK=false", m)
		}
	})

	t.Run("half-row endpoint never resolves", func(t *testing.T) {
		if m := ResolveIdentity(rows, "t9"); m.OK {
			t.Fatalf("a half row's trace must not resolve: %+v", m)
		}
	})

	t.Run("blank query and nil rows are misses", func(t *testing.T) {
		if m := ResolveIdentity(rows, "   "); m.OK {
			t.Fatalf("blank query resolved: %+v", m)
		}
		if m := ResolveIdentity(nil, "u1"); m.OK {
			t.Fatalf("nil rows resolved: %+v", m)
		}
	})
}

// LoadIdentityRows reads the store into its pre-fold rows (the form ResolveIdentity scans),
// yielding nil for a missing store and preserving append-only file order for a present one.
func TestLoadIdentityRows(t *testing.T) {
	dir := t.TempDir()

	if rows := LoadIdentityRows(dir); rows != nil {
		t.Fatalf("LoadIdentityRows on missing store = %v, want nil", rows)
	}

	content := `{"uuid":"u1","trace":"t1"}
{"uuid":"u2","trace":"t2","account":"worker-a"}
`
	if err := os.WriteFile(IdentityLedgerPath(dir), []byte(content), 0o644); err != nil {
		t.Fatalf("write store: %v", err)
	}
	rows := LoadIdentityRows(dir)
	if len(rows) != 2 || rows[0].UUID != "u1" || rows[1].Account != "worker-a" {
		t.Fatalf("LoadIdentityRows = %+v, want the two rows in file order", rows)
	}
	// Round-trips through ResolveIdentity end-to-end.
	if m := ResolveIdentity(rows, "u2"); !m.OK || m.Paired != "t2" || m.Row.Account != "worker-a" {
		t.Fatalf("resolve over loaded rows = %+v, want ok -> t2 (worker-a)", m)
	}
}

// FoldIdentityDriverPIDs answers "which process was witnessed driving this transcript?" —
// last recorded pid per uuid wins, an unrecorded pid contributes nothing and never clobbers a
// prior witness, and the join's Trace endpoint is not required for the (transcript, pid) fact.
func TestFoldIdentityDriverPIDs(t *testing.T) {
	cases := []struct {
		name string
		rows []IdentityRow
		want map[string]int
	}{
		{
			name: "nil rows yield an empty map",
			rows: nil,
			want: map[string]int{},
		},
		{
			name: "a witnessed pid is recorded per transcript",
			rows: []IdentityRow{{UUID: "u1", Trace: "t1", PID: 9001}, {UUID: "u2", Trace: "t2", PID: 9002}},
			want: map[string]int{"u1": 9001, "u2": 9002},
		},
		{
			// SessionStart re-fires on compact, so a transcript is observed repeatedly; the newest
			// witness is the one that owns it.
			name: "last recorded pid per uuid wins",
			rows: []IdentityRow{{UUID: "u1", Trace: "t1", PID: 9001}, {UUID: "u1", Trace: "t1", PID: 9007}},
			want: map[string]int{"u1": 9007},
		},
		{
			// The forward-compat case that motivates the whole field: every row written before it
			// existed decodes with PID 0. That is NOT RECORDED, and it must not erase a real one.
			name: "a row with no pid contributes nothing and clobbers nothing",
			rows: []IdentityRow{{UUID: "u1", Trace: "t1", PID: 9001}, {UUID: "u1", Trace: "t1"}, {UUID: "u2", Trace: "t2"}},
			want: map[string]int{"u1": 9001},
		},
		{
			name: "a negative pid is not a witness",
			rows: []IdentityRow{{UUID: "u1", Trace: "t1", PID: -1}},
			want: map[string]int{},
		},
		{
			// Unlike FoldIdentity, a missing trace does not disqualify the row: the fact folded
			// here is (transcript, driver pid), which the row still carries truthfully.
			name: "a row missing the trace endpoint still carries its pid",
			rows: []IdentityRow{{UUID: "u1", PID: 9001}},
			want: map[string]int{"u1": 9001},
		},
		{
			name: "a row missing the transcript has nothing to key on",
			rows: []IdentityRow{{Trace: "t1", PID: 9001}, {UUID: "  ", Trace: "t2", PID: 9002}},
			want: map[string]int{},
		},
		{
			// The consumer keys on the lowercased transcript stem, so the fold normalizes.
			name: "the transcript key is normalized to lower case",
			rows: []IdentityRow{{UUID: "U1-ABC", Trace: "t1", PID: 9001}},
			want: map[string]int{"u1-abc": 9001},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := FoldIdentityDriverPIDs(c.rows)
			if got == nil {
				t.Fatal("FoldIdentityDriverPIDs returned nil, want a non-nil map")
			}
			if len(got) != len(c.want) {
				t.Fatalf("got %v (len %d), want %v (len %d)", got, len(got), c.want, len(c.want))
			}
			for k, wv := range c.want {
				if gv, ok := got[k]; !ok || gv != wv {
					t.Fatalf("pid[%q] = %d (present=%v), want %d", k, gv, ok, wv)
				}
			}
		})
	}
}

// TestIdentityRowPIDWireForm pins the durable shape both directions: a row written before the
// pid field existed still decodes (as NOT RECORDED), and a row carrying one round-trips. The
// store is append-only, so an old row is not rewritable and the reader must handle it forever.
func TestIdentityRowPIDWireForm(t *testing.T) {
	dir := t.TempDir()
	content := `{"ts":"t","uuid":"u-legacy","trace":"guard","via":"guard-sessionstart"}
{"ts":"t","uuid":"u-witnessed","trace":"guard","pid":9001,"via":"guard-sessionstart"}
`
	if err := os.WriteFile(IdentityLedgerPath(dir), []byte(content), 0o644); err != nil {
		t.Fatalf("write store: %v", err)
	}
	rows := LoadIdentityRows(dir)
	if len(rows) != 2 {
		t.Fatalf("LoadIdentityRows = %+v, want 2 rows", rows)
	}
	if rows[0].PID != 0 {
		t.Fatalf("legacy row decoded PID = %d, want 0 (not recorded)", rows[0].PID)
	}
	if rows[1].PID != 9001 {
		t.Fatalf("forward row decoded PID = %d, want 9001", rows[1].PID)
	}
	// Both rows still join, so adding the field cost the store nothing it already answered.
	traceByUUID, _ := LoadIdentity(dir)
	if traceByUUID["u-legacy"] != "guard" || traceByUUID["u-witnessed"] != "guard" {
		t.Fatalf("the uuid<->trace join regressed: %v", traceByUUID)
	}
	pids := FoldIdentityDriverPIDs(rows)
	if _, ok := pids["u-legacy"]; ok {
		t.Fatalf("a legacy row must contribute no pid: %v", pids)
	}
	if pids["u-witnessed"] != 9001 {
		t.Fatalf("witnessed pid = %v, want 9001", pids)
	}
	// The pid is omitted from the wire form when nothing was witnessed, so an unwitnessed row
	// is byte-identical to the shape every prior reader already handles.
	raw, err := json.Marshal(IdentityRow{UUID: "u1", Trace: "t1"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(raw), "pid") {
		t.Fatalf("unwitnessed row carries a pid key: %s", raw)
	}
}

func assertMapEqual(t *testing.T, label string, got, want map[string]string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %v (len %d), want %v (len %d)", label, got, len(got), want, len(want))
	}
	for k, wv := range want {
		if gv, ok := got[k]; !ok || gv != wv {
			t.Fatalf("%s[%q] = %q (present=%v), want %q", label, k, gv, ok, wv)
		}
	}
}
