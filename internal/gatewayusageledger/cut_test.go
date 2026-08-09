package gatewayusageledger

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// cutTestRow builds a distinguishable row: every int64/uint64 counter field is
// set to base via reflection so a sum mismatch on ANY field — including one
// added after this test was written — fails the preservation check.
func cutTestRow(t *testing.T, kind, sessionType string, ms int64, base int64) Row {
	t.Helper()
	var c Counters
	cv := reflect.ValueOf(&c).Elem()
	for i := 0; i < cv.NumField(); i++ {
		f := cv.Field(i)
		switch f.Kind() {
		case reflect.Int64:
			f.SetInt(base)
		case reflect.Uint64:
			f.SetUint(uint64(base))
		case reflect.Bool:
			// Intent flags (#4349) fold by OR, so an all-true corpus keeps every field
			// true through a cut — the same "no counter total is lost" check as the
			// numeric fields, expressed in the only algebra a flag has.
			f.SetBool(true)
		case reflect.Map:
			m := reflect.MakeMap(f.Type())
			m.SetMapIndex(reflect.ValueOf("reason_a"), reflect.ValueOf(uint64(base)))
			f.Set(m)
		default:
			t.Fatalf("Counters field %s has kind %s that neither sumCounters nor this test knows — extend both", cv.Type().Field(i).Name, f.Kind())
		}
	}
	return NewRow(kind, sessionType, "test", "", 0, nil, c, time.UnixMilli(ms))
}

// totalCounters folds every row's Counters (expanding nothing — carryforward
// rows count via their summed Counters) into one grand total per field.
func totalCounters(t *testing.T, rows []Row) Counters {
	t.Helper()
	var total Counters
	for _, r := range rows {
		if err := sumCounters(&total, r.Counters); err != nil {
			t.Fatalf("sumCounters: %v", err)
		}
	}
	return total
}

func TestCutFoldsAndPreservesCounterSums(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "gateway-usage.jsonl")
	// 8 rows across (exit,serve) (exit,guard) (periodic,serve): fold all but 2.
	specs := []struct {
		kind, st string
		ms, base int64
	}{
		{"exit", "serve", 1000, 1},
		{"periodic", "serve", 2000, 2},
		{"exit", "guard", 3000, 3},
		{"exit", "serve", 4000, 4},
		{"periodic", "serve", 5000, 5},
		{"exit", "guard", 6000, 6},
		{"exit", "serve", 7000, 7},
		{"exit", "guard", 8000, 8},
	}
	for _, s := range specs {
		if err := Append(path, cutTestRow(t, s.kind, s.st, s.ms, s.base)); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	before := totalCounters(t, ReadLedgerFile(path))

	res, err := Cut(path, 2, false, time.UnixMilli(9000))
	if err != nil {
		t.Fatalf("Cut: %v", err)
	}
	if !res.Performed || res.DryRun {
		t.Fatalf("Cut should have performed a real cut: %+v", res)
	}
	if res.RowsBefore != 8 || res.FoldedRows != 6 || res.KeptRows != 2 {
		t.Fatalf("unexpected fold shape: %+v", res)
	}

	rows := ReadLedgerFile(path)
	if len(rows) != res.RowsAfter {
		t.Fatalf("RowsAfter=%d but re-read %d rows", res.RowsAfter, len(rows))
	}
	// Folded region held (exit,serve)x2 {1,4}, (periodic,serve)x2 {2,5}, (exit,guard)x2 {3,6}
	// -> exactly 3 carryforward rows, then the 2 kept rows (7000, 8000).
	var carries, kept []Row
	for _, r := range rows {
		if r.Kind == KindCarryforward {
			carries = append(carries, r)
		} else {
			kept = append(kept, r)
		}
	}
	if len(carries) != 3 || len(kept) != 2 {
		t.Fatalf("want 3 carryforward + 2 kept rows, got %d + %d", len(carries), len(kept))
	}
	for _, cr := range carries {
		if cr.Carryforward == nil || cr.Carryforward.FoldedRows != 2 {
			t.Fatalf("carryforward witness wrong: %+v", cr.Carryforward)
		}
		if cr.SessionID != "" || cr.PID != 0 {
			t.Fatalf("carryforward must not look like a live session: %+v", cr)
		}
	}
	if kept[0].UnixMillis != 7000 || kept[1].UnixMillis != 8000 {
		t.Fatalf("kept tail wrong: %d, %d", kept[0].UnixMillis, kept[1].UnixMillis)
	}

	// The load-bearing contract: whole-file counter totals are IDENTICAL across
	// the cut, on every field including maps.
	after := totalCounters(t, rows)
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("counter totals changed across cut:\nbefore %+v\nafter  %+v", before, after)
	}
	if res.BytesAfter >= res.BytesBefore {
		t.Fatalf("cut did not shrink the file: %d -> %d bytes", res.BytesBefore, res.BytesAfter)
	}
}

func TestCutReCutMergesCarryforward(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "gateway-usage.jsonl")
	for i := int64(1); i <= 6; i++ {
		if err := Append(path, cutTestRow(t, "exit", "serve", i*1000, i)); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	before := totalCounters(t, ReadLedgerFile(path))
	if _, err := Cut(path, 3, false, time.UnixMilli(7000)); err != nil {
		t.Fatalf("first Cut: %v", err)
	}
	// Append 2 more, cut again keeping 2: the old carryforward is inside the new
	// folded region and must MERGE, not stack.
	for i := int64(7); i <= 8; i++ {
		if err := Append(path, cutTestRow(t, "exit", "serve", i*1000, i)); err != nil {
			t.Fatalf("Append: %v", err)
		}
		if err := sumCounters(&before, cutTestRow(t, "exit", "serve", i*1000, i).Counters); err != nil {
			t.Fatalf("sumCounters: %v", err)
		}
	}
	if _, err := Cut(path, 2, false, time.UnixMilli(9000)); err != nil {
		t.Fatalf("second Cut: %v", err)
	}
	rows := ReadLedgerFile(path)
	var carries int
	for _, r := range rows {
		if r.Kind == KindCarryforward {
			carries++
			if r.Carryforward.FoldedRows != 6 || r.Carryforward.FirstUnixMillis != 1000 || r.Carryforward.LastUnixMillis != 6000 {
				t.Fatalf("merged carryforward witness wrong: %+v", r.Carryforward)
			}
		}
	}
	if carries != 1 {
		t.Fatalf("re-cut must merge into ONE carryforward row, got %d", carries)
	}
	after := totalCounters(t, rows)
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("counter totals changed across re-cut:\nbefore %+v\nafter  %+v", before, after)
	}
}

func TestCutNoOpAndDryRunWriteNothing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "gateway-usage.jsonl")

	// Missing file: clean no-op.
	res, err := Cut(path, 5, false, time.UnixMilli(1))
	if err != nil || res.Performed {
		t.Fatalf("missing-file Cut must be a clean no-op: %+v, %v", res, err)
	}

	for i := int64(1); i <= 3; i++ {
		if err := Append(path, cutTestRow(t, "exit", "serve", i*1000, i)); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	orig, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}

	// rows <= keep: no-op, bytes untouched.
	res, err = Cut(path, 3, false, time.UnixMilli(9000))
	if err != nil || res.Performed {
		t.Fatalf("small-file Cut must be a no-op: %+v, %v", res, err)
	}

	// Dry run on a cuttable file: full report, bytes untouched.
	res, err = Cut(path, 1, true, time.UnixMilli(9000))
	if err != nil {
		t.Fatalf("dry-run Cut: %v", err)
	}
	if !res.Performed || !res.DryRun || res.FoldedRows != 2 {
		t.Fatalf("dry run must report the would-be cut: %+v", res)
	}
	now, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(now) != string(orig) {
		t.Fatalf("dry run modified the ledger")
	}

	if _, err := Cut(path, -1, true, time.UnixMilli(1)); err == nil {
		t.Fatalf("negative keep must error")
	}
}

func TestCutPreservesUndecodableLines(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "gateway-usage.jsonl")
	if err := Append(path, cutTestRow(t, "exit", "serve", 1000, 1)); err != nil {
		t.Fatalf("Append: %v", err)
	}
	foreign := "not json at all"
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if _, err := f.WriteString(foreign + "\n"); err != nil {
		t.Fatalf("write: %v", err)
	}
	f.Close()
	for i := int64(2); i <= 3; i++ {
		if err := Append(path, cutTestRow(t, "exit", "serve", i*1000, i)); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if _, err := Cut(path, 1, false, time.UnixMilli(9000)); err != nil {
		t.Fatalf("Cut: %v", err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !strings.Contains(string(b), foreign) {
		t.Fatalf("cut destroyed an undecodable line it cannot account for:\n%s", b)
	}
}

func TestFoldTrendSkipsCarryforwardRows(t *testing.T) {
	carry := Row{Schema: Schema, Kind: KindCarryforward, SessionType: "serve", UnixMillis: 500,
		Counters:     Counters{InputTokens: 999999},
		Carryforward: &Carryforward{FoldedKind: "exit", FoldedRows: 10, FirstUnixMillis: 100, LastUnixMillis: 500}}
	a := NewRow("exit", "serve", "t", "", 0, nil, Counters{InputTokens: 10}, time.UnixMilli(1000))
	b := NewRow("exit", "serve", "t", "", 0, nil, Counters{InputTokens: 30}, time.UnixMilli(2000))

	trend, ok := FoldTrend([]Row{carry, a, b})
	if !ok {
		t.Fatalf("FoldTrend should fold the two real rows")
	}
	if trend.First.Kind == KindCarryforward || trend.Sessions != 2 || trend.DeltaInputTokens != 20 {
		t.Fatalf("carryforward leaked into the trend: %+v", trend)
	}
	if _, ok := FoldTrend([]Row{carry, a}); ok {
		t.Fatalf("one real row + a carryforward must not trend")
	}
}
