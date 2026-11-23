// fak-test:runtime fast est=10ms lane=default
// fak-test:justify why=contract when=changed:internal/perfledger/**
package perfledger

import (
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func newTestColumnStore(t *testing.T, limits ColumnStoreLimits) *ColumnStore {
	t.Helper()
	store, err := NewColumnStore(limits)
	if err != nil {
		t.Fatalf("NewColumnStore(%+v): %v", limits, err)
	}
	return store
}

func mustIngest(t *testing.T, store *ColumnStore, idx int, metrics map[string]any) {
	t.Helper()
	if err := store.Ingest(idx, float64(idx*10), float64(idx*10+5), metrics); err != nil {
		t.Fatalf("Ingest(%d): %v", idx, err)
	}
}

// sameFloats compares two columns treating NaN as equal to NaN.
func sameFloats(got, want []float64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if math.IsNaN(want[i]) != math.IsNaN(got[i]) {
			return false
		}
		if !math.IsNaN(want[i]) && got[i] != want[i] {
			return false
		}
	}
	return true
}

func TestColumnStoreSparseOutOfOrderIngest(t *testing.T) {
	store := newTestColumnStore(t, ColumnStoreLimits{InitialRows: 8})
	mustIngest(t, store, 3, map[string]any{"latency": 30.0, "tokens": 7})
	mustIngest(t, store, 0, map[string]any{"latency": 10.0})
	mustIngest(t, store, 5, map[string]any{"tokens": int64(9), "model": "m-a"})

	if store.Count() != 6 {
		t.Fatalf("Count = %d, want 6 (highest index + 1)", store.Count())
	}
	nan := math.NaN()
	if got, want := store.Numeric("latency"), []float64{10, nan, nan, 30, nan, nan}; !sameFloats(got, want) {
		t.Fatalf("latency = %v, want %v", got, want)
	}
	if got, want := store.Numeric("tokens"), []float64{nan, nan, nan, 7, nan, 9}; !sameFloats(got, want) {
		t.Fatalf("tokens = %v, want %v", got, want)
	}
	if got, want := store.Numeric("never_ingested"), []float64{nan, nan, nan, nan, nan, nan}; !sameFloats(got, want) {
		t.Fatalf("unknown tag = %v, want all NaN", got)
	}
	if store.NumericSum("latency") != 40 || store.NumericCount("latency") != 2 {
		t.Fatalf("latency sum/count = %v/%d, want 40/2", store.NumericSum("latency"), store.NumericCount("latency"))
	}
	if got, want := store.NumericTags(), []string{"latency", "tokens"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("NumericTags = %v, want %v", got, want)
	}
	if got, ok := store.Text("model", 5); !ok || got != "m-a" {
		t.Fatalf("Text(model, 5) = %q, %v", got, ok)
	}
	if _, ok := store.Text("model", 3); ok {
		t.Fatal("Text reported a value for a record that never wrote one")
	}

	// The returned column is a copy: a caller cannot reach into the store.
	store.Numeric("latency")[0] = -1
	if store.Numeric("latency")[0] != 10 {
		t.Fatal("mutating a returned column changed the store")
	}
}

func TestColumnStoreRewriteKeepsRunningStats(t *testing.T) {
	store := newTestColumnStore(t, ColumnStoreLimits{InitialRows: 4})
	mustIngest(t, store, 1, map[string]any{"latency": 10.0})
	mustIngest(t, store, 2, map[string]any{"latency": 5.0})
	mustIngest(t, store, 1, map[string]any{"latency": 25.0})
	if store.NumericSum("latency") != 30 || store.NumericCount("latency") != 2 {
		t.Fatalf("after rewrite sum/count = %v/%d, want 30/2", store.NumericSum("latency"), store.NumericCount("latency"))
	}
	mustIngest(t, store, 2, map[string]any{"latency": math.NaN()})
	if store.NumericSum("latency") != 25 || store.NumericCount("latency") != 1 {
		t.Fatalf("after NaN rewrite sum/count = %v/%d, want 25/1", store.NumericSum("latency"), store.NumericCount("latency"))
	}

	// Replacing the infinite cell below would leave Inf-Inf = NaN under
	// subtract-on-rewrite, and replacing the large one 0 instead of 1.
	mustIngest(t, store, 2, map[string]any{"latency": math.Inf(1)})
	if !math.IsInf(store.NumericSum("latency"), 1) || store.NumericCount("latency") != 2 {
		t.Fatalf("with an infinite cell sum/count = %v/%d, want +Inf/2", store.NumericSum("latency"), store.NumericCount("latency"))
	}
	mustIngest(t, store, 2, map[string]any{"latency": 1e16})
	mustIngest(t, store, 1, map[string]any{"latency": 1.0})
	mustIngest(t, store, 2, map[string]any{"latency": 0.0})
	if store.NumericSum("latency") != 1 || store.NumericCount("latency") != 2 {
		t.Fatalf("after replacing a large value sum/count = %v/%d, want 1/2", store.NumericSum("latency"), store.NumericCount("latency"))
	}
}

func TestColumnStoreNilValueIsSkipped(t *testing.T) {
	store := newTestColumnStore(t, ColumnStoreLimits{InitialRows: 4})
	mustIngest(t, store, 1, map[string]any{"latency": 2.0, "absent": nil})
	if store.Columns() != 1 || store.Count() != 2 || store.Numeric("latency")[1] != 2 {
		t.Fatalf("nil value was not skipped: columns=%d count=%d latency=%v", store.Columns(), store.Count(), store.Numeric("latency"))
	}
}

func TestColumnStoreGrowthBoundaries(t *testing.T) {
	// A write at exactly the capacity must grow: the last valid slot is
	// capacity-1.
	store := newTestColumnStore(t, ColumnStoreLimits{InitialRows: 2, MaxRows: 10})
	mustIngest(t, store, 0, map[string]any{"latency": 1.0})
	mustIngest(t, store, 2, map[string]any{"latency": 3.0})
	if store.Capacity() != 4 || store.Numeric("latency")[2] != 3 {
		t.Fatalf("write at idx == capacity: capacity=%d latency=%v, want 4 and a readable value", store.Capacity(), store.Numeric("latency"))
	}
	// Doubling from 4 would reach 16; the row limit of 10 is not a power of
	// two, so only the clamp keeps the store inside it.
	mustIngest(t, store, 9, map[string]any{"latency": 9.0})
	if store.Capacity() != 10 {
		t.Fatalf("Capacity = %d, want the 10-row limit", store.Capacity())
	}
	if got := store.Numeric("latency"); got[0] != 1 || got[2] != 3 || got[9] != 9 {
		t.Fatalf("values after clamped growth = %v", got)
	}
}

func TestColumnStoreGrowthBeyondInitialCapacity(t *testing.T) {
	store := newTestColumnStore(t, ColumnStoreLimits{InitialRows: 2, MaxRows: 64})
	mustIngest(t, store, 0, map[string]any{"latency": 1.0, "model": "m-a", "chunks": []float64{1, 2}})
	mustIngest(t, store, 1, map[string]any{"latency": 2.0, "model": "m-b"})
	if err := store.IngestCategorical(1, "worker", "w1"); err != nil {
		t.Fatalf("IngestCategorical: %v", err)
	}
	if store.Capacity() != 2 {
		t.Fatalf("Capacity = %d before growth, want 2", store.Capacity())
	}
	if len(store.setters) != 3 {
		t.Fatalf("%d cached setters before growth, want one per metric tag (3)", len(store.setters))
	}

	// Index 9 forces growth from 2 rows to 16. The setters cached above
	// hold the two-row arrays; a store that kept them would write the rows
	// below out of bounds or into arrays nothing reads.
	mustIngest(t, store, 9, map[string]any{"latency": 9.0, "model": "m-c", "chunks": []float64{3}})
	mustIngest(t, store, 4, map[string]any{"latency": 4.0})
	if store.Capacity() != 16 {
		t.Fatalf("Capacity = %d after growth, want 16", store.Capacity())
	}
	// Only the tags written since growth are cached again: latency, model and
	// chunks at row 9, then latency at row 4.
	if len(store.setters) != 3 {
		t.Fatalf("%d cached setters after growth, want 3 rebuilt", len(store.setters))
	}
	if store.Count() != 10 {
		t.Fatalf("Count = %d, want 10", store.Count())
	}

	nan := math.NaN()
	want := []float64{1, 2, nan, nan, 4, nan, nan, nan, nan, 9}
	if got := store.Numeric("latency"); !sameFloats(got, want) {
		t.Fatalf("latency after growth = %v, want %v", got, want)
	}
	if store.NumericSum("latency") != 16 || store.NumericCount("latency") != 4 {
		t.Fatalf("sum/count after growth = %v/%d, want 16/4", store.NumericSum("latency"), store.NumericCount("latency"))
	}
	for idx, wantText := range map[int]string{0: "m-a", 1: "m-b", 9: "m-c"} {
		if got, ok := store.Text("model", idx); !ok || got != wantText {
			t.Fatalf("Text(model, %d) = %q, %v; want %q", idx, got, ok, wantText)
		}
	}
	if _, ok := store.Text("model", 4); ok {
		t.Fatal("row added by growth reported a text value")
	}
	if got, want := store.Categorical("worker"), []int32{-1, 0, -1, -1, -1, -1, -1, -1, -1, -1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("worker codes after growth = %v, want %v", got, want)
	}
	if got, want := store.ListValues("chunks"), []float64{1, 2, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks values = %v, want %v", got, want)
	}
	// The categorical column must have grown with the rest: these index rows
	// that did not exist when it was created.
	if err := store.IngestCategorical(9, "worker", "w2"); err != nil {
		t.Fatalf("IngestCategorical after growth: %v", err)
	}
	wantMask := []bool{false, true, false, false, false, false, false, false, false, false}
	if got := store.MaskForCategorical("worker", "w1"); !reflect.DeepEqual(got, wantMask) {
		t.Fatalf("mask(w1) after growth = %v, want %v", got, wantMask)
	}
	if got := store.Categorical("worker")[9]; got != 1 {
		t.Fatalf("worker code at row 9 = %d, want 1", got)
	}

	// Growth stops at the row limit rather than doubling past it.
	mustIngest(t, store, 63, map[string]any{"latency": 63.0})
	if store.Capacity() != 64 {
		t.Fatalf("Capacity = %d at the row limit, want 64", store.Capacity())
	}
	if got := store.Numeric("latency"); got[63] != 63 || got[9] != 9 || got[0] != 1 {
		t.Fatalf("values lost growing to the row limit: [0]=%v [9]=%v [63]=%v", got[0], got[9], got[63])
	}
}

func TestColumnStoreRetainedSizeBound(t *testing.T) {
	limits := ColumnStoreLimits{InitialRows: 2, MaxRows: 8, MaxColumns: 3, MaxListLen: 2}
	store := newTestColumnStore(t, limits)
	mustIngest(t, store, 7, map[string]any{"a": 1.0, "b": "x", "c": []float64{1, 2}})

	refusals := []struct {
		name    string
		ingest  func() error
		wantErr string
	}{
		{"row_at_limit", func() error { return store.Ingest(8, 0, 1, map[string]any{"a": 1.0}) }, "exceeds row limit 8"},
		{"negative_row", func() error { return store.Ingest(-1, 0, 1, map[string]any{"a": 1.0}) }, "negative record index"},
		{"fourth_column", func() error { return store.Ingest(1, 0, 1, map[string]any{"d": 1.0}) }, "exceeds limit 3"},
		{"categorical_as_fourth_column", func() error { return store.IngestCategorical(1, "worker", "w1") }, "exceeds limit 3"},
		{"categorical_row_at_limit", func() error { return store.IngestCategorical(8, "a", "w1") }, "exceeds row limit 8"},
		{"list_too_long", func() error { return store.Ingest(1, 0, 1, map[string]any{"c": []float64{1, 2, 3}}) }, "limit 2"},
		{"second_list_for_record", func() error { return store.Ingest(7, 0, 1, map[string]any{"c": []float64{9}}) }, "already has a list for record 7"},
		{"kind_change", func() error { return store.Ingest(1, 0, 1, map[string]any{"a": "text"}) }, "is a numeric column, got a text value"},
		{"unsupported_type", func() error { return store.Ingest(1, 0, 1, map[string]any{"a": true}) }, "unsupported type bool"},
		{"categorical_on_metric_column", func() error { return store.IngestCategorical(1, "a", "w1") }, "not categorical"},
	}
	for _, tc := range refusals {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.ingest()
			if err == nil {
				t.Fatal("write beyond the store's limits was accepted")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error %q does not contain %q", err, tc.wantErr)
			}
		})
	}

	if store.Capacity() > limits.MaxRows || store.Columns() > limits.MaxColumns {
		t.Fatalf("store holds %d rows x %d columns, limits %d x %d", store.Capacity(), store.Columns(), limits.MaxRows, limits.MaxColumns)
	}
	if store.Count() != 8 {
		t.Fatalf("Count = %d after refused writes, want 8", store.Count())
	}
}

func TestColumnStoreRefusedRecordWritesNothing(t *testing.T) {
	store := newTestColumnStore(t, ColumnStoreLimits{InitialRows: 2, MaxRows: 64, MaxColumns: 2})
	mustIngest(t, store, 0, map[string]any{"a": 1.0})

	// One valid tag plus one that would be a third column: the valid half
	// must not land, whichever order the map yields them in.
	err := store.Ingest(40, 0, 1, map[string]any{"a": 2.0, "b": 3.0, "c": 4.0})
	if err == nil {
		t.Fatal("record needing a column past the limit was accepted")
	}
	if store.Count() != 1 || store.Capacity() != 2 || store.Columns() != 1 {
		t.Fatalf("refused record changed the store: count=%d capacity=%d columns=%d", store.Count(), store.Capacity(), store.Columns())
	}
	if store.NumericSum("a") != 1 || store.NumericCount("a") != 1 {
		t.Fatalf("refused record changed running stats: %v/%d", store.NumericSum("a"), store.NumericCount("a"))
	}
}

func TestColumnStoreLists(t *testing.T) {
	store := newTestColumnStore(t, ColumnStoreLimits{InitialRows: 4})
	mustIngest(t, store, 2, map[string]any{"itl": []float64{5, 6, 7}})
	mustIngest(t, store, 0, map[string]any{"itl": []float64{}})
	mustIngest(t, store, 1, map[string]any{"itl": []float64{8}})

	for idx, want := range map[int][]float64{2: {5, 6, 7}, 0: {}, 1: {8}} {
		got, ok := store.List("itl", idx)
		if !ok || !reflect.DeepEqual(got, want) {
			t.Fatalf("List(itl, %d) = %v, %v; want %v", idx, got, ok, want)
		}
	}
	if _, ok := store.List("itl", 3); ok {
		t.Fatal("List reported a value for a record that never wrote one")
	}
	if _, ok := store.List("missing", 0); ok {
		t.Fatal("List reported a value for an unknown tag")
	}
	if got, want := store.ListValues("itl"), []float64{5, 6, 7, 8}; !reflect.DeepEqual(got, want) {
		t.Fatalf("ListValues = %v, want %v (ingest order)", got, want)
	}
}

func TestColumnStoreCategoricalInterning(t *testing.T) {
	store := newTestColumnStore(t, ColumnStoreLimits{InitialRows: 4})
	for idx := 0; idx < 4; idx++ {
		mustIngest(t, store, idx, map[string]any{"latency": float64(idx)})
	}
	for idx, worker := range []string{"w1", "w2", "w1"} {
		if err := store.IngestCategorical(idx, "worker", worker); err != nil {
			t.Fatalf("IngestCategorical(%d): %v", idx, err)
		}
	}
	if got, want := store.Categorical("worker"), []int32{0, 1, 0, CategoricalMissing}; !reflect.DeepEqual(got, want) {
		t.Fatalf("codes = %v, want %v", got, want)
	}
	if got, want := store.CategoryStrings("worker"), []string{"w1", "w2"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("CategoryStrings = %v, want %v", got, want)
	}
	if got, want := store.MaskForCategorical("worker", "w1"), []bool{true, false, true, false}; !reflect.DeepEqual(got, want) {
		t.Fatalf("mask(w1) = %v, want %v", got, want)
	}
	for _, mask := range [][]bool{store.MaskForCategorical("worker", "never"), store.MaskForCategorical("no_such_tag", "w1")} {
		if !reflect.DeepEqual(mask, []bool{false, false, false, false}) {
			t.Fatalf("absent value or tag matched rows: %v", mask)
		}
	}
	if got, want := store.Categorical("no_such_tag"), []int32{-1, -1, -1, -1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("unknown categorical tag = %v, want all missing", got)
	}

	// Metadata alone does not make a record visible.
	lone := newTestColumnStore(t, ColumnStoreLimits{InitialRows: 4})
	if err := lone.IngestCategorical(2, "worker", "w1"); err != nil {
		t.Fatalf("IngestCategorical: %v", err)
	}
	if lone.Count() != 0 {
		t.Fatalf("Count = %d after metadata-only write, want 0", lone.Count())
	}
}

func TestColumnStoreCategoricalDistinctValueBound(t *testing.T) {
	store := newTestColumnStore(t, ColumnStoreLimits{InitialRows: 2, MaxRows: 2})
	for _, value := range []string{"a", "b"} {
		if err := store.IngestCategorical(0, "worker", value); err != nil {
			t.Fatalf("IngestCategorical(%q): %v", value, err)
		}
	}
	// Rewriting one row with ever-new values is the only way to out-grow the
	// row count; it must stop at the row limit.
	err := store.IngestCategorical(0, "worker", "c")
	if err == nil || !strings.Contains(err.Error(), "distinct values, limit 2") {
		t.Fatalf("third distinct value on a two-row store: err = %v", err)
	}
	if err := store.IngestCategorical(1, "worker", "a"); err != nil {
		t.Fatalf("re-using an interned value was refused: %v", err)
	}
}

func TestColumnStoreQueryTimeRange(t *testing.T) {
	store := newTestColumnStore(t, ColumnStoreLimits{InitialRows: 4})
	// mustIngest stamps record i with [10i, 10i+5]; slot 1 stays unwritten.
	mustIngest(t, store, 0, map[string]any{"latency": 1.0})
	mustIngest(t, store, 2, map[string]any{"latency": 1.0})
	mustIngest(t, store, 3, map[string]any{"latency": 1.0})

	cases := []struct {
		name       string
		start, end float64
		want       []bool
	}{
		{"covers_all_written", 0, 100, []bool{true, false, true, true}},
		{"inclusive_endpoints", 5, 20, []bool{true, false, true, false}},
		{"gap_between_records", 6, 19, []bool{false, false, false, false}},
		{"tail_only", 26, 40, []bool{false, false, false, true}},
	}
	for _, tc := range cases {
		if got := store.QueryTimeRange(tc.start, tc.end); !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("%s: QueryTimeRange(%v, %v) = %v, want %v", tc.name, tc.start, tc.end, got, tc.want)
		}
	}
}

func TestColumnStoreLimitDefaults(t *testing.T) {
	store := newTestColumnStore(t, ColumnStoreLimits{})
	if store.Capacity() != DefaultColumnStoreInitialRows {
		t.Fatalf("default Capacity = %d, want %d", store.Capacity(), DefaultColumnStoreInitialRows)
	}
	clamped := newTestColumnStore(t, ColumnStoreLimits{InitialRows: 100, MaxRows: 10})
	if clamped.Capacity() != 10 {
		t.Fatalf("Capacity = %d, want the initial size clamped to the 10-row limit", clamped.Capacity())
	}
	if _, err := NewColumnStore(ColumnStoreLimits{MaxRows: -1}); err == nil {
		t.Fatal("negative row limit was accepted")
	}
}

func TestColumnStoreGrowthAcrossDefaultCapacity(t *testing.T) {
	store := newTestColumnStore(t, ColumnStoreLimits{})
	mustIngest(t, store, 0, map[string]any{"ttft": 1.5, "n": int32(2), "f": float32(0.5)})
	mustIngest(t, store, 1023, map[string]any{"ttft": 2.5})
	if store.Capacity() != 1024 {
		t.Fatalf("Capacity = %d, want 1024: the last slot of the first block must not grow it", store.Capacity())
	}
	mustIngest(t, store, 5000, map[string]any{"ttft": 4.0})
	if store.Capacity() != 8192 {
		t.Fatalf("Capacity = %d after a jump to 5000, want 8192", store.Capacity())
	}
	// Written through a setter rebuilt after growth, into a slot that did not
	// exist when the first one was cached.
	mustIngest(t, store, 1500, map[string]any{"ttft": 8.0})
	mustIngest(t, store, 3, map[string]any{"ttft": 16.0})
	// A lower index never shrinks the count.
	if store.Count() != 5001 {
		t.Fatalf("Count = %d, want 5001", store.Count())
	}
	col := store.Numeric("ttft")
	for idx, want := range map[int]float64{0: 1.5, 3: 16, 1023: 2.5, 1500: 8, 5000: 4} {
		if col[idx] != want {
			t.Fatalf("ttft[%d] = %v, want %v", idx, col[idx], want)
		}
	}
	written := 0
	for _, v := range col {
		if !math.IsNaN(v) {
			written++
		}
	}
	if written != 5 {
		t.Fatalf("%d non-NaN cells, want 5: unwritten slots must stay NaN, not zero", written)
	}
	if store.NumericSum("ttft") != 32 || store.NumericCount("ttft") != 5 {
		t.Fatalf("sum/count = %v/%d, want 32/5", store.NumericSum("ttft"), store.NumericCount("ttft"))
	}
	if store.Numeric("n")[0] != 2 || store.Numeric("f")[0] != 0.5 {
		t.Fatalf("int32/float32 values = %v/%v, want 2/0.5", store.Numeric("n")[0], store.Numeric("f")[0])
	}
}

func TestColumnStoreMetadataGrowthRebindsSetters(t *testing.T) {
	store := newTestColumnStore(t, ColumnStoreLimits{InitialRows: 2, MaxRows: 64})
	mustIngest(t, store, 0, map[string]any{"latency": 1.0})
	if err := store.IngestCategorical(20, "worker", "w1"); err != nil {
		t.Fatalf("IngestCategorical: %v", err)
	}
	if store.Capacity() != 32 {
		t.Fatalf("Capacity = %d after metadata growth, want 32", store.Capacity())
	}
	mustIngest(t, store, 20, map[string]any{"latency": 3.0})
	if got := store.Numeric("latency"); got[0] != 1 || got[20] != 3 {
		t.Fatalf("latency [0]=%v [20]=%v after metadata-driven growth, want 1 and 3", got[0], got[20])
	}
	if got := store.Categorical("worker"); got[20] != 0 || got[0] != CategoricalMissing {
		t.Fatalf("worker codes [0]=%d [20]=%d, want missing and 0", got[0], got[20])
	}
}

func TestColumnStoreCategoricalCodesAreWideAndPerTag(t *testing.T) {
	const distinct = 40000
	store := newTestColumnStore(t, ColumnStoreLimits{})
	for i := 0; i < distinct; i++ {
		mustIngest(t, store, i, nil)
		if err := store.IngestCategorical(i, "correlation", strconv.Itoa(i)); err != nil {
			t.Fatalf("IngestCategorical(%d): %v", i, err)
		}
	}
	codes := store.Categorical("correlation")
	names := store.CategoryStrings("correlation")
	if len(names) != distinct {
		t.Fatalf("%d interned strings, want %d", len(names), distinct)
	}
	for i, code := range codes {
		if code != int32(i) || names[code] != strconv.Itoa(i) {
			t.Fatalf("row %d has code %d (%q), want %d", i, code, names[max(code, 0)], i)
		}
	}

	// A second tag starts its own table at zero, and "" is a value, not missing.
	for idx, v := range []string{"", "x", ""} {
		if err := store.IngestCategorical(idx, "tenant", v); err != nil {
			t.Fatalf("IngestCategorical(tenant, %q): %v", v, err)
		}
	}
	if got := store.Categorical("tenant")[:4]; !reflect.DeepEqual(got, []int32{0, 1, 0, CategoricalMissing}) {
		t.Fatalf("tenant codes = %v, want [0 1 0 -1]", got)
	}
	if got := store.CategoryStrings("tenant"); !reflect.DeepEqual(got, []string{"", "x"}) {
		t.Fatalf("tenant strings = %q", got)
	}
	if got := store.CategoryStrings("unknown"); len(got) != 0 {
		t.Fatalf("unknown tag has category strings: %q", got)
	}
	if empty := newTestColumnStore(t, ColumnStoreLimits{}); len(empty.Numeric("x")) != 0 || len(empty.QueryTimeRange(0, 1)) != 0 {
		t.Fatal("empty store returned non-empty columns")
	}
}
