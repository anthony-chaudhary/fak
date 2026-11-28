// Adapted from ai-dynamo/aiperf
// modified from the original): the session-indexed, NaN-sparse columnar store
// for per-record metrics. Each record writes to the slot named by its session index, in any
// arrival order; unwritten slots read as NaN. The numpy arrays become plain
// slices, the list backend becomes one CSR column, the source's metadata
// families are narrowed to the interned categorical column, and the time
// columns to start and end. Unlike the source the store is capped: it refuses
// a write it cannot hold instead of growing without limit, and refuses a value
// of an unsupported type instead of skipping it.
package perfledger

import (
	"fmt"
	"math"
	"sort"
)

// Default ColumnStoreLimits, applied to fields left at zero.
const (
	DefaultColumnStoreInitialRows = columnStoreInitialCapacity
	DefaultColumnStoreMaxRows     = 1 << 20
	DefaultColumnStoreMaxColumns  = 256
	DefaultColumnStoreMaxListLen  = 4096
)

// CategoricalMissing is the code of a row with no categorical value.
const CategoricalMissing int32 = -1

// ColumnStoreLimits bounds what a ColumnStore retains. Zero fields take the
// defaults above.
type ColumnStoreLimits struct {
	InitialRows int
	MaxRows     int
	MaxColumns  int
	MaxListLen  int
}

type columnKind uint8

const (
	columnNumeric columnKind = iota + 1
	columnText
	columnRagged
	columnCategorical
)

func (k columnKind) String() string {
	switch k {
	case columnNumeric:
		return "numeric"
	case columnText:
		return "text"
	case columnRagged:
		return "list"
	case columnCategorical:
		return "categorical"
	}
	return "unknown"
}

// raggedColumn is a CSR column: the values of the i-th ingested list are
// values[offsets[i]:offsets[i+1]] and belong to record rows[i].
type raggedColumn struct {
	values  []float64
	offsets []int
	rows    []int
	slot    map[int]int
}

// ColumnStore holds per-record metrics column-wise, indexed by session number.
//
// Retained size is bounded by rows x columns. Every numeric, text and
// categorical column, plus the two time columns, holds exactly Capacity()
// cells, and Capacity() never exceeds MaxRows; the number of named columns
// never exceeds MaxColumns. So dense storage is at most
// MaxRows x (MaxColumns + 2) cells. A list column holds at most one list of
// MaxListLen values per row, and a categorical column interns at most MaxRows
// distinct strings. The bound counts cells, where a text cell is a string
// header plus a presence flag: the bytes of text and category strings are the
// caller's.
//
// A ColumnStore is not safe for concurrent use.
type ColumnStore struct {
	limits   ColumnStoreLimits
	capacity int
	count    int

	kinds   map[string]columnKind
	numeric map[string][]float64
	sums    map[string]float64
	counts  map[string]int
	text    map[string][]string
	hasText map[string][]bool
	ragged  map[string]*raggedColumn

	categorical   map[string][]int32
	categories    map[string]map[string]int32
	categoryNames map[string][]string

	// setters caches one write closure per metric tag so a repeat tag skips
	// the kind dispatch and column lookup. A numeric or text closure holds the
	// column's backing array, which grow replaces, so grow must drop the cache.
	setters map[string]func(idx int, value any)

	startNS []float64
	endNS   []float64
}

// NewColumnStore returns an empty store with the given limits.
func NewColumnStore(limits ColumnStoreLimits) (*ColumnStore, error) {
	if limits.InitialRows < 0 || limits.MaxRows < 0 || limits.MaxColumns < 0 || limits.MaxListLen < 0 {
		return nil, fmt.Errorf("perfledger: ColumnStore limits must not be negative: %+v", limits)
	}
	if limits.MaxRows == 0 {
		limits.MaxRows = DefaultColumnStoreMaxRows
	}
	if limits.MaxColumns == 0 {
		limits.MaxColumns = DefaultColumnStoreMaxColumns
	}
	if limits.MaxListLen == 0 {
		limits.MaxListLen = DefaultColumnStoreMaxListLen
	}
	if limits.InitialRows == 0 {
		limits.InitialRows = DefaultColumnStoreInitialRows
	}
	if limits.InitialRows > limits.MaxRows {
		limits.InitialRows = limits.MaxRows
	}
	return &ColumnStore{
		limits:        limits,
		capacity:      limits.InitialRows,
		kinds:         map[string]columnKind{},
		numeric:       map[string][]float64{},
		sums:          map[string]float64{},
		counts:        map[string]int{},
		text:          map[string][]string{},
		hasText:       map[string][]bool{},
		ragged:        map[string]*raggedColumn{},
		categorical:   map[string][]int32{},
		categories:    map[string]map[string]int32{},
		categoryNames: map[string][]string{},
		setters:       map[string]func(int, any){},
		startNS:       nanColumn(limits.InitialRows),
		endNS:         nanColumn(limits.InitialRows),
	}, nil
}

func nanColumn(n int) []float64 {
	col := make([]float64, n)
	for i := range col {
		col[i] = math.NaN()
	}
	return col
}

// Count is the number of record slots in use: the highest ingested session
// index plus one.
func (s *ColumnStore) Count() int { return s.count }

// Capacity is the number of rows every dense column currently holds.
func (s *ColumnStore) Capacity() int { return s.capacity }

// Columns is the number of named columns, of every kind.
func (s *ColumnStore) Columns() int { return len(s.kinds) }

// Ingest writes one record to slot idx. Supported metric values are
// float64/float32/int/int32/int64 (numeric), string (text) and []float64
// (list); a tag keeps the kind of its first value. A NaN numeric value is
// stored as missing and stays out of the running sum and count, and a nil
// value is skipped as absent. The whole record is validated before anything
// is written, so a refused record leaves the store unchanged.
func (s *ColumnStore) Ingest(idx int, startNS, endNS float64, metrics map[string]any) error {
	if err := s.checkRow(idx); err != nil {
		return err
	}
	newColumns := 0
	for tag, value := range metrics {
		if value == nil {
			continue
		}
		kind, ok := valueKind(value)
		if !ok {
			return fmt.Errorf("perfledger: ColumnStore: metric %q has unsupported type %T", tag, value)
		}
		have, known := s.kinds[tag]
		switch {
		case !known:
			newColumns++
		case have != kind:
			return fmt.Errorf("perfledger: ColumnStore: metric %q is a %s column, got a %s value", tag, have, kind)
		}
		if kind != columnRagged {
			continue
		}
		if n := len(value.([]float64)); n > s.limits.MaxListLen {
			return fmt.Errorf("perfledger: ColumnStore: metric %q list has %d values, limit %d", tag, n, s.limits.MaxListLen)
		}
		if col := s.ragged[tag]; col != nil {
			if _, dup := col.slot[idx]; dup {
				return fmt.Errorf("perfledger: ColumnStore: metric %q already has a list for record %d", tag, idx)
			}
		}
	}
	if len(s.kinds)+newColumns > s.limits.MaxColumns {
		return fmt.Errorf("perfledger: ColumnStore: %d columns exceeds limit %d", len(s.kinds)+newColumns, s.limits.MaxColumns)
	}

	if idx >= s.capacity {
		s.grow(idx)
	}
	if idx >= s.count {
		s.count = idx + 1
	}
	s.startNS[idx] = startNS
	s.endNS[idx] = endNS
	for tag, value := range metrics {
		if value == nil {
			continue
		}
		set := s.setters[tag]
		if set == nil {
			set = s.newSetter(tag, value)
			s.setters[tag] = set
		}
		set(idx, value)
	}
	return nil
}

func (s *ColumnStore) checkRow(idx int) error {
	if idx < 0 {
		return fmt.Errorf("perfledger: ColumnStore: negative record index %d", idx)
	}
	if idx >= s.limits.MaxRows {
		return fmt.Errorf("perfledger: ColumnStore: record index %d exceeds row limit %d", idx, s.limits.MaxRows)
	}
	return nil
}

func valueKind(value any) (columnKind, bool) {
	switch value.(type) {
	case float64, float32, int, int32, int64:
		return columnNumeric, true
	case string:
		return columnText, true
	case []float64:
		return columnRagged, true
	}
	return 0, false
}

func (s *ColumnStore) newSetter(tag string, value any) func(int, any) {
	kind, _ := valueKind(value)
	s.kinds[tag] = kind
	switch kind {
	case columnText:
		col := s.text[tag]
		if col == nil {
			col = make([]string, s.capacity)
			s.text[tag] = col
			s.hasText[tag] = make([]bool, s.capacity)
		}
		present := s.hasText[tag]
		return func(idx int, v any) {
			col[idx] = v.(string)
			present[idx] = true
		}
	case columnRagged:
		col := s.ragged[tag]
		if col == nil {
			col = &raggedColumn{offsets: []int{0}, slot: map[int]int{}}
			s.ragged[tag] = col
		}
		return func(idx int, v any) {
			col.slot[idx] = len(col.rows)
			col.rows = append(col.rows, idx)
			col.values = append(col.values, v.([]float64)...)
			col.offsets = append(col.offsets, len(col.values))
		}
	default:
		// Numeric is the fallback store: columnNumeric, categorical, and the
		// unsupported kind 0 from valueKind all land in a float column.
		return s.numericSetter(tag)
	}
}

func (s *ColumnStore) numericSetter(tag string) func(int, any) {
	col := s.numeric[tag]
	if col == nil {
		col = nanColumn(s.capacity)
		s.numeric[tag] = col
	}
	return func(idx int, v any) {
		var x float64
		switch n := v.(type) {
		case float64:
			x = n
		case float32:
			x = float64(n)
		case int:
			x = float64(n)
		case int32:
			x = float64(n)
		case int64:
			x = float64(n)
		}
		rewrite := !math.IsNaN(col[idx])
		col[idx] = x
		if rewrite {
			// Subtracting the old value is wrong for an infinite cell and
			// drifts under rounding, so a rewrite re-adds the column.
			s.sums[tag], s.counts[tag] = sumPresent(col[:s.count])
			return
		}
		if !math.IsNaN(x) {
			s.sums[tag] += x
			s.counts[tag]++
		}
	}
}

// sumPresent returns the sum and count of the non-NaN cells of col.
func sumPresent(col []float64) (sum float64, n int) {
	for _, v := range col {
		if !math.IsNaN(v) {
			sum += v
			n++
		}
	}
	return sum, n
}

// grow doubles capacity until minIdx fits, capped at MaxRows.
func (s *ColumnStore) grow(minIdx int) {
	newCap := s.capacity
	for newCap <= minIdx {
		// Clamping before doubling keeps a huge row limit from overflowing.
		if newCap > s.limits.MaxRows/2 {
			newCap = s.limits.MaxRows
			break
		}
		newCap *= 2
	}

	regrow := func(old []float64) []float64 {
		col := nanColumn(newCap)
		copy(col, old)
		return col
	}
	s.startNS = regrow(s.startNS)
	s.endNS = regrow(s.endNS)
	for tag, old := range s.numeric {
		s.numeric[tag] = regrow(old)
	}
	for tag, old := range s.text {
		col := make([]string, newCap)
		copy(col, old)
		s.text[tag] = col
		present := make([]bool, newCap)
		copy(present, s.hasText[tag])
		s.hasText[tag] = present
	}
	for tag, old := range s.categorical {
		col := make([]int32, newCap)
		for i := copy(col, old); i < newCap; i++ {
			col[i] = CategoricalMissing
		}
		s.categorical[tag] = col
	}

	clear(s.setters)
	s.capacity = newCap
}

// Numeric returns a copy of the numeric column for tag, one cell per record
// slot, NaN where no value was ingested. An unknown tag reads as all NaN.
func (s *ColumnStore) Numeric(tag string) []float64 {
	col, ok := s.numeric[tag]
	if !ok {
		return nanColumn(s.count)
	}
	return append([]float64(nil), col[:s.count]...)
}

// NumericTags lists the numeric columns, sorted.
func (s *ColumnStore) NumericTags() []string {
	tags := make([]string, 0, len(s.numeric))
	for tag := range s.numeric {
		tags = append(tags, tag)
	}
	sort.Strings(tags)
	return tags
}

// NumericSum is the running sum of the non-NaN cells of a numeric column, up
// to floating-point rounding.
func (s *ColumnStore) NumericSum(tag string) float64 { return s.sums[tag] }

// NumericCount is the number of non-NaN cells of a numeric column.
func (s *ColumnStore) NumericCount(tag string) int { return s.counts[tag] }

// Text returns the text value of record idx and whether one was ingested.
func (s *ColumnStore) Text(tag string, idx int) (string, bool) {
	present := s.hasText[tag]
	if idx < 0 || idx >= s.count || idx >= len(present) || !present[idx] {
		return "", false
	}
	return s.text[tag][idx], true
}

// List returns a copy of the list ingested for record idx and whether one was.
func (s *ColumnStore) List(tag string, idx int) ([]float64, bool) {
	col := s.ragged[tag]
	if col == nil {
		return nil, false
	}
	i, ok := col.slot[idx]
	if !ok {
		return nil, false
	}
	return append([]float64{}, col.values[col.offsets[i]:col.offsets[i+1]]...), true
}

// ListValues returns a copy of every value of a list column, in ingest order.
func (s *ColumnStore) ListValues(tag string) []float64 {
	col := s.ragged[tag]
	if col == nil {
		return nil
	}
	return append([]float64(nil), col.values...)
}

// IngestCategorical writes an interned categorical metadata value to slot idx.
// Like the source's metadata path it does not advance Count: a record becomes
// visible through Ingest.
func (s *ColumnStore) IngestCategorical(idx int, tag, value string) error {
	if err := s.checkRow(idx); err != nil {
		return err
	}
	have, known := s.kinds[tag]
	switch {
	case known && have != columnCategorical:
		return fmt.Errorf("perfledger: ColumnStore: %q is a %s column, not categorical", tag, have)
	case !known && len(s.kinds)+1 > s.limits.MaxColumns:
		return fmt.Errorf("perfledger: ColumnStore: %d columns exceeds limit %d", len(s.kinds)+1, s.limits.MaxColumns)
	}
	code, interned := s.categories[tag][value]
	if !interned && len(s.categoryNames[tag]) >= s.limits.MaxRows {
		return fmt.Errorf("perfledger: ColumnStore: categorical %q has %d distinct values, limit %d", tag, len(s.categoryNames[tag]), s.limits.MaxRows)
	}

	if idx >= s.capacity {
		s.grow(idx)
	}
	col := s.categorical[tag]
	if col == nil {
		col = make([]int32, s.capacity)
		for i := range col {
			col[i] = CategoricalMissing
		}
		s.categorical[tag] = col
		s.categories[tag] = map[string]int32{}
		s.kinds[tag] = columnCategorical
	}
	if !interned {
		code = int32(len(s.categoryNames[tag]))
		s.categories[tag][value] = code
		s.categoryNames[tag] = append(s.categoryNames[tag], value)
	}
	col[idx] = code
	return nil
}

// Categorical returns a copy of the per-record codes for tag,
// CategoricalMissing where no value was ingested. Decode a code with
// CategoryStrings(tag)[code].
func (s *ColumnStore) Categorical(tag string) []int32 {
	out := make([]int32, s.count)
	col := s.categorical[tag]
	for i := copy(out, col[:min(s.count, len(col))]); i < len(out); i++ {
		out[i] = CategoricalMissing
	}
	return out
}

// CategoryStrings returns the code-to-string table of a categorical column.
func (s *ColumnStore) CategoryStrings(tag string) []string {
	return append([]string(nil), s.categoryNames[tag]...)
}

// MaskForCategorical marks the record slots whose tag equals value. A value
// that never appeared matches nothing, including rows with no value at all.
func (s *ColumnStore) MaskForCategorical(tag, value string) []bool {
	mask := make([]bool, s.count)
	code, ok := s.categories[tag][value]
	if !ok {
		return mask
	}
	col := s.categorical[tag]
	for i := range mask {
		mask[i] = col[i] == code
	}
	return mask
}

// QueryTimeRange marks the record slots whose [start, end] overlaps the
// inclusive window. Unwritten slots never match: every comparison against
// their NaN timestamps is false.
func (s *ColumnStore) QueryTimeRange(startNS, endNS float64) []bool {
	mask := make([]bool, s.count)
	for i := range mask {
		mask[i] = s.startNS[i] <= endNS && s.endNS[i] >= startNS
	}
	return mask
}
