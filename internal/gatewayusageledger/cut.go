package gatewayusageledger

// cut.go — the sanctioned fold-and-truncate compaction for the gateway-usage
// ledger (#3490). The ledger is append-only by design (doc.go), so every
// guard/serve session on every box grows .fak/nightrun/gateway-usage.jsonl
// forever, and every reader (ReadLedgerFile) pays a whole-file parse. Cut is
// the bounded answer, following the internal/journal Cut discipline (#2457)
// adapted to a plain (un-chained) JSONL file: fold everything older than the
// newest `keep` rows into per-(kind, session_type) CARRYFORWARD rows whose
// Counters are the exact field-wise SUM of the folded rows' counters, then
// atomically rewrite the file as carryforwards + kept tail. No history total
// is lost — a reader summing counters over the whole file sees the same
// per-kind totals before and after a cut — only per-row granularity of the
// folded prefix is given up.
//
// Cut is operator-invoked only (`fak nightrun cut`, dry-run by default). The
// write path (Append) stays untouched: sessions never trigger a cut, so a
// tracked ledger file never rewrites itself mid-fleet under a session's feet.

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"
	"time"
)

// KindCarryforward marks a synthetic row produced by Cut: the field-wise sum of
// the folded-away rows of one (kind, session_type) group. It is not a session
// snapshot — FoldTrend skips it, and per-session readers never match it (no
// SessionID, PID 0, timestamp = the newest folded row's).
const KindCarryforward = "carryforward"

// Carryforward is the fold witness a carryforward row carries: which row kind
// was folded, how many rows, and the time range they spanned. FoldedRows lets a
// reader that counts sessions (e.g. internal/auditusage) keep its totals true
// across a cut by expanding the carryforward back to its row count.
type Carryforward struct {
	FoldedKind      string `json:"folded_kind"`
	FoldedRows      int    `json:"folded_rows"`
	FirstUnixMillis int64  `json:"first_unix_millis"`
	LastUnixMillis  int64  `json:"last_unix_millis"`
}

// CutResult is the honest record of one Cut call — what was (or, on a dry run,
// would be) folded, in rows and bytes.
type CutResult struct {
	Path             string `json:"path"`
	Performed        bool   `json:"performed"`
	DryRun           bool   `json:"dry_run"`
	RowsBefore       int    `json:"rows_before"`
	RowsAfter        int    `json:"rows_after"`
	BytesBefore      int64  `json:"bytes_before"`
	BytesAfter       int64  `json:"bytes_after"`
	FoldedRows       int    `json:"folded_rows"`
	CarryforwardRows int    `json:"carryforward_rows"`
	KeptRows         int    `json:"kept_rows"`
	// PreservedLines counts undecodable/foreign lines in the folded region that
	// were kept verbatim rather than folded — Cut never destroys bytes it cannot
	// account for, matching ParseLedger's skip-don't-abort posture.
	PreservedLines int `json:"preserved_undecoded_lines,omitempty"`
}

// sumCounters adds add into dst field-wise. It walks the struct by reflection
// so a future Counters field is summed automatically instead of silently
// dropped; a field of a kind it does not know how to sum is a hard error (the
// carryforward's whole contract is that no counter total is lost).
func sumCounters(dst *Counters, add Counters) error {
	dv := reflect.ValueOf(dst).Elem()
	av := reflect.ValueOf(add)
	for i := 0; i < dv.NumField(); i++ {
		f, a := dv.Field(i), av.Field(i)
		switch f.Kind() {
		case reflect.Int64:
			f.SetInt(f.Int() + a.Int())
		case reflect.Uint64:
			f.SetUint(f.Uint() + a.Uint())
		case reflect.Bool:
			// An INTENT flag (ManagedCacheActive / DeferColdToolsArmed, #4349) is not a
			// quantity, so "sum" is logical OR: the folded window had the lever armed iff
			// ANY row in it did. That is the direction the carryforward contract needs —
			// OR cannot lose the evidence that a lever was on, whereas AND would erase an
			// armed session the moment it folded beside a never-armed one, and the whole
			// point of these flags is that an armed-and-inert session stays visible.
			//
			// The cost, stated: a carryforward's flag answers "was this lever EVER armed in
			// the window", not "was it armed all window". A per-session question must be
			// asked of an exit row (which is what cachevaluereport.FoldUsageRowsConfiguredButInert
			// reads — it skips carryforward rows precisely because they are cross-session sums).
			f.SetBool(f.Bool() || a.Bool())
		case reflect.Map:
			if a.Len() == 0 {
				continue
			}
			if f.IsNil() {
				f.Set(reflect.MakeMap(f.Type()))
			}
			iter := a.MapRange()
			for iter.Next() {
				var base uint64
				if cur := f.MapIndex(iter.Key()); cur.IsValid() {
					base = cur.Uint()
				}
				f.SetMapIndex(iter.Key(), reflect.ValueOf(base+iter.Value().Uint()))
			}
		default:
			return fmt.Errorf("gatewayusageledger: cannot sum Counters field %s (kind %s) — teach sumCounters about it",
				dv.Type().Field(i).Name, f.Kind())
		}
	}
	return nil
}

// carryGroup accumulates one (kind, session_type) fold group.
type carryGroup struct {
	kind, sessionType string
	rows              int
	first, last       int64
	counters          Counters
}

func (g *carryGroup) fold(rowCount int, first, last int64, c Counters) error {
	if g.rows == 0 || first < g.first {
		g.first = first
	}
	if last > g.last {
		g.last = last
	}
	g.rows += rowCount
	return sumCounters(&g.counters, c)
}

// Cut folds every ledger row older than the newest keep rows into carryforward
// rows and atomically rewrites path. dryRun computes the full result without
// writing anything. A missing file, or a file with <= keep rows, is a clean
// no-op (Performed=false, no error) — there is nothing to bound yet.
//
// A prior carryforward row inside the folded region is merged into its group
// (by its FoldedKind), so repeated cuts stay idempotent: the file never
// accumulates stacked carryforward rows for the same group.
func Cut(path string, keep int, dryRun bool, now time.Time) (CutResult, error) {
	res := CutResult{Path: path, DryRun: dryRun}
	if keep < 0 {
		return res, fmt.Errorf("gatewayusageledger: Cut keep must be >= 0, got %d", keep)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return res, nil // absent ledger is a clean first-run state, like ReadLedgerFile
	}
	res.BytesBefore = int64(len(b))

	lines := strings.Split(strings.TrimSuffix(string(b), "\n"), "\n")
	type parsed struct {
		line string
		row  Row
		ok   bool
	}
	items := make([]parsed, 0, len(lines))
	decoded := 0
	for _, ln := range lines {
		p := parsed{line: ln}
		if strings.TrimSpace(ln) != "" {
			var r Row
			if json.Unmarshal([]byte(ln), &r) == nil && r.Schema != "" {
				p.row, p.ok = r, true
				decoded++
			}
		}
		items = append(items, p)
	}
	res.RowsBefore = decoded
	if decoded <= keep {
		res.RowsAfter = decoded
		res.BytesAfter = res.BytesBefore
		res.KeptRows = decoded
		return res, nil
	}

	// Boundary over DECODED rows: everything before the (decoded-keep)-th decoded
	// row folds; that row and every line after it is kept byte-verbatim.
	foldRows := decoded - keep
	groups := map[string]*carryGroup{}
	var preserved []string
	var kept []string
	seen := 0
	for _, it := range items {
		inFoldRegion := seen < foldRows
		if it.ok && inFoldRegion {
			seen++
			r := it.row
			kind, rows, first, last := r.Kind, 1, r.UnixMillis, r.UnixMillis
			if r.Kind == KindCarryforward && r.Carryforward != nil {
				kind, rows = r.Carryforward.FoldedKind, r.Carryforward.FoldedRows
				first, last = r.Carryforward.FirstUnixMillis, r.Carryforward.LastUnixMillis
			}
			key := kind + "\x00" + r.SessionType
			g := groups[key]
			if g == nil {
				g = &carryGroup{kind: kind, sessionType: r.SessionType}
				groups[key] = g
			}
			if err := g.fold(rows, first, last, r.Counters); err != nil {
				return res, err
			}
			continue
		}
		if !it.ok && inFoldRegion {
			if strings.TrimSpace(it.line) == "" {
				continue // blank lines carry nothing — dropping them loses no bytes of record
			}
			preserved = append(preserved, it.line)
			continue
		}
		kept = append(kept, it.line)
	}

	ordered := make([]*carryGroup, 0, len(groups))
	for _, g := range groups {
		ordered = append(ordered, g)
	}
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].kind != ordered[j].kind {
			return ordered[i].kind < ordered[j].kind
		}
		return ordered[i].sessionType < ordered[j].sessionType
	})

	var out strings.Builder
	for _, ln := range preserved {
		out.WriteString(ln)
		out.WriteByte('\n')
	}
	for _, g := range ordered {
		row := Row{
			Schema:      Schema,
			Kind:        KindCarryforward,
			SessionType: g.sessionType,
			// PID 0 + no SessionID: synthetic, never mistakable for a live session.
			UnixMillis:  g.last,
			Counters:    g.counters,
			GeneratedAt: now.UTC().Format(time.RFC3339),
			Carryforward: &Carryforward{
				FoldedKind:      g.kind,
				FoldedRows:      g.rows,
				FirstUnixMillis: g.first,
				LastUnixMillis:  g.last,
			},
		}
		rb, err := json.Marshal(row)
		if err != nil {
			return res, err
		}
		out.Write(rb)
		out.WriteByte('\n')
	}
	for _, ln := range kept {
		out.WriteString(ln)
		out.WriteByte('\n')
	}

	res.Performed = true
	res.FoldedRows = foldRows
	res.CarryforwardRows = len(ordered)
	res.KeptRows = keep
	res.PreservedLines = len(preserved)
	res.RowsAfter = len(ordered) + keep
	res.BytesAfter = int64(out.Len())
	if dryRun {
		return res, nil
	}

	tmp := path + ".cut-tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return res, err
	}
	if _, err := f.WriteString(out.String()); err != nil {
		f.Close()
		os.Remove(tmp)
		return res, err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmp)
		return res, err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return res, err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return res, err
	}
	return res, nil
}
