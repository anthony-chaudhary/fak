// Package providercost imports authoritative provider billing rows and attributes them to roots.
package providercost

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

const Schema = "fak-provider-cost-ledger/1"

type MicroUSD int64
type Row struct {
	Schema         string    `json:"schema"`
	Provider       string    `json:"provider"`
	ProviderRowID  string    `json:"provider_row_id"`
	SessionID      string    `json:"session_id"`
	IntervalStart  string    `json:"interval_start"`
	IntervalEnd    string    `json:"interval_end"`
	BilledMicroUSD *MicroUSD `json:"billed_micro_usd,omitempty"`
	Currency       string    `json:"currency,omitempty"`
	ExportID       string    `json:"export_id"`
	ExportedAt     string    `json:"exported_at"`
	Source         string    `json:"source"`
}
type ImportReport struct {
	Schema        string `json:"schema"`
	Imported      int    `json:"imported"`
	Duplicates    int    `json:"duplicates"`
	UnknownAmount int    `json:"unknown_amount"`
}
type RootCost struct {
	RootRegistrationID string   `json:"root_registration_id"`
	Rows               int      `json:"rows"`
	BilledMicroUSD     MicroUSD `json:"billed_micro_usd"`
	AmountRows         int      `json:"amount_rows"`
}
type Coverage struct {
	TotalRows                int      `json:"total_rows"`
	AttributedRows           int      `json:"attributed_rows"`
	MissingRows              int      `json:"missing_rows"`
	AmbiguousRows            int      `json:"ambiguous_rows"`
	AmountRows               int      `json:"amount_rows"`
	AttributedAmountRows     int      `json:"attributed_amount_rows"`
	TotalBilledMicroUSD      MicroUSD `json:"total_billed_micro_usd"`
	AttributedBilledMicroUSD MicroUSD `json:"attributed_billed_micro_usd"`
}
type Report struct {
	Schema   string     `json:"schema"`
	Roots    []RootCost `json:"roots"`
	Coverage Coverage   `json:"coverage"`
}
type Reconciliation struct {
	Provider               string    `json:"provider"`
	ExpectedRows           int       `json:"expected_rows"`
	ImportedRows           int       `json:"imported_rows"`
	RowsMatch              bool      `json:"rows_match"`
	ExpectedBilledMicroUSD *MicroUSD `json:"expected_billed_micro_usd,omitempty"`
	ImportedBilledMicroUSD MicroUSD  `json:"imported_billed_micro_usd"`
	AmountMatch            *bool     `json:"amount_match,omitempty"`
}

func Validate(r Row) error {
	if r.Schema != Schema {
		return fmt.Errorf("schema must be %q", Schema)
	}
	for k, v := range map[string]string{"provider": r.Provider, "provider_row_id": r.ProviderRowID, "session_id": r.SessionID, "export_id": r.ExportID, "exported_at": r.ExportedAt, "source": r.Source} {
		if strings.TrimSpace(v) == "" {
			return fmt.Errorf("%s is required", k)
		}
	}
	if r.BilledMicroUSD != nil && *r.BilledMicroUSD < 0 {
		return errors.New("billed amount must be nonnegative")
	}
	if r.BilledMicroUSD != nil && strings.ToUpper(strings.TrimSpace(r.Currency)) != "USD" {
		return errors.New("billed amount currency must be USD")
	}
	return nil
}
func Read(path string) ([]Row, error) {
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return Parse(b)
}
func Parse(b []byte) ([]Row, error) {
	s := bufio.NewScanner(bytes.NewReader(b))
	s.Buffer(make([]byte, 1024), 1024*1024)
	out := []Row{}
	for line := 1; s.Scan(); line++ {
		if len(bytes.TrimSpace(s.Bytes())) == 0 {
			continue
		}
		var r Row
		d := json.NewDecoder(bytes.NewReader(s.Bytes()))
		d.DisallowUnknownFields()
		if err := d.Decode(&r); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		if err := Validate(r); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		out = append(out, r)
	}
	return out, s.Err()
}
func Import(dst string, src io.Reader) (ImportReport, error) {
	existing, err := Read(dst)
	if err != nil {
		return ImportReport{}, err
	}
	seen := map[string]bool{}
	for _, r := range existing {
		seen[r.Provider+"\x00"+r.ProviderRowID] = true
	}
	s := bufio.NewScanner(src)
	s.Buffer(make([]byte, 1024), 1024*1024)
	add := []Row{}
	rep := ImportReport{Schema: Schema}
	for s.Scan() {
		if len(bytes.TrimSpace(s.Bytes())) == 0 {
			continue
		}
		rows, err := Parse(append([]byte(nil), s.Bytes()...))
		if err != nil {
			return rep, err
		}
		r := rows[0]
		key := r.Provider + "\x00" + r.ProviderRowID
		if seen[key] {
			rep.Duplicates++
			continue
		}
		seen[key] = true
		add = append(add, r)
		rep.Imported++
		if r.BilledMicroUSD == nil {
			rep.UnknownAmount++
		}
	}
	if err := s.Err(); err != nil {
		return rep, err
	}
	if len(add) == 0 {
		return rep, nil
	}
	if err := os.MkdirAll(filepathDir(dst), 0700); err != nil {
		return rep, err
	}
	f, err := os.OpenFile(dst, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return rep, err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, r := range add {
		if err := enc.Encode(r); err != nil {
			return rep, err
		}
	}
	return rep, nil
}
func Fold(rows []Row, regs []sessionregistry.Record) Report {
	bySession := map[string]map[string]bool{}
	for _, r := range regs {
		sid := strings.TrimSpace(r.Identity.SessionID)
		if sid == "" {
			continue
		}
		if bySession[sid] == nil {
			bySession[sid] = map[string]bool{}
		}
		bySession[sid][r.RootRegistrationID] = true
	}
	roots := map[string]*RootCost{}
	rep := Report{Schema: Schema, Roots: []RootCost{}}
	for _, row := range rows {
		rep.Coverage.TotalRows++
		if row.BilledMicroUSD != nil {
			rep.Coverage.AmountRows++
			rep.Coverage.TotalBilledMicroUSD += *row.BilledMicroUSD
		}
		matches := bySession[row.SessionID]
		if len(matches) == 0 {
			rep.Coverage.MissingRows++
			continue
		}
		if len(matches) != 1 {
			rep.Coverage.AmbiguousRows++
			continue
		}
		var root string
		for root = range matches {
		}
		x := roots[root]
		if x == nil {
			x = &RootCost{RootRegistrationID: root}
			roots[root] = x
		}
		x.Rows++
		rep.Coverage.AttributedRows++
		if row.BilledMicroUSD != nil {
			x.AmountRows++
			x.BilledMicroUSD += *row.BilledMicroUSD
			rep.Coverage.AttributedAmountRows++
			rep.Coverage.AttributedBilledMicroUSD += *row.BilledMicroUSD
		}
	}
	for id := range roots {
		rep.Roots = append(rep.Roots, *roots[id])
	}
	sort.Slice(rep.Roots, func(i, j int) bool { return rep.Roots[i].RootRegistrationID < rep.Roots[j].RootRegistrationID })
	return rep
}
func Reconcile(rows []Row, provider string, expectedRows int, expected *MicroUSD) Reconciliation {
	r := Reconciliation{Provider: provider, ExpectedRows: expectedRows, ExpectedBilledMicroUSD: expected}
	for _, row := range rows {
		if row.Provider != provider {
			continue
		}
		r.ImportedRows++
		if row.BilledMicroUSD != nil {
			r.ImportedBilledMicroUSD += *row.BilledMicroUSD
		}
	}
	r.RowsMatch = r.ImportedRows == expectedRows
	if expected != nil {
		ok := r.ImportedBilledMicroUSD == *expected
		r.AmountMatch = &ok
	}
	return r
}
func filepathDir(p string) string {
	i := strings.LastIndexAny(p, "/\\")
	if i < 0 {
		return "."
	}
	return p[:i]
}
