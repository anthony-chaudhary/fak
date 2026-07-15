package negframe

import (
	"bufio"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"time"
)

const ReframeJournalSchema = "fak-negframe-reframe-pass/1"
const DefaultJournalMaxRows = 4096

// ReframeJournalRow is one bounded, content-free emit observation. It records
// counts and arm only; source/output prose never enters the journal.
type ReframeJournalRow struct {
	Schema            string `json:"schema"`
	UnixMillis        int64  `json:"unix_ms"`
	TraceID           string `json:"trace_id,omitempty"`
	Site              string `json:"site,omitempty"`
	Arm               string `json:"arm"` // control | treatment
	Applied           int    `json:"applied"`
	VerbatimFallback  int    `json:"verbatim_fallback"`
	ResidualNegatives int    `json:"residual_negatives"`
}

func NewReframeJournalRow(traceID, arm string, result ReframeResult, now time.Time) ReframeJournalRow {
	return NewReframeJournalSiteRow(traceID, "", arm, result, now)
}

func NewReframeJournalSiteRow(traceID, site, arm string, result ReframeResult, now time.Time) ReframeJournalRow {
	if arm != "treatment" {
		arm = "control"
	}
	return ReframeJournalRow{Schema: ReframeJournalSchema, UnixMillis: now.UnixMilli(), TraceID: traceID, Site: site, Arm: arm, Applied: result.Applied, VerbatimFallback: result.VerbatimFallback, ResidualNegatives: result.ResidualNegatives}
}

// AppendReframeJournal appends one JSONL row and retains at most maxRows newest
// rows. The rewrite uses a sibling temp plus rename; malformed prior rows are
// skipped, so a torn tail cannot poison future observations.
func AppendReframeJournal(path string, row ReframeJournalRow, maxRows int) error {
	if path == "" {
		return errors.New("negframe journal: empty path")
	}
	if maxRows <= 0 {
		maxRows = DefaultJournalMaxRows
	}
	var rows [][]byte
	if f, err := os.Open(path); err == nil {
		s := bufio.NewScanner(f)
		s.Buffer(make([]byte, 4096), 1<<20)
		for s.Scan() {
			line := append([]byte(nil), s.Bytes()...)
			var prior ReframeJournalRow
			if json.Unmarshal(line, &prior) == nil && prior.Schema == ReframeJournalSchema {
				rows = append(rows, line)
			}
		}
		_ = f.Close()
	} else if !os.IsNotExist(err) {
		return err
	}
	encoded, err := json.Marshal(row)
	if err != nil {
		return err
	}
	rows = append(rows, encoded)
	if len(rows) > maxRows {
		rows = rows[len(rows)-maxRows:]
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".reframe-journal-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	for _, line := range rows {
		if _, err = tmp.Write(append(line, '\n')); err != nil {
			tmp.Close()
			return err
		}
	}
	if err = tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}
