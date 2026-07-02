package dogfoodissues

// The bridge receipt is the durable witness that `fak dogfood-issues` actually
// checked a specific report against the tracker — the chain rung between "the
// dogfood packet found friction" and "that friction is a tracked issue". Without
// it, a report's ACTION findings can die in the evidence dir and the next
// outsider stumbles into the same friction the packet already witnessed. The
// receipt is written ONLY for modes that consulted the tracker (--live, which
// files/updates deduped issues, or --fetch-existing, which verifies the actions
// are already tracked); a pure dry-run proves nothing about the tracker and
// leaves no receipt. internal/dogfoodscore reads the receipt (by its stable name
// and schema, pinned by a paired test in each package) to grade the chain axis,
// which is what puts this rung on the superloop's worst-first walk.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

const (
	// ReceiptSchema tags the machine-readable bridge receipt.
	ReceiptSchema = "fak.dogfood-issues-receipt.v1"
	// ReceiptName is the receipt's file name, written next to the report it
	// witnessed so the co-location itself binds receipt to report.
	ReceiptName = "issues-sync.json"
	// ReceiptModeLive marks a receipt from a --live run (issues filed/updated).
	ReceiptModeLive = "live"
	// ReceiptModeFetchExisting marks a receipt from a --fetch-existing run (the
	// tracker was queried; actions were classified create-vs-update but nothing
	// was written).
	ReceiptModeFetchExisting = "fetch-existing"
)

// ReceiptRow is one action key's bridge outcome.
type ReceiptRow struct {
	Key    string `json:"key"`
	Action string `json:"action"`
	OK     bool   `json:"ok"`
	Number *int   `json:"number,omitempty"`
}

// Receipt is the persisted witness of one bridge run over one report.
type Receipt struct {
	Schema         string       `json:"schema"`
	Mode           string       `json:"mode"`
	Report         string       `json:"report"`
	WrittenAt      string       `json:"written_at"`
	Actions        int          `json:"actions"`
	PlannedCreates int          `json:"planned_creates"`
	PlannedUpdates int          `json:"planned_updates"`
	SyncedOK       int          `json:"synced_ok"`
	SyncedFailed   int          `json:"synced_failed"`
	Skipped        int          `json:"skipped"`
	Rows           []ReceiptRow `json:"rows,omitempty"`
}

// BuildReceipt folds a bridge Result into its receipt. mode must be one of the
// Receipt_Mode constants; on a live run the rows carry each gh outcome, on a
// fetch-existing run they carry the create/update classification (ok=true —
// nothing was attempted, so nothing failed).
func BuildReceipt(res Result, mode string, now time.Time) Receipt {
	if now.IsZero() {
		now = time.Now()
	}
	r := Receipt{
		Schema:    ReceiptSchema,
		Mode:      mode,
		Report:    res.Report,
		WrittenAt: now.UTC().Format(time.RFC3339),
		Actions:   len(res.Planned) + len(res.Skipped),
		Skipped:   len(res.Skipped),
	}
	for _, row := range res.Planned {
		if row.Action == "update" {
			r.PlannedUpdates++
		} else {
			r.PlannedCreates++
		}
	}
	if mode == ReceiptModeLive {
		for _, row := range res.Synced {
			if row.OK {
				r.SyncedOK++
			} else {
				r.SyncedFailed++
			}
			n := row.Number
			r.Rows = append(r.Rows, ReceiptRow{Key: row.Key, Action: row.Action, OK: row.OK, Number: n})
		}
	} else {
		for _, row := range res.Planned {
			r.Rows = append(r.Rows, ReceiptRow{Key: row.Key, Action: row.Action, OK: true, Number: row.Number})
		}
	}
	return r
}

// WriteReceipt persists the receipt as <dir>/issues-sync.json and returns the
// written path. dir is the report's own evidence dir.
func WriteReceipt(dir string, r Receipt) (string, error) {
	path := filepath.Join(dir, ReceiptName)
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
		return "", err
	}
	return path, nil
}

// NewestReport resolves the newest recent-feature dogfood report on this host —
// the default the CLI uses when no report path is given, so an operator (or a
// loop) never has to hand-locate an evidence stamp. It scans one level below
// <root>/.fak/recent-feature-dogfood for report.json files and picks the newest
// by mtime, mirroring the Python bridge's selection semantics.
func NewestReport(root string) (string, error) {
	dir := filepath.Join(root, ".fak", "recent-feature-dogfood")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", fmt.Errorf("no dogfood evidence under %s (run `make dogfood-recent` to produce a report): %w", dir, err)
	}
	type cand struct {
		path  string
		mtime time.Time
	}
	var cands []cand
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		p := filepath.Join(dir, e.Name(), "report.json")
		info, err := os.Stat(p)
		if err != nil {
			continue
		}
		cands = append(cands, cand{path: p, mtime: info.ModTime()})
	}
	if len(cands) == 0 {
		return "", fmt.Errorf("no report.json under %s (run `make dogfood-recent` to produce one)", dir)
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].mtime.After(cands[j].mtime) })
	return cands[0].path, nil
}
