package accountprobe

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// LedgerEntry is one probe_ledger.jsonl record. Only the fields the roster fold reads
// are modeled; unknown keys are ignored on decode.
type LedgerEntry struct {
	TS          string `json:"ts"`
	Account     string `json:"account"`
	Tag         string `json:"tag"`
	Status      string `json:"status"`
	PrevStatus  string `json:"prev_status"`
	Flip        bool   `json:"flip"`
	Reset       string `json:"reset"`
	Weekly      string `json:"weekly"`
	BlockReason string `json:"block_reason"`
	Reason      string `json:"reason"`
}

// RegDir resolves the fleet registry dir the probe ledger lives under. It honors a
// FLEET_REG_DIR override (which the fleet sets in production, so the Go reader and the
// Python writer agree), else falls back to tools/_registry relative to the working
// directory — the fak binary runs from the clone root. Mirrors account_probe.reg_dir.
func RegDir() string {
	if v := strings.TrimSpace(os.Getenv("FLEET_REG_DIR")); v != "" {
		return v
	}
	return filepath.Join("tools", "_registry")
}

// ProbeLedgerPath returns the probe_ledger.jsonl path under rd, defaulting rd to
// RegDir() when empty. Mirrors account_probe.probe_ledger_path.
func ProbeLedgerPath(rd string) string {
	if rd == "" {
		rd = RegDir()
	}
	return filepath.Join(rd, "probe_ledger.jsonl")
}

// ReadLedger reads every JSON line of a probe ledger, skipping blank and malformed
// lines; a missing/unreadable file yields an empty slice (never an error). Mirrors
// account_probe._read_ledger's best-effort read.
func ReadLedger(path string) []LedgerEntry {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	var out []LedgerEntry
	sc := bufio.NewScanner(f)
	// Probe lines are small, but raise the buffer ceiling so a long block_reason line
	// is not silently dropped as "too long".
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		ln := strings.TrimSpace(sc.Text())
		if ln == "" {
			continue
		}
		var e LedgerEntry
		if err := json.Unmarshal([]byte(ln), &e); err != nil {
			continue
		}
		out = append(out, e)
	}
	return out
}

// LastProbeByAccount returns the most-recent ledger entry per account (basename ->
// entry). The file is append-ordered, so the last write wins. Mirrors
// account_probe.last_probe_by_account.
func LastProbeByAccount(rd string) map[string]LedgerEntry {
	latest := map[string]LedgerEntry{}
	for _, e := range ReadLedger(ProbeLedgerPath(rd)) {
		if e.Account != "" {
			latest[e.Account] = e
		}
	}
	return latest
}

// RecentProbeAgeMin returns the minutes since account was last probed, measured from
// now, or nil if the account was never probed or its timestamp is unparseable. Mirrors
// account_probe.recent_probe_age_min (now is injected for determinism; pass
// time.Now().UTC() in production).
func RecentProbeAgeMin(account, rd string, now time.Time) *float64 {
	e, ok := LastProbeByAccount(rd)[account]
	if !ok || e.TS == "" {
		return nil
	}
	when := parseLedgerTime(e.TS)
	if when == nil {
		return nil
	}
	age := now.Sub(*when).Seconds() / 60.0
	return &age
}

// parseLedgerTime parses an ISO-8601 probe timestamp, anchoring a naive (tz-less) time
// to UTC — mirroring datetime.fromisoformat(ts.replace("Z","+00:00")) with the
// tzinfo-None → UTC fixup. Returns nil on an unknown format.
func parseLedgerTime(raw string) *time.Time {
	s := strings.Replace(strings.TrimSpace(raw), "Z", "+00:00", 1)
	if s == "" {
		return nil
	}
	for _, layout := range []string{
		"2006-01-02T15:04:05.999999999-07:00",
		"2006-01-02T15:04:05-07:00",
		"2006-01-02T15:04:05.999999999",
		"2006-01-02T15:04:05",
	} {
		if t, err := time.Parse(layout, s); err == nil {
			u := t.UTC()
			return &u
		}
	}
	return nil
}
