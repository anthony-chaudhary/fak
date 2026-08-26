package studymonitor

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const ImportSchema = "fak-study-import/1"

type ImportStatus string

const (
	ImportEligible ImportStatus = "candidate"
	ImportImported ImportStatus = "imported"
	ImportHeld     ImportStatus = "held"
	ImportRejected ImportStatus = "rejected"
)

type SourceLineage struct {
	Path     string `json:"path"`
	Date     string `json:"date"`
	Revision string `json:"revision"`
}

type ImportRecord struct {
	Schema  string        `json:"schema"`
	ID      string        `json:"id"`
	Kind    string        `json:"kind"`
	Title   string        `json:"title"`
	Source  string        `json:"source,omitempty"`
	Lineage SourceLineage `json:"lineage"`
}

type ImportEntry struct {
	Path      string         `json:"path"`
	Status    ImportStatus   `json:"status"`
	Reason    string         `json:"reason"`
	RecordIDs []string       `json:"record_ids,omitempty"`
	Records   []ImportRecord `json:"records,omitempty"`
}

type ImportLedger struct {
	Schema    string        `json:"schema"`
	DryRun    bool          `json:"dry_run"`
	Attempted int           `json:"attempted"`
	Eligible  int           `json:"eligible"`
	Imported  int           `json:"imported"`
	Held      int           `json:"held"`
	Rejected  int           `json:"rejected"`
	Entries   []ImportEntry `json:"entries"`
}

// ImportTracked reconciles tracked study prose and study-monitor registries into
// an isolated content-addressed store. Dry runs perform the same discovery and
// parsing without writing records.
func ImportTracked(repoRoot, storeDir string, dryRun bool) (ImportLedger, error) {
	paths, err := trackedStudyPaths(repoRoot)
	if err != nil {
		return ImportLedger{}, err
	}
	ledger := ImportLedger{Schema: ImportSchema, DryRun: dryRun, Entries: make([]ImportEntry, 0, len(paths))}
	for _, path := range paths {
		entry := importTrackedPath(repoRoot, path)
		if entry.Status == ImportEligible && !dryRun {
			for i := range entry.Records {
				id, writeErr := writeImportRecord(storeDir, entry.Records[i])
				if writeErr != nil {
					return ImportLedger{}, fmt.Errorf("import %s: %w", path, writeErr)
				}
				entry.Records[i].ID = id
				entry.RecordIDs = append(entry.RecordIDs, id)
			}
			entry.Status = ImportImported
			entry.Reason = "records stored"
		}
		if dryRun {
			for _, record := range entry.Records {
				entry.RecordIDs = append(entry.RecordIDs, recordID(record))
			}
		}
		ledger.Entries = append(ledger.Entries, entry)
	}
	ledger.reconcile()
	return ledger, nil
}

func (l *ImportLedger) reconcile() {
	l.Attempted = len(l.Entries)
	for _, entry := range l.Entries {
		switch entry.Status {
		case ImportEligible:
			l.Eligible++
		case ImportImported:
			l.Imported++
		case ImportHeld:
			l.Held++
		case ImportRejected:
			l.Rejected++
		}
	}
}

func trackedStudyPaths(repoRoot string) ([]string, error) {
	cmd := exec.Command("git", "-C", repoRoot, "ls-files", "--", "docs/research", "docs/notes", "*study-monitor*.json", "*study_monitor*.json")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("discover tracked study artifacts: %w", err)
	}
	var paths []string
	for _, raw := range strings.Split(strings.ReplaceAll(string(out), "\\", "/"), "\n") {
		path := strings.TrimSpace(raw)
		if path == "" {
			continue
		}
		base := strings.ToLower(filepath.Base(path))
		if strings.HasPrefix(path, "docs/research/") ||
			(strings.HasPrefix(path, "docs/notes/") && strings.Contains(base, "concept")) ||
			(strings.Contains(base, "study-monitor") || strings.Contains(base, "study_monitor")) {
			paths = append(paths, path)
		}
	}
	sort.Strings(paths)
	return paths, nil
}

func importTrackedPath(repoRoot, path string) ImportEntry {
	entry := ImportEntry{Path: path}
	data, err := os.ReadFile(filepath.Join(repoRoot, filepath.FromSlash(path)))
	if err != nil {
		entry.Status, entry.Reason = ImportRejected, "read failed: "+err.Error()
		return entry
	}
	lower := strings.ToLower(path)
	if strings.HasSuffix(lower, ".md") {
		record, reason := parseProseRecord(path, data)
		if reason != "" {
			entry.Status, entry.Reason = ImportHeld, reason
			return entry
		}
		entry.Status, entry.Reason, entry.Records = ImportEligible, "deterministic prose metadata", []ImportRecord{record}
		return entry
	}
	if strings.HasSuffix(lower, ".json") {
		var registry Registry
		if err := json.Unmarshal(data, &registry); err != nil {
			entry.Status, entry.Reason = ImportRejected, "invalid monitor registry: "+err.Error()
			return entry
		}
		if err := registry.Validate(); err != nil {
			entry.Status, entry.Reason = ImportRejected, "invalid monitor registry: "+err.Error()
			return entry
		}
		for _, repo := range registry.Repositories {
			entry.Records = append(entry.Records, ImportRecord{Schema: ImportSchema, Kind: "monitor-registry", Title: repo.Repository, Source: repo.URL, Lineage: SourceLineage{Path: path, Date: repo.LastChecked, Revision: repo.CheckedRevision}})
		}
		entry.Status, entry.Reason = ImportEligible, "validated monitor registry"
		return entry
	}
	entry.Status, entry.Reason = ImportRejected, "unsupported artifact type"
	return entry
}

func parseProseRecord(path string, data []byte) (ImportRecord, string) {
	meta := map[string]string{}
	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "# ") && meta["title"] == "" {
			meta["title"] = strings.TrimSpace(strings.TrimPrefix(line, "# "))
		}
		for _, key := range []string{"source", "source-revision", "observed-at"} {
			prefix := key + ":"
			if strings.HasPrefix(strings.ToLower(line), prefix) {
				meta[key] = strings.TrimSpace(line[len(prefix):])
			}
		}
	}
	missing := make([]string, 0, 4)
	for _, key := range []string{"title", "source", "source-revision", "observed-at"} {
		if meta[key] == "" {
			missing = append(missing, key)
		}
	}
	if len(missing) > 0 {
		return ImportRecord{}, "ambiguous prose: missing " + strings.Join(missing, ", ")
	}
	if _, err := time.Parse("2006-01-02", meta["observed-at"]); err != nil {
		return ImportRecord{}, "ambiguous prose: observed-at must be YYYY-MM-DD"
	}
	return ImportRecord{Schema: ImportSchema, Kind: "prose", Title: meta["title"], Source: meta["source"], Lineage: SourceLineage{Path: path, Date: meta["observed-at"], Revision: meta["source-revision"]}}, ""
}

func writeImportRecord(storeDir string, record ImportRecord) (string, error) {
	if storeDir == "" {
		return "", errors.New("store directory is required for live import")
	}
	id := recordID(record)
	record.ID = id
	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return "", err
	}
	data = append(data, '\n')
	if err := os.MkdirAll(storeDir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(storeDir, id+".json")
	if existing, err := os.ReadFile(path); err == nil {
		if string(existing) != string(data) {
			return "", errors.New("content-address collision")
		}
		return id, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	return id, os.WriteFile(path, data, 0o644)
}

func recordID(record ImportRecord) string {
	record.ID = ""
	data, _ := json.Marshal(record)
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}
