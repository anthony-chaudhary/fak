package memorycotravel

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/rolloutmode"
)

// Invariant: memory cotravel evaluation is fail-closed and bounded.
// Guard: rollout gate defaults to shadow mode to prevent unintended disk writes across workspace boundaries.
// Contract: memory files are transferred only when allowed by the active strategy and gate.

const (
	// DefaultGate is the safe rollout mode used when no environment or option override is provided.
	DefaultGate = string(rolloutmode.Shadow)
	// DefaultStrategy is the fallback merge policy that preserves existing destination files.
	DefaultStrategy = "additive"
	// DefaultLedgerCap specifies the maximum byte threshold before the JSONL audit ledger undergoes single-generation rotation.
	DefaultLedgerCap = 8 * 1024 * 1024
)

// Decision computes the planned transfer action ("copy" or "skip") for a given source and destination file.
type Decision func(src, dst string) string

// PlanItem records the prospective transfer action and destination file presence for a single memory entry.
type PlanItem struct {
	Name      string `json:"name"`
	Action    string `json:"action"`
	DstExists bool   `json:"dst_exists"`
}

// Record captures an auditable memory synchronization execution including gate, strategy, plan, and outcome.
type Record struct {
	Timestamp    string     `json:"ts"`
	Session      string     `json:"session"`
	Slug         string     `json:"slug"`
	DstSlug      string     `json:"dst_slug"`
	Gate         string     `json:"gate"`
	Strategy     string     `json:"strategy"`
	SrcMemory    string     `json:"src_memory"`
	DstMemory    string     `json:"dst_memory"`
	SrcHasMemory bool       `json:"src_has_memory"`
	Plan         []PlanItem `json:"plan"`
	Copied       []string   `json:"copied"`
	Skipped      []string   `json:"skipped"`
	WouldCopy    []string   `json:"would_copy,omitempty"`
	Note         string     `json:"note,omitempty"`
}

// Strategies maps named resolution policies to their corresponding Decision implementation functions.
var Strategies = map[string]Decision{
	"additive":     Additive,
	"source_wins":  SourceWins,
	"newest_mtime": NewestMtime,
}

var memoryCotravelModes = []rolloutmode.Mode{rolloutmode.Off, rolloutmode.Shadow, rolloutmode.On}

func parseGate(raw string) rolloutmode.Mode {
	raw = strings.ToLower(strings.TrimSpace(raw))
	// Accept the retired applied-rung spelling at the environment boundary while
	// rendering and recording only the shared canonical `on` rung (#6090).
	if raw == "live" {
		raw = rolloutmode.On.String()
	}
	m, _ := rolloutmode.ParseIn(raw, memoryCotravelModes, rolloutmode.Shadow)
	return m
}

// Gate returns the active rollout gate string parsed from FAK_MEMORY_COTRAVEL.
func Gate() string { return parseGate(os.Getenv("FAK_MEMORY_COTRAVEL")).String() }

// StrategyName returns the configured resolution policy name parsed from FAK_MEMORY_MERGE.
func StrategyName() string {
	s := strings.ToLower(strings.TrimSpace(os.Getenv("FAK_MEMORY_MERGE")))
	if _, ok := Strategies[s]; ok {
		return s
	}
	return DefaultStrategy
}

// Differ compares two files on disk and returns true if their byte contents differ or either read fails.
func Differ(src, dst string) bool {
	a, err := os.ReadFile(src)
	if err != nil {
		return true
	}
	b, err := os.ReadFile(dst)
	if err != nil {
		return true
	}
	return !bytes.Equal(a, b)
}

// Additive implements a non-destructive conflict policy that copies only when the destination file is missing.
func Additive(_src, dst string) string {
	if _, err := os.Stat(dst); err == nil {
		return "skip"
	}
	return "copy"
}

// SourceWins implements a source-preferred conflict policy that copies when the source and destination differ.
func SourceWins(src, dst string) string {
	if Differ(src, dst) {
		return "copy"
	}
	return "skip"
}

// NewestMtime implements a timestamp-based conflict policy that copies when the source file has a newer modification time.
func NewestMtime(src, dst string) string {
	dstInfo, err := os.Stat(dst)
	if err != nil {
		return "copy"
	}
	srcInfo, err := os.Stat(src)
	if err != nil {
		return "skip"
	}
	if srcInfo.ModTime().After(dstInfo.ModTime()) {
		return "copy"
	}
	return "skip"
}

// PlanOneDir inspects the source memory directory and evaluates decision rules against corresponding destination paths.
func PlanOneDir(srcMem, dstMem string, decide Decision) []PlanItem {
	info, err := os.Stat(srcMem)
	if err != nil || !info.IsDir() {
		return nil
	}
	paths, _ := filepath.Glob(filepath.Join(srcMem, "*.md"))
	sort.Strings(paths)
	plan := make([]PlanItem, 0, len(paths))
	for _, src := range paths {
		name := filepath.Base(src)
		dst := filepath.Join(dstMem, name)
		if sameAbs(src, dst) {
			plan = append(plan, PlanItem{Name: name, Action: "skip", DstExists: true})
			continue
		}
		_, statErr := os.Stat(dst)
		plan = append(plan, PlanItem{Name: name, Action: decide(src, dst), DstExists: statErr == nil})
	}
	return plan
}

// Options provides parameter overrides for memory cotravel execution including destination slug, gate, and strategy.
type Options struct {
	DstSlug  string
	Gate     string
	Strategy string
}

// CotravelMemory executes memory synchronization from source to destination configuration roots under the specified options.
func CotravelMemory(srcCfg, dstCfg, slug, sid string, opts Options) Record {
	g := parseGate(opts.Gate).String()
	if strings.TrimSpace(opts.Gate) == "" {
		g = Gate()
	}
	strat := opts.Strategy
	if strat == "" {
		strat = StrategyName()
	}
	decide := Strategies[strat]
	if decide == nil {
		strat = DefaultStrategy
		decide = Additive
	}
	dstSlug := opts.DstSlug
	if dstSlug == "" {
		dstSlug = slug
	}
	srcMem := filepath.Join(srcCfg, "projects", slug, "memory")
	dstMem := filepath.Join(dstCfg, "projects", dstSlug, "memory")
	rec := Record{
		Timestamp:    time.Now().UTC().Format("2006-01-02T15:04:05Z"),
		Session:      sid,
		Slug:         slug,
		DstSlug:      dstSlug,
		Gate:         g,
		Strategy:     strat,
		SrcMemory:    srcMem,
		DstMemory:    dstMem,
		SrcHasMemory: isDir(srcMem),
	}
	if g == rolloutmode.Off.String() {
		rec.Plan = []PlanItem{}
		rec.Copied = []string{}
		rec.Skipped = []string{}
		rec.Note = "gate=off (no-op)"
		return rec
	}
	rec.Plan = PlanOneDir(srcMem, dstMem, decide)
	for _, item := range rec.Plan {
		if item.Action == "copy" {
			rec.WouldCopy = append(rec.WouldCopy, item.Name)
		} else {
			rec.Skipped = append(rec.Skipped, item.Name)
		}
	}
	if g == rolloutmode.Shadow.String() {
		rec.Copied = []string{}
		AppendLedger(rec)
		return rec
	}
	if len(rec.WouldCopy) > 0 {
		_ = os.MkdirAll(dstMem, 0o755)
	}
	for _, name := range rec.WouldCopy {
		if err := copyFile(filepath.Join(srcMem, name), filepath.Join(dstMem, name)); err == nil {
			rec.Copied = append(rec.Copied, name)
		}
	}
	AppendLedger(rec)
	return rec
}

// LedgerPath resolves the audit log location using FAK_MEMORY_COTRAVEL_LEDGER or home directory defaults.
func LedgerPath() string {
	if p := os.Getenv("FAK_MEMORY_COTRAVEL_LEDGER"); strings.TrimSpace(p) != "" {
		return p
	}
	home := os.Getenv("FLEET_USER_HOME")
	if home == "" {
		if h, err := os.UserHomeDir(); err == nil {
			home = h
		}
	}
	return filepath.Join(home, ".claude", "fak-memory-cotravel-ledger.jsonl")
}

// AppendLedger records an auditable synchronization entry to the JSONL log file, performing rotation if needed.
func AppendLedger(row Record) {
	path := LedgerPath()
	if path == "" {
		return
	}
	_ = rotateIfNeeded(path)
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	var lines []string
	if f, err := os.Open(path); err == nil {
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			if strings.TrimSpace(sc.Text()) != "" {
				lines = append(lines, sc.Text())
			}
		}
		_ = f.Close()
	}
	data, err := json.Marshal(row)
	if err != nil {
		return
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), "*.tmp")
	if err != nil {
		return
	}
	tmpName := tmp.Name()
	for _, line := range lines {
		_, _ = tmp.WriteString(line + "\n")
	}
	_, _ = tmp.Write(append(data, '\n'))
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return
	}
	if err := os.Rename(tmpName, path); err != nil {
		_ = os.Remove(tmpName)
	}
}

// ReadLedger deserializes and returns all recorded synchronization entries from the active ledger path.
func ReadLedger() []map[string]any {
	path := LedgerPath()
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	var rows []map[string]any
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row map[string]any
		if json.Unmarshal([]byte(line), &row) == nil {
			rows = append(rows, row)
		}
	}
	return rows
}

func rotateIfNeeded(path string) error {
	info, err := os.Stat(path)
	if err == nil && info.Size() > DefaultLedgerCap {
		return os.Rename(path, path+".1")
	}
	return nil
}

func sameAbs(a, b string) bool {
	aa, errA := filepath.Abs(a)
	bb, errB := filepath.Abs(b)
	return errA == nil && errB == nil && aa == bb
}

func isDir(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func copyFile(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	info, _ := os.Stat(src)
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(dst, data, 0o644); err != nil {
		return err
	}
	if info != nil {
		_ = os.Chtimes(dst, info.ModTime(), info.ModTime())
	}
	return nil
}
