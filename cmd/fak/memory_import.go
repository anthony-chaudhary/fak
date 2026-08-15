package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/memq"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

const claudeMemoryImportSchema = "fak-claude-memory-import/1"

var claudeImportBeforeResnapshot = func() {}

var (
	privateMemoryPattern = regexp.MustCompile(`(?i)(api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|secret|bearer\s+[a-z0-9._-]+|authorization:|cookie:|private[_ -]?key|account roster|oauth)`)
	sessionMemoryPattern = regexp.MustCompile(`(?i)(^|[-_ ])(session|transcript|conversation|scratch|orphan|todo|wip|handoff|current[-_ ]?run)([-_ .]|$)`)
	dateMemoryPattern    = regexp.MustCompile(`\b20\d\d[-/]\d\d[-/]\d\d\b`)
	wordPattern          = regexp.MustCompile(`[a-z0-9]+`)
)

type claudeImportSnapshot struct {
	Files int       `json:"files"`
	Bytes int64     `json:"bytes"`
	Hash  string    `json:"hash"`
	At    time.Time `json:"captured_at"`
}

type claudeImportReceipt struct {
	Schema      string               `json:"schema"`
	Mode        string               `json:"mode"`
	Source      claudeImportSnapshot `json:"source"`
	Destination string               `json:"destination,omitempty"`
	Counts      map[string]int       `json:"counts"`
	Reasons     map[string]int       `json:"reasons"`
	Accounted   int                  `json:"accounted_files"`
	Unexplained int                  `json:"unexplained_files"`
	Applied     int                  `json:"applied_cells"`
	Refusal     string               `json:"refusal,omitempty"`
}

type claudeSourceFile struct {
	path, name, hash string
	size             int64
	mtime            int64
	body             []byte
}

type importDecision struct {
	file                  claudeSourceFile
	class, reason, target string
}

func runMemoryImportClaude(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("memory import-claude", flag.ContinueOnError)
	fs.SetOutput(stderr)
	source := fs.String("source", "", "Claude project memory directory")
	destination := fs.String("destination", "", "explicit destination memory directory")
	apply := fs.Bool("apply", false, "write audited durable cells (default is dry-run)")
	consent := fs.String("consent-scope", "", "promotion consent scope (required with --apply)")
	producer := fs.String("producer", "", "promotion producer identity (required with --apply)")
	capture := fs.String("capture-time", "", "RFC3339 source capture time (required with --apply)")
	jsonOut := fs.Bool("json", true, "emit JSON receipt")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *source == "" {
		fmt.Fprintln(stderr, "memory import-claude: --source is required")
		return 2
	}
	if *apply && (*destination == "" || *consent == "" || *producer == "" || *capture == "") {
		fmt.Fprintln(stderr, "memory import-claude: --apply requires --destination, --consent-scope, --producer, and --capture-time")
		return 2
	}
	var captured time.Time
	var err error
	if *apply {
		captured, err = time.Parse(time.RFC3339, *capture)
		if err != nil {
			fmt.Fprintf(stderr, "memory import-claude: invalid --capture-time: %v\n", err)
			return 2
		}
	}
	receipt, decisions, err := inspectClaudeMemory(*source, *destination)
	if err != nil {
		if receipt.Schema == "" {
			receipt.Schema, receipt.Mode = claudeMemoryImportSchema, "dry-run"
		}
		receipt.Refusal = err.Error()
		_ = json.NewEncoder(stdout).Encode(receipt)
		return 1
	}
	if *apply {
		receipt.Mode = "apply"
		for _, d := range decisions {
			if d.class != "importable" {
				continue
			}
			if destinationHasImportedCell(*destination, d.file.hash) {
				receipt.Counts["importable"]--
				receipt.Counts["duplicate"]++
				receipt.Reasons["destination_exact"]++
				continue
			}
			if err := applyClaudeMemoryCell(*destination, d, *consent, *producer, captured); err != nil {
				receipt.Refusal = err.Error()
				_ = json.NewEncoder(stdout).Encode(receipt)
				return 1
			}
			receipt.Applied++
		}
	}
	if *jsonOut {
		if err := json.NewEncoder(stdout).Encode(receipt); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	return 0
}

func inspectClaudeMemory(source, destination string) (claudeImportReceipt, []importDecision, error) {
	r := claudeImportReceipt{Schema: claudeMemoryImportSchema, Mode: "dry-run", Destination: destination, Counts: map[string]int{}, Reasons: map[string]int{}}
	files, snap, err := snapshotClaudeMemory(source)
	if err != nil {
		return r, nil, err
	}
	r.Source = snap
	destBodies, err := destinationMemoryBodies(destination)
	if err != nil {
		return r, nil, err
	}
	seen := append([][]byte(nil), destBodies...)
	decisions := make([]importDecision, 0, len(files))
	for _, f := range files {
		d := classifyClaudeMemory(f)
		if d.class == "importable" {
			for _, prior := range seen {
				if normalizedMemory(string(prior)) == normalizedMemory(string(f.body)) {
					d.class, d.reason = "duplicate", "exact_duplicate"
					break
				}
				if nearDuplicateMemory(prior, f.body) {
					d.class, d.reason = "duplicate", "near_duplicate"
					break
				}
			}
		}
		if d.class == "importable" {
			stale, detail := staleMemoryClaims(source, f.name)
			if stale {
				d.class, d.reason = "stale", detail
			}
		}
		if d.class == "importable" {
			seen = append(seen, f.body)
			d.target = safeMemoryTarget(f.name, f.hash)
		}
		decisions = append(decisions, d)
		r.Counts[d.class]++
		r.Reasons[d.reason]++
	}
	r.Accounted = len(decisions)
	r.Unexplained = snap.Files - r.Accounted
	claudeImportBeforeResnapshot()
	_, after, err := snapshotClaudeMemory(source)
	if err != nil {
		return r, decisions, err
	}
	if snap.Files != after.Files || snap.Bytes != after.Bytes || snap.Hash != after.Hash {
		return r, decisions, errors.New("SOURCE_CHANGED: Claude memory directory mutated during inspection")
	}
	if r.Unexplained != 0 {
		return r, decisions, fmt.Errorf("UNEXPLAINED_FILES: %d", r.Unexplained)
	}
	return r, decisions, nil
}

func snapshotClaudeMemory(dir string) ([]claudeSourceFile, claudeImportSnapshot, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, claudeImportSnapshot{}, err
	}
	var files []claudeSourceFile
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			return nil, claudeImportSnapshot{}, err
		}
		body, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, claudeImportSnapshot{}, err
		}
		h := sha256.Sum256(body)
		files = append(files, claudeSourceFile{path: filepath.Join(dir, e.Name()), name: e.Name(), hash: hex.EncodeToString(h[:]), size: info.Size(), mtime: info.ModTime().UTC().UnixNano(), body: body})
	}
	sort.Slice(files, func(i, j int) bool { return files[i].name < files[j].name })
	h := sha256.New()
	var total int64
	for _, f := range files {
		total += f.size
		fmt.Fprintf(h, "%s\x00%d\x00%d\x00%s\n", f.name, f.size, f.mtime, f.hash)
	}
	return files, claudeImportSnapshot{Files: len(files), Bytes: total, Hash: hex.EncodeToString(h.Sum(nil)), At: time.Now().UTC()}, nil
}

func classifyClaudeMemory(f claudeSourceFile) importDecision {
	d := importDecision{file: f, class: "importable", reason: "durable_project_knowledge"}
	ext := strings.ToLower(filepath.Ext(f.name))
	if ext != ".md" && ext != ".txt" {
		d.class, d.reason = "unsupported", "unsupported_extension"
		return d
	}
	if strings.EqualFold(f.name, "MEMORY.md") {
		d.class, d.reason = "unsupported", "source_index"
		return d
	}
	lowerName, body := strings.ToLower(f.name), string(f.body)
	if privateMemoryPattern.MatchString(lowerName) || privateMemoryPattern.MatchString(body) {
		d.class, d.reason = "private", "private_material"
		return d
	}
	if sessionMemoryPattern.MatchString(lowerName) || strings.Contains(strings.ToLower(body), "<transcript") || strings.Contains(strings.ToLower(body), "assistant:") && strings.Contains(strings.ToLower(body), "user:") {
		d.class, d.reason = "session-specific", "session_or_transcript_residue"
		return d
	}
	if len(strings.TrimSpace(body)) < 80 {
		d.class, d.reason = "session-specific", "insufficient_durable_content"
		return d
	}
	if dateMemoryPattern.MatchString(lowerName) && (strings.Contains(lowerName, "handoff") || strings.Contains(lowerName, "status")) {
		d.class, d.reason = "session-specific", "dated_run_state"
	}
	return d
}

func destinationMemoryBodies(dir string) ([][]byte, error) {
	if dir == "" {
		return nil, nil
	}
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out [][]byte
	for _, e := range entries {
		if e.IsDir() || strings.EqualFold(e.Name(), "MEMORY.md") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, nil
}

func staleMemoryClaims(dir, name string) (bool, string) {
	b, label := notesMemoryBackend(dir)
	if b == nil {
		return false, ""
	}
	cells, err := b.Cells(context.Background())
	if err != nil {
		return false, ""
	}
	for _, c := range cells {
		if filepath.Base(c.ID) != name {
			continue
		}
		findings, err := b.Verify(context.Background(), c.ID)
		if err != nil {
			return true, "freshness_check_error"
		}
		for _, f := range findings {
			if f.Status != recall.ArtifactFresh {
				return true, "stale_concrete_claim"
			}
		}
	}
	_ = label
	return false, ""
}

func normalizedMemory(s string) string {
	return strings.Join(wordPattern.FindAllString(strings.ToLower(s), -1), " ")
}
func nearDuplicateMemory(a, b []byte) bool {
	aSet, bSet := tokenSet(a), tokenSet(b)
	if len(aSet) < 12 || len(bSet) < 12 {
		return false
	}
	inter := 0
	for w := range aSet {
		if _, ok := bSet[w]; ok {
			inter++
		}
	}
	union := len(aSet) + len(bSet) - inter
	return union > 0 && float64(inter)/float64(union) >= 0.82
}
func tokenSet(b []byte) map[string]struct{} {
	m := map[string]struct{}{}
	for _, w := range wordPattern.FindAllString(strings.ToLower(string(b)), -1) {
		if len(w) > 2 {
			m[w] = struct{}{}
		}
	}
	return m
}
func safeMemoryTarget(name, hash string) string {
	base := strings.TrimSuffix(strings.ToLower(name), filepath.Ext(name))
	var out []rune
	for _, r := range base {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			out = append(out, r)
		} else if len(out) > 0 && out[len(out)-1] != '-' {
			out = append(out, '-')
		}
	}
	clean := strings.Trim(string(out), "-")
	if clean == "" {
		clean = "claude-memory"
	}
	return clean + "-" + hash[:8] + ".md"
}

func destinationHasImportedCell(destination, sourceHash string) bool {
	prefix := []byte("  source_sha256: " + sourceHash)
	entries, err := os.ReadDir(destination)
	if err != nil {
		return false
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		body, err := os.ReadFile(filepath.Join(destination, entry.Name()))
		if err == nil && bytes.Contains(body, prefix) {
			return true
		}
	}
	return false
}

func applyClaudeMemoryCell(destination string, d importDecision, consent, producer string, captured time.Time) error {
	if err := os.MkdirAll(destination, 0o755); err != nil {
		return err
	}
	target := filepath.Join(destination, d.target)
	if _, err := os.Stat(target); err == nil {
		return fmt.Errorf("DESTINATION_EXISTS: %s", d.target)
	} else if !os.IsNotExist(err) {
		return err
	}
	audit := fmt.Sprintf("---\nfak_promotion_audit:\n  source_sha256: %s\n  source_bytes: %d\n  durability_class: durable_project_knowledge\n  consent_scope: %q\n  producer: %q\n  captured_at: %s\n---\n\n", d.file.hash, d.file.size, consent, producer, captured.UTC().Format(time.RFC3339))
	return os.WriteFile(target, append([]byte(audit), d.file.body...), 0o600)
}

// Keep memq linked here so importer freshness behavior remains tied to the same
// concrete backend used by memory recall rather than a second parser.
var _ *memq.NotesBackend
