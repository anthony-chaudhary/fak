package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const codexLoopArchiveSchema = "fak.sessions.codex_loop_archive.v1"

type codexLoopArchiveManifest struct {
	Schema      string `json:"schema"`
	ArchivedAt  string `json:"archived_at"`
	Source      string `json:"source"`
	Destination string `json:"destination"`
	SHA256      string `json:"sha256"`
	SessionID   string `json:"session_id"`
	Verdict     string `json:"verdict"`
	Reason      string `json:"reason"`
	DryRun      bool   `json:"dry_run,omitempty"`
	Idempotent  bool   `json:"idempotent,omitempty"`
}

var codexLoopArchiveNow = time.Now

func runSessionsCodexLoopArchive(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("sessions codex-loop archive", flag.ContinueOnError)
	fs.SetOutput(stderr)
	path := fs.String("path", "", "terminal Codex rollout JSONL to archive")
	codexHome := fs.String("codex-home", "", "Codex home directory (default: $CODEX_HOME or ~/.codex)")
	dryRun := fs.Bool("dry-run", false, "prove eligibility and print the manifest without moving bytes")
	asJSON := fs.Bool("json", false, "emit stable JSON")
	if code, done := parseFlagsRejectArgs(fs, args, stderr); done {
		return code
	}
	if strings.TrimSpace(*path) == "" {
		fmt.Fprintln(stderr, "fak sessions codex-loop archive: --path is required")
		return 2
	}
	manifest, err := archiveTerminalCodexLoop(*path, *codexHome, *dryRun)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop archive: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, manifest, "fak sessions codex-loop archive")
	}
	action := "archived"
	if manifest.DryRun {
		action = "would archive"
	} else if manifest.Idempotent {
		action = "already archived"
	}
	fmt.Fprintf(stdout, "%s terminal Codex LOOP %s -> %s (sha256:%s)\n", action, manifest.Source, manifest.Destination, manifest.SHA256)
	return 0
}

func archiveTerminalCodexLoop(source, codexHome string, dryRun bool) (codexLoopArchiveManifest, error) {
	home, err := resolvedCodexLoopHome(codexHome)
	if err != nil {
		return codexLoopArchiveManifest{}, err
	}
	source, err = filepath.Abs(source)
	if err != nil {
		return codexLoopArchiveManifest{}, err
	}
	sessionsRoot := filepath.Join(home, "sessions")
	if !pathWithinCodexSessions(source, sessionsRoot) {
		return codexLoopArchiveManifest{}, fmt.Errorf("refuse %s: source is outside %s", source, sessionsRoot)
	}
	archiveRoot := filepath.Join(home, "archived-sessions", "codex-loop")
	if prior, ok := findCodexLoopArchiveManifest(archiveRoot, source); ok {
		prior.Idempotent = true
		return prior, nil
	}
	diagnosis, err := diagnoseCodexLoopPath(source)
	if err != nil {
		return codexLoopArchiveManifest{}, fmt.Errorf("diagnose source: %w", err)
	}
	if diagnosis.Verdict != "LOOP" {
		return codexLoopArchiveManifest{}, fmt.Errorf("refuse %s: verdict is %s, not LOOP", source, diagnosis.Verdict)
	}
	rows, err := readSessionRows()
	if err != nil {
		return codexLoopArchiveManifest{}, fmt.Errorf("refuse %s: lifecycle evidence unavailable: %w", source, err)
	}
	state := loopStateForSession(rows, diagnosis.SessionID)
	if state != loopStateTerminal {
		return codexLoopArchiveManifest{}, fmt.Errorf("refuse %s: session %q lifecycle is %s; only independently registered terminal sessions can be archived", source, diagnosis.SessionID, state)
	}
	raw, err := os.ReadFile(source)
	if err != nil {
		return codexLoopArchiveManifest{}, err
	}
	sum := sha256.Sum256(raw)
	digest := hex.EncodeToString(sum[:])
	base := strings.TrimSuffix(filepath.Base(source), filepath.Ext(source))
	sourceSum := sha256.Sum256([]byte(filepath.Clean(source)))
	destination := filepath.Join(archiveRoot, base+"-"+digest[:12]+"-"+hex.EncodeToString(sourceSum[:4])+".jsonl")
	manifest := codexLoopArchiveManifest{
		Schema: codexLoopArchiveSchema, ArchivedAt: codexLoopArchiveNow().UTC().Format(time.RFC3339Nano),
		Source: source, Destination: destination, SHA256: digest, SessionID: diagnosis.SessionID,
		Verdict: diagnosis.Verdict, Reason: diagnosis.Reason, DryRun: dryRun,
	}
	if dryRun {
		return manifest, nil
	}
	if err := os.MkdirAll(archiveRoot, 0o700); err != nil {
		return codexLoopArchiveManifest{}, err
	}
	if existing, err := os.ReadFile(destination); err == nil {
		existingSum := sha256.Sum256(existing)
		if existingSum != sum {
			return codexLoopArchiveManifest{}, fmt.Errorf("archive destination collision: %s", destination)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return codexLoopArchiveManifest{}, err
	} else if err := os.Rename(source, destination); err != nil {
		return codexLoopArchiveManifest{}, fmt.Errorf("move rollout: %w", err)
	}
	body, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return codexLoopArchiveManifest{}, err
	}
	if err := os.WriteFile(strings.TrimSuffix(destination, ".jsonl")+".manifest.json", append(body, '\n'), 0o600); err != nil {
		// Roll back the move so a successful archive always has its witness.
		_ = os.Rename(destination, source)
		return codexLoopArchiveManifest{}, fmt.Errorf("write manifest: %w", err)
	}
	return manifest, nil
}

func findCodexLoopArchiveManifest(root, source string) (codexLoopArchiveManifest, bool) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return codexLoopArchiveManifest{}, false
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".manifest.json") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(root, entry.Name()))
		if err != nil {
			continue
		}
		var manifest codexLoopArchiveManifest
		if json.Unmarshal(raw, &manifest) == nil && manifest.Schema == codexLoopArchiveSchema && filepath.Clean(manifest.Source) == filepath.Clean(source) {
			if _, err := os.Stat(manifest.Destination); err == nil {
				return manifest, true
			}
		}
	}
	return codexLoopArchiveManifest{}, false
}

func pathWithinCodexSessions(path, sessionsRoot string) bool {
	root, err := filepath.Abs(sessionsRoot)
	if err != nil {
		return false
	}
	candidate, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(root, candidate)
	return err == nil && rel != "." && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}
