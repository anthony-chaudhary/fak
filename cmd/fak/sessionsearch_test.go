package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionsearch"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

func TestRunSessionSearch(t *testing.T) {
	t.Chdir(t.TempDir())

	t.Run("json empty results array", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		rc := runSessionSearch(&stdout, &stderr, []string{"--query", "fak", "--json"})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
		}
		var env sessionSearchResultsEnvelope
		if err := json.Unmarshal(stdout.Bytes(), &env); err != nil {
			t.Fatalf("failed to unmarshal json envelope: %v, output: %s", err, stdout.String())
		}
		if env.Schema != "fak.sessionsearch-results/1" {
			t.Errorf("expected schema %q, got %q", "fak.sessionsearch-results/1", env.Schema)
		}
		if env.Query != "fak" {
			t.Errorf("expected query %q, got %q", "fak", env.Query)
		}
		if env.Total != 0 {
			t.Errorf("expected total 0, got %d", env.Total)
		}
		if env.Hits == nil || len(env.Hits) != 0 {
			t.Errorf("expected empty non-nil hits slice, got %#v", env.Hits)
		}
	})

	t.Run("default text", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		rc := runSessionSearch(&stdout, &stderr, []string{"--query", "fak"})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
		}
		if !strings.Contains(stdout.String(), `sessionsearch: query "fak" matched 0 hit(s)`) {
			t.Errorf("expected matching query text, got: %s", stdout.String())
		}
	})

	t.Run("with journal file", func(t *testing.T) {
		tmpDir := t.TempDir()
		journalPath := filepath.Join(tmpDir, "journal.jsonl")
		content := `{"kind":"spawn","call_id":"c1","session":"interactive-42","tool":"slow_fetch","at_unix_ms":1700000000000,"deadline_ms":30000}
{"kind":"kill","call_id":"c1","session":"interactive-42","reason":"TOOL_DEADLINE_EXCEEDED","at_unix_ms":1700000032000}`
		if err := os.WriteFile(journalPath, []byte(content), 0o600); err != nil {
			t.Fatalf("failed to write journal: %v", err)
		}
		var stdout, stderr bytes.Buffer
		rc := runSessionSearch(&stdout, &stderr, []string{"--journal", journalPath, "--query", "deadline", "--json"})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
		}
		var env sessionSearchResultsEnvelope
		if err := json.Unmarshal(stdout.Bytes(), &env); err != nil {
			t.Fatalf("failed to unmarshal json envelope: %v, output: %s", err, stdout.String())
		}
		if env.Schema != "fak.sessionsearch-results/1" {
			t.Errorf("expected schema %q, got %q", "fak.sessionsearch-results/1", env.Schema)
		}
		if env.Query != "deadline" {
			t.Errorf("expected query %q, got %q", "deadline", env.Query)
		}
		if env.Total != 1 {
			t.Errorf("expected total 1, got %d", env.Total)
		}
		if !strings.Contains(stdout.String(), "TOOL_DEADLINE_EXCEEDED") {
			t.Errorf("expected matched event in json output, got: %s", stdout.String())
		}
	})

	t.Run("invalid flag", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		rc := runSessionSearch(&stdout, &stderr, []string{"--invalid-flag-xyz"})
		if rc != 2 {
			t.Fatalf("expected rc 2 on invalid flag, got %d", rc)
		}
	})
}

func TestRunSessionSearch_DefaultJournalAndSchemaEnvelope(t *testing.T) {
	tmpDir := t.TempDir()
	t.Chdir(tmpDir)

	journalDir := filepath.Join(tmpDir, ".fak", "toolproc")
	if err := os.MkdirAll(journalDir, 0o755); err != nil {
		t.Fatalf("failed to create default journal dir: %v", err)
	}

	content := `{"kind":"spawn","call_id":"c1","session":"sess-1","tool":"read_file","at_unix_ms":1700000000000}
{"kind":"exit","call_id":"c1","session":"sess-1","status":"ok","at_unix_ms":1700000001000}`
	journalPath := filepath.Join(journalDir, "journal.jsonl")
	if err := os.WriteFile(journalPath, []byte(content), 0o600); err != nil {
		t.Fatalf("failed to write default journal: %v", err)
	}

	var stdout, stderr bytes.Buffer
	rc := runSessionSearch(&stdout, &stderr, []string{"--query", "read_file", "--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
	}

	var env sessionSearchResultsEnvelope
	if err := json.Unmarshal(stdout.Bytes(), &env); err != nil {
		t.Fatalf("failed to unmarshal json envelope: %v\noutput: %s", err, stdout.String())
	}

	if env.Schema != "fak.sessionsearch-results/1" {
		t.Errorf("expected schema %q, got %q", "fak.sessionsearch-results/1", env.Schema)
	}
	if env.Query != "read_file" {
		t.Errorf("expected query %q, got %q", "read_file", env.Query)
	}
	if env.Total != 1 {
		t.Errorf("expected total 1, got %d", env.Total)
	}
	if len(env.Hits) != 1 {
		t.Fatalf("expected 1 hit, got %d", len(env.Hits))
	}
	if !strings.Contains(env.Hits[0].Doc.Text, "read_file") {
		t.Errorf("expected hit doc text to contain %q, got %q", "read_file", env.Hits[0].Doc.Text)
	}
}

func TestRunSessionSearch_WindowsCompactionShareMode(t *testing.T) {
	tmpDir := t.TempDir()
	journalPath := filepath.Join(tmpDir, "journal.jsonl")

	initial := []byte(`{"kind":"spawn","call_id":"c1","session":"s1","tool":"read_file","at_unix_ms":1000}
{"kind":"exit","call_id":"c1","session":"s1","status":"ok","at_unix_ms":2000}
{"kind":"spawn","call_id":"c2","session":"s1","tool":"list_files","at_unix_ms":3000}
{"kind":"exit","call_id":"c2","session":"s1","status":"ok","at_unix_ms":4000}
`)
	if err := os.WriteFile(journalPath, initial, 0o600); err != nil {
		t.Fatalf("failed to write initial journal: %v", err)
	}

	// 1. Open the journal using toolproc.OpenShareDelete (the opener used by sessionsearch).
	f, err := toolproc.OpenShareDelete(journalPath)
	if err != nil {
		t.Fatalf("toolproc.OpenShareDelete: %v", err)
	}
	defer f.Close()

	// Parse documents to simulate an active reader.
	docs, err := sessionsearch.DocsFromJournal(f)
	if err != nil {
		t.Fatalf("DocsFromJournal: %v", err)
	}
	if len(docs) == 0 {
		t.Fatalf("expected non-empty docs from initial journal")
	}

	// 2. Perform compaction swap while f is held open.
	// On Windows, if the file was opened with standard os.Open (lacking FILE_SHARE_DELETE),
	// this rename/compaction swap fails with ERROR_ACCESS_DENIED. With OpenShareDelete,
	// the swap succeeds.
	compacted, err := toolproc.CompactJournalFile(journalPath, 0, 1)
	if err != nil {
		t.Fatalf("CompactJournalFile while journal held open by sessionsearch reader: %v", err)
	}
	if !compacted {
		t.Fatalf("expected journal to be compacted")
	}

	// 3. runSessionSearch itself must succeed when invoked on the compacted journal.
	var stdout, stderr bytes.Buffer
	rc := runSessionSearch(&stdout, &stderr, []string{"--journal", journalPath, "--query", "list_files", "--json"})
	if rc != 0 {
		t.Fatalf("runSessionSearch failed with rc %d, stderr: %s", rc, stderr.String())
	}
	var env sessionSearchResultsEnvelope
	if err := json.Unmarshal(stdout.Bytes(), &env); err != nil {
		t.Fatalf("failed to unmarshal json envelope: %v, output: %s", err, stdout.String())
	}
	if env.Total != 1 {
		t.Fatalf("expected 1 hit for list_files after compaction, got %d", env.Total)
	}

	// 4. On Windows, explicitly prove the contrast: opening with os.Open without
	// FILE_SHARE_DELETE locks out the compaction swap with ERROR_ACCESS_DENIED.
	if runtime.GOOS == "windows" {
		lockedJournal := filepath.Join(tmpDir, "locked_journal.jsonl")
		if err := os.WriteFile(lockedJournal, initial, 0o600); err != nil {
			t.Fatalf("failed to write locked journal: %v", err)
		}
		lockedFile, err := os.Open(lockedJournal)
		if err != nil {
			t.Fatalf("os.Open: %v", err)
		}
		defer lockedFile.Close()

		_, lockErr := toolproc.CompactJournalFile(lockedJournal, 0, 1)
		if lockErr == nil {
			t.Errorf("expected compaction swap to fail with access denied when held open by os.Open on Windows, but it succeeded")
		}
	}
}

func TestRunSessionSearch_DefaultJournalDisappearsGracefully(t *testing.T) {
	tmpDir := t.TempDir()
	t.Chdir(tmpDir)

	journalDir := filepath.Join(tmpDir, ".fak", "toolproc")
	if err := os.MkdirAll(journalDir, 0o755); err != nil {
		t.Fatalf("failed to create default journal dir: %v", err)
	}
	journalPath := filepath.Join(journalDir, "journal.jsonl")

	// 1. Absent default journal: returns 0 hits and exit code 0.
	var stdout, stderr bytes.Buffer
	rc := runSessionSearch(&stdout, &stderr, []string{"--query", "fak", "--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0 with absent default journal, got %d, stderr: %s", rc, stderr.String())
	}
	var env sessionSearchResultsEnvelope
	if err := json.Unmarshal(stdout.Bytes(), &env); err != nil {
		t.Fatalf("failed to unmarshal json envelope: %v", err)
	}
	if env.Total != 0 {
		t.Fatalf("expected total 0, got %d", env.Total)
	}

	// 2. Concurrent churn: rapidly create and rotate/remove default journal while searching.
	// In the TOCTOU window between os.Stat and OpenShareDelete, OpenShareDelete encounters
	// ErrNotExist when the journal is rotated/removed. With the fix, runSessionSearch must
	// handle this gracefully and never return exit code 2.
	rotatedPath := filepath.Join(tmpDir, "journal.rotated.jsonl")
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		content := []byte(`{"kind":"spawn","call_id":"c1","session":"s1","tool":"churn_tool","at_unix_ms":1000}` + "\n")
		for {
			select {
			case <-stop:
				return
			default:
				if err := os.WriteFile(journalPath, content, 0o600); err == nil {
					_ = os.Rename(journalPath, rotatedPath)
					_ = os.Remove(rotatedPath)
				}
			}
		}
	}()

	for i := 0; i < 100; i++ {
		stdout.Reset()
		stderr.Reset()
		rc := runSessionSearch(&stdout, &stderr, []string{"--query", "churn_tool", "--json"})
		if rc != 0 {
			close(stop)
			<-done
			t.Fatalf("iteration %d: expected rc 0 during concurrent disappearance, got %d, stderr: %s", i, rc, stderr.String())
		}
	}
	close(stop)
	<-done

	// 3. Contrast: an explicit non-existent --journal flag must still return exit code 2.
	stdout.Reset()
	stderr.Reset()
	missingExplicit := filepath.Join(tmpDir, "missing_explicit.jsonl")
	rc = runSessionSearch(&stdout, &stderr, []string{"--journal", missingExplicit, "--query", "churn_tool"})
	if rc != 2 {
		t.Fatalf("expected rc 2 for explicit missing journal, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "open journal") {
		t.Fatalf("expected 'open journal' error in stderr, got: %s", stderr.String())
	}
}

func TestRunSessionSearch_ConcurrentTornLineTolerance(t *testing.T) {
	tmpDir := t.TempDir()
	journalPath := filepath.Join(tmpDir, "journal.jsonl")

	// Journal with two valid events followed by a truncated / torn trailing write without newline.
	content := []byte(`{"kind":"spawn","call_id":"c1","session":"s1","tool":"search_index","at_unix_ms":1000}
{"kind":"exit","call_id":"c1","session":"s1","status":"ok","at_unix_ms":2000}
{"kind":"spawn","call_id":"c2","session":"s1","tool":"torn_wr`)

	if err := os.WriteFile(journalPath, content, 0o600); err != nil {
		t.Fatalf("failed to write journal: %v", err)
	}

	var stdout, stderr bytes.Buffer
	rc := runSessionSearch(&stdout, &stderr, []string{"--journal", journalPath, "--query", "search_index", "--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0 despite trailing torn line, got %d, stderr: %s", rc, stderr.String())
	}

	var env sessionSearchResultsEnvelope
	if err := json.Unmarshal(stdout.Bytes(), &env); err != nil {
		t.Fatalf("failed to unmarshal json envelope: %v, output: %s", err, stdout.String())
	}
	if env.Total != 1 {
		t.Fatalf("expected 1 hit for valid event, got %d", env.Total)
	}
	if !strings.Contains(env.Hits[0].Doc.Text, "search_index") {
		t.Fatalf("expected hit to contain search_index, got: %s", env.Hits[0].Doc.Text)
	}

	// Querying for terms present only in the torn line yields 0 hits, but succeeds with rc 0.
	stdout.Reset()
	stderr.Reset()
	rc = runSessionSearch(&stdout, &stderr, []string{"--journal", journalPath, "--query", "torn_wr", "--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0 for torn line query, got %d, stderr: %s", rc, stderr.String())
	}
	env = sessionSearchResultsEnvelope{}
	if err := json.Unmarshal(stdout.Bytes(), &env); err != nil {
		t.Fatalf("failed to unmarshal json envelope: %v", err)
	}
	if env.Total != 0 {
		t.Fatalf("expected total 0 for torn write terms, got %d", env.Total)
	}

	// Also verify default journal path with trailing incomplete event.
	t.Chdir(tmpDir)
	defaultJournalDir := filepath.Join(tmpDir, ".fak", "toolproc")
	if err := os.MkdirAll(defaultJournalDir, 0o755); err != nil {
		t.Fatalf("failed to create default journal dir: %v", err)
	}
	defaultJournalPath := filepath.Join(defaultJournalDir, "journal.jsonl")
	defaultContent := []byte(`{"kind":"spawn","call_id":"c1","session":"s1","tool":"default_tool","at_unix_ms":1000}
{"kind":"spawn"}` + "\n")
	if err := os.WriteFile(defaultJournalPath, defaultContent, 0o600); err != nil {
		t.Fatalf("failed to write default journal: %v", err)
	}

	stdout.Reset()
	stderr.Reset()
	rc = runSessionSearch(&stdout, &stderr, []string{"--query", "default_tool", "--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0 with trailing incomplete event in default journal, got %d, stderr: %s", rc, stderr.String())
	}
	env = sessionSearchResultsEnvelope{}
	if err := json.Unmarshal(stdout.Bytes(), &env); err != nil {
		t.Fatalf("failed to unmarshal json envelope: %v", err)
	}
	if env.Total != 1 {
		t.Fatalf("expected 1 hit from default journal, got %d", env.Total)
	}

	// Contrast: corruption in the middle of the journal must fail with rc 2.
	middleCorruptPath := filepath.Join(tmpDir, "middle_corrupt.jsonl")
	middleCorruptContent := []byte(`{"kind":"spawn","call_id":"c1","session":"s1","tool":"valid_first","at_unix_ms":1000}
{"kind":"bogus_kind","at_unix_ms":1500}
{"kind":"spawn","call_id":"c2","session":"s1","tool":"valid_second","at_unix_ms":2000}` + "\n")
	if err := os.WriteFile(middleCorruptPath, middleCorruptContent, 0o600); err != nil {
		t.Fatalf("failed to write middle corrupt journal: %v", err)
	}
	stdout.Reset()
	stderr.Reset()
	rc = runSessionSearch(&stdout, &stderr, []string{"--journal", middleCorruptPath, "--query", "valid_first"})
	if rc != 2 {
		t.Fatalf("expected rc 2 for middle-corrupt journal, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "read journal") {
		t.Fatalf("expected 'read journal' error in stderr, got: %s", stderr.String())
	}
}
