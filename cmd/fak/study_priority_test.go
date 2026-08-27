package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studyprio"
)

func TestStudyPriorityBuildWiresSourceAndOutputs(t *testing.T) {
	wantLedger := studyprio.Ledger{}
	wantSummary := studyprio.Summary{}
	var gotOptions studyprio.BuildOptions
	writes := map[string][]byte{}
	ops := studyPriorityOperations{
		build: func(options studyprio.BuildOptions) (studyprio.Ledger, studyprio.Summary, error) {
			gotOptions = options
			return wantLedger, wantSummary, nil
		},
		marshalLedger: func(ledger studyprio.Ledger) ([]byte, error) {
			if !reflect.DeepEqual(ledger, wantLedger) {
				t.Fatalf("ledger = %+v, want %+v", ledger, wantLedger)
			}
			return []byte("ledger-bytes"), nil
		},
		marshalSummary: func(summary studyprio.Summary) ([]byte, error) {
			if !reflect.DeepEqual(summary, wantSummary) {
				t.Fatalf("summary = %+v, want %+v", summary, wantSummary)
			}
			return []byte("# Study priority\n"), nil
		},
		write: func(path string, data []byte) error {
			writes[path] = append([]byte(nil), data...)
			return nil
		},
	}
	var stdout, stderr bytes.Buffer
	code := runStudyPriorityWithOperations(&stdout, &stderr, []string{
		"build",
		"--source-ledger", "source.json",
		"--ledger", "priority.json",
		"--summary", "README.md",
	}, ops)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if gotOptions.SourceLedgerPath != "source.json" {
		t.Fatalf("options = %+v", gotOptions)
	}
	if string(writes["priority.json"]) != "ledger-bytes" || string(writes["README.md"]) != "# Study priority\n" {
		t.Fatalf("writes = %q", writes)
	}
	if !strings.Contains(stdout.String(), "priority.json") || !strings.Contains(stdout.String(), "README.md") {
		t.Fatalf("stdout = %q", stdout.String())
	}
}

func TestWriteStudyPriorityFileCreatesParentAndReplacesOutput(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "ledger.json")
	if err := writeStudyPriorityFile(path, []byte("first")); err != nil {
		t.Fatal(err)
	}
	if err := writeStudyPriorityFile(path, []byte("second")); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "second" {
		t.Fatalf("output = %q, want replacement", got)
	}
}

func TestStudyPriorityValidateWiresAllArtifacts(t *testing.T) {
	var gotOptions studyprio.ValidateOptions
	ops := studyPriorityOperations{
		validate: func(options studyprio.ValidateOptions) error {
			gotOptions = options
			return nil
		},
	}
	var stdout, stderr bytes.Buffer
	code := runStudyPriorityWithOperations(&stdout, &stderr, []string{
		"validate",
		"--source-ledger", "source.json",
		"--ledger", "priority.json",
		"--summary", "README.md",
	}, ops)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if gotOptions.SourceLedgerPath != "source.json" || gotOptions.LedgerPath != "priority.json" || gotOptions.SummaryPath != "README.md" {
		t.Fatalf("options = %+v", gotOptions)
	}
	if !strings.Contains(stdout.String(), "valid study-priority ledger priority.json") {
		t.Fatalf("stdout = %q", stdout.String())
	}
}

func TestStudyPriorityUsageAndFailures(t *testing.T) {
	t.Run("missing required flag", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runStudyPriorityWithOperations(&stdout, &stderr, []string{"build", "--source-ledger", "source.json"}, studyPriorityOperations{})
		if code != 2 || !strings.Contains(stderr.String(), "--summary PATH") {
			t.Fatalf("code=%d stderr=%q", code, stderr.String())
		}
	})

	t.Run("unknown subcommand", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runStudyPriorityWithOperations(&stdout, &stderr, []string{"rank"}, studyPriorityOperations{})
		if code != 2 || !strings.Contains(stderr.String(), "<build|validate>") {
			t.Fatalf("code=%d stderr=%q", code, stderr.String())
		}
	})

	t.Run("validation failure", func(t *testing.T) {
		ops := studyPriorityOperations{
			validate: func(studyprio.ValidateOptions) error { return errors.New("checksum mismatch") },
		}
		var stdout, stderr bytes.Buffer
		code := runStudyPriorityWithOperations(&stdout, &stderr, []string{
			"validate", "--source-ledger", "source.json", "--ledger", "priority.json", "--summary", "README.md",
		}, ops)
		if code != 1 || !strings.Contains(stderr.String(), "study-priority: validate: checksum mismatch") {
			t.Fatalf("code=%d stderr=%q", code, stderr.String())
		}
	})
}
