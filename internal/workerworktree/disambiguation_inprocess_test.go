package workerworktree

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/conceptcatalog/scorecard"
)

func copyScorecardFixture(t *testing.T) string {
	t.Helper()
	src := filepath.Join("..", "conceptcatalog", "scorecard", "testdata", "fixture")
	dst := filepath.Join(t.TempDir(), "workspace")
	walkErr := filepath.WalkDir(src, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, relErr := filepath.Rel(src, p)
		if relErr != nil {
			return relErr
		}
		if d.IsDir() {
			return os.MkdirAll(filepath.Join(dst, rel), 0o755)
		}
		b, readErr := os.ReadFile(p)
		if readErr != nil {
			return readErr
		}
		return os.WriteFile(filepath.Join(dst, rel), bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n")), 0o644)
	})
	if walkErr != nil {
		t.Fatal(walkErr)
	}
	return dst
}

// fak-test:runtime fast est=1s
func TestRunAnalyzerReturnsOnlyTheVerdictSliceInProcess(t *testing.T) {
	ws := copyScorecardFixture(t)
	generated := filepath.Join(t.TempDir(), "generated")
	raw, err := runAnalyzer(context.Background(), ws, generated)
	if err != nil {
		t.Fatal(err)
	}
	full := scorecard.Collect(scorecard.ResolvePath(ws), "")
	whole, err := full.JSON()
	if err != nil {
		t.Fatal(err)
	}
	if len(raw)*10 > len(whole) {
		t.Fatalf("analyzer returned %d bytes against a %d-byte payload; want only the verdict slice", len(raw), len(whole))
	}
	var got scorecard.Summary
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	want := full.Summary()
	if got.OK != want.OK || got.Reason != want.Reason || got.Corpus.ClarityDefects != want.Corpus.ClarityDefects ||
		got.Corpus.CoverageDebt != want.Corpus.CoverageDebt || got.Corpus.Coverage.CoveragePct != want.Corpus.Coverage.CoveragePct ||
		len(got.Corpus.Coverage.PerFamily) != len(want.Corpus.Coverage.PerFamily) {
		t.Fatalf("summary = %+v, want %+v", got, want)
	}
	docs, err := full.DocFolder("")
	if err != nil {
		t.Fatal(err)
	}
	for name, body := range docs {
		b, readErr := os.ReadFile(filepath.Join(generated, name))
		if readErr != nil || !bytes.Equal(b, body) {
			t.Fatalf("generated %s does not match the fold: %v", name, readErr)
		}
	}
}

func TestRunAnalyzerHonoursCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	generated := filepath.Join(t.TempDir(), "generated")
	if _, err := runAnalyzer(ctx, copyScorecardFixture(t), generated); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled analyzer = %v, want context.Canceled", err)
	}
	if _, err := os.Stat(generated); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("a cancelled analyzer must not write the doc folder: %v", err)
	}
}
