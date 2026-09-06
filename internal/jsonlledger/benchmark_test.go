package jsonlledger

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type benchRow struct {
	Date   string `json:"date"`
	RunID  string `json:"run_id"`
	Status string `json:"status"`
	N      int    `json:"n"`
}

func BenchmarkParse(b *testing.B) {
	for _, count := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("%d_rows", count), func(b *testing.B) {
			var sb strings.Builder
			for i := 0; i < count; i++ {
				sb.WriteString(fmt.Sprintf(`{"date":"2026-09-%02d","run_id":"RID-%04d","status":"OK","n":%d}`+"\n", (i%28)+1, i, i))
			}
			content := sb.String()
			b.SetBytes(int64(len(content)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rows := Parse[benchRow](content, nil)
				if len(rows) != count {
					b.Fatalf("want %d, got %d", count, len(rows))
				}
			}
		})
	}

	b.Run("Filtered", func(b *testing.B) {
		var sb strings.Builder
		for i := 0; i < 100; i++ {
			sb.WriteString(fmt.Sprintf(`{"date":"2026-09-%02d","run_id":"RID-%04d","status":"OK","n":%d}`+"\n", (i%28)+1, i, i))
		}
		content := sb.String()
		b.SetBytes(int64(len(content)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rows := Parse[benchRow](content, func(r benchRow) bool {
				return r.N%2 == 0
			})
			if len(rows) != 50 {
				b.Fatalf("want 50, got %d", len(rows))
			}
		}
	})
}

func BenchmarkLatestBefore(b *testing.B) {
	for _, count := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("%d_rows", count), func(b *testing.B) {
			prior := make([]benchRow, count)
			for i := 0; i < count; i++ {
				prior[i] = benchRow{
					Date:   fmt.Sprintf("2026-09-%02dT%02d:00:00Z", (i%28)+1, i%24),
					RunID:  fmt.Sprintf("RID-%04d", i),
					Status: "OK",
					N:      i,
				}
			}
			ref := benchRow{Date: "2026-12-31T23:59:59Z", RunID: "RID-REF", N: -1}
			date := func(r benchRow) string { return r.Date }
			tiebreak := func(r benchRow) string { return r.RunID }

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				latest, ok := LatestBefore(ref, prior, date, tiebreak)
				if !ok {
					b.Fatal("expected row")
				}
				_ = latest
			}
		})
	}
}

func BenchmarkTailFold(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "tailfold_bench.jsonl")

	var sb strings.Builder
	for i := 0; i < 500; i++ {
		sb.WriteString(fmt.Sprintf(`{"date":"2026-09-%02d","run_id":"RID-%04d","status":"OK","n":%d}`+"\n", (i%28)+1, i, i))
	}
	baseContent := sb.String()
	if err := os.WriteFile(path, []byte(baseContent), 0o644); err != nil {
		b.Fatal(err)
	}

	step := func(acc int, _ benchRow) int {
		return acc + 1
	}

	b.Run("Full", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ck, err := TailFold(path, Checkpoint[int]{}, 0, step)
			if err != nil {
				b.Fatal(err)
			}
			if ck.State != 500 {
				b.Fatalf("want 500, got %d", ck.State)
			}
		}
	})

	initialCk, err := TailFold(path, Checkpoint[int]{}, 0, step)
	if err != nil {
		b.Fatal(err)
	}

	b.Run("NoChange", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ck, err := TailFold(path, initialCk, initialCk.State, step)
			if err != nil {
				b.Fatal(err)
			}
			if ck.State != 500 {
				b.Fatalf("want 500, got %d", ck.State)
			}
		}
	})

	var deltaSb strings.Builder
	for i := 500; i < 510; i++ {
		deltaSb.WriteString(fmt.Sprintf(`{"date":"2026-09-%02d","run_id":"RID-%04d","status":"OK","n":%d}`+"\n", (i%28)+1, i, i))
	}
	if err := os.WriteFile(path, []byte(baseContent+deltaSb.String()), 0o644); err != nil {
		b.Fatal(err)
	}

	b.Run("Delta_10_rows", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ck, err := TailFold(path, initialCk, initialCk.State, step)
			if err != nil {
				b.Fatal(err)
			}
			if ck.State != 510 {
				b.Fatalf("want 510, got %d", ck.State)
			}
		}
	})
}

func BenchmarkFirst(b *testing.B) {
	line := `{"date":"2026-09-05T12:00:00Z","run_id":"RID-123","status":"OK","n":1}` + "\n"
	b.Run("Immediate", func(b *testing.B) {
		r := strings.NewReader(line)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			r.Reset(line)
			row, found, err := First[benchRow](r)
			if err != nil || !found {
				b.Fatalf("First failed: %v, found: %v", err, found)
			}
			_ = row
		}
	})

	blankLines := "\n\n\n" + line
	b.Run("LeadingBlankLines", func(b *testing.B) {
		r := strings.NewReader(blankLines)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			r.Reset(blankLines)
			row, found, err := First[benchRow](r)
			if err != nil || !found {
				b.Fatalf("First failed: %v, found: %v", err, found)
			}
			_ = row
		}
	})
}

func BenchmarkAppendValidated(b *testing.B) {
	row := benchRow{
		Date:   "2026-09-05T12:00:00Z",
		RunID:  "RID-123456",
		Status: "OK",
		N:      42,
	}
	validate := func(r benchRow) error {
		if r.Date == "" || r.RunID == "" {
			return errors.New("missing key fields")
		}
		return nil
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := AppendValidated(io.Discard, row, validate); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAppendBounded(b *testing.B) {
	line := []byte(`{"date":"2026-09-05","run_id":"RID-001","status":"OK","n":1}` + "\n")

	b.Run("AppendOnly", func(b *testing.B) {
		dir := b.TempDir()
		path := filepath.Join(dir, "append.jsonl")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := AppendBounded(path, line, DefaultActiveBytes); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("WithRotation", func(b *testing.B) {
		dir := b.TempDir()
		path := filepath.Join(dir, "rotate.jsonl")
		maxBytes := int64(len(line) + 10)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := AppendBounded(path, line, maxBytes); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkReadTail(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "tail_bench.jsonl")
	var sb strings.Builder
	for i := 0; i < 1000; i++ {
		sb.WriteString(`{"date":"2026-09-05T12:00:00Z","run_id":"RID-000","status":"OK","n":1}` + "\n")
	}
	if err := os.WriteFile(path, []byte(sb.String()), 0o644); err != nil {
		b.Fatal(err)
	}

	b.Run("SmallWindow_1KB", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out := ReadTail(path, 1024)
			if len(out) == 0 {
				b.Fatal("empty tail")
			}
		}
	})

	b.Run("LargeWindow_32KB", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out := ReadTail(path, 32*1024)
			if len(out) == 0 {
				b.Fatal("empty tail")
			}
		}
	})
}
