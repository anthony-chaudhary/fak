package ctxmmu_test

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	_ "github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/toolbound"
)

// fak-test:runtime fast est=10ms lane=default
func TestClosedFileWindows(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "main.go")
	if err := os.WriteFile(path, []byte(strings.Repeat("package sample // a long inspection line\n", 40)), 0600); err != nil {
		t.Fatal(err)
	}
	reader := toolbound.NewWindowedFileReader(root)
	view, err := reader.OpenView(path, 1, 30)
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(view.Content)
	otherRange, err := reader.GotoView(2)
	if err != nil {
		t.Fatal(err)
	}
	small, err := reader.OpenView(path, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	page := func(turn int) ctxmmu.TokenPage {
		return ctxmmu.TokenPage{TurnIndex: turn, Kind: ctxmmu.PageKindToolResult,
			Role: "tool", ToolName: "view", Content: bytes.Clone(body), Resident: true}
	}
	cfg := ctxmmu.CompactorConfig{WindowSizeK: 1, CASPageOut: true,
		DeduplicateFileWindows: true, FileWindowTools: []string{"view", "other_view"}}
	for _, tc := range []struct {
		name   string
		change func(*ctxmmu.TokenPage, *ctxmmu.TokenPage, *ctxmmu.CompactorConfig)
		close  bool
	}{
		{"identical", func(_, _ *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) {}, true},
		{"active_latest", func(_, b *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) { b.TurnIndex = 6 }, true},
		{"pinned_latest", func(_, b *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) { b.Pinned = true }, true},
		{"pinned_older", func(a, _ *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) { a.Pinned = true }, false},
		{"active_older", func(a, b *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) { a.TurnIndex = 6; b.TurnIndex = 6 }, false},
		{"changed_content", func(_, b *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) {
			b.Content = bytes.Replace(b.Content, []byte("package sample"), []byte("package newer"), 1)
		}, false},
		{"different_range", func(_, b *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) { b.Content = []byte(otherRange.Content) }, false},
		{"different_path", func(_, b *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) {
			b.Content = bytes.Replace(b.Content, []byte("main.go"), []byte("Main.go"), 1)
		}, false},
		{"different_tool", func(_, b *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) { b.ToolName = "other_view" }, false},
		{"malformed_latest", func(_, b *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) { b.Content = b.Content[:len(b.Content)-1] }, false},
		{"user_spoof", func(_, b *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) { b.Kind = ctxmmu.PageKindUser; b.Role = "user" }, false},
		{"untrusted_tool", func(_, b *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) { b.ToolName = "bash" }, false},
		{"continuation", func(_, b *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) { b.IsContinuation = true; b.TurnIndex = 6 }, false},
		{"no_token_savings", func(a, _ *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) { a.Tokens = 1 }, false},
		{"small_window", func(a, b *ctxmmu.TokenPage, _ *ctxmmu.CompactorConfig) {
			a.Content = []byte(small.Content)
			b.Content = []byte(small.Content)
		}, false},
		{"default_off", func(_, _ *ctxmmu.TokenPage, c *ctxmmu.CompactorConfig) {
			c.DeduplicateFileWindows = false
			c.VerboseThresholdBytes = 100000
			c.VerboseThresholdTokens = 100000
		}, false},
		{"no_allowlist", func(_, _ *ctxmmu.TokenPage, c *ctxmmu.CompactorConfig) {
			c.FileWindowTools = nil
			c.VerboseThresholdBytes = 100000
			c.VerboseThresholdTokens = 100000
		}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, b, config := page(2), page(5), cfg
			tc.change(&a, &b, &config)
			original := bytes.Clone(a.Content)
			latest := bytes.Clone(b.Content)
			pages := []ctxmmu.TokenPage{a, b, {TurnIndex: 6, Kind: ctxmmu.PageKindUser, Content: []byte("continue")}}
			c := ctxmmu.NewCompactor(config)
			scan := c.Scan(pages)
			var report ctxmmu.CompactionReport
			got, err := c.CompactInPlace(pages, &report)
			if err != nil {
				t.Fatal(err)
			}
			closed := strings.Contains(got[0].Tombstone.Summary, "File window closed:")
			if closed != tc.close {
				t.Fatalf("closed = %v, want %v", closed, tc.close)
			}
			if !tc.close && tc.name != "continuation" && !bytes.Equal(got[0].Content, original) {
				t.Fatal("non-redundant older view changed")
			}
			if tc.close {
				if !bytes.Equal(got[1].Content, latest) || got[1].Tombstone.Active {
					t.Fatal("newest view changed")
				}
				if scan.ReclaimableBytes != report.BytesReclaimed || scan.ReclaimableTokens != report.TokensReclaimed || report.BytesReclaimed <= 0 || report.TokensReclaimed <= 0 {
					t.Fatalf("scan/compaction accounting mismatch: %+v / %+v", scan, report)
				}
				restored, err := c.ReFault(context.Background(), &got[0])
				if err != nil || !bytes.Equal(restored, original) {
					t.Fatalf("CAS restoration differs: %v", err)
				}
				before := bytes.Clone(got[0].Content)
				var again ctxmmu.CompactionReport
				if _, err := c.CompactInPlace(got, &again); err != nil || again.TombstonesCreated != 0 || !bytes.Equal(got[0].Content, before) {
					t.Fatalf("second compaction changed closed view: %+v, %v", again, err)
				}
			}
		})
	}
}
