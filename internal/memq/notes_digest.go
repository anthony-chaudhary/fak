package memq

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/memoryread"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// RenderNotesDigest renders MEMORY.md plus linked fact bodies through the same
// trust gate `fak memory recall` applies per query (#2429): every fact body is
// paged in via NotesBackend.Materialize, so a sealed or stale-claim note never
// enters the digest wearing the authority of a fact. This is the session-start
// counterpart to memoryread.RenderDigest (which reads raw bytes off disk with no
// screen and no claim probe), wired into `fak memory-read`. Byte-budget and
// overflow-naming behavior otherwise match RenderDigest exactly.
func RenderNotesDigest(storeDir string, indexOnly bool, maxBytes int) string {
	dir := storeDir
	indexPath := filepath.Join(dir, "MEMORY.md")
	if fi, err := os.Stat(dir); err == nil && !fi.IsDir() {
		indexPath = dir
		dir = filepath.Dir(dir)
	}
	indexBytes, err := os.ReadFile(indexPath)
	if err != nil {
		return fmt.Sprintf("(no committed memory mirror at %s - fresh node or scrubbed clone; nothing to orient from)\n", filepath.ToSlash(storeDir))
	}
	indexText := string(indexBytes)
	parts := []string{
		"# Fleet memory (committed mirror: " + memoryread.StoreRel + ") - read-only orientation",
		"",
		strings.TrimRight(indexText, "\n"),
	}
	if indexOnly {
		parts = append(parts, "")
		return strings.Join(parts, "\n") + "\n"
	}

	backend, _ := NewNotesBackend(dir) // a missing/partial store yields an empty corpus, never an error
	ctx := context.Background()
	cells, _ := backend.Cells(ctx)

	parts = append(parts, "", "---", "")
	budget := maxBytes
	emitted := 0
	var overflow, withheld []string
	for _, cell := range cells {
		title, fname := cell.Attrs["title"], cell.ID
		body, merr := backend.Materialize(ctx, cell.ID)
		if merr != nil {
			withheld = append(withheld, fmt.Sprintf("%s (%s): %s", title, fname, withholdReason(ctx, backend, cell, merr)))
			continue
		}
		block := fmt.Sprintf("## %s (%s)\n\n%s\n", title, fname, strings.TrimRight(string(body), "\n"))
		if budget-len(block) < 0 && emitted > 0 {
			overflow = append(overflow, fmt.Sprintf("%s (%s)", title, fname))
			continue
		}
		parts = append(parts, block)
		budget -= len(block)
		emitted++
	}
	// A fact file the index links but the store no longer carries on disk — the
	// gate never sees it as a cell at all, so count it the same way RenderDigest
	// names an unreadable entry: named, never a silent drop.
	if unreadable := len(memoryread.ParseIndex(indexText)) - len(cells); unreadable > 0 {
		parts = append(parts, fmt.Sprintf("...(%d fact file(s) unreadable and skipped)", unreadable))
	}
	if len(overflow) > 0 {
		parts = append(parts, fmt.Sprintf("%s: %d fact file(s) past the %d-byte budget: %s - read them directly from %s/",
			memoryread.OverflowReason, len(overflow), maxBytes, strings.Join(overflow, ", "), memoryread.StoreRel))
	}
	if len(withheld) > 0 {
		parts = append(parts, "withheld (never injected as fact):")
		for _, w := range withheld {
			parts = append(parts, "  - "+w)
		}
	}
	return strings.TrimRight(strings.Join(parts, "\n"), "\n") + "\n"
}

// withholdReason names the refusal evidence for a fact the digest could not page
// in — the failing claim for a stale note, or the sealed descriptor — so the
// digest footer never reads as an anonymous drop. A sealed note surfaces only as
// the same `[sealed memory note: N bytes]` descriptor the scan stamps on its
// Cell (#2429): the byte count is the whole disclosure, never a reason string
// that could grow to paraphrase the body.
func withholdReason(ctx context.Context, b *NotesBackend, cell Cell, err error) string {
	if errors.Is(err, ErrStale) {
		if findings, verr := b.Verify(ctx, cell.ID); verr == nil {
			for _, f := range findings {
				if f.Status == recall.ArtifactStale {
					return fmt.Sprintf("stale %s %q: %s", f.Claim.Kind, f.Claim.Value, f.Detail)
				}
			}
		}
		return "stale recall artifact"
	}
	if errors.Is(err, ErrSealed) {
		return fmt.Sprintf("[sealed memory note: %d bytes]", cell.Bytes)
	}
	return err.Error()
}
