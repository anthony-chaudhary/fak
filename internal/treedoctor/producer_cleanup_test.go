package treedoctor

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestCleanScratchProducerPreservesIgnoredSiblingByteForByte(t *testing.T) {
	repo := t.TempDir()
	target := filepath.Join(repo, "_scratch", "selected")
	sibling := filepath.Join(repo, "_scratch", "peer", "state.bin")
	mustWrite(t, filepath.Join(target, "root.txt"), "selected root\n")
	mustWrite(t, filepath.Join(target, "nested", "data.bin"), "selected nested\x00\xff")
	mustWrite(t, sibling, "peer bytes\x00\x01\xff")

	before, err := os.ReadFile(sibling)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := CleanScratchProducer(repo, "selected")
	if err != nil {
		t.Fatalf("CleanScratchProducer: %v", err)
	}
	if receipt.Verdict != ScratchProducerReaped || receipt.RemovedCount != 4 {
		t.Fatalf("receipt = %+v, want reaped with 4 removed entries", receipt)
	}
	if receipt.ResolvedTarget != target {
		t.Fatalf("resolved target = %q, want %q", receipt.ResolvedTarget, target)
	}
	if _, err := os.Stat(target); !os.IsNotExist(err) {
		t.Fatalf("selected producer survived: %v", err)
	}
	after, err := os.ReadFile(sibling)
	if err != nil {
		t.Fatalf("peer producer was removed: %v", err)
	}
	if string(after) != string(before) {
		t.Fatalf("peer producer changed: before=%q after=%q", before, after)
	}
}

func TestCleanScratchProducerRefusesUnsafeTargets(t *testing.T) {
	repo := t.TempDir()
	for _, producer := range []string{
		"",
		".",
		"..",
		"../outside",
		`..\outside`,
		"_scratch",
		"_scratch/selected",
		"one/two",
		`one\two`,
		"selected*",
		"selected?",
		"[selected]",
		filepath.Join(repo, "_scratch", "selected"),
	} {
		t.Run(producer, func(t *testing.T) {
			receipt, err := CleanScratchProducer(repo, producer)
			if !errors.Is(err, ErrUnsafeScratchProducer) {
				t.Fatalf("error = %v, want ErrUnsafeScratchProducer", err)
			}
			if receipt.Verdict != ScratchProducerRefused || receipt.RemovedCount != 0 {
				t.Fatalf("receipt = %+v, want refused with zero removals", receipt)
			}
		})
	}
}

func TestCleanScratchProducerRefusesReparseEscapeBeforeDeleting(t *testing.T) {
	repo := t.TempDir()
	target := filepath.Join(repo, "_scratch", "selected")
	safe := filepath.Join(target, "safe.txt")
	outside := t.TempDir()
	outsideFile := filepath.Join(outside, "outside.txt")
	mustWrite(t, safe, "keep on refusal\n")
	mustWrite(t, outsideFile, "outside bytes\n")
	if err := os.Symlink(outside, filepath.Join(target, "escape")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	receipt, err := CleanScratchProducer(repo, "selected")
	if !errors.Is(err, ErrUnsafeScratchProducer) {
		t.Fatalf("error = %v, want ErrUnsafeScratchProducer", err)
	}
	if receipt.Verdict != ScratchProducerRefused || receipt.RemovedCount != 0 {
		t.Fatalf("receipt = %+v, want refused before deletion", receipt)
	}
	for _, path := range []string{safe, outsideFile} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("refusal removed %s: %v", path, err)
		}
	}
}

func TestCleanScratchProducerMissingDirectoryIsIdempotent(t *testing.T) {
	repo := t.TempDir()
	receipt, err := CleanScratchProducer(repo, "absent")
	if err != nil {
		t.Fatalf("CleanScratchProducer: %v", err)
	}
	if receipt.Verdict != ScratchProducerAbsent || receipt.RemovedCount != 0 {
		t.Fatalf("receipt = %+v, want absent with zero removals", receipt)
	}
	if receipt.ResolvedTarget != filepath.Join(repo, "_scratch", "absent") {
		t.Fatalf("resolved target = %q", receipt.ResolvedTarget)
	}
}
