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

func TestCleanScratchProducerUnlinksNestedLinksWithoutTraversingTargets(t *testing.T) {
	repo := t.TempDir()
	target := filepath.Join(repo, "_scratch", "selected")
	nested := filepath.Join(target, "nested")
	mustWrite(t, filepath.Join(target, "root.txt"), "producer root\n")
	mustWrite(t, filepath.Join(nested, "local.txt"), "producer nested\n")

	outside := t.TempDir()
	externalFile := filepath.Join(outside, "external.bin")
	externalNestedFile := filepath.Join(outside, "directory-target", "deeper", "sentinel.bin")
	mustWrite(t, externalFile, "external bytes\x00\x01\xff")
	mustWrite(t, externalNestedFile, "nested external bytes\x00\xfe")
	externalFileBefore, err := os.ReadFile(externalFile)
	if err != nil {
		t.Fatal(err)
	}
	externalNestedBefore, err := os.ReadFile(externalNestedFile)
	if err != nil {
		t.Fatal(err)
	}

	if err := os.Symlink(externalFile, filepath.Join(target, "external-file-link")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if err := os.Symlink(filepath.Join(outside, "directory-target"), filepath.Join(nested, "external-directory-link")); err != nil {
		t.Skipf("directory symlinks unavailable: %v", err)
	}

	receipt, err := CleanScratchProducer(repo, "selected")
	if err != nil {
		t.Fatalf("CleanScratchProducer: %v", err)
	}
	if receipt.Verdict != ScratchProducerReaped || receipt.RemovedCount != 6 {
		t.Fatalf("receipt = %+v, want reaped with 6 removed entries", receipt)
	}
	if _, err := os.Lstat(target); !os.IsNotExist(err) {
		t.Fatalf("selected producer survived: %v", err)
	}
	assertFileBytes(t, externalFile, externalFileBefore)
	assertFileBytes(t, externalNestedFile, externalNestedBefore)
}

func TestCleanScratchProducerRefusesProducerRootLinkBeforeDeleting(t *testing.T) {
	repo := t.TempDir()
	outside := t.TempDir()
	externalFile := filepath.Join(outside, "outside.txt")
	mustWrite(t, externalFile, "outside bytes\n")
	before, err := os.ReadFile(externalFile)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(repo, "_scratch"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(repo, "_scratch", "selected")); err != nil {
		t.Skipf("directory symlinks unavailable: %v", err)
	}

	receipt, err := CleanScratchProducer(repo, "selected")
	if !errors.Is(err, ErrUnsafeScratchProducer) {
		t.Fatalf("error = %v, want ErrUnsafeScratchProducer", err)
	}
	if receipt.Verdict != ScratchProducerRefused || receipt.RemovedCount != 0 {
		t.Fatalf("receipt = %+v, want refused before deletion", receipt)
	}
	if _, err := os.Lstat(filepath.Join(repo, "_scratch", "selected")); err != nil {
		t.Fatalf("refusal removed producer root link: %v", err)
	}
	assertFileBytes(t, externalFile, before)
}

func assertFileBytes(t *testing.T, path string, want []byte) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if string(got) != string(want) {
		t.Fatalf("%s changed: before=%q after=%q", path, want, got)
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
