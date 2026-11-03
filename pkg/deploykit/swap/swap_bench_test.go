package swap

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/deploykit"
)

// benchPayloadSize is the payload every swap benchmark moves: a few KB, so each iteration
// stays cheap even on a busy shared trunk.
const benchPayloadSize = 4 << 10

// benchPayload returns benchPayloadSize deterministic bytes whose last byte carries seed, so
// successive calls give distinct content-addressed digests.
func benchPayload(seed byte) []byte {
	p := make([]byte, benchPayloadSize)
	for i := range p {
		p[i] = byte('a' + i%26)
	}
	p[len(p)-1] = seed
	return p
}

// benchWrite writes data to path, creating parent directories.
func benchWrite(b *testing.B, path string, data []byte) {
	b.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		b.Fatal(err)
	}
}

// BenchmarkPriorSlotPath measures the pure string math of the content-addressed rollback slot
// name, for an .exe target and a plain target.
func BenchmarkPriorSlotPath(b *testing.B) {
	sum := sha256.Sum256(benchPayload(0))
	sha12 := fmt.Sprintf("%x", sum)[:12]
	exe := filepath.Join("install", "router.exe")
	plain := filepath.Join("install", "fak-server")
	if got := PriorSlotPath(exe, sum); got != filepath.Join("install", "router.deploy-prior-"+sha12+".exe") {
		b.Fatalf("fixture PriorSlotPath(.exe) = %s, want install/router.deploy-prior-%s.exe", got, sha12)
	}
	if got := PriorSlotPath(plain, sum); got != filepath.Join("install", "fak-server.deploy-prior-"+sha12) {
		b.Fatalf("fixture PriorSlotPath(plain) = %s, want install/fak-server.deploy-prior-%s", got, sha12)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = PriorSlotPath(exe, sum)
		_ = PriorSlotPath(plain, sum)
	}
}

// BenchmarkSavePrior measures SavePrior's full stage-hash-sync-swap path: each iteration gives
// the target fresh bytes so the slot is always written rather than reused, and prunes the tree
// back down every 16 saves so the directory stays bounded.
func BenchmarkSavePrior(b *testing.B) {
	dir := b.TempDir()
	target := filepath.Join(dir, "app")
	payload := benchPayload(0)
	benchWrite(b, target, payload)
	slot, err := SavePrior(target)
	if err != nil {
		b.Fatalf("fixture SavePrior: %v", err)
	}
	if want := PriorSlotPath(target, sha256.Sum256(payload)); slot != want {
		b.Fatalf("fixture SavePrior slot = %s, want %s", slot, want)
	}
	if got, err := os.ReadFile(slot); err != nil || !bytes.Equal(got, payload) {
		b.Fatalf("fixture slot bytes mismatch: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		body := benchPayload(byte(i))
		if err := os.WriteFile(target, body, 0o644); err != nil {
			b.Fatal(err)
		}
		if _, err := SavePrior(target); err != nil {
			b.Fatal(err)
		}
		if i%16 == 15 {
			if _, err := PruneSlots(target, deploykit.RollbackPolicy{KeepSlots: 4}); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// benchSlotTree writes n content-addressed prior slots of target directly, bypassing
// SavePrior's staging, with distinct deterministic modification times: slots[0] is oldest and
// slots[n-1] newest.
func benchSlotTree(b *testing.B, dir, target string, n int) []string {
	b.Helper()
	base := time.Now().Add(-time.Hour)
	var slots []string
	for i := 0; i < n; i++ {
		body := benchPayload(byte(i))
		path := PriorSlotPath(target, sha256.Sum256(body))
		benchWrite(b, path, body)
		at := base.Add(time.Duration(i) * time.Minute)
		if err := os.Chtimes(path, at, at); err != nil {
			b.Fatal(err)
		}
		slots = append(slots, path)
	}
	return slots
}

// BenchmarkPriorSlots measures listing and newest-first sorting of a target's prior slots over
// a tree of eight slots with distinct modification times.
func BenchmarkPriorSlots(b *testing.B) {
	dir := b.TempDir()
	target := filepath.Join(dir, "app")
	benchWrite(b, target, benchPayload(0xff)) // the live target is never listed as a slot
	slots := benchSlotTree(b, dir, target, 8)
	got := PriorSlots(target)
	if len(got) != 8 {
		b.Fatalf("fixture PriorSlots = %d slots, want 8", len(got))
	}
	for i, s := range got {
		if s.Path != slots[7-i] || s.Kind != KindPrior {
			b.Fatalf("fixture PriorSlots[%d] = %+v, want newest-first %s", i, s, slots[7-i])
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = PriorSlots(target)
	}
}

// BenchmarkPruneSlots measures pruning a tree of four prior slots down to two. PruneSlots
// removes what it reports, so the tree is rebuilt each iteration; the rebuild is included in
// the measurement.
func BenchmarkPruneSlots(b *testing.B) {
	dir := b.TempDir()
	target := filepath.Join(dir, "app")
	benchWrite(b, target, benchPayload(0xff))
	slots := benchSlotTree(b, dir, target, 4)
	// PruneSlots walks the newest-first listing beyond keep, so it removes the second-oldest
	// before the oldest.
	removed, err := PruneSlots(target, deploykit.RollbackPolicy{KeepSlots: 2})
	if err != nil || len(removed) != 2 || removed[0] != slots[1] || removed[1] != slots[0] {
		b.Fatalf("fixture PruneSlots = (%v, %v), want the two oldest %v", removed, err, []string{slots[1], slots[0]})
	}
	if left := PriorSlots(target); len(left) != 2 || left[0].Path != slots[3] {
		b.Fatalf("fixture leftover slots = %v, want the two newest kept", left)
	}
	rebuild := func() {
		for _, p := range slots {
			// The fixture prune already removed two of the four.
			if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
				b.Fatal(err)
			}
		}
		benchSlotTree(b, dir, target, 4)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rebuild()
		if _, err := PruneSlots(target, deploykit.RollbackPolicy{KeepSlots: 2}); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSwapSmallFiles measures one atomic swap of a few-KB candidate over an installed
// file: each iteration writes fresh source and target bytes, swaps, and checks the result.
func BenchmarkSwapSmallFiles(b *testing.B) {
	dir := b.TempDir()
	src, dst := filepath.Join(dir, "candidate"), filepath.Join(dir, "app")
	first := benchPayload(0)
	benchWrite(b, src, first)
	benchWrite(b, dst, benchPayload(0xff))
	if err := Swap(src, dst); err != nil {
		b.Fatalf("fixture Swap: %v", err)
	}
	if got, err := os.ReadFile(dst); err != nil || !bytes.Equal(got, first) {
		b.Fatalf("fixture target after Swap: %v", err)
	}
	if _, err := os.Stat(src); !errors.Is(err, os.ErrNotExist) {
		b.Fatalf("fixture Swap must consume src: stat err = %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		body := benchPayload(byte(i))
		if err := os.WriteFile(src, body, 0o644); err != nil {
			b.Fatal(err)
		}
		if err := os.WriteFile(dst, benchPayload(byte(i+1)), 0o644); err != nil {
			b.Fatal(err)
		}
		if err := Swap(src, dst); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTransactionSmallFiles measures an all-or-nothing activation of three small targets:
// each iteration restores the pre-state bytes and activates the transaction, asserting the
// Updated result.
func BenchmarkTransactionSmallFiles(b *testing.B) {
	root := b.TempDir()
	names := []string{"a", "b", "c"}
	var copies []Copy
	var targets []string
	for _, name := range names {
		target := filepath.Join(root, "install", name)
		source := filepath.Join(root, "candidates", name)
		benchWrite(b, target, []byte("old-"+name))
		benchWrite(b, source, benchPayload(byte(name[0])))
		copies = append(copies, Copy{Source: source, Target: target})
		targets = append(targets, target)
	}
	res := Transaction(copies)
	u, ok := res.(Updated)
	if !ok || u.Changed != 3 || u.Err != nil {
		b.Fatalf("fixture Transaction = %#v, want Updated with 3 changed", res)
	}
	for i, target := range targets {
		if got, err := os.ReadFile(target); err != nil || !bytes.Equal(got, benchPayload(byte(names[i][0]))) {
			b.Fatalf("fixture %s after Transaction: %v", names[i], err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j, target := range targets {
			if err := os.WriteFile(target, []byte("old-"+names[j]), 0o644); err != nil {
				b.Fatal(err)
			}
		}
		res := Transaction(copies)
		u, ok := res.(Updated)
		if !ok || u.Changed != 3 || u.Err != nil {
			b.Fatalf("iteration %d: Transaction = %#v, want Updated with 3 changed", i, res)
		}
	}
}

// BenchmarkLockAcquireRelease measures the single-flight lock's acquire and release over a
// temp-dir lock file.
func BenchmarkLockAcquireRelease(b *testing.B) {
	dir := b.TempDir()
	release, err := Lock(dir, "bench")
	if err != nil {
		b.Fatalf("fixture Lock: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "bench.deploy.lock")); err != nil {
		b.Fatalf("fixture lock file: %v", err)
	}
	if _, err := Lock(dir, "bench"); !errors.Is(err, ErrBusy) {
		b.Fatalf("fixture second Lock err = %v, want ErrBusy", err)
	}
	release()
	again, err := Lock(dir, "bench")
	if err != nil {
		b.Fatalf("fixture Lock after release: %v", err)
	}
	again()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		release, err := Lock(dir, "bench")
		if err != nil {
			b.Fatal(err)
		}
		release()
	}
}
