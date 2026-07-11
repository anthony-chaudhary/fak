package loopmgr

import (
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestRotateSealsAndBoundsActiveFile is the rotation-seam proof for issue #3465: Rotate
// moves the hot active file into an immutable sealed segment (bounding the file Append
// locks and growthgate caps) and records a manifest seal, while appends continue into a
// fresh active segment that Load reads on its own.
func TestRotateSealsAndBoundsActiveFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	for i := 0; i < 3; i++ {
		if _, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", RunID: "a" + strconv.Itoa(i)}); err != nil {
			t.Fatalf("seed #%d: %v", i, err)
		}
	}

	res, err := Rotate(path, 0)
	if err != nil {
		t.Fatalf("Rotate: %v", err)
	}
	if !res.Rotated || res.SealedEvents != 3 || res.SealedIndex != 1 {
		t.Fatalf("rotate result = %+v, want rotated index 1 with 3 sealed events", res)
	}
	// The active file was renamed into the sealed segment — the hot file is now bounded.
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("active ledger still present after rotate: %v", err)
	}
	if fi, err := os.Stat(res.SealedPath); err != nil || fi.Size() == 0 {
		t.Fatalf("sealed segment missing/empty: %v", err)
	}

	// Post-rotation appends land in a fresh active segment (seq restarts at 1).
	for i := 0; i < 2; i++ {
		ev, err := Append(path, Event{LoopID: "l", Kind: EventAdmit, Source: "s", RunID: "b" + strconv.Itoa(i)})
		if err != nil {
			t.Fatalf("post-rotate append #%d: %v", i, err)
		}
		if ev.Seq != uint64(i+1) {
			t.Fatalf("post-rotate append seq = %d, want %d (fresh segment)", ev.Seq, i+1)
		}
	}

	active, err := Load(path)
	if err != nil {
		t.Fatalf("Load active: %v", err)
	}
	if len(active) != 2 {
		t.Fatalf("Load active = %d events, want 2 (active segment only)", len(active))
	}

	all, err := LoadAll(path)
	if err != nil {
		t.Fatalf("LoadAll: %v", err)
	}
	if len(all) != 5 {
		t.Fatalf("LoadAll = %d events, want 5 (3 sealed + 2 active)", len(all))
	}
}

func TestRotateNoOpCases(t *testing.T) {
	dir := t.TempDir()

	missing := filepath.Join(dir, "missing.jsonl")
	if res, err := Rotate(missing, 0); err != nil || res.Rotated {
		t.Fatalf("Rotate(absent) = %+v, %v; want no-op", res, err)
	}

	path := filepath.Join(dir, "loops.jsonl")
	if _, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s"}); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if res, err := Rotate(path, 1<<30); err != nil || res.Rotated {
		t.Fatalf("Rotate(under threshold) = %+v, %v; want no-op", res, err)
	}
	if segs, _ := sealedSegments(path); len(segs) != 0 {
		t.Fatalf("under-threshold rotate created %d sealed segment(s), want 0", len(segs))
	}
}

// TestLoadAllDetectsSwappedSegment proves LoadAll catches a broken seam even when the
// swapped-in segment is internally valid: replacing a sealed segment with a different
// self-verifying chain leaves its final hash mismatched against the manifest seal, and
// LoadAll must reject it.
func TestLoadAllDetectsSwappedSegment(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "loops.jsonl")
	for i := 0; i < 2; i++ {
		if _, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", RunID: "g" + strconv.Itoa(i)}); err != nil {
			t.Fatalf("seed genesis #%d: %v", i, err)
		}
	}
	res, err := Rotate(path, 0)
	if err != nil {
		t.Fatalf("Rotate: %v", err)
	}
	if _, err := Append(path, Event{LoopID: "l", Kind: EventAdmit, Source: "s", RunID: "h"}); err != nil {
		t.Fatalf("post-rotate append: %v", err)
	}
	if _, err := LoadAll(path); err != nil {
		t.Fatalf("LoadAll pre-tamper: %v", err)
	}

	// Build a different, internally-valid chain and swap it in for the sealed segment.
	other := filepath.Join(dir, "other.jsonl")
	for i := 0; i < 2; i++ {
		if _, err := Append(other, Event{LoopID: "l", Kind: EventFire, Source: "s", RunID: "x" + strconv.Itoa(i)}); err != nil {
			t.Fatalf("build other chain #%d: %v", i, err)
		}
	}
	otherBytes, err := os.ReadFile(other)
	if err != nil {
		t.Fatalf("read other: %v", err)
	}
	if err := os.WriteFile(res.SealedPath, otherBytes, 0o644); err != nil {
		t.Fatalf("overwrite sealed: %v", err)
	}
	if _, err := Load(res.SealedPath); err != nil {
		t.Fatalf("swapped sealed segment should self-verify: %v", err)
	}
	if _, err := LoadAll(path); err == nil || !strings.Contains(err.Error(), "manifest seal") {
		t.Fatalf("LoadAll err = %v, want a manifest-seal mismatch", err)
	}
}

// TestLoadAllDetectsTamperedManifest proves the manifest is itself a tamper-evident
// hash chain: editing a seal row breaks its self-hash and LoadAll refuses to trust it.
func TestLoadAllDetectsTamperedManifest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	for i := 0; i < 2; i++ {
		if _, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", RunID: strconv.Itoa(i)}); err != nil {
			t.Fatalf("seed #%d: %v", i, err)
		}
	}
	if _, err := Rotate(path, 0); err != nil {
		t.Fatalf("Rotate: %v", err)
	}
	manifest := path + manifestSuffix
	body, err := os.ReadFile(manifest)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	// Flip the recorded event count without recomputing the row hash.
	tampered := strings.Replace(string(body), `"events":2`, `"events":9`, 1)
	if tampered == string(body) {
		t.Fatalf("manifest did not contain the expected events token")
	}
	if err := os.WriteFile(manifest, []byte(tampered), 0o644); err != nil {
		t.Fatalf("write tampered manifest: %v", err)
	}
	if _, err := LoadAll(path); err == nil || !strings.Contains(err.Error(), "manifest") {
		t.Fatalf("LoadAll err = %v, want a manifest chain break", err)
	}
}

func TestLoadAllEqualsLoadWhenUnrotated(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	for i := 0; i < 4; i++ {
		if _, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", RunID: strconv.Itoa(i)}); err != nil {
			t.Fatalf("seed #%d: %v", i, err)
		}
	}
	load, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	all, err := LoadAll(path)
	if err != nil {
		t.Fatalf("LoadAll: %v", err)
	}
	if len(all) != len(load) || len(all) != 4 {
		t.Fatalf("LoadAll=%d Load=%d, want both 4", len(all), len(load))
	}
	for i := range all {
		if all[i].Hash != load[i].Hash {
			t.Fatalf("event %d hash differs between LoadAll and Load", i)
		}
	}
}

func TestRotateRefusesBrokenTail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	for i := 0; i < 2; i++ {
		if _, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", RunID: strconv.Itoa(i)}); err != nil {
			t.Fatalf("seed #%d: %v", i, err)
		}
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	tampered := strings.Replace(string(body), `"fire"`, `"admit"`, 1)
	if err := os.WriteFile(path, []byte(tampered), 0o644); err != nil {
		t.Fatalf("write tampered: %v", err)
	}
	if _, err := Rotate(path, 0); err == nil {
		t.Fatalf("Rotate sealed a ledger that does not verify; want refusal")
	}
	if segs, _ := sealedSegments(path); len(segs) != 0 {
		t.Fatalf("broken-tail rotate created %d sealed segment(s), want 0", len(segs))
	}
}

// TestRotateChainAcrossThreeSegments walks a longer history — rotate, append, rotate,
// append — and asserts the manifest chains all seals and LoadAll returns the full
// ordered history verified across every seam.
func TestRotateChainAcrossThreeSegments(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	appendN := func(tag string, n int) {
		for i := 0; i < n; i++ {
			if _, err := Append(path, Event{LoopID: "l", Kind: EventHeartbeat, Source: "s", RunID: tag + strconv.Itoa(i)}); err != nil {
				t.Fatalf("append %s#%d: %v", tag, i, err)
			}
		}
	}
	appendN("s1-", 3)
	if res, err := Rotate(path, 0); err != nil || !res.Rotated {
		t.Fatalf("rotate 1 = %+v, %v", res, err)
	}
	appendN("s2-", 2)
	if res, err := Rotate(path, 0); err != nil || !res.Rotated {
		t.Fatalf("rotate 2 = %+v, %v", res, err)
	}
	appendN("s3-", 4)

	segs, err := sealedSegments(path)
	if err != nil {
		t.Fatalf("sealedSegments: %v", err)
	}
	if len(segs) != 2 {
		t.Fatalf("sealed segments = %d, want 2", len(segs))
	}
	seals, err := loadSeals(path)
	if err != nil {
		t.Fatalf("loadSeals: %v", err)
	}
	if len(seals) != 2 || seals[0].PrevSealHash != "" || seals[1].PrevSealHash != seals[0].Hash {
		t.Fatalf("manifest chain not linked: %+v", seals)
	}
	all, err := LoadAll(path)
	if err != nil {
		t.Fatalf("LoadAll across three segments: %v", err)
	}
	if len(all) != 9 {
		t.Fatalf("LoadAll = %d events, want 9 (3+2+4)", len(all))
	}
}

// TestShouldRotate pins the cheap lock-free trigger a production auto-rotation wiring
// gates on: it reports due only once the active file has grown to the bound, and it is a
// pure stat (no lock, no chain read), so keeping it off the append hot path adds none of
// the O(N) cost issue #3465 raised about the growing ledger.
func TestShouldRotate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")

	// Absent file: not due, size 0, no error.
	if due, sz, err := ShouldRotate(path, 1); err != nil || due || sz != 0 {
		t.Fatalf("ShouldRotate(absent) = %v,%d,%v; want false,0,nil", due, sz, err)
	}

	if _, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", RunID: "0"}); err != nil {
		t.Fatalf("seed: %v", err)
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	size := fi.Size()

	// minBytes<=0 means "no bound configured" -> never auto-due.
	if due, _, err := ShouldRotate(path, 0); err != nil || due {
		t.Fatalf("ShouldRotate(minBytes=0) = %v,%v; want false,nil", due, err)
	}
	// Under the bound: not due, but the real size is still reported.
	if due, sz, err := ShouldRotate(path, size+1); err != nil || due || sz != size {
		t.Fatalf("ShouldRotate(under) = %v,%d,%v; want false,%d,nil", due, sz, err, size)
	}
	// At the bound: due (>= is the boundary).
	if due, sz, err := ShouldRotate(path, size); err != nil || !due || sz != size {
		t.Fatalf("ShouldRotate(at) = %v,%d,%v; want true,%d,nil", due, sz, err, size)
	}
}

// TestRotateBoundsActiveFileBelowThreshold is the on-disk-bound proof for issue #3465:
// once the active ledger crosses the configured size bound, ShouldRotate flags it and
// Rotate seals it away, dropping the hot active file back below the bound while LoadAll
// still returns the full verified history across the seam. This is the loop that keeps
// the file Append flock-serializes and growthgate size-caps (ClassLoops warns at 8 MiB)
// bounded no matter how much total history accumulates. The 512-byte bound here is a
// scale-invariant stand-in for that production threshold so the test stays fast.
func TestRotateBoundsActiveFileBelowThreshold(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	const bound = 512

	n := 0
	for {
		if _, err := Append(path, Event{LoopID: "l", Kind: EventHeartbeat, Source: "s", RunID: strconv.Itoa(n)}); err != nil {
			t.Fatalf("seed #%d: %v", n, err)
		}
		n++
		due, sz, err := ShouldRotate(path, bound)
		if err != nil {
			t.Fatalf("ShouldRotate: %v", err)
		}
		if due {
			if sz < bound {
				t.Fatalf("flagged due at size %d < bound %d", sz, bound)
			}
			break
		}
		if n > 10000 {
			t.Fatalf("active file never crossed %d bytes after %d appends", bound, n)
		}
	}

	res, err := Rotate(path, bound)
	if err != nil {
		t.Fatalf("Rotate: %v", err)
	}
	if !res.Rotated || res.SealedEvents != n {
		t.Fatalf("rotate = %+v, want rotated with %d sealed events", res, n)
	}

	// The hot active file is now bounded: it was renamed into the sealed segment, so a
	// fresh (empty) one starts on the next append and ShouldRotate goes quiet.
	if due, sz, err := ShouldRotate(path, bound); err != nil || due {
		t.Fatalf("post-rotate ShouldRotate = %v,%d,%v; want not-due (bounded)", due, sz, err)
	}

	// No history was lost to the bound: LoadAll still returns all n events, verified
	// across the seal.
	all, err := LoadAll(path)
	if err != nil {
		t.Fatalf("LoadAll: %v", err)
	}
	if len(all) != n {
		t.Fatalf("LoadAll = %d events, want %d (full history across the seal)", len(all), n)
	}
}

// TestRotateIfDue pins the lock-cheap activation primitive for issue #3465: under the
// bound it is a pure no-op that seals nothing (so a per-tick sweep never takes the append
// lock on the common pass that would re-strain the hot path the issue flags), and once
// the active file crosses the bound it delegates to Rotate and seals the segment while
// LoadAll still returns the full verified history across the seam.
func TestRotateIfDue(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")

	// Absent file: not due, no-op, no error.
	if res, err := RotateIfDue(path, 1); err != nil || res.Rotated {
		t.Fatalf("RotateIfDue(absent) = %+v, %v; want no-op", res, err)
	}

	for i := 0; i < 3; i++ {
		if _, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", RunID: strconv.Itoa(i)}); err != nil {
			t.Fatalf("seed #%d: %v", i, err)
		}
	}

	// Under the bound: no-op, and crucially nothing is sealed.
	if res, err := RotateIfDue(path, 1<<30); err != nil || res.Rotated {
		t.Fatalf("RotateIfDue(under) = %+v, %v; want no-op", res, err)
	}
	if segs, _ := sealedSegments(path); len(segs) != 0 {
		t.Fatalf("under-bound RotateIfDue sealed %d segment(s), want 0", len(segs))
	}

	// minBytes<=0 ("no bound configured"): never seals.
	if res, err := RotateIfDue(path, 0); err != nil || res.Rotated {
		t.Fatalf("RotateIfDue(minBytes=0) = %+v, %v; want no-op", res, err)
	}

	// At/over the bound: due -> delegates to Rotate and seals the active segment.
	res, err := RotateIfDue(path, 1)
	if err != nil {
		t.Fatalf("RotateIfDue(due): %v", err)
	}
	if !res.Rotated || res.SealedEvents != 3 {
		t.Fatalf("RotateIfDue(due) = %+v, want rotated with 3 sealed events", res)
	}
	if segs, _ := sealedSegments(path); len(segs) != 1 {
		t.Fatalf("due RotateIfDue sealed %d segment(s), want 1", len(segs))
	}
	all, err := LoadAll(path)
	if err != nil {
		t.Fatalf("LoadAll: %v", err)
	}
	if len(all) != 3 {
		t.Fatalf("LoadAll = %d events, want 3 (full history across the seal)", len(all))
	}
}
