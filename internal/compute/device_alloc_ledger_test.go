package compute

import (
	"context"
	_ "embed"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

//go:embed testdata/cuda_alloc_ledger_test.cpp
var cudaAllocLedgerHarness []byte

func allocLedgerCell(s DeviceAllocSnapshot, class DeviceAllocClass, kind DeviceAllocKind) DeviceAllocCell {
	for _, c := range s.Cells {
		if c.Class == class && c.Kind == kind {
			return c
		}
	}
	return DeviceAllocCell{Class: class, Kind: kind}
}

// runAllocLedgerHarness compiles cuda/alloc_ledger.h against a host-malloc
// fake allocator and returns each labelled raw read.
func runAllocLedgerHarness(t *testing.T) map[int]DeviceAllocSnapshot {
	t.Helper()
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	dir := t.TempDir()
	for name, data := range map[string][]byte{"alloc_ledger.h": cudaAllocLedgerHeader, "ledger_test.cpp": cudaAllocLedgerHarness} {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	binary := filepath.Join(dir, "alloc-ledger")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	args := []string{"-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread", "-I", dir, filepath.Join(dir, "ledger_test.cpp"), "-o", binary}
	if out, err := exec.CommandContext(ctx, compiler, args...).CombinedOutput(); err != nil {
		t.Fatalf("compile alloc ledger harness: %v\n%s", err, out)
	}
	out, err := exec.CommandContext(ctx, binary).CombinedOutput()
	if err != nil {
		t.Fatalf("alloc ledger harness: %v\n%s", err, out)
	}
	reads := map[int]DeviceAllocSnapshot{}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		f := strings.Fields(line)
		if len(f) != 3+DeviceAllocCells*5 || f[0] != "read" {
			t.Fatalf("harness line has %d fields, want %d", len(f), 3+DeviceAllocCells*5)
		}
		vals := make([]int64, 0, len(f)-1)
		for _, s := range f[1:] {
			v, err := strconv.ParseInt(s, 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			vals = append(vals, v)
		}
		reads[int(vals[0])] = BuildDeviceAllocSnapshot(vals[2:], vals[1], nil, 10000, 8000, true)
	}
	return reads
}

// fak-test:runtime medium est=4s lane=default
func TestDeviceAllocLedgerFakeAllocatorContract(t *testing.T) {
	r := runAllocLedgerHarness(t)
	// One snapshot per labelled read the fixture emits: count the print_read(n) call sites, not the definition.
	wantReads := len(regexp.MustCompile(`print_read\(\d+\)`).FindAllString(string(cudaAllocLedgerHarness), -1))
	if len(r) != wantReads {
		t.Fatalf("reads=%d, want %d", len(r), wantReads)
	}

	one := r[1]
	if got := allocLedgerCell(one, AllocOther, AllocDeviceLocal); got.LiveBytes != 96 || got.LiveCount != 2 {
		t.Fatalf("untagged allocations: other live=%d count=%d, want 96/2", got.LiveBytes, got.LiveCount)
	}
	if one.DeviceLocalLiveBytes != 1596 || one.ManagedLiveBytes != 300 || one.LiveBytes != 1896 || one.LiveCount != 5 {
		t.Fatalf("read1 sums device=%d managed=%d live=%d count=%d", one.DeviceLocalLiveBytes, one.ManagedLiveBytes, one.LiveBytes, one.LiveCount)
	}
	if one.UsedBytes != 2000 || one.UnaccountedBytes != 2000-1596 {
		t.Fatalf("read1 used=%d unaccounted=%d", one.UsedBytes, one.UnaccountedBytes)
	}

	two := r[2]
	w := allocLedgerCell(two, AllocWeights, AllocDeviceLocal)
	if w.LiveBytes != 0 || w.PeakBytes != 1000 || w.Frees != 1 {
		t.Fatalf("free must decrement and keep peak: %+v", w)
	}
	if idle := allocLedgerCell(two, AllocIdlePool, AllocDeviceLocal); idle.LiveBytes != 500 || allocLedgerCell(two, AllocKVCache, AllocDeviceLocal).LiveBytes != 0 {
		t.Fatalf("pool return must move bytes kv->idle_pool: idle=%d", idle.LiveBytes)
	}
	if two.UntrackedFrees != 1 || two.LiveBytes != 896 {
		t.Fatalf("read2 untracked=%d live=%d", two.UntrackedFrees, two.LiveBytes)
	}

	three := r[3]
	w3 := allocLedgerCell(three, AllocWeights, AllocDeviceLocal)
	if w3.LiveBytes != 200 || w3.LiveCount != 1 || w3.PeakBytes != 1000 {
		t.Fatalf("re-track double-counted or peak regressed: %+v", w3)
	}

	four := r[4]
	act := allocLedgerCell(four, AllocActivationsScratch, AllocDeviceLocal)
	if act.LiveBytes != 0 || act.Allocs != 8000 || act.Frees != 8000 || act.PeakBytes < 16 || act.PeakBytes > 64 {
		t.Fatalf("concurrent pairs must conserve: %+v", act)
	}
	if four.LiveBytes != three.LiveBytes {
		t.Fatalf("concurrent traffic changed live sum %d -> %d", three.LiveBytes, four.LiveBytes)
	}

	five := r[5]
	if five.LiveBytes != 0 || five.LiveCount != 0 {
		t.Fatalf("all frees must return live to zero: live=%d count=%d", five.LiveBytes, five.LiveCount)
	}
	for i := 1; i < 5; i++ {
		for _, before := range r[i].Cells {
			if after := allocLedgerCell(r[i+1], before.Class, before.Kind); after.PeakBytes < before.PeakBytes {
				t.Fatalf("peak %s/%s fell %d -> %d", before.ClassName, before.KindName, before.PeakBytes, after.PeakBytes)
			}
		}
	}
}

// fak-test:runtime fast est=1s
func TestDeviceAllocLedgerClassABIMatchesHeader(t *testing.T) {
	header := string(cudaAllocLedgerHeader)
	for i := DeviceAllocClass(0); i < deviceAllocClassCount; i++ {
		want := "FAK_ALLOC_" + strings.ToUpper(i.String()) + " = " + strconv.Itoa(int(i)) + ","
		if !strings.Contains(header, want) {
			t.Fatalf("header lacks %q", want)
		}
	}
	for k := DeviceAllocKind(0); k < deviceAllocKindCount; k++ {
		want := "FAK_ALLOC_" + strings.ToUpper(k.String()) + " = " + strconv.Itoa(int(k)) + ","
		if !strings.Contains(header, want) {
			t.Fatalf("header lacks %q", want)
		}
	}
	counts := regexp.MustCompile(`FAK_ALLOC_(CLASS|KIND)_COUNT = (\d+)`).FindAllStringSubmatch(header, -1)
	if len(counts) != 2 || counts[0][2] != strconv.Itoa(int(deviceAllocClassCount)) || counts[1][2] != strconv.Itoa(int(deviceAllocKindCount)) {
		t.Fatalf("class/kind counts drifted: %v", counts)
	}
}

// fak-test:runtime fast est=1s
func TestDeviceAllocClassForMapsBackendClasses(t *testing.T) {
	for _, tc := range []struct {
		class MemoryClass
		site  string
		panel bool
		want  DeviceAllocClass
	}{
		{MemoryUnknown, "dalloc", false, AllocOther},
		{MemoryClass("novel"), "x", false, AllocOther},
		{MemoryWeights, "dallocWeight", false, AllocWeights},
		{MemoryOffload, "dallocManaged", false, AllocWeights},
		{MemoryWeights, "q4k-dequant-shadow", false, AllocWeightsDequantShadow},
		{MemoryKVCache, "kv-cache", false, AllocKVCache},
		{MemoryKVCache, "qwen35-recurrent", false, AllocRecurrentState},
		{MemoryScratchpad, "transient", false, AllocActivationsScratch},
		{MemoryScratchpad, "transient", true, AllocPrefillScratch},
		{MemoryScratchpad, "qwen35-sequence-q", false, AllocPrefillScratch},
		{MemoryActivation, "V4.1 shared attention output", false, AllocAttentionScratch},
		{MemoryActivation, "V4.1 router gate logits", false, AllocSampling},
	} {
		if got := DeviceAllocClassFor(tc.class, tc.site, tc.panel); got != tc.want {
			t.Fatalf("%s/%q panel=%v -> %s, want %s", tc.class, tc.site, tc.panel, got, tc.want)
		}
	}
	if DeviceAllocClass(-1).String() != "other" || DeviceAllocClass(99).String() != "other" {
		t.Fatal("out-of-range class must read as other")
	}
}

// fak-test:runtime fast est=1s
func TestBuildDeviceAllocSnapshotUnknownMemInfo(t *testing.T) {
	raw := make([]int64, DeviceAllocCells*5)
	raw[int(AllocKVCache)*int(deviceAllocKindCount)*5] = 4096
	raw[int(AllocKVCache)*int(deviceAllocKindCount)*5+3] = 1
	s := BuildDeviceAllocSnapshot(raw, 0, []DeviceAllocSite{{Site: "small", LiveBytes: 1, Allocs: 1}, {Site: "big", LiveBytes: 9, Allocs: 1}, {Site: "idle"}}, 0, FreeUnknown, false)
	if s.UnaccountedBytes != 0 || s.UsedBytes != 0 || s.ClassLiveBytes(AllocKVCache) != 4096 || len(s.Cells) != 1 {
		t.Fatalf("unknown mem info snapshot: %+v", s)
	}
	if len(s.Sites) != 2 || s.Sites[0].Site != "big" {
		t.Fatalf("sites must drop idle and sort by live bytes: %+v", s.Sites)
	}
}
