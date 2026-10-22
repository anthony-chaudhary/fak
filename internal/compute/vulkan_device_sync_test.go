//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"os"
	"testing"
	"time"
)

type vulkanSyncCellKey struct{ site, stage string }

func vulkanSyncCells(t *testing.T, before, after DeviceWork) map[vulkanSyncCellKey]DeviceSyncSite {
	t.Helper()
	if !before.SyncObserved || !after.SyncObserved {
		t.Fatalf("SyncObserved before=%v after=%v, want both true", before.SyncObserved, after.SyncObserved)
	}
	delta, ok := DeviceSyncSitesDelta(before.SyncSites, after.SyncSites)
	if !ok {
		t.Fatal("sync cells went backwards between two reads")
	}
	out := make(map[vulkanSyncCellKey]DeviceSyncSite, len(delta))
	for _, d := range delta {
		out[vulkanSyncCellKey{d.Site, d.Stage}] = d
	}
	return out
}

// goClockSlack bounds how far Go's monotonic wall can trail the shim's
// steady_clock across one interval: two Go clock ticks (~0.5ms each on Windows,
// nanoseconds on Linux).
func goClockSlack() time.Duration {
	var tick time.Duration
	for range 3 {
		a := time.Now()
		b := time.Now()
		for b.Sub(a) == 0 {
			b = time.Now()
		}
		tick = max(tick, b.Sub(a))
	}
	return 2 * tick
}

// vulkanPipelinedCycles mirrors batchCycle's eligibility for a plain dense
// window: the process-start env gates, not a runtime toggle.
func vulkanPipelinedCycles() bool {
	return os.Getenv("FAK_VULKAN_PIPELINE_FLUSH") != "0" &&
		os.Getenv("FAK_VULKAN_TIMESTAMP_PROFILE") != "1" &&
		!VulkanHazardLoweringArmed(os.Getenv)
}

// A fixed submit/readback pattern books exactly one sync per host wait, to the
// entry point that waited and the kernel it waited on, with the readback bytes
// on the readback cell; the idle clock grows by at least the injected host idle
// and never by more than wall.
// fak-test:runtime fast est=1s lane=default
func TestVulkanHostSyncSitesCountExactly(t *testing.T) {
	v := vk(t)
	const n = 1024
	c := cpu()
	host := make([]float32, n)
	for i := range host {
		host[i] = float32(i%7) - 3
	}
	x := v.Upload(NewF32(c, []int{n}, host), F32)
	y := v.Upload(NewF32(c, []int{n}, host), F32)
	defer v.Free(x)
	defer v.Free(y)
	_ = v.Read(y)

	const idle = 20 * time.Millisecond
	wallStart := time.Now()
	before := readVulkanDeviceWork()

	v.AddInPlace(y, x)
	time.Sleep(idle)
	_ = v.Read(y)

	v.BeginBatch()
	for range 3 {
		v.AddInPlace(y, x)
	}
	v.FlushBatch()
	_ = v.Read(y)

	v.BeginBatch()
	v.AddInPlace(y, x)
	v.CycleBatch()
	v.AddInPlace(y, x)
	v.CycleBatch()
	v.AddInPlace(y, x)
	v.FlushBatch()
	got := v.Read(y)

	directBefore := VulkanDirectWeightUploadBytes()
	z := v.Upload(NewF32(c, []int{n}, host), F32)
	defer v.Free(z)
	staged := VulkanDirectWeightUploadBytes()-directBefore != n*4

	after := readVulkanDeviceWork()
	wall := time.Since(wallStart)

	want := map[vulkanSyncCellKey]uint64{
		{SyncSiteSubmitWait, "add"}:       1,
		{SyncSiteReadback, SyncStageNone}: 3,
	}
	if vulkanPipelinedCycles() {
		want[vulkanSyncCellKey{SyncSiteBatchFlush, "add"}] = 2
		want[vulkanSyncCellKey{SyncSitePipelineDrain, "add"}] = 2
	} else {
		want[vulkanSyncCellKey{SyncSiteBatchFlush, "add"}] = 4
	}
	if staged {
		want[vulkanSyncCellKey{SyncSiteUpload, SyncStageNone}] = 1
	}
	t.Logf("pipelined cycles=%v staged upload=%v", vulkanPipelinedCycles(), staged)
	cells := vulkanSyncCells(t, before, after)
	if len(cells) != len(want) {
		t.Fatalf("sync cells = %+v, want exactly %v", cells, want)
	}
	for key, syncs := range want {
		cell, ok := cells[key]
		if !ok || cell.Syncs != syncs || cell.WaitNS == 0 {
			t.Fatalf("cell %v = %+v (present %v), want %d syncs with nonzero wait", key, cell, ok, syncs)
		}
		wantD2H := uint64(0)
		if key.site == SyncSiteReadback {
			wantD2H = 3 * n * 4
		}
		if cell.D2HBytes != wantD2H {
			t.Fatalf("cell %v D2HBytes = %d, want %d", key, cell.D2HBytes, wantD2H)
		}
	}
	gotSyncs, gotWait := after.HostSyncs()
	baseSyncs, baseWait := before.HostSyncs()
	var wantSyncs uint64
	for _, s := range want {
		wantSyncs += s
	}
	if gotSyncs-baseSyncs != wantSyncs || gotWait <= baseWait {
		t.Fatalf("HostSyncs delta = %d (wait %d->%d), want %d", gotSyncs-baseSyncs, baseWait, gotWait, wantSyncs)
	}

	idleDelta := time.Duration(after.DeviceIdleNS - before.DeviceIdleNS)
	if slack := goClockSlack(); after.DeviceIdleNS < before.DeviceIdleNS || idleDelta < idle-slack || idleDelta > wall+slack {
		t.Fatalf("device idle delta = %v over wall %v (+%v clock slack), want at least %v", idleDelta, wall, slack, idle)
	}
	var wantPipelined uint64
	if vulkanPipelinedCycles() {
		wantPipelined = 2
	}
	if after.PipelinedSubmits-before.PipelinedSubmits != wantPipelined || after.PipelinedDrains-before.PipelinedDrains != wantPipelined ||
		after.PipelinedSubmits != after.PipelinedDrains {
		t.Fatalf("pipelined submits %d->%d drains %d->%d, want %d each and none outstanding",
			before.PipelinedSubmits, after.PipelinedSubmits, before.PipelinedDrains, after.PipelinedDrains, wantPipelined)
	}
	if _, ok := DeviceIdleDelta(before, after); ok != (wantPipelined == 0) {
		t.Fatalf("DeviceIdleDelta ok=%v over a window with %d pipelined batches", ok, wantPipelined)
	}
	if len(got) != n {
		t.Fatalf("read %d values, want %d", len(got), n)
	}
	for i, x0 := range host {
		if want := x0 + 7*x0; got[i] != want {
			t.Fatalf("y[%d] = %v, want %v", i, got[i], want)
		}
	}
}

// With nothing submitted the idle clock runs at wall speed between reads and
// books no sync.
// fak-test:runtime fast est=50ms lane=default
func TestVulkanDeviceIdleClockRunsWhileIdle(t *testing.T) {
	v := vk(t)
	c := cpu()
	x := v.Upload(NewF32(c, []int{4}, []float32{1, 2, 3, 4}), F32)
	defer v.Free(x)
	_ = v.Read(x)
	const gap = 10 * time.Millisecond
	wallStart := time.Now()
	first := readVulkanDeviceWork()
	time.Sleep(gap)
	second := readVulkanDeviceWork()
	wall := time.Since(wallStart)
	if cells := vulkanSyncCells(t, first, second); len(cells) != 0 {
		t.Fatalf("idle interval booked syncs %+v", cells)
	}
	ns, ok := DeviceIdleDelta(first, second)
	if d, slack := time.Duration(ns), goClockSlack(); !ok || d < gap-slack || d > wall+slack {
		t.Fatalf("idle clock advanced %v (ok=%v) over wall %v, want at least %v", d, ok, wall, gap)
	}
}

// Off the pipelined path every host wait follows its own submit, so idle
// intervals and submit-to-signal waits partition the shim's wall between two
// reads: a submit that leaves the idle clock running, or a wait booked as idle,
// breaks the sum.
// fak-test:runtime fast est=300ms lane=default
func TestVulkanDeviceIdleAndSyncWaitsPartitionSynchronousWall(t *testing.T) {
	v := vk(t)
	const n = 1024
	c := cpu()
	host := make([]float32, n)
	x := v.Upload(NewF32(c, []int{n}, host), F32)
	y := v.Upload(NewF32(c, []int{n}, host), F32)
	defer v.Free(x)
	defer v.Free(y)
	_ = v.Read(y)
	slack := goClockSlack()
	waitedSince := func(before DeviceWork) time.Duration {
		var ns uint64
		for _, cell := range vulkanSyncCells(t, before, readVulkanDeviceWork()) {
			ns += cell.WaitNS
		}
		return time.Duration(ns)
	}

	const idle = 5 * time.Millisecond
	outerStart := time.Now()
	before := readVulkanDeviceWork()
	innerStart := time.Now()
	time.Sleep(idle)
	for rounds := 0; rounds < 4096 && waitedSince(before) <= 4*slack; rounds += 16 {
		for range 16 {
			v.AddInPlace(y, x)
		}
	}
	v.BeginBatch()
	v.AddInPlace(y, x)
	v.AddInPlace(y, x)
	v.FlushBatch()
	_ = v.Read(y)
	innerEnd := time.Now()
	after := readVulkanDeviceWork()
	outerEnd := time.Now()

	idleNS, ok := DeviceIdleDelta(before, after)
	if !ok {
		t.Fatalf("DeviceIdleDelta refused a synchronous window: before %+v after %+v", before, after)
	}
	var waitNS uint64
	for _, cell := range vulkanSyncCells(t, before, after) {
		waitNS += cell.WaitNS
	}
	if time.Duration(waitNS) <= 4*slack {
		t.Fatalf("host waits %v do not exceed 4x clock slack %v, so the partition cannot discriminate", time.Duration(waitNS), slack)
	}
	sum := time.Duration(idleNS + waitNS)
	inner, outer := innerEnd.Sub(innerStart), outerEnd.Sub(outerStart)
	if time.Duration(idleNS) < idle-slack || sum < inner-slack || sum > outer+slack {
		t.Fatalf("idle %v + waits %v = %v, want idle at least %v and the sum within [%v, %v] +/- %v",
			time.Duration(idleNS), time.Duration(waitNS), sum, idle, inner, outer, slack)
	}
}

// While a pipelined batch is outstanding the idle clock is closed, exactly, and
// DeviceIdleDelta refuses the window; once the flush drains it the clock runs
// again. Off the pipelined path each cycle waits, so the clock runs throughout.
// fak-test:runtime fast est=50ms lane=default
func TestVulkanDeviceIdleClockClosedWhileBatchOutstanding(t *testing.T) {
	v := vk(t)
	const n = 1024
	c := cpu()
	host := make([]float32, n)
	x := v.Upload(NewF32(c, []int{n}, host), F32)
	y := v.Upload(NewF32(c, []int{n}, host), F32)
	defer v.Free(x)
	defer v.Free(y)
	_ = v.Read(y)

	const gap = 10 * time.Millisecond
	slack := goClockSlack()
	v.BeginBatch()
	v.AddInPlace(y, x)
	v.CycleBatch()
	v.AddInPlace(y, x)
	v.CycleBatch()
	first := readVulkanDeviceWork()
	time.Sleep(gap)
	second := readVulkanDeviceWork()
	v.FlushBatch()
	settled := readVulkanDeviceWork()
	time.Sleep(gap)
	third := readVulkanDeviceWork()

	if vulkanPipelinedCycles() {
		if second.DeviceIdleNS != first.DeviceIdleNS {
			t.Fatalf("idle clock moved %d->%d with a pipelined batch outstanding", first.DeviceIdleNS, second.DeviceIdleNS)
		}
		if first.PipelinedSubmits != first.PipelinedDrains+1 {
			t.Fatalf("pipelined submits %d drains %d, want exactly one outstanding", first.PipelinedSubmits, first.PipelinedDrains)
		}
		if _, ok := DeviceIdleDelta(first, second); ok {
			t.Fatal("DeviceIdleDelta accepted a window that began with a pipelined batch outstanding")
		}
	} else if ns, ok := DeviceIdleDelta(first, second); !ok || time.Duration(ns) < gap-slack {
		t.Fatalf("idle advanced %v (ok=%v) across %v with no batch outstanding", time.Duration(ns), ok, gap)
	}
	if settled.PipelinedSubmits != settled.PipelinedDrains {
		t.Fatalf("pipelined submits %d drains %d after flush, want none outstanding", settled.PipelinedSubmits, settled.PipelinedDrains)
	}
	if ns, ok := DeviceIdleDelta(settled, third); !ok || time.Duration(ns) < gap-slack {
		t.Fatalf("idle advanced %v (ok=%v) across %v after the flush drained", time.Duration(ns), ok, gap)
	}
}

// Inside a V4.1 step window a checked op's status flush books to flush_status at
// its kernel, a checked readback of an empty window to readback/none with its
// bytes, and the closing flush to step_end at the last recorded kernel.
// fak-test:runtime fast est=1s lane=default
func TestVulkanV41StepSyncSitesBookFlushStatusReadbackAndStepEnd(t *testing.T) {
	v := vk(t)
	const width = 256
	if !v.SupportsV41ExactRMSNorm(width, 1e-6) {
		t.Skip("V4.1 exact RMSNorm module unavailable on this device")
	}
	c := cpu()
	host := make([]float32, width)
	for i := range host {
		host[i] = float32(i%5) + 1
	}
	x := v.Upload(NewF32(c, []int{1, width}, host), F32)
	w := v.Upload(NewF32(c, []int{width}, host), F32)
	defer v.Free(x)
	defer v.Free(w)
	_ = v.Read(w)

	before := readVulkanDeviceWork()
	if !v.BeginV41StepBatch() {
		t.Fatal("V4.1 step window refused on a quiescent context")
	}
	out, err := v.V41ExactRMSNorm(x, w, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Free(out)
	if got := v.Read(out); len(got) != width {
		t.Fatalf("read %d values, want %d", len(got), width)
	}
	v.AddInPlace(w, w)
	if err := v.EndV41StepBatch(); err != nil {
		t.Fatal(err)
	}
	after := readVulkanDeviceWork()

	want := map[vulkanSyncCellKey]DeviceSyncSite{
		{SyncSiteFlushStatus, "v41_exact_rmsnorm"}: {Syncs: 1},
		{SyncSiteReadback, SyncStageNone}:          {Syncs: 1, D2HBytes: width * 4},
		{SyncSiteStepEnd, "add"}:                   {Syncs: 1},
	}
	cells := vulkanSyncCells(t, before, after)
	if len(cells) != len(want) {
		t.Fatalf("sync cells = %+v, want exactly %v", cells, want)
	}
	for key, wc := range want {
		cell, ok := cells[key]
		if !ok || cell.Syncs != wc.Syncs || cell.D2HBytes != wc.D2HBytes || cell.WaitNS == 0 {
			t.Fatalf("cell %v = %+v (present %v), want %d syncs, %d D2H bytes, nonzero wait", key, cell, ok, wc.Syncs, wc.D2HBytes)
		}
	}
}
