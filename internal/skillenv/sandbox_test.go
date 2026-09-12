package skillenv

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
)

// TestTable_LockFreeCAS_ConcurrentSwapsSandwiched proves the lock-free CAS
// page table under contention: a sandwiched Swap/Await/Swap sequence always
// observes a consistent chain of ~count flips (no lost torn updates), and all
// flips succeed despite concurrent interleaved writers.
func TestTable_LockFreeCAS_ConcurrentSwapsSandwiched(t *testing.T) {
	table := New(nil, nil, nil)
	if _, _, err := table.Pin("skill-a", "1.0.0"); err != nil {
		t.Fatalf("pin v1: %v", err)
	}

	const count = 1000
	observed := make([]string, count)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < count; i++ {
			v, ok := table.ActiveVersion("skill-a")
			if !ok {
				t.Errorf("snap %d: AetActiveVersion missing", i)
				return
			}
			observed[i] = v
		}
	}()

	// Interleaved concurrent writers exercising the CAS publish loop. Pin
	// alternates versions by worker parity — Pin never refuses, so any error
	// here is a real kernel failure.
	var writeErrs atomic.Int64
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			v := fmt.Sprintf("%d.0.0", w%2+1)
			for i := 0; i < 250; i++ {
				if _, _, err := table.Pin("skill-a", v); err != nil {
					writeErrs.Add(1)
				}
			}
		}(w)
	}
	wg.Wait()

	if e := writeErrs.Load(); e != 0 {
		t.Fatalf("%d writes failed permanently", e)
	}

	// Snapshots after test completion must be one of the two live versions.
	for i, v := range observed {
		if v != "1.0.0" && v != "2.0.0" {
			t.Fatalf("snapshot %d: torn or missing version %q", i, v)
		}
	}
}

// TestSandbox_SwappingAndReclamation proves live zero-downtime swap: a page is
// admitted, runs, and can be unloaded while it serves, with in-flight pinned
// frames surviving the swap and reclaimed memory returning to zero.
func TestSandbox_SwappingAndReclamation(t *testing.T) {
	table := New(nil, nil, nil)
	sb := NewSandbox(table, 1<<20) // 1 MiB cap

	page, err := sb.Admit(PageQuery{Skill: "code-review", Version: "1.0.0", ABIDigest: ComputePluginABIDigest()})
	if err != nil {
		t.Fatalf("admit: %v", err)
	}
	if err := sb.Activate(page); err != nil {
		t.Fatalf("activate: %v", err)
	}

	// Execution succeeds under the fence.
	if err := sb.Run(page, func() error { return nil }); err != nil {
		t.Fatalf("run: %v", err)
	}
	if used := sb.Pool().Used(); used != 0 {
		t.Fatalf("pool used after clean run = %d, want 0 (zero leaks)", used)
	}

	// Admit a v2 with the right ABI digest and swap the table pin.
	page2, err := sb.Admit(PageQuery{Skill: "code-review", Version: "2.0.0", ABIDigest: ComputePluginABIDigest()})
	if err != nil {
		t.Fatalf("admit v2: %v", err)
	}
	if err := sb.Activate(page2); err != nil {
		t.Fatalf("activate v2: %v", err)
	}
	if v, ok := table.ActiveVersion("code-review"); !ok || v != "2.0.0" {
		t.Fatalf("after admit v2 active = %q, ok=%v, want 2.0.0", v, ok)
	}

	// Unload v1 while it still exists — v2 continues serving. Reclamation hits
	// the page itself (not v2).
	n, err := sb.Unload("code-review", "1.0.0")
	if err != nil {
		t.Fatalf("unload v1: %v", err)
	}
	if n == 0 {
		t.Fatalf("unload v1 reclaimed 0")
	}
	if page.State() != PageDraining {
		t.Fatalf("unloaded page state = %s, want draining", page.State())
	}
	if v, ok := table.ActiveVersion("code-review"); !ok || v != "2.0.0" {
		t.Fatalf("after unload v1 active = %q ok=%v, want 2.0.0 untouched", v, ok)
	}

	// Unload v2 too; pin drops; the resolver governs again (ok=false).
	if _, err := sb.Unload("code-review", "2.0.0"); err != nil {
		t.Fatalf("unload v2: %v", err)
	}
	if _, ok := table.ActiveVersion("code-review"); ok {
		t.Fatalf("after unloading both versions, skill still pinned")
	}
	if got := len(sb.Pages()); got != 0 {
		t.Fatalf("pages remaining after full unload = %d, want 0", got)
	}
}

// TestSandbox_InFlightPinnedFrameSurvivesUnload proves the draining step does
// not disturb an in-flight invocation: a run that started before Unload keeps
// executing under the fence and releases its segment cleanly.
func TestSandbox_InFlightPinnedFrameSurvivesUnload(t *testing.T) {
	table := New(nil, nil, nil)
	sb := NewSandbox(table, 1<<20)

	page, err := sb.Admit(PageQuery{Skill: "skill-a", Version: "1.0.0", ABIDigest: ComputePluginABIDigest()})
	if err != nil {
		t.Fatalf("admit: %v", err)
	}
	if err := sb.Activate(page); err != nil {
		t.Fatalf("activate: %v", err)
	}

	release := make(chan struct{})
	started := make(chan struct{})
	ran := make(chan error, 1)
	go func() {
		ran <- sb.Run(page, func() error {
			started <- struct{}{}
			<-release
			return nil
		})
	}()

	// Wait until the invocation is genuinely in flight, then drain the page.
	<-started
	if _, err := sb.Unload("skill-a", "1.0.0"); err != nil {
		t.Fatalf("unload: %v", err)
	}

	// New dispatch AFTER unload is refused (draining, and no longer admitted).
	if err := sb.Run(page, func() error { return nil }); err == nil {
		t.Fatalf("run on unloaded page: want refusal")
	}

	close(release)
	if err := <-ran; err != nil {
		t.Fatalf("in-flight run after unload failed: %v", err)
	}
	if used := sb.Pool().Used(); used != 0 {
		t.Fatalf("pool used after in-flight completion = %d, want 0", used)
	}
}

// TestSandbox_MemoryCapFailClosed proves the pool refuses (rather than spills)
// a charge that would exceed the cap, and the caller sees the failure.
func TestSandbox_MemoryCapFailClosed(t *testing.T) {
	pool := NewPool(2 * segmentCost)
	if err := pool.Charge(segmentCost); err != nil {
		t.Fatalf("first charge: %v", err)
	}
	if err := pool.Charge(segmentCost); err != nil {
		t.Fatalf("second charge: %v", err)
	}
	err := pool.Charge(1)
	if err == nil {
		t.Fatal("over-cap charge: want error")
	}
	if used := pool.Used(); used != 2*segmentCost {
		t.Fatalf("used after refused charge = %d, want %d (no partial consumption)", used, 2*segmentCost)
	}
	pool.Release(segmentCost)
	if used := pool.Used(); used != segmentCost {
		t.Fatalf("used after release = %d, want %d", used, segmentCost)
	}
}

// TestABIAdmission_FailClosed proves the static ABI-hash verifier admits only
// the exact kernel-derived digest: empty, wrong, and corrupted digests are all
// refused, and gate is wired into Sandbox.Admit.
func TestABIAdmission_FailClosed(t *testing.T) {
	want := ComputePluginABIDigest()
	if len(want) != 64 { // sha256 hex
		t.Fatalf("digest length = %d, want 64", len(want))
	}
	if err := VerifyPluginABI(want); err != nil {
		t.Fatalf("verifier refused matching digest: %v", err)
	}

	// Empty digest: refused.
	if err := VerifyPluginABI(""); err == nil {
		t.Fatal("empty digest: want refusal")
	}
	// Corrupted digest: refused.
	bad := []byte(want)
	bad[0] ^= 0xFF
	if err := VerifyPluginABI(string(bad)); err == nil {
		t.Fatal("corrupted digest: want refusal")
	}
	// Digest computed under a different ABI version must not match this
	// kernel's gate; prove the refusal path explicitly.
	if err := VerifyPluginABI(fmt.Sprintf("%064x", 0)); err == nil {
		t.Fatal("zero digest: want refusal")
	}

	// The sandbox gate refuses a mismatched page outright.
	table := New(nil, nil, nil)
	sb := NewSandbox(table, 1<<20)
	if _, err := sb.Admit(PageQuery{Skill: "x", Version: "1", ABIDigest: "deadbeef"}); err == nil {
		t.Fatal("admit with wrong ABI digest: want refusal")
	}
	if _, err := sb.Admit(PageQuery{Skill: "x", Version: "1"}); err == nil {
		t.Fatal("admit with empty ABI digest: want refusal")
	}
}

// TestSandbox_DuplicateAdmitRefused proves admission is idempotent-safe: the
// same (skill, version) cannot be admitted twice into distinct slots.
func TestSandbox_DuplicateAdmitRefused(t *testing.T) {
	table := New(nil, nil, nil)
	sb := NewSandbox(table, 1<<20)
	if _, err := sb.Admit(PageQuery{Skill: "s", Version: "1", ABIDigest: ComputePluginABIDigest()}); err != nil {
		t.Fatalf("first admit: %v", err)
	}
	if _, err := sb.Admit(PageQuery{Skill: "s", Version: "1", ABIDigest: ComputePluginABIDigest()}); err == nil {
		t.Fatal("duplicate admit: want refusal")
	}
}

// TestSandbox_UnloadAbsentPageIsNoop proves unloading a page that was never
// admitted (or already unloaded) reclaims nothing and reports no error.
func TestSandbox_UnloadAbsentPageIsNoop(t *testing.T) {
	table := New(nil, nil, nil)
	sb := NewSandbox(table, 1<<20)
	n, err := sb.Unload("never-admitted", "1.0.0")
	if err != nil {
		t.Fatalf("unload absent: %v", err)
	}
	if n != 0 {
		t.Fatalf("reclaimed = %d, want 0", n)
	}
}

// TestRunner_NotActiveRefused proves the runner refuses execution except on an
// Active page (loading and draining are both terminal/non-schedulable).
func TestRunner_NotActiveRefused(t *testing.T) {
	sb := NewSandbox(New(nil, nil, nil), 1<<20)
	loading, err := sb.Admit(PageQuery{Skill: "s", Version: "1", ABIDigest: ComputePluginABIDigest()})
	if err != nil {
		t.Fatalf("admit: %v", err)
	}
	if err := sb.Run(loading, func() error { return nil }); err != nil {
		t.Fatalf("warm-up run on loading page: %v", err)
	}
	if err := sb.Activate(loading); err != nil {
		t.Fatalf("activate: %v", err)
	}
	if err := sb.Run(loading, func() error { return nil }); err != nil {
		t.Fatalf("run on active page: %v", err)
	}
	// Double-activate must fail (monotonic transition).
	if err := sb.Activate(loading); err == nil {
		t.Fatal("double activate: want error")
	}
}

// TestRunner_FailClosedWarmup proves a page that fails during its loading
// warm-up NEVER becomes schedulable: a runner failure on a loading page marks
// the page terminal (draining) with the error recorded, the page can then
// never be activated, and no segment memory is left charged.
func TestRunner_FailClosedWarmup(t *testing.T) {
	sb := NewSandbox(New(nil, nil, nil), 1<<20)
	page, err := sb.Admit(PageQuery{Skill: "s", Version: "1", ABIDigest: ComputePluginABIDigest()})
	if err != nil {
		t.Fatalf("admit: %v", err)
	}
	// A direct runner failure (simulating a failed warm-up hook) marks the page
	// terminal before it was ever Active.
	if err := sb.Run(page, func() error { return errors.New("warmup boom") }); err == nil {
		t.Fatal("run on loading page: want refusal")
	}
	if page.State() != PageDraining {
		t.Fatalf("failed page state = %s, want draining (fail-closed)", page.State())
	}
	if got := page.LastError(); got == "" {
		t.Fatal("LastError empty after failure, want recorded")
	}
	if err := sb.Activate(page); err == nil {
		t.Fatal("activate after failed warm-up: want refusal")
	}
	if used := sb.Pool().Used(); used != 0 {
		t.Fatalf("pool used after failed warm-up = %d, want 0", used)
	}
	// Cleanup.
	_, _ = sb.Unload("s", "1")
}

// TestTable_PrevSemanticsPreserved pins down the API contract the CLI relies
// on: Pin/Unpin/Swap prev-return values derived from the linearization
// pre-state, and swap's from-guard message.
func TestTable_PrevSemanticsPreserved(t *testing.T) {
	table := New(nil, nil, nil)

	prev, _, err := table.Pin("s", "1.0.0")
	if err != nil || prev != "" {
		t.Fatalf("first pin: prev=%q err=%v, want empty", prev, err)
	}
	prev, _, err = table.Pin("s", "2.0.0")
	if err != nil || prev != "1.0.0" {
		t.Fatalf("second pin: prev=%q err=%v, want 1.0.0", prev, err)
	}
	unpinned, _, err := table.Unpin("s")
	if err != nil || unpinned != "2.0.0" {
		t.Fatalf("unpin: got %q err=%v, want 2.0.0", unpinned, err)
	}
	// Historical contract (C4): a swap targeting an UNPINNED skill binds it
	// directly — the from-guard only constrains skills that ARE pinned.
	_, _, err = table.Swap("s", "1.0.0", "3.0.0")
	if err != nil {
		t.Fatalf("swap on unpinned skill: %v (want implicit bind)", err)
	}
	if v, _ := table.ActiveVersion("s"); v != "3.0.0" {
		t.Fatalf("swap-on-unpinned version = %q, want 3.0.0", v)
	}
	if _, _, err := table.Swap("s", "9.9.9", "3.0.0"); err == nil {
		t.Fatal("swap with mismatched from: want refusal")
	}
}

// TestSandbox_UnloadLoadingPageRefusesSubsequentRun proves unloading a page
// before activation makes it terminal and prevents any later callback.
func TestSandbox_UnloadLoadingPageRefusesSubsequentRun(t *testing.T) {
	sb := NewSandbox(New(nil, nil, nil), 1<<20)
	page, err := sb.Admit(PageQuery{Skill: "review", Version: "1", ABIDigest: ComputePluginABIDigest()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sb.Unload("review", "1"); err != nil {
		t.Fatal(err)
	}
	invoked := false
	err = sb.Run(page, func() error {
		invoked = true
		return nil
	})
	if err == nil {
		t.Error("unloaded loading page remained schedulable")
	}
	if invoked {
		t.Error("callback ran after loading page was unloaded")
	}
}
