package kvmmu

// Host-tier pins for the SEAM-01 physical-pool charge/release ledger, adapted
// from the DeepSeek-V4.1-Flash Engram host-offload H200 recipe (vLLM
// --engram-config cpu_offload + use_thp at TP4; SGLang placement cpu / lookup
// peer_read equivalent). Source: vllm-project/vllm,
// recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml @ e740badc2e3df13a5c2abfca04ed0ac917985c5f
// (Apache-2.0), ADAPT verdict — no upstream code copied. The geometry mirrors the
// published recipe per rank: ~81.2 GiB resident weights + 38.5 GiB KV on the
// device row, a 23.6 GiB pinned Engram shard on the host row, against >=256 GB
// playbook host RAM. Software evidence only: no pinned memory is allocated and
// no hardware claim is made.

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const (
	// hostTierBytes models the provider-playbook host RAM ceiling (>=256 GB).
	hostTierBytes = int64(256) << 30
	// deviceTierBytes models the per-GPU resident footprint after offload
	// (81.2 GiB weights + 38.5 GiB KV, rounded to exact bytes).
	deviceTierBytes = int64(81)<<30 + int64(2)<<30
	// engramShardBytes models one rank's pinned Engram tables (23.6 GiB).
	engramShardBytes = int64(23)<<30 + int64(600)<<20
)

func newHostTierLedger() *PhysicalPoolLedger {
	return NewPhysicalPoolLedger(PoolCapacity{DeviceBytes: deviceTierBytes, HostBytes: hostTierBytes})
}

func TestPoolLedgerHostTierChargeReleaseAccounting(t *testing.T) {
	l := newHostTierLedger()

	if err := l.Charge("engram-rank0", RowHost, engramShardBytes); err != nil {
		t.Fatalf("Charge(engram-rank0, host) = %v, want nil", err)
	}
	if got := l.RowUsed(RowHost); got != engramShardBytes {
		t.Fatalf("RowUsed(host) = %d, want %d", got, engramShardBytes)
	}
	if got := l.RowFree(RowHost); got != hostTierBytes-engramShardBytes {
		t.Fatalf("RowFree(host) = %d, want %d", got, hostTierBytes-engramShardBytes)
	}
	if got := l.RowUsed(RowDevice); got != 0 {
		t.Fatalf("RowUsed(device) = %d, want 0 (host shard must not land on the device row)", got)
	}
	if got := l.TotalCharged(); got != engramShardBytes {
		t.Fatalf("TotalCharged() = %d, want %d", got, engramShardBytes)
	}

	// A second host shard lands under its own row; releasing the first restores
	// exactly its bytes — one byte, one row, one charge (MBL-087 double-charge guard).
	kvShardBytes := int64(38)<<30 + int64(500)<<20
	if err := l.Charge("kv-host-shard", RowHost, kvShardBytes); err != nil {
		t.Fatalf("Charge(kv-host-shard, host) = %v, want nil", err)
	}
	if got := l.RowUsed(RowHost); got != engramShardBytes+kvShardBytes {
		t.Fatalf("RowUsed(host) = %d, want %d", got, engramShardBytes+kvShardBytes)
	}
	released, ok := l.Release("engram-rank0")
	if !ok || released != engramShardBytes {
		t.Fatalf("Release(engram-rank0) = (%d, %v), want (%d, true)", released, ok, engramShardBytes)
	}
	if got := l.RowUsed(RowHost); got != kvShardBytes {
		t.Fatalf("RowUsed(host) after release = %d, want %d", got, kvShardBytes)
	}
	if got := l.RowFree(RowHost); got != hostTierBytes-kvShardBytes {
		t.Fatalf("RowFree(host) after release = %d, want %d", got, hostTierBytes-kvShardBytes)
	}
	if got := l.TotalCharged(); got != kvShardBytes {
		t.Fatalf("TotalCharged() after release = %d, want %d", got, kvShardBytes)
	}
	if _, ok := l.Release("engram-rank0"); ok {
		t.Fatal("second Release(engram-rank0) reported success, want (0, false)")
	}
	if n, ok := l.Release("never-charged"); ok || n != 0 {
		t.Fatalf("Release(never-charged) = (%d, %v), want (0, false)", n, ok)
	}

	// Structural sentinels: closed vocabulary, asserted with errors.Is.
	if err := l.Charge("", RowHost, 1); !errors.Is(err, ErrEmptyOwner) {
		t.Fatalf("Charge(empty owner) = %v, want ErrEmptyOwner", err)
	}
	if err := l.Charge("x", PoolRowClass("gpu"), 1); !errors.Is(err, ErrUnknownRowClass) {
		t.Fatalf("Charge(unknown class) = %v, want ErrUnknownRowClass", err)
	}
	if err := l.Charge("x", RowHost, 0); !errors.Is(err, ErrNonPositiveBytes) {
		t.Fatalf("Charge(zero bytes) = %v, want ErrNonPositiveBytes", err)
	}
	if err := l.Charge("kv-host-shard", RowHost, 1); !errors.Is(err, ErrDuplicateOwner) {
		t.Fatalf("Charge(duplicate owner) = %v, want ErrDuplicateOwner", err)
	}
	if got := l.RowUsed(RowHost); got != kvShardBytes {
		t.Fatalf("refused charges leaked into RowUsed(host): got %d, want %d", got, kvShardBytes)
	}
}

func TestPoolLedgerHostTierAdmissionDeclaresHostRAMRequirement(t *testing.T) {
	l := newHostTierLedger()

	// Leave exactly 1 GiB of host headroom, then ask for 1 GiB + 1 B more:
	// host RAM is a requirement, not a free window, so the admission math must
	// refuse it against the measured host peak.
	fill := hostTierBytes - (1 << 30)
	if err := l.Charge("engram-bulk", RowHost, fill); err != nil {
		t.Fatalf("Charge(engram-bulk, host) = %v, want nil", err)
	}
	overWant := int64(1)<<30 + 1
	err := l.Admit(compute.MemoryPlan{{Class: compute.MemoryOffload, Bytes: overWant, Scope: compute.MemoryScopeHost}})
	var fe *compute.FitError
	if !errors.As(err, &fe) {
		t.Fatalf("Admit(host-over) = %v, want typed *compute.FitError", err)
	}
	if fe.Verdict != compute.FitTooBig {
		t.Fatalf("FitError.Verdict = %v, want FitTooBig", fe.Verdict)
	}
	if fe.Scope != compute.MemoryScopeHost {
		t.Fatalf("FitError.Scope = %v, want host", fe.Scope)
	}
	if fe.Want != fill+overWant {
		t.Fatalf("FitError.Want = %d, want %d (post-demand host total)", fe.Want, fill+overWant)
	}
	if fe.Avail != hostTierBytes {
		t.Fatalf("FitError.Avail = %d, want %d (measured host peak)", fe.Avail, hostTierBytes)
	}

	// The refusal left the ledger untouched.
	if got := l.RowUsed(RowHost); got != fill {
		t.Fatalf("RowUsed(host) after refused Admit = %d, want %d", got, fill)
	}

	// Device-scoped demands are judged against the device row alone: an exact-fit
	// device plan still admits while the host row sits near its peak.
	if err := l.Admit(compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: deviceTierBytes}}); err != nil {
		t.Fatalf("Admit(exact-fit device plan) = %v, want nil", err)
	}
	// An empty plan admits.
	if err := l.Admit(nil); err != nil {
		t.Fatalf("Admit(nil) = %v, want nil", err)
	}
	// A device demand that overruns the device row refuses with the device scope:
	// fill the device row to its measured peak, then ask one more byte.
	if err := l.Charge("weights-device", RowDevice, deviceTierBytes); err != nil {
		t.Fatalf("Charge(weights-device) = %v, want nil", err)
	}
	err = l.Admit(compute.MemoryPlan{{Class: compute.MemoryKVCache, Bytes: 1, Scope: compute.MemoryScopeDevice}})
	if !errors.As(err, &fe) || fe.Verdict != compute.FitTooBig || fe.Scope != compute.MemoryScopeDevice {
		t.Fatalf("Admit(over device row) = %v, want typed device FitTooBig", err)
	}
	if fe.Want != deviceTierBytes+1 || fe.Avail != deviceTierBytes {
		t.Fatalf("FitError.(Want, Avail) = (%d, %d), want (%d, %d)", fe.Want, fe.Avail, deviceTierBytes+1, deviceTierBytes)
	}
}

func TestPoolLedgerHostTierChargeRefusalAtMeasuredHostPeak(t *testing.T) {
	l := newHostTierLedger()

	// Charge the host row to exactly its measured peak.
	if err := l.Charge("engram-full", RowHost, hostTierBytes); err != nil {
		t.Fatalf("Charge to exact host peak = %v, want nil", err)
	}
	// One byte past the measured host peak refuses with the typed sentinel.
	err := l.Charge("engram-over", RowHost, 1)
	var fe *compute.FitError
	if !errors.As(err, &fe) {
		t.Fatalf("Charge(past peak) = %v, want typed *compute.FitError", err)
	}
	if fe.Verdict != compute.FitTooBig || fe.Scope != compute.MemoryScopeHost {
		t.Fatalf("FitError = %+v, want host-scope FitTooBig", fe)
	}
	if fe.Want != hostTierBytes+1 || fe.Avail != hostTierBytes {
		t.Fatalf("FitError.(Want, Avail) = (%d, %d), want (%d, %d)", fe.Want, fe.Avail, hostTierBytes+1, hostTierBytes)
	}
	// The refused charge did not land.
	if got := l.RowUsed(RowHost); got != hostTierBytes {
		t.Fatalf("RowUsed(host) = %d, want %d", got, hostTierBytes)
	}
	if got := l.TotalCharged(); got != hostTierBytes {
		t.Fatalf("TotalCharged() = %d, want %d", got, hostTierBytes)
	}

	// Joint single-pool view: device and host rows sum against one pool (MBL-084).
	if err := l.Charge("weights-device", RowDevice, deviceTierBytes); err != nil {
		t.Fatalf("Charge(weights-device) = %v, want nil", err)
	}
	if got := l.TotalCharged(); got != hostTierBytes+deviceTierBytes {
		t.Fatalf("TotalCharged() = %d, want %d (joint pool sum)", got, hostTierBytes+deviceTierBytes)
	}
}
