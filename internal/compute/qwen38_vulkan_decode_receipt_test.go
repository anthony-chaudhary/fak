package compute

import (
	"errors"
	"strings"
	"testing"
)

func TestQwen38VulkanDecodeReceiptValidAndComparable(t *testing.T) {
	packet := NewQwen38VulkanDecodePacket([]int32{151644, 8948, 198}, 4)
	receipt := validQwen38VulkanDecodeReceipt(t, packet)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}

	candidate := receipt
	candidate.ElapsedNanoseconds--
	candidate.Counters.DispatchSubmits++
	if err := CompareQwen38VulkanDecodeReceipts(receipt, candidate); err != nil {
		t.Fatalf("CompareQwen38VulkanDecodeReceipts() error = %v", err)
	}
}

func TestQwen38VulkanDecodeReceiptRejectsContractInconsistencies(t *testing.T) {
	packet := NewQwen38VulkanDecodePacket([]int32{1, 2}, 4)
	base := validQwen38VulkanDecodeReceipt(t, packet)

	tests := []struct {
		name string
		edit func(*Qwen38VulkanDecodeReceipt)
		want string
	}{
		{"model identity", func(r *Qwen38VulkanDecodeReceipt) { r.Packet.ModelGGUFSHA256 = strings.Repeat("0", 64) }, "model digest"},
		{"backend", func(r *Qwen38VulkanDecodeReceipt) { r.Backend = "cpu" }, "engine identity"},
		{"boundary", func(r *Qwen38VulkanDecodeReceipt) { r.GeneratedTokens-- }, "work boundary"},
		{"output hash", func(r *Qwen38VulkanDecodeReceipt) { r.OutputTokenIDsSHA256 = strings.Repeat("0", 64) }, "output token digest"},
		{"dispatch total", func(r *Qwen38VulkanDecodeReceipt) { r.Counters.ComputeDispatches++ }, "compute dispatch total"},
		{"transfer", func(r *Qwen38VulkanDecodeReceipt) { r.Counters.H2D.Count = 0 }, "h2d count/bytes"},
		{"q4 stage", func(r *Qwen38VulkanDecodeReceipt) { r.Counters.Q4KStageCalls = 0 }, "stage calls/bytes"},
		{"tensor home", func(r *Qwen38VulkanDecodeReceipt) { r.Counters.TensorHome.CopiedBytes = r.Counters.D2D.Bytes + 1 }, "exceed d2d"},
		{"memory", func(r *Qwen38VulkanDecodeReceipt) { r.PeakDeviceMemoryBytes = r.PeakProcessMemoryBytes + 1 }, "exceeds process peak"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receipt := base
			tt.edit(&receipt)
			if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestCompareQwen38VulkanDecodeReceiptsRejectsMismatches(t *testing.T) {
	packet := NewQwen38VulkanDecodePacket([]int32{7, 8}, 4)
	parent := validQwen38VulkanDecodeReceipt(t, packet)

	t.Run("packet", func(t *testing.T) {
		candidatePacket := NewQwen38VulkanDecodePacket([]int32{7, 9}, 4)
		candidate := validQwen38VulkanDecodeReceipt(t, candidatePacket)
		if err := CompareQwen38VulkanDecodeReceipts(parent, candidate); err == nil || !strings.Contains(err.Error(), "packet mismatch") {
			t.Fatalf("comparison error = %v, want packet mismatch", err)
		}
	})

	t.Run("output", func(t *testing.T) {
		candidate := parent
		candidate.OutputTokenIDs = []int32{11, 12, 13, 99}
		candidate.OutputTokenIDsSHA256 = Qwen38VulkanTokenIDsSHA256(candidate.OutputTokenIDs)
		if err := CompareQwen38VulkanDecodeReceipts(parent, candidate); err == nil || !strings.Contains(err.Error(), "output mismatch") {
			t.Fatalf("comparison error = %v, want output mismatch", err)
		}
	})

	t.Run("work boundary", func(t *testing.T) {
		candidate := parent
		candidate.Packet.GeneratedTokenLimit = 3
		candidate.OutputTokenIDs = candidate.OutputTokenIDs[:3]
		candidate.GeneratedTokens = 3
		candidate.OutputTokenIDsSHA256 = Qwen38VulkanTokenIDsSHA256(candidate.OutputTokenIDs)
		candidate.PacketSHA256 = mustQwen38VulkanPacketDigest(t, candidate.Packet)
		if err := CompareQwen38VulkanDecodeReceipts(parent, candidate); err == nil || !strings.Contains(err.Error(), "packet mismatch") {
			t.Fatalf("comparison error = %v, want packet mismatch for unequal boundary", err)
		}
	})
}

func TestExecuteQwen38VulkanDecodeContractCleansUpExactlyOnce(t *testing.T) {
	tests := []struct {
		name      string
		run       func() (int, error)
		wantPanic bool
	}{
		{"success", func() (int, error) { return 7, nil }, false},
		{"error", func() (int, error) { return 0, errors.New("decode failed") }, false},
		{"panic", func() (int, error) { panic("decode panic") }, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cleanups := 0
			func() {
				defer func() {
					gotPanic := recover() != nil
					if gotPanic != tt.wantPanic {
						t.Fatalf("panic = %v, want %v", gotPanic, tt.wantPanic)
					}
				}()
				_, _ = ExecuteQwen38VulkanDecodeContract(func() { cleanups++ }, tt.run)
			}()
			if cleanups != 1 {
				t.Fatalf("cleanup calls = %d, want 1", cleanups)
			}
		})
	}
}

func TestQwen38VulkanDecodePacketRequiresPositiveBoundary(t *testing.T) {
	packet := NewQwen38VulkanDecodePacket([]int32{1}, 0)
	if err := packet.Validate(); err == nil || !strings.Contains(err.Error(), "must be positive") {
		t.Fatalf("Validate() error = %v, want positive-boundary refusal", err)
	}
}

func validQwen38VulkanDecodeReceipt(t *testing.T, packet Qwen38VulkanDecodePacket) Qwen38VulkanDecodeReceipt {
	t.Helper()
	output := []int32{11, 12, 13, 14}
	if len(output) != packet.GeneratedTokenLimit {
		t.Fatalf("test output length %d != packet boundary %d", len(output), packet.GeneratedTokenLimit)
	}
	return Qwen38VulkanDecodeReceipt{
		Schema:                 Qwen38VulkanDecodeReceiptSchema,
		Packet:                 packet,
		PacketSHA256:           mustQwen38VulkanPacketDigest(t, packet),
		Backend:                Qwen38VulkanDecodeBackend,
		Runtime:                Qwen38VulkanDecodeRuntime,
		Fallback:               Qwen38VulkanDecodeFallback,
		OutputTokenIDs:         output,
		OutputTokenIDsSHA256:   Qwen38VulkanTokenIDsSHA256(output),
		GeneratedTokens:        len(output),
		ElapsedNanoseconds:     50_000,
		PeakProcessMemoryBytes: 8 << 30,
		PeakDeviceMemoryBytes:  6 << 30,
		Counters: Qwen38VulkanDecodeCounters{
			ComputeDispatches:      40,
			Q4KMatmulDispatches:    30,
			OtherComputeDispatches: 10,
			DispatchSubmits:        8,
			H2D:                    Qwen38VulkanTransferCounters{Count: 3, Bytes: 3072},
			D2H:                    Qwen38VulkanTransferCounters{Count: 1, Bytes: 4},
			D2D:                    Qwen38VulkanTransferCounters{Count: 2, Bytes: 2048},
			Q4KStageCalls:          4,
			Q4KStageBytes:          4096,
			TensorHome: Qwen38VulkanTensorHomeCounters{
				Hits: 8, Admissions: 2, Bypasses: 1, ResidentBytes: 2048, CopiedBytes: 2048,
			},
		},
	}
}

func mustQwen38VulkanPacketDigest(t *testing.T, packet Qwen38VulkanDecodePacket) string {
	t.Helper()
	digest, err := packet.Digest()
	if err != nil {
		t.Fatalf("packet.Digest() error = %v", err)
	}
	return digest
}
