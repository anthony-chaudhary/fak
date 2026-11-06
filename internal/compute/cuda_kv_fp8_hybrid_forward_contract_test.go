package compute

import (
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestCUDAKVFP8HybridForwardSourceContract(t *testing.T) {
	adapter := readCUDASource(t, "cuda_kv_fp8_hybrid_forward_qualification.go")
	for _, part := range []string{"a.be.graphOnce != nil", "cudaGraphOnceOwner != nil", "req.KV != nil",
		"validateQwen35SequenceGeometry(req)", "tensor.buf.(*cudaBuf)", "tensor.Quant != nil",
		"b.n != bytes", "b.class != MemoryKVCache", "b.graphOncePinned", "invalid: &b.invalid",
		"C.fcuda_kv_fp8_qwen_admit_owned()", "a.appendOperations()"} {
		if !strings.Contains(adapter, part) {
			t.Fatalf("missing concrete state/admission binding %q", part)
		}
	}
	for _, bad := range []string{"cudaMu.Lock(", "ForwardExecutionOwner().Lock(", "makeTensor(", "devTr(", "NewKV(", "qwen35SequencePoisonStates("} {
		if strings.Contains(adapter, bad) {
			t.Fatalf("unsafe replacement/lock/public path %q", bad)
		}
	}
	core := readCUDASource(t, "kv_fp8_hybrid_forward.go")
	admit := strings.Index(core, "if err := admit();")
	reserve := strings.Index(core, "err = kvFP8AppendEager(")
	if admit < 0 || reserve < admit {
		t.Fatal("admission must precede full append reservation")
	}
	for _, part := range []string{"f.owner.poisonNumerical()", "atomic.StoreUint32(s.invalid, 1)", "flight.retain()",
		"if !normalReturn || err != nil", "ops.stop(kvFP8AppendFailure)", "return tx.checked()"} {
		if !strings.Contains(core, part) {
			t.Fatal("missing whole-forward boundary", part)
		}
	}
}
