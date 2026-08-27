package model

import "testing"

func TestAdmitCUDAAsyncArchitectureGate(t *testing.T) {
	bad, err := AdmitCUDAAsync(CUDAAsyncEnvelope{Architecture: 80, Shape: "decode-b1", TMA: true, AcceptedTokens: 1, Fallback: "none"})
	if err != nil {
		t.Fatal(err)
	}
	if bad.Admitted || bad.Refusal != "TMA_REQUIRES_SM90" {
		t.Fatalf("bad=%+v", bad)
	}
	ok, err := AdmitCUDAAsync(CUDAAsyncEnvelope{Architecture: 90, Shape: "decode-b1", AsyncCopy: true, TMA: true, WarpSpecialized: true, Persistent: true, SharedReuseBytes: 4096, PhysicalBytes: 8192, Nanoseconds: 1000, Joules: .2, AcceptedTokens: 4, Fallback: "none"})
	if err != nil {
		t.Fatal(err)
	}
	if !ok.Admitted || ok.Engine != "fak-native-cuda" || ok.PhysicalBytesPerAccepted != 2048 || ok.NanosecondsPerAccepted != 250 || ok.JoulesPerAccepted != .05 {
		t.Fatalf("ok=%+v", ok)
	}
}
func TestAdmitCUDAAsyncRefusesFallback(t *testing.T) {
	r, _ := AdmitCUDAAsync(CUDAAsyncEnvelope{Architecture: 90, Shape: "x", AcceptedTokens: 1, Fallback: "cpu"})
	if r.Admitted || r.Refusal != "FALLBACK_PRESENT" {
		t.Fatalf("receipt=%+v", r)
	}
}
