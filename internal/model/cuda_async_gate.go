package model

import "fmt"

type CUDAAsyncEnvelope struct {
	Architecture     int     `json:"sm"`
	Shape            string  `json:"shape"`
	AsyncCopy        bool    `json:"async_copy"`
	TMA              bool    `json:"tma"`
	WarpSpecialized  bool    `json:"warp_specialized"`
	Persistent       bool    `json:"persistent"`
	SharedReuseBytes int64   `json:"shared_reuse_bytes"`
	PhysicalBytes    int64   `json:"physical_bytes"`
	Nanoseconds      int64   `json:"nanoseconds"`
	Joules           float64 `json:"joules"`
	AcceptedTokens   int     `json:"accepted_tokens"`
	Fallback         string  `json:"fallback"`
}
type CUDAAsyncReceipt struct {
	Schema                   string  `json:"schema"`
	Engine                   string  `json:"engine"`
	Architecture             int     `json:"sm"`
	Shape                    string  `json:"shape"`
	Admitted                 bool    `json:"admitted"`
	Refusal                  string  `json:"refusal,omitempty"`
	AsyncCopy                bool    `json:"async_copy"`
	TMA                      bool    `json:"tma"`
	WarpSpecialized          bool    `json:"warp_specialized"`
	Persistent               bool    `json:"persistent"`
	SharedReuseBytes         int64   `json:"shared_reuse_bytes"`
	PhysicalBytesPerAccepted float64 `json:"physical_bytes_per_accepted_token"`
	NanosecondsPerAccepted   float64 `json:"nanoseconds_per_accepted_token"`
	JoulesPerAccepted        float64 `json:"joules_per_accepted_token"`
	QualityConstraint        string  `json:"quality_constraint"`
	StopRule                 string  `json:"stop_rule"`
	Rollback                 string  `json:"rollback"`
}

func AdmitCUDAAsync(e CUDAAsyncEnvelope) (CUDAAsyncReceipt, error) {
	if e.Architecture < 0 || e.Shape == "" || e.SharedReuseBytes < 0 || e.PhysicalBytes < 0 || e.Nanoseconds < 0 || e.Joules < 0 || e.AcceptedTokens < 0 {
		return CUDAAsyncReceipt{}, fmt.Errorf("model: invalid CUDA async envelope")
	}
	r := CUDAAsyncReceipt{Schema: "fak-cuda-async/1", Engine: "fak-native-cuda", Architecture: e.Architecture, Shape: e.Shape, AsyncCopy: e.AsyncCopy, TMA: e.TMA, WarpSpecialized: e.WarpSpecialized, Persistent: e.Persistent, SharedReuseBytes: e.SharedReuseBytes, QualityConstraint: "exact native logits/tokens with fallback=none", StopRule: "refuse unsupported architecture/shape; reject net regression", Rollback: "baseline CUDA kernel"}
	if e.TMA && e.Architecture < 90 {
		r.Refusal = "TMA_REQUIRES_SM90"
		return r, nil
	}
	if e.Fallback != "none" {
		r.Refusal = "FALLBACK_PRESENT"
		return r, nil
	}
	if e.AcceptedTokens == 0 {
		r.Refusal = "NO_ACCEPTED_TOKEN_DENOMINATOR"
		return r, nil
	}
	r.Admitted = true
	r.PhysicalBytesPerAccepted = float64(e.PhysicalBytes) / float64(e.AcceptedTokens)
	r.NanosecondsPerAccepted = float64(e.Nanoseconds) / float64(e.AcceptedTokens)
	r.JoulesPerAccepted = e.Joules / float64(e.AcceptedTokens)
	return r, nil
}
