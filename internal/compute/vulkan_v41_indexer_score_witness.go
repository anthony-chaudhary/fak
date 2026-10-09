//go:build v41_indexer_witness && vulkan && (windows || linux) && cgo

package compute

/*
// This declaration is available only in the explicitly tagged witness build.
// The matching native archive must itself be built with the same definition;
// a production archive lacks these symbols and intentionally cannot link it.
#define FAK_V41_INDEXER_SCORE_WITNESS
#include "vulkan_backend.h"
*/
import "C"

// These unexported helpers are absent from every ordinary production Go build.
// They cannot change capability admission or select a model execution route.
func (v *vulkanBackend) v41IndexerScoreWitnessReady() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_v41_indexer_score_witness_ready() != 0
}

func (v *vulkanBackend) v41IndexerScoreWitness(q, keys, weights Tensor, rows, heads, headDim int) (Tensor, error) {
	return v.v41IndexerScoreWithNative(q, keys, weights, rows, heads, headDim,
		func() bool { return C.fvk_v41_indexer_score_witness_ready() != 0 },
		func() string { return C.GoString(C.fvk_v41_indexer_score_unavailable_reason()) },
		v41IndexerScoreWitnessStatusLocked)
}

func v41IndexerScoreWitnessStatusLocked(q, keys, weights, out *vulkanBuf, rows, heads, headDim int) int {
	return int(C.fvk_v41_indexer_score_witness_f32(q.ptr, keys.ptr, weights.ptr, out.ptr, C.int(rows), C.int(heads), C.int(headDim)))
}
