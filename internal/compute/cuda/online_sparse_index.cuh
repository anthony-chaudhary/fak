// online_sparse_index.cuh - structure-only contract for MInference online sparse-index approx.
//
// diagonal-sum (calc_index) top-k per request, without full-attention recompute. No upstream
// code is copied.
//
// SKELETON ONLY: constants. No hardware run, no speed claim. The Go oracle in
// online_sparse_index_test.go reads these #defines.
#ifndef FAK_ONLINE_SPARSE_INDEX_CUH
#define FAK_ONLINE_SPARSE_INDEX_CUH

#define FAK_ONLINE_SPARSE_TOP_V 2
#define FAK_ONLINE_SPARSE_TOP_S 2
#define FAK_ONLINE_SPARSE_LAST_Q_ONLY 1

#endif  // FAK_ONLINE_SPARSE_INDEX_CUH
