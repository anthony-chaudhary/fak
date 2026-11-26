package compute

// Inkling2TreeVerifyDevice runs a verify chunk laid out as a token tree
// instead of a chain. parent is a device table of T int32: row 0 the root
// (-1), every other row's parent an earlier row. Row t is the token at
// position pos+depth(t) along its root path while its cache row and log
// column stay those of row t, so each op is its Inkling2VerifyDevice chain op
// evaluated along row t's root path; a chain table (parent[t] = t-1) gives the
// chain op's bits. A path is a kept root path: increasing rows from the root.
type Inkling2TreeVerifyDevice interface {
	// Ink2TConvLog is Ink2PConvLog with row t's history taken from its nearest
	// ancestors, then hist; hist is left as it is for Ink2TConvCommit.
	Ink2TConvLog(x, w, dyn, hist, log, out, resid, parent Ink2Ref, n, heads, kn int, norm float32, T int) error
	// Ink2TCombineConvLog is Ink2PCombineConvLog on Ink2TConvLog.
	Ink2TCombineConvLog(down, wts Ink2Ref, k int, w, hist, log, out, resid, parent Ink2Ref, n, kn int, norm float32, T int) error
	// Ink2TConvCommit sets hist[c][j] = log[c][cols[j]], the past columns
	// metalgemm.TreeConvCommitCols names for a kept path (or the int32 at
	// colsDev when it is set).
	Ink2TConvCommit(hist, log, colsDev Ink2Ref, channels, past, L int, cols []int) error
	// Ink2TQKVPostDev is Ink2PQKVPostDev with row t's position pos+depth(t).
	Ink2TQKVPostDev(q, r, k, v, qclip, rclip, kcache, vcache, posDev, parent Ink2Ref, p Ink2AttnParams, cap, T int) error
	// Ink2TAttendDev is Ink2PAttendDev with row t attending the committed
	// prefix and its own root path only. A local ring needs window+T-1 rows.
	Ink2TAttendDev(q, r, sinks, relBias, kcache, vcache, out, posDev, parent Ink2Ref, p Ink2AttnParams, extent, window, cap, T int) error
	// Ink2TKVCompact moves a kept path's cache rows pos+path[i] to pos+i in
	// kcache and (when set) vcache, rings of cap rows of width floats; pathDev,
	// when set, holds [count, rows...] in 1+32 int32 instead of path.
	Ink2TKVCompact(kcache, vcache, pathDev, posDev Ink2Ref, width, cap int, path []int) error
	// Ink2TGatherRows sets out[i] = in[path[i]] for rows of width floats, in's
	// rows inStride apart; rows bounds the rows in holds.
	Ink2TGatherRows(in, out, pathDev Ink2Ref, width, inStride, rows int, path []int) error
}
