package compute

// cudaPagedKVWriteMaxPages keeps the by-value CUDA page map below the legacy 4 KiB
// kernel-argument limit. The C adapter independently enforces the same 64-page cap.
const cudaPagedKVWriteMaxPages = 64

// pagedKVAppendBatches is the paged writer's submission order, not an atomic KV
// transaction. prepare may publish allocated/COW table entries, but the caller
// commits its row count only after this function returns. beforePrepare identifies
// arena growth or COW: already prepared rows must be submitted before either can
// allocate, copy or retire storage. Ordinary free-list/private-page preparation
// performs no CUDA work. A panic from any callback stops without replay or rollback.
//
// write borrows ids only for the call. The CUDA adapter copies them into a by-value
// kernel argument before returning; it must never retain this Go array's address.
func pagedKVAppendBatches(start, n, blockTokens int, beforePrepare func(int) bool,
	prepare func(int) int32, write func(ids []int32, src, off, rows int)) {
	if start < 0 || n <= 0 || blockTokens <= 0 || start > int(^uint(0)>>1)-n {
		panic("cuda: paged KV append range out of bounds")
	}
	if off := start % blockTokens; n <= blockTokens-off {
		// Decode/tail writes need one ID, not bulk staging. Keep the admission
		// callback before preparation even though there is no pending group.
		blk := start / blockTokens
		beforePrepare(blk)
		ids := [1]int32{prepare(blk)}
		if ids[0] < 0 {
			panic("cuda: paged KV append prepared an invalid page")
		}
		write(ids[:], 0, off, n)
		return
	}
	var ids [cudaPagedKVWriteMaxPages]int32
	count, batchSrc, batchOff, rows := 0, 0, 0, 0
	flush := func() {
		if count != 0 {
			write(ids[:count], batchSrc, batchOff, rows)
			count, rows = 0, 0
		}
	}
	for src := 0; src < n; {
		pos := start + src
		blk, off := pos/blockTokens, pos%blockTokens
		if beforePrepare(blk) {
			flush()
		}
		id := prepare(blk)
		if id < 0 {
			panic("cuda: paged KV append prepared an invalid page")
		}
		if count == 0 {
			batchSrc, batchOff = src, off
		}
		take := min(blockTokens-off, n-src)
		ids[count] = id
		count++
		rows += take
		src += take
		if count == len(ids) {
			flush()
		}
	}
	flush()
}
