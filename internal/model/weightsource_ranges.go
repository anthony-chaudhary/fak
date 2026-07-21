package model

// weightsource_ranges.go — a shared, source-agnostic parallel range reader for the weight
// loader (issue #3250, child of the ZML-inspiration epic #3236).
//
// WHY. The GGUF/safetensors loaders read tensor payloads sequentially from ONE local
// io.ReaderAt today (ggufload.WeightSource.r, weightsource.go's per-name views). A local
// file at ~2.8 GB/s hides that, but the VFS child (#3249) will front the same seam with a
// remote source (HTTP/HF/S3), where a single serial stream is latency-bound and leaves the
// link far from saturated. ZML's answer (zml/io/vfs/parallel_read.zig) is ONE shared reader
// that takes the byte ranges a load needs and fetches them with bounded parallelism — the
// same helper serves file, HTTP, HF, S3, GCS because it only ever speaks io.ReaderAt.
//
// This file is that helper, deliberately narrow: it fans a set of (offset,len) reads across
// a bounded worker set over an io.ReaderAt, first error cancels the rest, and — because each
// range owns its OWN destination buffer and io.ReaderAt.ReadAt is defined safe for concurrent
// use — the bytes it produces are identical to reading the same ranges one at a time in a
// sequential loop. The HTTP `Range:` fan-out that makes this pay off over the network is
// explicitly deferred to #3249; here the source is any io.ReaderAt (a local file, or an
// in-memory blob in the tests), NO network.
//
// STDLIB-ONLY. The module carries no golang.org/x/sync, so the "errgroup first-error-cancel"
// semantic is built by hand: an atomic cursor hands out range indices, a shared abort channel
// is closed once by the first failing worker (sync.Once), and every worker checks abort before
// claiming its next range — so after the first error no NEW read is started, and the first
// error is the one returned. No context type is used (and no ctx-named identifier), keeping the
// surface a single pure function over the io.ReaderAt seam.

import (
	"fmt"
	"io"
	"sync"
	"sync/atomic"
)

// Range is one contiguous read the caller wants served: len(Dst) bytes drawn from the source
// starting at Offset, deposited into Dst. The caller owns Dst (typically a sub-slice of a
// larger reconstruction buffer, mirroring ZML's per-shard writer spans), so ReadRanges never
// allocates and different ranges writing into disjoint sub-slices of the same backing array is
// safe — each worker touches only its own Dst.
type Range struct {
	Offset int64  // byte offset of the span within the source
	Dst    []byte // destination; exactly len(Dst) bytes are read at Offset
}

// ReadRanges serves every range from src concurrently, using at most parallelism workers, and
// returns the first read error (if any) or nil when all ranges were filled. It is the shared
// parallel range reader the remote weight sources of #3249 will fan their tensor spans through.
//
// CONTRACT.
//   - Byte-identity: each range's bytes are read with src.ReadAt(r.Dst, r.Offset), exactly as
//     a sequential loop would. io.ReaderAt requires ReadAt be safe for concurrent callers and
//     to never retain or mutate the input slice beyond the call, and distinct ranges own
//     distinct Dst buffers, so the parallel fill is byte-for-byte the sequential fill. A short
//     read (n < len(Dst) with nil error) is treated as an error, so a truncated source fails
//     loudly instead of leaving a partially-filled buffer to be mistaken for good weights.
//   - First-error-cancels: the first failing worker records its error and closes a shared abort
//     channel; every worker checks abort before claiming another range, so no NEW read starts
//     after the first failure. In-flight reads are allowed to finish (ReadAt is not
//     interruptible), then ReadRanges returns that first error. This is the errgroup semantic
//     without the dependency.
//   - parallelism <= 0 is treated as 1 (a plain sequential fill); more workers than ranges is
//     capped to len(ranges). An empty range set is a no-op success.
func ReadRanges(src io.ReaderAt, ranges []Range, parallelism int) error {
	n := len(ranges)
	if n == 0 {
		return nil
	}
	if parallelism < 1 {
		parallelism = 1
	}
	if parallelism > n {
		parallelism = n
	}

	// cursor hands out the next range index atomically; workers race to claim indices so the
	// set is drained regardless of per-range cost (a big tensor span next to a small one does
	// not stall a worker on a fixed stripe). abort is closed exactly once by the first failing
	// worker; firstErr is published under errOnce so the returned error is deterministically the
	// first failure, not whichever worker happened to write last.
	var (
		cursor   int64
		firstErr error
		errOnce  sync.Once
		abort    = make(chan struct{})
		wg       sync.WaitGroup
	)
	fail := func(err error) {
		errOnce.Do(func() {
			firstErr = err
			close(abort)
		})
	}

	wg.Add(parallelism)
	for w := 0; w < parallelism; w++ {
		go func() {
			defer wg.Done()
			for {
				// Stop claiming new work once any worker has failed. Checked BEFORE the
				// cursor bump so a cancelled run does no further reads than it must.
				select {
				case <-abort:
					return
				default:
				}
				i := atomic.AddInt64(&cursor, 1) - 1
				if i >= int64(n) {
					return
				}
				r := ranges[i]
				got, err := src.ReadAt(r.Dst, r.Offset)
				if err != nil {
					fail(fmt.Errorf("model: ReadRanges: read %d bytes at offset %d: %w", len(r.Dst), r.Offset, err))
					return
				}
				if got != len(r.Dst) {
					fail(fmt.Errorf("model: ReadRanges: short read at offset %d: got %d of %d bytes", r.Offset, got, len(r.Dst)))
					return
				}
			}
		}()
	}
	wg.Wait()
	return firstErr
}
