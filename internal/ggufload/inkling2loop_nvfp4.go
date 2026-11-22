package ggufload

import (
	"fmt"
	"runtime"
	"sync"
)

// repackInkling2NVFP4 converts rows of ggml NVFP4 blocks into compute.NewNVFP4's
// layout: packed E2M1 [rows, cols/2] with even K in the low nibble, then one
// E4M3FN scale byte per 16 values. ggml's doubled E2M1 table and halved UE4M3
// scale cancel, so codes and scale bits carry over unchanged and the global
// scale is 1.
func repackInkling2NVFP4(raw []byte, rows, cols int) (packed, scales []byte, err error) {
	if rows <= 0 || cols <= 0 || cols%qkNVFP4 != 0 {
		return nil, nil, fmt.Errorf("inkling2_loop: NVFP4 repack of [%d %d] needs cols divisible by %d", rows, cols, qkNVFP4)
	}
	blocks := rows * (cols / qkNVFP4)
	if len(raw) != blocks*blockNVFP4Bytes {
		return nil, nil, fmt.Errorf("inkling2_loop: NVFP4 repack has %d bytes, want %d", len(raw), blocks*blockNVFP4Bytes)
	}
	const subs = qkNVFP4 / qkNVFP4Sub
	packed = make([]byte, blocks*qkNVFP4/2)
	scales = make([]byte, blocks*subs)
	// A whole expert stack is ~10^7 blocks; split it across cores.
	workers := max(1, min(runtime.GOMAXPROCS(0), blocks/4096))
	errs := make([]error, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			errs[w] = repackInkling2NVFP4Blocks(raw, packed, scales, lo, hi)
		}(blocks*w/workers, blocks*(w+1)/workers)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, nil, err
		}
	}
	return packed, scales, nil
}

func repackInkling2NVFP4Blocks(raw, packed, scales []byte, lo, hi int) error {
	const subs = qkNVFP4 / qkNVFP4Sub
	const half = qkNVFP4Sub / 2
	for b := lo; b < hi; b++ {
		blk := raw[b*blockNVFP4Bytes : (b+1)*blockNVFP4Bytes]
		qs := blk[subs:]
		for s := 0; s < subs; s++ {
			sc, err := inkling2UE4M3ToE4M3FN(blk[s])
			if err != nil {
				return fmt.Errorf("inkling2_loop: NVFP4 block %d sub-block %d: %w", b, s, err)
			}
			scales[b*subs+s] = sc
			src := qs[s*half : (s+1)*half]
			dst := packed[(b*subs+s)*half : (b*subs+s+1)*half]
			for i := 0; i < half; i++ {
				dst[i] = inkling2NVFP4Nibble(src, 2*i) | inkling2NVFP4Nibble(src, 2*i+1)<<4
			}
		}
	}
	return nil
}

// inkling2NVFP4Nibble reads element e of a ggml 16-value sub-block, whose byte
// j holds element j low and element j+8 high.
func inkling2NVFP4Nibble(src []byte, e int) byte {
	if e < qkNVFP4Sub/2 {
		return src[e] & 0x0F
	}
	return src[e-qkNVFP4Sub/2] >> 4
}

// inkling2UE4M3ToE4M3FN maps a ggml UE4M3 scale byte to the E4M3FN byte that
// decodes to the same magnitude. ggml ignores the sign bit and reads 0x7F as
// zero; 0xFF decodes to 480 there, which E4M3FN cannot hold.
func inkling2UE4M3ToE4M3FN(x byte) (byte, error) {
	if x == 0x7F {
		return 0, nil
	}
	if s := x & 0x7F; s != 0x7F {
		return s, nil
	}
	return 0, fmt.Errorf("UE4M3 scale 0x%02X has no E4M3FN equivalent", x)
}
