//go:build cuda && cgo

package compute

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"testing"
	"unsafe"
)

// One resident witness compares the candidate to the exact retained seven-launch
// CUDA composition, including prefix/suffix sentinels and publication metadata.
// FAK_CUDA_GDN_REQUIRED=1 makes absent hardware fail rather than skip.
// fak-test:runtime integration est=5s lane=optin
func TestCUDAQwen35AttentionPostprocessMatchesComposition(t *testing.T) {
	be := cudaGDNBackend(t)
	if !cudaQwen35AttentionPostprocessEnabled {
		if os.Getenv(cudaGDNRequiredEnv) == "1" {
			t.Fatal("set FAK_CUDA_QWEN35_ATTN_POSTPROCESS=1 to witness the selected candidate")
		}
		t.Skip("experimental attention postprocess disabled")
	}
	t.Cleanup(be.Recycle)
	for _, tc := range []struct{ tokens, start, rotary int }{{1, 0, 2}, {7, 5, 64}, {128, 0, 64}, {129, 5, 256}} {
		t.Run(fmt.Sprintf("P%d-start%d-rope%d", tc.tokens, tc.start, tc.rotary), func(t *testing.T) {
			const hd, qw, kw = 256, 24 * 256, 4 * 256
			const eps, theta = float32(1e-6), float64(1e7)
			rng := qwen35PanelLCG(0x0616c0da)
			upload := func(shape []int, values []float32) Tensor {
				x := be.UploadClass(NewF32(Default(), shape, values), F32, MemoryScratchpad, "qwen35-postprocess-witness")
				t.Cleanup(func() { be.Free(x) })
				return x
			}
			q := upload([]int{tc.tokens, qw}, rng.values(tc.tokens*qw, 0.5))
			k := upload([]int{tc.tokens, kw}, rng.values(tc.tokens*kw, 0.5))
			v := upload([]int{tc.tokens, kw}, rng.values(tc.tokens*kw, 0.5))
			qn := upload([]int{hd}, rng.values(hd, 1))
			kn := upload([]int{hd}, rng.values(hd, 1))
			capacity := (tc.start + tc.tokens + 3) * kw
			sentinel := make([]float32, capacity)
			for i := range sentinel {
				sentinel[i] = -12345.75
			}
			makeKV := func() (*cudaKV, []Tensor) {
				buffers := []Tensor{upload([]int{capacity}, sentinel), upload([]int{capacity}, sentinel), upload([]int{capacity}, sentinel)}
				kv := &cudaKV{be: be, cfg: KVConfig{NumLayers: 1, NumKVHeads: 4, HeadDim: hd}, Kraw: make([]dslice, 1), K: make([]dslice, 1), V: make([]dslice, 1)}
				for i, dst := range []*dslice{&kv.Kraw[0], &kv.K[0], &kv.V[0]} {
					*dst = dslice{ptr: buffers[i].buf.(*cudaBuf).ptr, len: tc.start * kw, cap: capacity}
				}
				for pos := 0; pos < tc.start; pos++ {
					kv.pos = append(kv.pos, pos)
				}
				return kv, buffers // uploads, not kv.Free, own these test allocations
			}
			baseline, wantCache := makeKV()
			candidate, gotCache := makeKV()
			var want, got Tensor
			var launchDelta uint64
			be.ResetHostXfer()
			be.ResetH2DXfer()
			err := func() error {
				cudaMu.Lock()
				defer cudaMu.Unlock()
				qNorm, _, err := be.qwen35SequenceRMSNormLocked(q, qn, tc.tokens*24, eps, "postprocess-oracle-q")
				if err != nil {
					return err
				}
				kNorm, _, err := be.qwen35SequenceRMSNormLocked(k, kn, tc.tokens*4, eps, "postprocess-oracle-k")
				if err != nil {
					return err
				}
				var kr Tensor
				want, _, kr, _, err = be.qwen35SequenceRoPELocked(qNorm, kNorm, tc.tokens, tc.start, 24, 4, hd, tc.rotary, theta)
				if err != nil {
					return err
				}
				if err = be.qwen35SequenceAppendKVLocked(baseline, 0, kNorm, kr, v, tc.start, tc.tokens); err != nil {
					return err
				}
				// An output/output alias must refuse before any enqueue or metadata write.
				original := candidate.Kraw[0]
				candidate.Kraw[0] = candidate.K[0]
				beforeRefusal := cudaQwen35AttentionPostprocessLaunchesLocked()
				_, _, selected, aliasErr := be.qwen35SequenceAttentionPostprocessLocked(q, k, v, qn, kn, candidate, 0, tc.tokens, tc.start, tc.rotary, eps, theta)
				candidate.Kraw[0] = original
				if !selected || aliasErr == nil || cudaQwen35AttentionPostprocessLaunchesLocked() != beforeRefusal || candidate.Len() != tc.start {
					return fmt.Errorf("alias refusal failed: %v", aliasErr)
				}
				before := cudaQwen35AttentionPostprocessLaunchesLocked()
				var used bool
				got, _, used, err = be.qwen35SequenceAttentionPostprocessLocked(q, k, v, qn, kn, candidate, 0, tc.tokens, tc.start, tc.rotary, eps, theta)
				if err != nil {
					return err
				}
				if !used {
					return fmt.Errorf("candidate declined on acceptance device")
				}
				launchDelta = cudaQwen35AttentionPostprocessLaunchesLocked() - before
				return be.qwen35SequenceSyncLocked()
			}()
			if err != nil {
				t.Fatal(err)
			}
			if launchDelta != 1 {
				t.Fatalf("candidate enqueues=%d, want 1", launchDelta)
			}
			if be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
				t.Fatal("postprocess staged through the host")
			}
			for i, pair := range [][2]Tensor{{want, got}, {wantCache[0], gotCache[0]}, {wantCache[1], gotCache[1]}, {wantCache[2], gotCache[2]}} {
				a, b := be.Read(pair[0]), be.Read(pair[1])
				if len(a) != len(b) {
					t.Fatalf("buffer %d length mismatch", i)
				}
				for j := range a {
					if math.IsNaN(float64(b[j])) || math.IsInf(float64(b[j]), 0) || math.Float32bits(a[j]) != math.Float32bits(b[j]) {
						t.Fatalf("buffer %d element %d: baseline %08x candidate %08x", i, j, math.Float32bits(a[j]), math.Float32bits(b[j]))
					}
				}
			}
			if candidate.Len() != tc.start+tc.tokens || candidate.Kraw[0].len != baseline.Kraw[0].len || candidate.K[0].len != baseline.K[0].len || candidate.V[0].len != baseline.V[0].len {
				t.Fatal("KV publication differs from composition")
			}
		})
	}
}

// Additive typed witness; the original F32 acceptance test above stays intact.
// All cache allocations below have their actual dtype byte size. Raw copies are
// made only after the completion fence, outside the no-host-transfer interval.
// fak-test:runtime integration est=10s lane=optin
func TestCUDAQwen35AttentionPostprocessTypedMatchesComposition(t *testing.T) {
	be := cudaGDNBackend(t)
	if !cudaQwen35AttentionPostprocessEnabled {
		if os.Getenv(cudaGDNRequiredEnv) == "1" {
			t.Fatal("set FAK_CUDA_QWEN35_ATTN_POSTPROCESS=1 to witness the selected candidate")
		}
		t.Skip("experimental attention postprocess disabled")
	}
	t.Cleanup(be.Recycle)
	for _, storage := range []struct {
		dtype     Dtype
		precision KVPrecision
	}{{F16, KVPrecisionF16}, {BF16, KVPrecisionBF16}} {
		for _, tc := range []struct{ tokens, start, rotary int }{{1, 0, 2}, {7, 5, 64}, {128, 0, 64}, {129, 5, 256}} {
			t.Run(fmt.Sprintf("%s/P%d-start%d-rope%d", storage.dtype, tc.tokens, tc.start, tc.rotary), func(t *testing.T) {
				const hd, qw, kw = 256, 6144, 1024
				const eps, theta = float32(1e-6), float64(1e7)
				rng := qwen35PanelLCG(0x0616c0da)
				upload := func(shape []int, values []float32) Tensor {
					x := be.UploadClass(NewF32(Default(), shape, values), F32, MemoryScratchpad, "qwen35-postprocess-typed-witness")
					t.Cleanup(func() { be.Free(x) })
					return x
				}
				q := upload([]int{tc.tokens, qw}, rng.values(tc.tokens*qw, 0.5))
				k := upload([]int{tc.tokens, kw}, rng.values(tc.tokens*kw, 0.5))
				values := rng.values(tc.tokens*kw, 0.5)
				// Half and BF16 even/odd ties and signed zero travel through V's
				// real conversion path without normalization obscuring their bits.
				for i, bits := range []uint32{0x80000000, 0, 0x3f801000, 0x3f803000, 0x3f808000, 0x3f818000} {
					values[i] = math.Float32frombits(bits)
				}
				v := upload([]int{tc.tokens, kw}, values)
				qn := upload([]int{hd}, rng.values(hd, 1))
				kn := upload([]int{hd}, rng.values(hd, 1))
				rows := tc.start + tc.tokens + 3
				capacity := rows * kw
				sentinel := make([]float32, capacity)
				for i := range sentinel {
					sentinel[i] = -123.75
				}
				seed := upload([]int{rows, kw}, sentinel)
				allocate := func(n int, dtype Dtype) Tensor {
					cudaMu.Lock()
					b, err := be.dallocDeviceOnlyClass(n*dtype.Bytes(), MemoryKVCache, "postprocess-typed-witness-cache")
					cudaMu.Unlock()
					if err != nil {
						t.Fatal(err)
					}
					x := Tensor{Dtype: dtype, Shape: []int{n}, buf: b, be: be}
					t.Cleanup(func() { be.Free(x) })
					if b.n != n*dtype.Bytes() || b.managed {
						t.Fatal("cache allocation is not exact strict typed storage")
					}
					return x
				}
				makeKV := func() (*cudaKV, []Tensor) {
					buffers := []Tensor{allocate(capacity, F32), allocate(capacity, storage.dtype), allocate(capacity, storage.dtype)}
					kv := &cudaKV{be: be, cfg: KVConfig{NumLayers: 1, NumKVHeads: 4, HeadDim: hd, Precision: storage.precision}, Kraw: make([]dslice, 1), K: make([]dslice, 1), V: make([]dslice, 1)}
					for i, dst := range []*dslice{&kv.Kraw[0], &kv.K[0], &kv.V[0]} {
						*dst = dslice{ptr: buffers[i].buf.(*cudaBuf).ptr, cap: capacity, dtype: buffers[i].Dtype}
					}
					cudaMu.Lock()
					err := be.qwen35SequenceAppendKVLocked(kv, 0, seed, seed, seed, 0, rows)
					if err == nil {
						err = be.qwen35SequenceSyncLocked()
					}
					cudaMu.Unlock()
					if err != nil {
						t.Fatal(err)
					}
					for _, dst := range []*dslice{&kv.Kraw[0], &kv.K[0], &kv.V[0]} {
						dst.len = tc.start * kw
					}
					kv.pos = kv.pos[:tc.start]
					return kv, buffers
				}
				baseline, wantCache := makeKV()
				candidate, gotCache := makeKV()
				readBytes := func(x Tensor) []byte {
					b := x.buf.(*cudaBuf)
					out := make([]byte, x.Numel()*x.Dtype.Bytes())
					cudaMu.Lock()
					status := cudaReadNative.copy(b.device, unsafe.Pointer(&out[0]), b.ptr, uintptr(len(out)))
					cudaMu.Unlock()
					if status != 0 {
						t.Fatalf("completed cache raw read: CUDA status %d", status)
					}
					return out
				}
				beforeBytes := [][]byte{readBytes(gotCache[0]), readBytes(gotCache[1]), readBytes(gotCache[2])}
				short := allocate((tc.start+tc.tokens)*kw-1, storage.dtype)
				shortBefore := readBytes(short)
				// Every refusal must leave the exact bytes, frontier, positions and
				// launch count untouched. None is replayed as a successful operation.
				for _, failure := range []string{"mixed", "config", "shadow", "unsupported-dtype", "short-bytes", "frontier", "alias", "allocation", "capture", "unsupported-precision"} {
					cudaMu.Lock()
					oldRaw, oldK, oldV, oldPrecision := candidate.Kraw[0], candidate.K[0], candidate.V[0], candidate.cfg.Precision
					switch failure {
					case "mixed":
						candidate.V[0].dtype = F32
					case "config":
						candidate.cfg.Precision = KVPrecisionF32
					case "shadow":
						candidate.Kraw[0].dtype = storage.dtype
					case "unsupported-dtype":
						candidate.K[0].dtype, candidate.V[0].dtype = Q8_0, Q8_0
					case "short-bytes":
						candidate.V[0].ptr = short.buf.(*cudaBuf).ptr
					case "frontier":
						candidate.K[0].len++
					case "alias":
						candidate.V[0].ptr = candidate.K[0].ptr
					case "allocation":
						qwen35GDNStrictAllocFailAfter = 0
					case "capture":
						be.capturing = true
					case "unsupported-precision":
						candidate.cfg.Precision = KVPrecisionQ8
					}
					before := cudaQwen35AttentionPostprocessLaunchesLocked()
					out, outBuf, used, err := be.qwen35SequenceAttentionPostprocessLocked(q, k, v, qn, kn, candidate, 0, tc.tokens, tc.start, tc.rotary, eps, theta)
					after := cudaQwen35AttentionPostprocessLaunchesLocked()
					// Observe publication before restoring intentionally malformed metadata.
					positions := len(candidate.pos)
					lengths := []int{candidate.Kraw[0].len, candidate.K[0].len, candidate.V[0].len}
					var shortAfter []byte
					shortCopyStatus := 0
					if failure == "short-bytes" {
						shortAfter = make([]byte, len(shortBefore))
						shortCopyStatus = cudaReadNative.copy(0, unsafe.Pointer(&shortAfter[0]), candidate.V[0].ptr, uintptr(len(shortAfter)))
					}
					candidate.Kraw[0], candidate.K[0], candidate.V[0], candidate.cfg.Precision = oldRaw, oldK, oldV, oldPrecision
					be.capturing, qwen35GDNStrictAllocFailAfter = false, -1
					cudaMu.Unlock()
					decline := failure == "capture" || failure == "unsupported-precision"
					if used == decline || (err == nil) != decline || out.Buf() != nil || outBuf != nil || after != before || positions != tc.start {
						t.Fatalf("%s refusal: used=%v err=%v launches=%d/%d positions=%d", failure, used, err, before, after, positions)
					}
					for i, n := range lengths {
						want := tc.start * kw
						if failure == "frontier" && i == 1 {
							want++
						}
						if n != want {
							t.Fatalf("%s published plane %d length=%d", failure, i, n)
						}
						if !bytes.Equal(readBytes(gotCache[i]), beforeBytes[i]) {
							t.Fatalf("%s changed plane %d bytes", failure, i)
						}
					}
					if failure == "short-bytes" && (shortCopyStatus != 0 || !bytes.Equal(shortAfter, shortBefore)) {
						t.Fatalf("short-bytes target changed before metadata restoration (copy status %d)", shortCopyStatus)
					}
				}
				var want, got, wantAttention, gotAttention Tensor
				var launchDelta uint64
				be.ResetHostXfer()
				be.ResetH2DXfer()
				err := func() error {
					cudaMu.Lock()
					defer cudaMu.Unlock()
					qNorm, _, err := be.qwen35SequenceRMSNormLocked(q, qn, tc.tokens*24, eps, "typed-postprocess-oracle-q")
					if err != nil {
						return err
					}
					kNorm, _, err := be.qwen35SequenceRMSNormLocked(k, kn, tc.tokens*4, eps, "typed-postprocess-oracle-k")
					if err != nil {
						return err
					}
					var kr Tensor
					want, _, kr, _, err = be.qwen35SequenceRoPELocked(qNorm, kNorm, tc.tokens, tc.start, 24, 4, hd, tc.rotary, theta)
					if err != nil {
						return err
					}
					if err = be.qwen35SequenceAppendKVLocked(baseline, 0, kNorm, kr, v, tc.start, tc.tokens); err != nil {
						return err
					}
					before := cudaQwen35AttentionPostprocessLaunchesLocked()
					var used bool
					got, _, used, err = be.qwen35SequenceAttentionPostprocessLocked(q, k, v, qn, kn, candidate, 0, tc.tokens, tc.start, tc.rotary, eps, theta)
					if err != nil {
						return err
					}
					if !used {
						return fmt.Errorf("typed candidate declined on acceptance device")
					}
					launchDelta = cudaQwen35AttentionPostprocessLaunchesLocked() - before
					wantAttention, _, err = be.qwen35SequenceAttentionLocked(want, baseline, 0, tc.tokens, tc.start, 24, 1.0/16)
					if err != nil {
						return err
					}
					gotAttention, _, err = be.qwen35SequenceAttentionLocked(got, candidate, 0, tc.tokens, tc.start, 24, 1.0/16)
					if err != nil {
						return err
					}
					return be.qwen35SequenceSyncLocked()
				}()
				if err != nil {
					t.Fatal(err)
				}
				if launchDelta != 1 || be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
					t.Fatal("typed fusion did not perform exactly one resident enqueue")
				}
				for i := range gotCache {
					gotBytes := readBytes(gotCache[i])
					if !bytes.Equal(gotBytes, readBytes(wantCache[i])) {
						t.Fatalf("plane %d raw byte mismatch", i)
					}
					width := gotCache[i].Dtype.Bytes() * kw
					if !bytes.Equal(gotBytes[:tc.start*width], beforeBytes[i][:tc.start*width]) || !bytes.Equal(gotBytes[(tc.start+tc.tokens)*width:], beforeBytes[i][(tc.start+tc.tokens)*width:]) {
						t.Fatalf("plane %d prefix/suffix changed", i)
					}
				}
				for i, pair := range [][2]Tensor{{want, got}, {wantCache[0], gotCache[0]}, {wantAttention, gotAttention}} {
					a, b := be.Read(pair[0]), be.Read(pair[1])
					if len(a) != len(b) {
						t.Fatal("F32 comparison length mismatch")
					}
					for j := range a {
						if math.IsNaN(float64(a[j])) || math.IsInf(float64(a[j]), 0) || math.IsNaN(float64(b[j])) || math.IsInf(float64(b[j]), 0) || math.Float32bits(a[j]) != math.Float32bits(b[j]) {
							t.Fatalf("F32 buffer %d element %d: %08x != %08x (both must be finite)", i, j, math.Float32bits(a[j]), math.Float32bits(b[j]))
						}
					}
				}
				if candidate.Len() != tc.start+tc.tokens || len(candidate.pos) != len(baseline.pos) {
					t.Fatal("typed frontier differs")
				}
				for i := range candidate.pos {
					if candidate.pos[i] != baseline.pos[i] {
						t.Fatal("typed positions differ")
					}
				}
				for _, dst := range []*dslice{&candidate.Kraw[0], &candidate.K[0], &candidate.V[0]} {
					if dst.len != (tc.start+tc.tokens)*kw {
						t.Fatal("typed plane frontier differs")
					}
				}
				if tc.tokens == 1 {
					// Storage permits nonfinite V just like the retained typed append.
					// Compare exact NaN/Inf bytes, but do not claim finite attention for
					// these intentionally nonfinite inputs or feed them to that oracle.
					nonfinite := append([]float32(nil), values...)
					for i, bits := range []uint32{0x7f800001, 0xffc12345, 0x7f800000, 0xff800000} {
						nonfinite[i] = math.Float32frombits(bits)
					}
					nv := upload([]int{tc.tokens, kw}, nonfinite)
					nbase, nbuffers := makeKV()
					ncandidate, ncbuffers := makeKV()
					err := func() error {
						cudaMu.Lock()
						defer cudaMu.Unlock()
						qNorm, _, err := be.qwen35SequenceRMSNormLocked(q, qn, 24, eps, "typed-nonfinite-oracle-q")
						if err != nil {
							return err
						}
						kNorm, _, err := be.qwen35SequenceRMSNormLocked(k, kn, 4, eps, "typed-nonfinite-oracle-k")
						if err != nil {
							return err
						}
						_, _, kr, _, err := be.qwen35SequenceRoPELocked(qNorm, kNorm, 1, tc.start, 24, 4, hd, tc.rotary, theta)
						if err != nil {
							return err
						}
						if err = be.qwen35SequenceAppendKVLocked(nbase, 0, kNorm, kr, nv, tc.start, 1); err != nil {
							return err
						}
						_, _, used, err := be.qwen35SequenceAttentionPostprocessLocked(q, k, nv, qn, kn, ncandidate, 0, 1, tc.start, tc.rotary, eps, theta)
						if err != nil {
							return err
						}
						if !used {
							return fmt.Errorf("nonfinite storage candidate declined")
						}
						return be.qwen35SequenceSyncLocked()
					}()
					if err != nil {
						t.Fatal(err)
					}
					narrow := readBytes(ncbuffers[2])
					if !bytes.Equal(narrow, readBytes(nbuffers[2])) {
						t.Fatal("nonfinite V bytes differ from typed append")
					}
					exponent, mantissa := uint16(0x7c00), uint16(0x03ff)
					if storage.dtype == BF16 {
						exponent, mantissa = 0x7f80, 0x007f
					}
					for i := 0; i < 4; i++ {
						u := uint16(narrow[2*i]) | uint16(narrow[2*i+1])<<8
						// CUDA half may canonicalize NaNs. Its exact bytes were compared
						// above with the retained intrinsic; BF16 preserves their sign.
						checkSign := i >= 2 || storage.dtype == BF16
						if u&exponent != exponent || ((u&mantissa != 0) != (i < 2)) || (checkSign && (u&0x8000 != 0) != (i == 1 || i == 3)) {
							t.Fatalf("nonfinite value %d lost NaN/Inf/sign: %04x", i, u)
						}
					}
				}
			})
		}
	}
}
