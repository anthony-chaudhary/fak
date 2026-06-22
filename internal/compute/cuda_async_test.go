//go:build cuda

package compute

import "testing"

// cuda_async_test.go — the `-tags cuda` witness for issue #482 (async CUDA backend: ops
// enqueue on g_stream and return Buffers with Ready()==false until fenced; Read/Argmax are the
// ONLY host fences; Argmax runs ON-DEVICE and returns just the token id; Caps.Async=true). It
// is the device twin of the parity idea in cuda_test.go, but the comparison is SYNCHRONOUS-
// device-path vs ASYNC-device-path on the SAME backend, proving two things per token:
//
//	1. PARITY — the greedy-decode token id from the async path (on-device Argmax over the
//	   device-resident logits) equals the id from the synchronous path (Read the full logits
//	   vector host-ward, then host argmax) for every prompt + generated token.
//	2. NO FULL-LOGITS HOST COPY — across an async step only the argmax id (one int) crosses the
//	   bus: the device->host byte counter reads sizeof(int), not vocab*4, AND the logits Buffer
//	   is Ready()==false (device-resident, unfenced) right up until the Argmax fence, then flips
//	   to Ready() once Argmax has drained the stream.
//
// Skips cleanly when no CUDA device is registered. The win32 dev host has no CUDA toolkit; the
// actual RUN is the residual handed to a GPU node via tools/run_482_acceptance_on_gpu.sh.

// idBytes is what fcuda_argmax_f32 copies host-ward per token: one C int (the token id). C int
// is 4 bytes on every target platform (LP64 / LLP64), and the witness asserts exactly this.
const idBytes uint64 = 4

func asyncWitnessCfg() (synthCfg, map[string][]float32) {
	cfg := synthCfg{H: 64, L: 3, nH: 8, nKV: 2, hd: 8, I: 172, vocab: 96, eps: 1e-5, theta: 10000}
	return cfg, synthHostWeights(cfg)
}

func cudaTBOrSkip(tb testing.TB) *cudaBackend {
	cb, ok := Pick("cuda").(*cudaBackend)
	if !ok {
		tb.Skip("cuda backend not registered (no reachable CUDA device)")
	}
	return cb
}

// TestCUDAAsyncArgmaxParityAndResidency is the #482 acceptance witness (see file header).
func TestCUDAAsyncArgmaxParityAndResidency(t *testing.T) {
	cb := cudaOrSkip(t)
	if !cb.Caps().Async {
		t.Fatalf("#482: cuda backend must advertise Caps.Async=true (got Async=false)")
	}
	cfg, host := asyncWitnessCfg()
	mSync := newSynth(cb, cfg, host)  // synchronous device path: Read logits + host argmax
	mAsync := newSynth(cb, cfg, host) // async path: on-device Argmax; logits never Read
	wantFull := uint64(cfg.vocab) * 4 // a full-logits Read crosses vocab f32 host-ward

	step := func(tag string, tok int) int {
		// synchronous device path: the full logits vector crosses host-ward, host picks argmax.
		cb.ResetHostXfer()
		lSync := mSync.step(tok)
		syncBytes := cb.HostXferBytes()
		idSync := argmaxF32(lSync)

		// async device path: logits stay device-resident; on-device Argmax; only the id crosses.
		cb.ResetHostXfer()
		ldev := mAsync.stepDev(tok)
		readyPre := ldev.Ready()
		_, hostable := cb.Host(ldev)
		idAsync := cb.Argmax(ldev)
		asyncBytes := cb.HostXferBytes()
		readyPost := ldev.Ready()

		// (1) parity: the on-device Argmax id matches the synchronous Read+host-argmax id.
		if idAsync != idSync {
			t.Fatalf("%s: async on-device Argmax id=%d != synchronous device-path id=%d", tag, idAsync, idSync)
		}
		// (2) async contract: the logits are device-resident and unready BEFORE the fence...
		if readyPre {
			t.Errorf("%s: logits Ready()==true before the Argmax fence — async Buffers must be unready until fenced", tag)
		}
		if hostable {
			t.Errorf("%s: device logits are host-addressable — a resident tensor's Host() must be (nil,false)", tag)
		}
		// ...and ready AFTER it (the fence drained the stream that produced them).
		if !readyPost {
			t.Errorf("%s: logits Ready()==false after the Argmax fence — the fence must materialize prior work", tag)
		}
		// (2) only the token id crosses on the async path; the full vector crosses on the sync path.
		if asyncBytes != idBytes {
			t.Errorf("%s: async step copied %d B device->host; want exactly the token id (%d B), not the full logits vector", tag, asyncBytes, idBytes)
		}
		if syncBytes != wantFull {
			t.Errorf("%s: synchronous step copied %d B device->host; want the full logits vector (%d B)", tag, syncBytes, wantFull)
		}
		return idSync
	}

	prompt := []int{5, 17, 42, 3, 88, 11}
	var next int
	for i, id := range prompt {
		next = step("prompt"+itoaC(i), id)
	}
	const nGen = 8
	for s := 0; s < nGen; s++ {
		next = step("gen"+itoaC(s), next)
	}
	t.Logf("#482 async witness: %d prompt + %d greedy steps, async==sync ids; async D2H=%d B/token (id only) vs sync %d B/token (full logits, %dx); device=%s tier=%s class=%s",
		len(prompt), nGen, idBytes, wantFull, wantFull/idBytes, cb.Name(), cb.Tier(), cb.Class())
}

// benchCtxCap bounds the KV context during the decode benchmarks so attention cost stays at
// steady state (O(ctx) per token, not O(b.N)) and VRAM does not grow without limit — both
// benchmarks use the same cap so the sync-vs-async tok/s delta is apples-to-apples.
const benchCtxCap = 256

// resetKV frees the grown KV buffers and starts a fresh empty cache at pos 0 (cheap: the
// non-graph cudaKV preallocates nothing). Used by the benchmarks to bound context growth.
func (m *synthModel) resetKV() {
	if f, ok := m.kv.(interface{ Free() }); ok {
		f.Free()
	}
	c := m.cfg
	m.kv = m.be.NewKV(KVConfig{NumLayers: c.L, NumKVHeads: c.nKV, HeadDim: c.hd, RopeTheta: c.theta})
	m.pos = 0
}

// BenchmarkCUDASyncDecode measures greedy decode with the SYNCHRONOUS device path: the full
// logits vector is Read host-ward each token, then the host picks the argmax. Recycle() at the
// token boundary keeps allocation at steady state, as the model loop does.
func BenchmarkCUDASyncDecode(b *testing.B) {
	cb := cudaTBOrSkip(b)
	cfg, host := asyncWitnessCfg()
	m := newSynth(cb, cfg, host)
	next := argmaxF32(m.step(5)) // warm: populate buffer pool + weight cache
	cb.Recycle()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		next = argmaxF32(m.step(next))
		cb.Recycle()
		if m.pos >= benchCtxCap {
			b.StopTimer()
			m.resetKV()
			b.StartTimer()
		}
	}
	b.StopTimer()
	_ = next
}

// BenchmarkCUDAAsyncDecode measures greedy decode with the ASYNC device path: the logits stay
// device-resident and on-device Argmax returns just the token id (no full-logits Read). The
// sync-vs-async tok/s delta is what tools/run_482_acceptance_on_gpu.sh reports.
func BenchmarkCUDAAsyncDecode(b *testing.B) {
	cb := cudaTBOrSkip(b)
	cfg, host := asyncWitnessCfg()
	m := newSynth(cb, cfg, host)
	next := cb.Argmax(m.stepDev(5)) // warm
	cb.Recycle()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		next = cb.Argmax(m.stepDev(next))
		cb.Recycle()
		if m.pos >= benchCtxCap {
			b.StopTimer()
			m.resetKV()
			b.StartTimer()
		}
	}
	b.StopTimer()
	_ = next
}
