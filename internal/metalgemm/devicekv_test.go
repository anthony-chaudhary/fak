//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"testing"
)

// TestProjectionGraphQwenDeviceKVParityAndWalk is the #13087 device-resident KV
// witness at the metalgemm layer. It SKIPS cleanly when Metal is unavailable, so
// it is a no-op on a non-GPU host and a real GPU check where a Metal device
// exists.
//
//   - "parity":      FullAttentionDevice's output/KRaw/KPost match the host path
//     (FullAttention with the SAME prefix seeded as a host slice) on a panel-shaped
//     multi-row encode, so moving the KV append onto the device is numerically
//     transparent.
//   - "panel_walk":  two panels append into ONE persistent DeviceKV pair. Panel 2
//     reads its prefix on the device — with NO host re-upload between panels — and
//     its attention output must match a host reference whose prefix is panel 1's
//     device-resident KPost/V rows. This proves the pair stays live across panels,
//     which is the property that removes the per-panel host readback + prefix
//     re-upload.
func TestProjectionGraphQwenDeviceKVParityAndWalk(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	const input, nH, nKV, hd, rotary = 256, 2, 1, 32, 16
	const qwidth, kvwidth = nH * hd, nKV * hd
	scale, qkEps := float32(1/math.Sqrt(float64(hd))), float32(1e-6)

	qgateWeight := UploadQ4K(q4kTestRaw(2*qwidth, input, 1308701), 2*qwidth, input)
	kWeight := UploadQ4K(q4kTestRaw(kvwidth, input, 1308702), kvwidth, input)
	vWeight := UploadQ4K(q4kTestRaw(kvwidth, input, 1308703), kvwidth, input)
	if qgateWeight == nil || kWeight == nil || vWeight == nil {
		t.Fatal("device-KV Q4_K upload")
	}
	qnorm, knorm := make([]float32, hd), make([]float32, hd)
	for i := range qnorm {
		qnorm[i], knorm[i] = 0.87+float32(i%7)*0.05, 0.91+float32(i%5)*0.04
	}
	prefixK, prefixV := make([]float32, 8*kvwidth), make([]float32, 8*kvwidth)
	for i := range prefixK {
		prefixK[i], prefixV[i] = float32((i*7)%29-14)*0.02, float32((i*11)%31-15)*0.017
	}

	// ropePhases builds deterministic per-row cos/sin for `rows` positions starting
	// at absolute position `base`.
	ropePhases := func(rows, base int) ([]float32, []float32) {
		cosv, sinv := make([]float32, rows*(rotary/2)), make([]float32, rows*(rotary/2))
		for row := 0; row < rows; row++ {
			for dim := 0; dim < rotary/2; dim++ {
				angle := float64((base+row+1)*(dim+1)) * 0.003
				cosv[row*(rotary/2)+dim], sinv[row*(rotary/2)+dim] = float32(math.Cos(angle)), float32(math.Sin(angle))
			}
		}
		return cosv, sinv
	}
	// encodePanel runs one P-row panel through `g`, seeding `prefixK`/`prefixV` at
	// `base` on the chosen path, and returns the packed terminal readback.
	encodePanel := func(g *ProjectionGraph, x []float32, rows, base int, kv *DeviceKV, hostPrefix bool) []float32 {
		t.Helper()
		q, err := g.EncodeQ4K(qgateWeight)
		if err != nil {
			t.Fatal(err)
		}
		k, err := g.EncodeQ4K(kWeight)
		if err != nil {
			t.Fatal(err)
		}
		v, err := g.EncodeQ4K(vWeight)
		if err != nil {
			t.Fatal(err)
		}
		q2, gate, err := g.SplitGatedQ(q, qwidth, hd)
		if err != nil {
			t.Fatal(err)
		}
		cosv, sinv := ropePhases(rows, base)
		var att Qwen35GraphAttentionResult
		if hostPrefix {
			att, err = g.FullAttention(q2, k, v, gate, qnorm, knorm, cosv, sinv, prefixK, prefixV, base, nH, nKV, hd, rotary, scale, qkEps, true, true)
		} else {
			att, err = g.FullAttentionDevice(q2, k, v, gate, kv, 0, qnorm, knorm, cosv, sinv, base, nH, nKV, hd, rotary, scale, qkEps, true, true)
		}
		if err != nil {
			t.Fatal(err)
		}
		out, _, err := g.FinishRead(att.Output, att.KRaw, att.KPost, att.V)
		if err != nil {
			t.Fatal(err)
		}
		// Flatten [Output, KRaw, KPost, V] so each slot is indexable by row*width.
		flat := make([]float32, 0, len(out[0])+3*len(out[1]))
		for _, s := range out {
			flat = append(flat, s...)
		}
		return flat
	}
	// The packed readback is [Output(rows*qwidth), KRaw, KPost, V(rows*kvwidth each)];
	// slot 0 is qwidth-wide and slots 1..3 are kvwidth-wide, which differ whenever
	// nH != nKV, so the offset of each slot must be accumulated, not multiplied.
	slot := func(flat []float32, which, rows int) []float32 {
		offset := 0
		if which > 0 {
			offset = rows * qwidth
		}
		n := rows * qwidth
		if which > 0 {
			n = rows * kvwidth
			offset += (which - 1) * n
		}
		return flat[offset : offset+n]
	}
	closeEnough := func(name string, got, ref []float32) {
		t.Helper()
		for i := range ref {
			if d := math.Abs(float64(got[i] - ref[i])); d > 1e-4 {
				t.Fatalf("%s[%d] got=%g want=%g delta=%g", name, i, got[i], ref[i], d)
			}
		}
	}
	// concat copies both inputs into a fresh slice. A bare append(a, b...) would
	// reuse a's spare capacity and alias the parent readback, so a later slot read
	// from the same parent would observe an earlier concatenation's writes.
	concat := func(parts ...[]float32) []float32 {
		n := 0
		for _, p := range parts {
			n += len(p)
		}
		out := make([]float32, 0, n)
		for _, p := range parts {
			out = append(out, p...)
		}
		return out
	}

	const rows, base = 32, 8
	x := q4kTestVector(rows*input, 1308710)

	t.Run("parity", func(t *testing.T) {
		gh, err := BeginProjectionGraph(x, nil, nil, rows, input)
		if err != nil {
			t.Fatal(err)
		}
		host := encodePanel(gh, x, rows, base, nil, true)
		gh.Free()

		kv := NewDeviceKV(1, base+rows, kvwidth)
		if kv == nil {
			t.Fatal("NewDeviceKV returned nil")
		}
		defer kv.Close()
		for side, src := range map[int][]float32{0: prefixK, 1: prefixK, 2: prefixV} {
			if err := kv.Upload(side, src); err != nil {
				t.Fatalf("upload side %d: %v", side, err)
			}
		}
		gd, err := BeginProjectionGraph(x, nil, nil, rows, input)
		if err != nil {
			t.Fatal(err)
		}
		dev := encodePanel(gd, x, rows, base, kv, false)
		gd.Free()
		closeEnough("attention output", slot(dev, 0, rows), slot(host, 0, rows))
		closeEnough("K raw", slot(dev, 1, rows), slot(host, 1, rows))
		closeEnough("K post", slot(dev, 2, rows), slot(host, 2, rows))
	})

	t.Run("panel_walk", func(t *testing.T) {
		// One persistent pair sized for two panels plus the seeded prefix.
		kv := NewDeviceKV(1, base+2*rows, kvwidth)
		if kv == nil {
			t.Fatal("NewDeviceKV returned nil")
		}
		defer kv.Close()
		for side, src := range map[int][]float32{0: prefixK, 1: prefixK, 2: prefixV} {
			if err := kv.Upload(side, src); err != nil {
				t.Fatalf("upload side %d: %v", side, err)
			}
		}
		// Panel 1 appends at base; panel 2 appends at base+rows reading the pair the
		// device already holds — no host re-upload between them.
		g1, err := BeginProjectionGraph(x, nil, nil, rows, input)
		if err != nil {
			t.Fatal(err)
		}
		panel1 := encodePanel(g1, x, rows, base, kv, false)
		g1.Free()

		// Host reference for panel 2: its prefix is the seeded prefix followed by
		// panel 1's device-resident KPost/V rows.
		refK := concat(prefixK, slot(panel1, 2, rows))
		refV := concat(prefixV, slot(panel1, 3, rows))
		gh, err := BeginProjectionGraph(x, nil, nil, rows, input)
		if err != nil {
			t.Fatal(err)
		}
		q, err := gh.EncodeQ4K(qgateWeight)
		if err != nil {
			t.Fatal(err)
		}
		k, err := gh.EncodeQ4K(kWeight)
		if err != nil {
			t.Fatal(err)
		}
		v, err := gh.EncodeQ4K(vWeight)
		if err != nil {
			t.Fatal(err)
		}
		q2, gate, err := gh.SplitGatedQ(q, qwidth, hd)
		if err != nil {
			t.Fatal(err)
		}
		cosv, sinv := ropePhases(rows, base+rows)
		att, err := gh.FullAttention(q2, k, v, gate, qnorm, knorm, cosv, sinv, refK, refV, base+rows, nH, nKV, hd, rotary, scale, qkEps, true, true)
		if err != nil {
			t.Fatal(err)
		}
		hostOut, _, err := gh.FinishRead(att.Output, att.KRaw, att.KPost, att.V)
		if err != nil {
			t.Fatal(err)
		}
		gh.Free()
		host2 := concat(hostOut[0], hostOut[1], hostOut[2], hostOut[3])

		g2, err := BeginProjectionGraph(x, nil, nil, rows, input)
		if err != nil {
			t.Fatal(err)
		}
		panel2 := encodePanel(g2, x, rows, base+rows, kv, false)
		g2.Free()
		closeEnough("panel2 attention output", slot(panel2, 0, rows), slot(host2, 0, rows))

		// The pair now holds BOTH panels' rows after the seeded prefix, on the
		// device, with no host round-trip in between.
		for _, side := range []struct {
			name string
			id   int
			host []float32
		}{
			// concat copies into a fresh slice: appending onto slot(...) would alias
			// the panel1 flat buffer's spare capacity and clobber the sibling slots.
			{name: "KRaw", id: 0, host: concat(slot(panel1, 1, rows), slot(panel2, 1, rows))},
			{name: "KPost", id: 1, host: concat(slot(panel1, 2, rows), slot(panel2, 2, rows))},
			{name: "V", id: 2, host: concat(slot(panel1, 3, rows), slot(panel2, 3, rows))},
		} {
			got := make([]float32, 2*rows*kvwidth)
			if err := kv.DownloadRegion(side.id, base*kvwidth, got); err != nil {
				t.Fatalf("download %s walk rows: %v", side.name, err)
			}
			closeEnough("device "+side.name, got, side.host)
		}
	})
}
