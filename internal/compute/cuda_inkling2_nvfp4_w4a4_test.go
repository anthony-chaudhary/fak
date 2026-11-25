//go:build cuda && cgo

package compute

import (
	"bytes"
	"math"
	"math/rand"
	"os"
	"testing"
)

// fak-test:runtime integration est=3s lane=optin
// The device activation quantizer must equal its host port byte for byte, and the
// native W4A4 grouped expert GEMM must equal a float64 dot of the dequantized weights
// against those activations (accumulation error only), and stay close to the
// fp32-activation product. The structured case uses one-hot weight rows (one product
// per output) with distinct per-row and per-block scales, including the edge codes 0,
// 0x01, 0x07 and 0x7E, and activations whose block magnitudes differ per token and
// block, so a wrong A/B/SFA/SFB lane mapping changes values outright.
func TestCUDAInk2PExpertNVFP4W4A4MatchesReference(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	if !cb.ink2pNVFP4W4A4Available() {
		if os.Getenv(cudaNVFP4RequireEnv) == "1" {
			t.Fatal("loaded CUDA image has no native NVFP4 W4A4 kernels; build with --arch sm_120f")
		}
		t.Skip("loaded CUDA image has no sm_120a/sm_120f NVFP4 W4A4 kernels; set " + cudaNVFP4RequireEnv + "=1 on HIL to fail instead of skip")
	}
	t.Setenv("FAK_INKLING2_PREFILL_OLD", "")
	r := rand.New(rand.NewSource(41))
	const nexp = 4
	for _, c := range []struct {
		name                        string
		rows, in, tokens, k         int
		perPair, scaled, structured bool
	}{
		{"gate_up shared x", 512, 256, 48, 2, false, true, false},
		{"down per-pair x", 256, 2304, 40, 2, true, false, false},
		{"structured one-hot", 256, 256, 44, 2, false, true, true},
	} {
		npairs := c.tokens * c.k
		// Expert 0 takes 36 tokens (two pair tiles), expert 3 none.
		ids := make([]int, npairs)
		for tok := 0; tok < c.tokens; tok++ {
			if tok < 36 {
				ids[tok*c.k], ids[tok*c.k+1] = 0, 1+r.Intn(2)
			} else {
				ids[tok*c.k], ids[tok*c.k+1] = 1, 2
			}
		}
		counts := make([]int, nexp)
		for _, id := range ids {
			counts[id]++
		}
		if counts[0] <= 32 || counts[3] != 0 {
			t.Fatalf("%s: routing counts %v", c.name, counts)
		}

		packed := make([]byte, nexp*c.rows*c.in/2)
		scales := make([]byte, nexp*c.rows*c.in/16)
		if c.structured {
			for o := 0; o < nexp*c.rows; o++ {
				k := (o*37 + o/c.rows*11) % c.in
				nib := byte(o%7 + 1)
				if o%2 == 1 {
					nib |= 8
				}
				packed[o*c.in/2+k/2] = nib << (4 * (k % 2))
				edge := []byte{0, 0x01, 0x07, 0x7e}
				for b := 0; b < c.in/16; b++ {
					sc := byte(0x18 + (o*7+b*3)%0x50)
					if o%9 < len(edge) {
						sc = edge[(o%9+b)%len(edge)]
					}
					scales[o*c.in/16+b] = sc
				}
			}
		} else {
			r.Read(packed)
			for i := range scales {
				scales[i] = byte(0x20 + r.Intn(0x30))
			}
		}
		stack := cudaNVFP4Resident(t, cb, []int{nexp * c.rows, c.in}, packed, scales, 1)
		w, err := DecodeNVFP4Reference(nexp*c.rows, c.in, packed, scales, 1)
		if err != nil {
			t.Fatal(err)
		}

		xDiv, xRows := c.k, c.tokens
		if c.perPair {
			xDiv, xRows = 1, npairs
		}
		x := make([]float32, xRows*c.in)
		for row := 0; row < xRows; row++ {
			for k := 0; k < c.in; k++ {
				// The fp32 row scale cancels a per-token factor; a per-token, per-block
				// exponent makes the activation codes themselves differ by token and block.
				e := (k/16)%5 - 2
				if c.structured {
					e = (row+k/16)%7 - 3
				}
				x[row*c.in+k] = float32(r.NormFloat64() * (1 + 0.25*float64(row)) * math.Ldexp(1, e))
			}
		}
		esc := make([]float32, nexp)
		for i := range esc {
			esc[i] = 0.5 + r.Float32()
		}
		idf := make([]float32, npairs)
		for i, id := range ids {
			idf[i] = math.Float32frombits(uint32(id))
		}
		alloc := func(n int) Tensor {
			tt, err := cb.Ink2Alloc(n)
			if err != nil {
				t.Fatal(err)
			}
			return tt
		}
		dx, dids, des, scratch := alloc(len(x)), alloc(npairs), alloc(nexp), alloc(Ink2PGroupScratch(npairs, nexp))
		for _, wr := range []struct {
			dst Tensor
			src []float32
		}{{dx, x}, {dids, idf}, {des, esc}} {
			if err := cb.Ink2Write(Ink2At(wr.dst, 0), wr.src); err != nil {
				t.Fatal(err)
			}
		}
		var scale Ink2Ref
		if c.scaled {
			scale = Ink2At(des, 0)
		}
		run := func(arm string) []float32 {
			t.Setenv("FAK_INKLING2_PREFILL_NVFP4_W4A4", arm)
			n := npairs * c.rows
			y := alloc(n)
			nan := make([]float32, n)
			for i := range nan {
				nan[i] = float32(math.NaN())
			}
			if err := cb.Ink2Write(Ink2At(y, 0), nan); err != nil {
				t.Fatal(err)
			}
			ok, err := cb.Ink2PExpertGrouped(stack, c.rows, nexp, Ink2At(dids, 0), Ink2At(dx, 0), xDiv, c.in, Ink2At(y, 0), npairs, scale, Ink2At(scratch, 0), false)
			if err != nil || !ok {
				t.Fatalf("%s arm %s: ok=%v err=%v", c.name, arm, ok, err)
			}
			out, err := cb.Ink2Read(Ink2At(y, 0), n)
			if err != nil {
				t.Fatal(err)
			}
			return out
		}
		got := run("1")
		devQ, devCodes, devRS, err := cb.ink2pNVFP4W4A4Quant(xRows, c.in)
		if err != nil {
			t.Fatal(err)
		}
		int8Arm := run("0")

		type quantRow struct {
			q, codes []byte
			rs       float32
		}
		qx := make([]quantRow, xRows)
		for row := range qx {
			qx[row].q, qx[row].codes, qx[row].rs = ink2pNVF4QuantRow(x[row*c.in : (row+1)*c.in])
			dq, dc := devQ[row*c.in/2:(row+1)*c.in/2], devCodes[row*c.in/16:(row+1)*c.in/16]
			if math.Float32bits(devRS[row]) != math.Float32bits(qx[row].rs) || !bytes.Equal(dc, qx[row].codes) || !bytes.Equal(dq, qx[row].q) {
				t.Fatalf("%s x row %d: device quantizer differs from host port: row scale %v vs %v, codes %x vs %x, nibbles %x vs %x",
					c.name, row, devRS[row], qx[row].rs, dc, qx[row].codes, dq, qx[row].q)
			}
		}
		// One product per output pins the structured case to fp32 rounding; the
		// mxf4nvf4 accumulator's internal precision over chained k64 steps is not
		// documented, so random cases get headroom and log the observed error.
		tol := 2e-4
		if c.structured {
			tol = 1e-6
		}
		var maxRel, num, den, num8 float64
		for i, id := range ids {
			xr := i / xDiv
			es := 1.0
			if c.scaled {
				es = float64(esc[id])
			}
			for o := 0; o < c.rows; o++ {
				wrow := w[(id*c.rows+o)*c.in : (id*c.rows+o+1)*c.in]
				var dot, mag, dot32 float64
				for k, wv := range wrow {
					p := float64(wv) * ink2pNVF4Dequant(qx[xr].q, qx[xr].codes, qx[xr].rs, k)
					dot += p
					mag += math.Abs(p)
					dot32 += float64(wv) * float64(x[xr*c.in+k])
				}
				dot, mag, dot32 = dot*es, mag*es, dot32*es
				g := float64(got[i*c.rows+o])
				if math.IsNaN(g) || math.Abs(g-dot) > tol*mag+1e-30 {
					t.Fatalf("%s pair %d (expert %d, x row %d) row %d: device %v, host %v (sum|w*xq| %v)", c.name, i, id, xr, o, g, dot, mag)
				}
				if mag > 0 {
					maxRel = math.Max(maxRel, math.Abs(g-dot)/mag)
				}
				num += (g - dot32) * (g - dot32)
				d8 := float64(int8Arm[i*c.rows+o]) - dot32
				num8 += d8 * d8
				den += dot32 * dot32
			}
		}
		frob, frob8 := math.Sqrt(num/den), math.Sqrt(num8/den)
		t.Logf("%s: max |dev-host|/sum|w*xq| %.3g; rel Frobenius vs fp32 activations: w4a4 %.4f, int8 arm %.4f", c.name, maxRel, frob, frob8)
		if !c.structured && frob >= 0.15 {
			t.Fatalf("%s: W4A4 relative Frobenius error vs fp32 activations %.4f >= 0.15", c.name, frob)
		}
	}
}
