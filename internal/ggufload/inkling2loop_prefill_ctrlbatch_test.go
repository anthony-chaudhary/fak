package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type ink2CtrlBatchCall struct {
	cur, msg, down, gain, forget []ink2Ref
	g, out                       ink2Ref
	n, rank, T                   int
}

// ink2CtrlBatchDev is a prefill device that only records Ink2PCtrlBatch calls.
type ink2CtrlBatchDev struct {
	compute.Inkling2PrefillDevice
	calls []ink2CtrlBatchCall
}

func (d *ink2CtrlBatchDev) Ink2PCtrlBatch(cur, msg, down, gain, forget []ink2Ref, g, out ink2Ref, n, rank, T int) error {
	d.calls = append(d.calls, ink2CtrlBatchCall{cur, msg, down, gain, forget, g, out, n, rank, T})
	return nil
}

type ink2NoBatchDev struct{ compute.Inkling2PrefillDevice }

func ink2CtrlBatchFixture(pd compute.Inkling2PrefillDevice) (*ink2Prefill, *ink2Ctrl) {
	const hidden, T = 64, 5
	s := &inkling2Resident{hp: model.Inkling2LoopHParams{Hidden: hidden}}
	p := &ink2Prefill{s: s, pd: pd, T: T, scrFloats: 1 << 16}
	c := &ink2Ctrl{rank: 32}
	for j := range c.down {
		c.down[j] = compute.Ink2At(compute.Tensor{}, 1000*(j+1))
		c.gain[j] = compute.Ink2At(compute.Tensor{}, 1000*(j+1)+100)
		c.forget[j] = compute.Ink2At(compute.Tensor{}, 1000*(j+1)+200)
	}
	return p, c
}

func ink2CtrlBatchOperands() (cur, msg []ink2Ref) {
	for j := 0; j < 3; j++ {
		cur = append(cur, compute.Ink2At(compute.Tensor{}, 10+j))
		msg = append(msg, compute.Ink2At(compute.Tensor{}, 20+j))
	}
	return cur, msg
}

func TestInkling2PrefillControlBatchIsOptIn(t *testing.T) {
	cur, msg := ink2CtrlBatchOperands()

	t.Setenv("FAK_INKLING2_PREFILL_CTRL_BATCH", "")
	dev := &ink2CtrlBatchDev{}
	p, c := ink2CtrlBatchFixture(dev)
	if _, ok := p.controlBatch(cur, msg, c); ok || len(dev.calls) != 0 {
		t.Fatalf("unset toggle: ok=%v calls=%d, want the three per-controller calls", ok, len(dev.calls))
	}

	t.Setenv("FAK_INKLING2_PREFILL_CTRL_BATCH", "1")
	p, c = ink2CtrlBatchFixture(&ink2NoBatchDev{})
	if _, ok := p.controlBatch(cur, msg, c); ok {
		t.Fatal("a device without Ink2PCtrlBatch must fall back to per-controller calls")
	}
}

func TestInkling2PrefillControlBatchIsControllerMajor(t *testing.T) {
	t.Setenv("FAK_INKLING2_PREFILL_CTRL_BATCH", "1")
	cur, msg := ink2CtrlBatchOperands()
	dev := &ink2CtrlBatchDev{}
	p, c := ink2CtrlBatchFixture(dev)
	out, ok := p.controlBatch(cur, msg, c)
	if !ok || len(dev.calls) != 1 {
		t.Fatalf("ok=%v calls=%d, want one batched call", ok, len(dev.calls))
	}
	call := dev.calls[0]
	h, T := p.s.hp.Hidden, p.T
	if call.n != h || call.rank != c.rank || call.T != T {
		t.Fatalf("shape n=%d rank=%d T=%d, want %d %d %d", call.n, call.rank, call.T, h, c.rank, T)
	}
	if len(call.cur) != 3 || len(call.down) != 3 || len(call.gain) != 3 || len(call.forget) != 3 {
		t.Fatalf("operand counts cur=%d down=%d gain=%d forget=%d, want 3", len(call.cur), len(call.down), len(call.gain), len(call.forget))
	}
	for j := 0; j < 3; j++ {
		// fixture operands carry distinct offsets, so Off names each one
		if call.cur[j].Off != cur[j].Off || call.msg[j].Off != msg[j].Off || call.down[j].Off != c.down[j].Off ||
			call.gain[j].Off != c.gain[j].Off || call.forget[j].Off != c.forget[j].Off {
			t.Fatalf("controller %d operands are not controller %d's", j, j)
		}
		if want := call.out.Off + j*T*h; out[j].Off != want {
			t.Fatalf("out[%d].Off=%d, want controller-major offset %d", j, out[j].Off, want)
		}
	}
	if call.out.Off-call.g.Off < 3*T*c.rank {
		t.Fatalf("g scratch %d floats overlaps out, want >= %d", call.out.Off-call.g.Off, 3*T*c.rank)
	}
	if p.s.err != nil {
		t.Fatal(p.s.err)
	}
}
