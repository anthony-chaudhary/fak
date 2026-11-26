//go:build darwin && arm64 && cgo

package compute

var _ Inkling2DFlash2Device = (*metalBackend)(nil)

func (c *metalBackend) Ink2DFQKRope(x, w, dst, posDev Ink2Ref, rows, heads, dim, cap, base int, ring bool, eps float32, theta float64) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, w, dst, posDev)
	if err != nil {
		return ink2MetalErr("draft qk rope", err)
	}
	return ink2MetalErr("draft qk rope", d.DraftQKRope(r[0], r[1], r[2], r[3], rows, heads, dim, cap, base, ring, eps, theta))
}

func (c *metalBackend) Ink2DFRingStore(src, ring, posDev Ink2Ref, rows, width, cap, base int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(src, ring, posDev)
	if err != nil {
		return ink2MetalErr("draft ring store", err)
	}
	return ink2MetalErr("draft ring store", d.DraftRingStore(r[0], r[1], r[2], rows, width, cap, base))
}

func (c *metalBackend) Ink2DFSiluMul(g, u, out Ink2Ref, n int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(g, u, out)
	if err != nil {
		return ink2MetalErr("draft silu mul", err)
	}
	return ink2MetalErr("draft silu mul", d.DraftSiluMul(r[0], r[1], r[2], n))
}

func (c *metalBackend) Ink2DFHead(w Tensor, x, y Ink2Ref, tokens int) (bool, error) {
	if len(w.Shape) != 2 {
		return false, nil
	}
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, y)
	if err != nil {
		return false, ink2MetalErr("draft head", err)
	}
	wv, _, err := ink2MetalWeight(w, false)
	if err != nil {
		return false, ink2MetalErr("draft head", err)
	}
	ok, err := d.DraftHead(wv, r[0], wv.Cols, r[1], wv.Rows, tokens)
	return ok, ink2MetalErr("draft head", err)
}
