//go:build darwin && arm64 && cgo

package compute

import (
	"errors"
	"unsafe"
)

// metalQwen35SequenceGraph is the caller-owned command-buffer spine used by the
// Qwen sequence-prefill implementation. It deliberately exposes only resident
// F32 primitives: callers may compose all token rows without falling back to
// the synchronous Backend methods, then commit and wait exactly once.
type metalQwen35SequenceGraph struct {
	owner *metalCommandOwner
}

func beginMetalQwen35SequenceGraph() (*metalQwen35SequenceGraph, error) {
	owner, err := beginMetalCommand()
	if err != nil {
		return nil, err
	}
	return &metalQwen35SequenceGraph{owner: owner}, nil
}

func (g *metalQwen35SequenceGraph) rmsNorm(x, weight, y unsafe.Pointer, rows, width int, eps float32) error {
	if g == nil || g.owner == nil {
		return errMetalOwnerTerminal
	}
	return g.owner.encodeRMSNorm(x, weight, y, rows, width, eps)
}

func (g *metalQwen35SequenceGraph) projection(weight, x, y unsafe.Pointer, out, in, tokens int) error {
	if g == nil || g.owner == nil {
		return errMetalOwnerTerminal
	}
	return g.owner.encodeMatMul(weight, x, y, out, in, tokens)
}

func (g *metalQwen35SequenceGraph) swiGLU(gate, up, y unsafe.Pointer, elements int) error {
	if g == nil || g.owner == nil {
		return errMetalOwnerTerminal
	}
	return g.owner.encodeSwiGLU(gate, up, y, elements)
}

func (g *metalQwen35SequenceGraph) residual(dst, src unsafe.Pointer, elements int) error {
	if g == nil || g.owner == nil {
		return errMetalOwnerTerminal
	}
	return g.owner.encodeAdd(dst, src, elements)
}

func (g *metalQwen35SequenceGraph) finish() (metalCommandReceipt, error) {
	if g == nil || g.owner == nil {
		return metalCommandReceipt{}, errMetalOwnerTerminal
	}
	owner := g.owner
	g.owner = nil
	return owner.finish()
}

func (g *metalQwen35SequenceGraph) abort() error {
	if g == nil || g.owner == nil {
		return errMetalOwnerTerminal
	}
	owner := g.owner
	g.owner = nil
	return owner.abort()
}

func (g *metalQwen35SequenceGraph) fail(err error) error {
	if err == nil {
		return nil
	}
	if abortErr := g.abort(); abortErr != nil && !errors.Is(abortErr, errMetalOwnerTerminal) {
		return errors.Join(err, abortErr)
	}
	return err
}
