//go:build darwin && arm64 && cgo

package metalgemm

import (
	_ "embed"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
)

//go:embed ink2_dec_lean.metal
var ink2LeanSource string

// The decode op names in FAK_INK2_LEGACY, each keeping its family's first kernels:
// "mix", "ctrl" (the controller's kernels and its normalize), "rms" and "route";
// "ctrlmix" keeps float32 controller weights and a mix compiled for any state count.
// Names of other families are left to their own parsers.
type ink2DecLegacyOps struct{ mix, ctrl, rms, route, ctrlmix bool }

func parseInk2OpsLegacy(s string) (l ink2DecLegacyOps) {
	for _, name := range strings.Split(s, ",") {
		switch strings.ToLower(strings.TrimSpace(name)) {
		case "mix":
			l.mix = true
		case "ctrl":
			l.ctrl = true
		case "rms":
			l.rms = true
		case "route":
			l.route = true
		case "ctrlmix":
			l.ctrlmix = true
		}
	}
	return l
}

var ink2OpsLegacy = parseInk2OpsLegacy(os.Getenv(Ink2LegacyEnv))

// These mirror the limits compiled into ink2_dec_lean.metal.
const (
	ink2LeanMaxExperts = 4 * ink2OpsRouteThreads
)

// ink2LeanCtrlRanks are the controller ranks ink2_ctrl_apply_lean is built for.
var ink2LeanCtrlRanks = []int{16, 32, 48, 64, 96, 128}

// ink2LeanBuild is one compilation of the lean source: a thread of a reducing
// kernel names rows elements of its column, and exact drops the bounds checks
// for a width of exactly that many rows.
type ink2LeanBuild struct {
	rows  int
	exact bool
}

var ink2LeanRows = []int{8, 9, 12, 16}

// The pipeline cache keys on a source's storage, so each build's text is made once.
var ink2LeanSources = func() map[ink2LeanBuild]string {
	m := map[ink2LeanBuild]string{}
	for _, rows := range ink2LeanRows {
		head := fmt.Sprintf("#define INK2L_ROWS %d\n", rows)
		m[ink2LeanBuild{rows, false}] = head + ink2LeanSource
		m[ink2LeanBuild{rows, true}] = head + "#define INK2L_EXACT 1\n" + ink2LeanSource
	}
	return m
}()

// ink2LeanBuildFor picks the build with the fewest rows that cover width n,
// or reports that none does.
func ink2LeanBuildFor(n int) (ink2LeanBuild, bool) {
	for _, rows := range ink2LeanRows {
		if n <= rows*ink2OpsThreads {
			return ink2LeanBuild{rows, n == rows*ink2OpsThreads}, true
		}
	}
	return ink2LeanBuild{}, false
}

// leanKernel returns a kernel of ink2_dec_lean.metal for operands of width n,
// or false when the lean source has no build that wide.
func (d *Ink2Device) leanKernel(name string, n int) (Ink2Pipeline, bool, error) {
	build, ok := ink2LeanBuildFor(n)
	if !ok {
		return Ink2Pipeline{}, false, nil
	}
	p, err := d.PipelinePrecise(ink2LeanSources[build], name)
	if err != nil {
		return Ink2Pipeline{}, false, err
	}
	if p.ThreadWidth() != 32 || p.MaxThreads() < ink2OpsThreads {
		return Ink2Pipeline{}, false, fmt.Errorf("metalgemm: inkling2 kernel %s needs 32-wide SIMD groups of %d threads", name, ink2OpsThreads)
	}
	return p, true, nil
}

// opsOrLean returns the lean kernel when it is wanted and has a build for
// width n, and the first kernel otherwise.
func (d *Ink2Device) opsOrLean(first, lean string, legacy bool, n int) (Ink2Pipeline, bool, error) {
	if !legacy {
		if p, ok, err := d.leanKernel(lean, n); ok || err != nil {
			return p, ok, err
		}
	}
	p, err := d.opsKernel(first, true)
	return p, false, err
}

// ctrlApplyKernel returns the controller apply kernel for width n and rank.
func (d *Ink2Device) ctrlApplyKernel(n, rank int) (Ink2Pipeline, error) {
	if !ink2OpsLegacy.ctrl && slices.Contains(ink2LeanCtrlRanks, rank) {
		if build, ok := ink2LeanBuildFor(n); ok {
			return d.PipelinePrecise(ink2LeanSources[build], fmt.Sprintf("ink2_ctrl_apply_lean_%d", rank))
		}
	}
	return d.opsKernel("ink2_ctrl_apply", false)
}

// ink2CtrlBF16 reports whether a controller of width n and rank runs on
// bfloat16 weights.
func ink2CtrlBF16(n, rank int) bool {
	_, ok := ink2LeanBuildFor(n)
	return ok && !ink2OpsLegacy.ctrl && !ink2OpsLegacy.ctrlmix && slices.Contains(ink2LeanCtrlRanks, rank)
}

// mixLeanName is the lean mix kernel for ns states.
func mixLeanName(ns int) string {
	if ink2OpsLegacy.ctrlmix {
		return "ink2_mix_lean"
	}
	return fmt.Sprintf("ink2_mix_lean_%d", ns)
}

// PackBF16 returns the n float32 values at src as bfloat16, two to a word in
// element order, or false when the bfloat16 controller is off or a value is
// not exactly a bfloat16.
func (d *Ink2Device) PackBF16(src Ink2Ref, n int) ([]float32, bool, error) {
	if ink2OpsLegacy.ctrl || ink2OpsLegacy.ctrlmix {
		return nil, false, nil
	}
	v, err := d.Read(src, n)
	if err != nil {
		return nil, false, err
	}
	words := make([]uint32, (n+1)/2)
	for i, x := range v {
		b := math.Float32bits(x)
		if b&0xffff != 0 {
			return nil, false, nil
		}
		words[i/2] |= (b >> 16) << (16 * (i % 2))
	}
	out := make([]float32, len(words))
	for i, w := range words {
		out[i] = math.Float32frombits(w)
	}
	return out, true, nil
}
