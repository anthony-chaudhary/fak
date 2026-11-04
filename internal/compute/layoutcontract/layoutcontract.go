// Package layoutcontract is the declarative registry of tensor layout
// contracts for every quantized kernel family present in-tree: Q4_K, Q8_0,
// NVFP4 and MXFP4.
//
// WHY THIS EXISTS. High-performance quantized kernels expect weights and
// scales pre-packed or swizzled in specific tile layouts at load time. Port a
// compute kernel without its loader's exact packing logic and every matmul
// pays a runtime transpose/repack that erases the kernel speedup — silently,
// because both sides still produce the right numbers. Each Contract below is
// the closed fact set that separates "the loader feeds the kernel what it
// expects" from "the loader feeds it something else and hopes": storage
// order, interleave width, scale layout, padding rule, whether the packing
// happens natively at load or through a runtime transform, and a pinned
// path:line@sha source ref so drift shows up as a stale ref to re-pin, not as
// a surprise.
//
// A Transform pair (forward + inverse byte permutation) is registered where a
// Go implementation exists in-tree; the round-trip test drives every pair
// through the same uniform Shape/[]byte convention, so an offset bug in any
// repack is caught by the identity witness instead of by a hardware run.
package layoutcontract

import (
	"errors"
	"fmt"
	"regexp"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// ErrUnknownFamily is the fail-closed refusal of Get/Must for a family the
// registry does not register. errors.Is pins it.
var ErrUnknownFamily = errors.New("layoutcontract: unknown quantized kernel family")

// StorageOrder names the byte order a kernel consumes a weight payload in.
type StorageOrder string

const (
	// RowMajor stores row r entirely before row r+1; the checkpoint/resident
	// default.
	RowMajor StorageOrder = "row-major"
	// TileSwizzled groups a payload into fixed tiles whose bytes are permuted
	// so each tile loads contiguously (NVFP4's 128x4 scale image).
	TileSwizzled StorageOrder = "tile-swizzled"
	// ColumnInterleaved groups rows in blocks of the interleave width and lays
	// each super-block field out lane by lane (Q4_K repack).
	ColumnInterleaved StorageOrder = "column-interleaved"
)

// ScaleLayout names where a family's per-block scales live relative to the
// weight payload.
type ScaleLayout string

const (
	// ScaleLinear keeps scales in a separate row-major array aligned to the
	// weight grid (Q8_0's per-32 f32, MXFP4's per-32 E8M0).
	ScaleLinear ScaleLayout = "linear"
	// ScaleTiled128x4 is the TensorRT-LLM NVFP4 scale image: 128-row by
	// 4-column tiles, zero-padded to complete the final tiles.
	ScaleTiled128x4 ScaleLayout = "128x4-tiled"
	// ScalePackedInRecord packs the scales into the weight record itself
	// (Q4_K's 12 packed 6-bit scale/min bytes per 256-weight super-block).
	ScalePackedInRecord ScaleLayout = "packed-in-record"
)

// PackingPath names how the kernel-ready layout is produced.
type PackingPath string

const (
	// PackingNative: the loader writes kernel-ready bytes directly; no repack
	// step stands between checkpoint and kernel.
	PackingNative PackingPath = "native"
	// PackingRuntime: a registered Go transform repacks the loaded bytes into
	// the kernel layout; skipping it is the layout-mismatch bug this registry
	// exists to make visible.
	PackingRuntime PackingPath = "runtime"
)

// Shape is the uniform geometry argument of a Transform call. Rows and Cols
// name the logical [rows, cols] payload; Width is the column-interleave
// width, 1 when the transform has none.
type Shape struct {
	Rows  int
	Cols  int
	Width int
}

// Transform is a registered forward/inverse byte-layout pair. The pairs this
// registry wraps share ONE body per direction in their source, so the
// round-trip witness catches an offset bug instead of two hand-written
// inverses agreeing on the same mistake.
type Transform struct {
	// Name names the wrapped Go pair, e.g. "bench.RepackQ4KInterleaved /
	// bench.UnrepackQ4KInterleaved".
	Name string
	// Source pins the forward function's definition site.
	Source SourceRef
	// Forward maps the entry's stored layout to the kernel layout.
	Forward func(Shape, []byte) ([]byte, error)
	// Inverse undoes Forward.
	Inverse func(Shape, []byte) ([]byte, error)
}

// Present reports whether the entry carries a Go transform pair.
func (t Transform) Present() bool { return t.Forward != nil && t.Inverse != nil }

// SourceRef pins a contract fact to its definition site at a commit, so a
// future move or rewrite shows up as a stale ref to re-pin, not as silent
// drift.
type SourceRef struct {
	Path string // repo-relative file path
	Line int    // 1-indexed line at SHA
	SHA  string // full commit SHA the pin was cut at
}

// String renders the ref as path:line@short-sha.
func (r SourceRef) String() string {
	short := r.SHA
	if len(short) > 12 {
		short = short[:12]
	}
	return fmt.Sprintf("%s:%d@%s", r.Path, r.Line, short)
}

// Contract is one quantized kernel family's tensor layout contract: the
// closed set of facts a loader must satisfy for a kernel to read its bytes
// without a runtime transpose or repack.
type Contract struct {
	// Family is the closed family tag ("Q4_K", "Q8_0", "NVFP4", "MXFP4").
	Family string
	// WeightOrder is the order the kernel consumes the weight payload in.
	WeightOrder StorageOrder
	// InterleaveWidth is the column-interleave width; 0 means the width is
	// resolved at runtime from host CPU features.
	InterleaveWidth int
	// ScaleLayout is where the family's per-block scales live.
	ScaleLayout ScaleLayout
	// PaddingRule is the closed shape rule the kernel accepts.
	PaddingRule string
	// PackingPath says how the kernel-ready layout is produced.
	PackingPath PackingPath
	// Source pins the layout fact's definition site.
	Source SourceRef
	// Transform is the Go repack pair, zero when no implementation exists.
	Transform Transform
}

// Validate checks one entry against the registry's closed invariants and
// returns the first violated one. The integrity test runs it over every
// entry, so a future entry author inherits the same bar.
func (c Contract) Validate() error {
	if c.Family == "" {
		return errors.New("layoutcontract: entry has no family tag")
	}
	switch c.WeightOrder {
	case RowMajor, TileSwizzled, ColumnInterleaved:
	default:
		return fmt.Errorf("layoutcontract: %s: storage order %q is not a closed value", c.Family, c.WeightOrder)
	}
	if c.InterleaveWidth < 0 {
		return fmt.Errorf("layoutcontract: %s: interleave width %d is negative", c.Family, c.InterleaveWidth)
	}
	switch c.ScaleLayout {
	case ScaleLinear, ScaleTiled128x4, ScalePackedInRecord:
	default:
		return fmt.Errorf("layoutcontract: %s: scale layout %q is not a closed value", c.Family, c.ScaleLayout)
	}
	if c.PaddingRule == "" {
		return fmt.Errorf("layoutcontract: %s: missing padding rule", c.Family)
	}
	switch c.PackingPath {
	case PackingNative, PackingRuntime:
	default:
		return fmt.Errorf("layoutcontract: %s: packing path %q is not a closed value", c.Family, c.PackingPath)
	}
	if err := c.Source.validate(c.Family); err != nil {
		return err
	}
	if (c.Transform.Forward == nil) != (c.Transform.Inverse == nil) {
		return fmt.Errorf("layoutcontract: %s: transform pair is half-registered", c.Family)
	}
	if c.Transform.Present() {
		if c.Transform.Name == "" {
			return fmt.Errorf("layoutcontract: %s: transform pair has no name", c.Family)
		}
		if err := c.Transform.Source.validate(c.Family + " transform"); err != nil {
			return err
		}
	}
	return nil
}

var shaPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)

func (r SourceRef) validate(where string) error {
	if r.Path == "" || r.Line < 1 || !shaPattern.MatchString(r.SHA) {
		return fmt.Errorf("layoutcontract: %s: source ref %s is not a pinned path:line@sha", where, r)
	}
	return nil
}

// pinSHA is the commit the SourceRefs below were cut at; re-pin an entry when
// its cited line moves.
const pinSHA = "9bd0f1516ae06f8e2848f07b8b9a63c78c920efd"

// registry holds the closed family set. Order is presentation order only;
// Families sorts.
var registry = []Contract{
	{
		// Q4_K: the resident bytes ARE the GGUF bytes (row-major 144-byte
		// super-blocks), which the production decode kernels read verbatim —
		// hence native packing. The column-interleaved variant is a runtime
		// repack available through the registered bench pair (#5285 spike);
		// its width is host-resolved, not a build constant.
		Family:          "Q4_K",
		WeightOrder:     RowMajor,
		InterleaveWidth: 0, // host-resolved: bench.Q4KInterleaveWidth -> 8 (AVX2/AVX512), 4 (NEON), 1 (scalar)
		ScaleLayout:     ScalePackedInRecord,
		PaddingRule:     "reduction dim is a multiple of 256 (one 144-byte GGUF super-block); the interleaved repack is a pure permutation, same length as the row-major input",
		PackingPath:     PackingNative,
		Source:          SourceRef{Path: "internal/model/quant_q4k.go", Line: 22, SHA: pinSHA},
		Transform: Transform{
			Name:   "bench.RepackQ4KInterleaved / bench.UnrepackQ4KInterleaved",
			Source: SourceRef{Path: "internal/bench/q4krepack.go", Line: 163, SHA: pinSHA},
			Forward: func(s Shape, b []byte) ([]byte, error) {
				return bench.RepackQ4KInterleaved(b, s.Rows, s.Cols, s.Width)
			},
			Inverse: func(s Shape, b []byte) ([]byte, error) {
				return bench.UnrepackQ4KInterleaved(b, s.Rows, s.Cols, s.Width)
			},
		},
	},
	{
		// Q8_0: row-major int8 codes plus a separate row-major f32 scale per
		// 32-element block; NewQ8 wraps prequantized bytes directly, so no
		// repack step exists and none is registered.
		Family:          "Q8_0",
		WeightOrder:     RowMajor,
		InterleaveWidth: 1,
		ScaleLayout:     ScaleLinear,
		PaddingRule:     "K is a multiple of the 32-element scale block; codes are exact out*in int8 bytes and scales exact out*(in/32) f32, no padding",
		PackingPath:     PackingNative,
		Source:          SourceRef{Path: "internal/compute/cpuref.go", Line: 65, SHA: pinSHA},
	},
	{
		// NVFP4: the weight nibbles stay row-major (even K in the low nibble,
		// in a multiple of 16), but the kernel-facing block-scale image is the
		// TensorRT-LLM 128x4 tiled layout, which the loader's linear scales do
		// not already hold — the registered swizzle pair produces it. The pair
		// is exercised by the round-trip witness; no loader call site invokes
		// it yet, which is exactly the open seam this entry makes visible.
		Family:          "NVFP4",
		WeightOrder:     RowMajor,
		InterleaveWidth: 1,
		ScaleLayout:     ScaleTiled128x4,
		PaddingRule:     "in is a multiple of 16 (weight bytes are exact out*in/2); the 128x4-tiled scale image is zero-padded to complete the final 128-row and 4-column tiles",
		PackingPath:     PackingRuntime,
		Source:          SourceRef{Path: "internal/compute/nvfp4_scale_layout.go", Line: 26, SHA: pinSHA},
		Transform: Transform{
			Name:   "compute.SwizzleNVFP4Scales / compute.UnswizzleNVFP4Scales",
			Source: SourceRef{Path: "internal/compute/nvfp4_scale_layout.go", Line: 26, SHA: pinSHA},
			Forward: func(s Shape, b []byte) ([]byte, error) {
				return compute.SwizzleNVFP4Scales(s.Rows, s.Cols, b)
			},
			Inverse: func(s Shape, b []byte) ([]byte, error) {
				return compute.UnswizzleNVFP4Scales(s.Rows, s.Cols, b)
			},
		},
	},
	{
		// MXFP4: routed-expert GEMV format — I8-packed E2M1 (low nibble = even
		// K) with row-major raw E8M0 scales, one per 32 unpacked K values.
		// Checkpoints ship the packed bytes kernel-ready and the kernels decode
		// inline, so no repack step exists and none is registered.
		Family:          "MXFP4",
		WeightOrder:     RowMajor,
		InterleaveWidth: 1,
		ScaleLayout:     ScaleLinear,
		PaddingRule:     "O positive and I even (weight bytes are exact O*I/2); the scale row rounds up to 32, ceil(I/32) raw E8M0 bytes per row",
		PackingPath:     PackingNative,
		Source:          SourceRef{Path: "internal/compute/fp8_block32.go", Line: 457, SHA: pinSHA},
	},
}

// All returns every registered contract, in registry order.
func All() []Contract {
	out := make([]Contract, len(registry))
	copy(out, registry)
	return out
}

// Families returns the registered family tags, sorted.
func Families() []string {
	out := make([]string, len(registry))
	for i, c := range registry {
		out[i] = c.Family
	}
	sort.Strings(out)
	return out
}

// Get returns the contract for family, failing closed with ErrUnknownFamily
// for anything the registry does not register.
func Get(family string) (Contract, error) {
	for _, c := range registry {
		if c.Family == family {
			return c, nil
		}
	}
	return Contract{}, fmt.Errorf("%w: %q", ErrUnknownFamily, family)
}

// Must is Get without the error, for tests and init-time wiring.
func Must(family string) Contract {
	c, err := Get(family)
	if err != nil {
		panic(err)
	}
	return c
}
