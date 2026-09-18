package ggufload

import (
	"errors"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/rotationmeta"
)

// gguf_prism_hadamard.go — read and interpret the prism-ml rotated-basis weight
// contract declared by Ternary-Bonsai-2-27B GGUFs.
//
// Those checkpoints fold a blockwise (1024) normalized Sylvester Walsh-Hadamard
// rotation into the STORED ternary weights and declare the matching activation
// transform as GGUF metadata under the `prism.hadamard.*` prefix. A runtime that
// loads the weights but ignores the rotation produces garbage (the model card states
// stock llama.cpp does exactly that with Q2_0). This reader therefore does not merely
// parse the keys: it converts them into the neutral rotationmeta Descriptor and
// adjudicates against the caller's declared capabilities, so a session either has a
// supported transform or is told to refuse — never silently ignores the rotation.
//
// Wire contract (parsed from Ternary-Bonsai-2-27B-PTQ1_0.gguf at
// prism-ml/Ternary-Bonsai-2-27B-gguf@6ed5e12b):
//
//	prism.hadamard.version               = 1                              (uint32)
//	prism.hadamard.block_size            = 1024                           (uint32)
//	prism.hadamard.transform             = "normalized-sylvester-walsh-hadamard"
//	prism.hadamard.axis                  = "input-last-dimension"
//	prism.hadamard.sign_mode             = "explicit"
//	prism.hadamard.sign_widths           = [5120, 6144, 17408]            (array<int>)
//	prism.hadamard.sign_values           = [+/-1 x 28672]                 (array<int>)
//	prism.hadamard.weight_names          = [... 401 rotated tensors]      (array<string>)
//	prism.hadamard.inverse_weight_names  = ["token_embd.weight"]          (array<string>)
//	prism.hadamard.gdn_v_grouped         = true                           (bool)

// PrismHadamard is the parsed, self-consistent subset of the prism.hadamard.*
// metadata that a runtime needs to apply (or refuse) the rotation. Presence of this
// struct on a File means the file DECLARES the rotated-basis contract; it does not
// imply the runtime can execute it — that is what Descriptor + adjudication decide.
type PrismHadamard struct {
	Version      int
	BlockSize    int
	Transform    string
	Axis         string
	SignMode     string
	SignWidths   []int
	SignValues   []int
	WeightNames  []string
	InverseNames []string
	GDNVGrouped  bool
}

// ErrPrismHadamardMalformed is returned when a file declares the prism.hadamard.*
// contract but its keys are incomplete or internally inconsistent. The loader must
// treat this as a hard refusal: applying a partially-read rotation is worse than
// not loading, because the failure is silent garbage rather than a loud error.
var ErrPrismHadamardMalformed = errors.New("ggufload: prism.hadamard metadata is malformed")

// PrismHadamardMeta reads the prism.hadamard.* keys. It returns (nil, nil) for a file
// that does not declare the contract at all (the common case — only Bonsai-2 family
// GGUFs carry these keys). When any prism.hadamard.* key is present, the contract is
// required to be complete, so a missing or inconsistent key returns
// ErrPrismHadamardMalformed instead of a nil struct.
func (f *File) PrismHadamardMeta() (*PrismHadamard, error) {
	if !f.hasPrismHadamardKey() {
		return nil, nil
	}
	h := &PrismHadamard{}
	var ok bool

	v, ok := f.Uint64("prism.hadamard.version")
	if !ok {
		return nil, malformedf("version")
	}
	h.Version = int(v)
	bs, ok := f.Uint64("prism.hadamard.block_size")
	if !ok {
		return nil, malformedf("block_size")
	}
	h.BlockSize = int(bs)
	if h.Transform, ok = f.String("prism.hadamard.transform"); !ok {
		return nil, malformedf("transform")
	}
	if h.Axis, ok = f.String("prism.hadamard.axis"); !ok {
		return nil, malformedf("axis")
	}
	if h.SignMode, ok = f.String("prism.hadamard.sign_mode"); !ok {
		return nil, malformedf("sign_mode")
	}
	if h.SignWidths, ok = f.IntArray("prism.hadamard.sign_widths"); !ok {
		return nil, malformedf("sign_widths")
	}
	if h.SignValues, ok = f.SignedIntArray("prism.hadamard.sign_values"); !ok {
		return nil, malformedf("sign_values")
	}
	if h.WeightNames, ok = f.StringArray("prism.hadamard.weight_names"); !ok {
		return nil, malformedf("weight_names")
	}
	// inverse_weight_names and gdn_v_grouped are optional refinements: a file that
	// omits them still declares a usable rotation, and their absence changes no byte
	// of the transform. Read them best-effort.
	h.InverseNames, _ = f.StringArray("prism.hadamard.inverse_weight_names")
	h.GDNVGrouped, _ = f.Bool("prism.hadamard.gdn_v_grouped")

	// The declared transform and axis are a closed vocabulary for version 1. An
	// unrecognized transform is a different rotation we cannot promise to invert, so
	// it is malformed-for-this-runtime rather than silently accepted.
	if h.Transform != "normalized-sylvester-walsh-hadamard" || h.Axis != "input-last-dimension" {
		return nil, malformedf("transform/axis")
	}
	if h.SignMode != "explicit" {
		return nil, malformedf("sign_mode")
	}
	if h.Version != 1 {
		return nil, malformedf("version")
	}
	return h, nil
}

// Descriptor converts the parsed contract into the neutral rotationmeta descriptor so
// it can be adjudicated against declared runtime capabilities in one place.
func (h *PrismHadamard) Descriptor() rotationmeta.Descriptor {
	prov, _ := rotationmeta.PinnedProvenance(rotationmeta.RecipePrismHadamard, "prism-bonsai2-gguf-v1")
	return rotationmeta.Descriptor{
		ContractVersion: rotationmeta.ContractVersion,
		Recipe:          rotationmeta.RecipePrismHadamard,
		RecipeVersion:   "prism-bonsai2-gguf-v1",
		Provenance:      prov,
		ArtifactFormat:  "gguf",
		Transforms: []rotationmeta.Transform{{
			Name:      "activation-last-dim",
			Placement: rotationmeta.PlacementOnline,
			Fusion:    PrismHadamardFusion,
		}},
		Sign: &rotationmeta.SignVector{Widths: h.SignWidths, Values: h.SignValues},
	}
}

// PrismHadamardFusion is the runtime capability name a session must declare to apply
// the Bonsai-2 activation rotation. It names the block size so a future block-size
// variant gets a distinct token rather than silently reusing this one.
const PrismHadamardFusion = "hadamard/h1024-activation"

func (f *File) hasPrismHadamardKey() bool {
	for k := range f.Metadata {
		if strings.HasPrefix(k, "prism.hadamard.") {
			return true
		}
	}
	return false
}

// SignedIntArray reads a GGUF metadata array of SIGNED integers into []int. It exists
// because IntArray rejects negative values (correct for head counts, wrong for a +/-
// 1 sign vector). Returns (nil,false) on a non-array or any non-integer item.
func (f *File) SignedIntArray(key string) ([]int, bool) {
	return metadataArray(f, key, func(item Value) (int, bool) {
		switch item.Type {
		case TypeInt8:
			return int(item.Value.(int8)), true
		case TypeInt16:
			return int(item.Value.(int16)), true
		case TypeInt32:
			return int(item.Value.(int32)), true
		case TypeInt64:
			return int(item.Value.(int64)), true
		case TypeUint8:
			return int(item.Value.(uint8)), true
		case TypeUint16:
			return int(item.Value.(uint16)), true
		case TypeUint32:
			return int(item.Value.(uint32)), true
		case TypeUint64:
			return int(item.Value.(uint64)), true
		default:
			return 0, false
		}
	})
}

func malformedf(key string) error {
	return errors.Join(ErrPrismHadamardMalformed, errors.New(key))
}
