package ggufload

import (
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Inkling2SpecDefault is the speculative decode a caller turns on when
// FAK_INKLING2_SPEC is unset; the zero value keeps plain decode.
type Inkling2SpecDefault struct {
	Depth int
	Draft string
}

// Inkling2MetalSpecDefault is the default-on speculation of the resident Metal
// target. The draft is auto: DFlash2 with its per-step width chooser when its
// checkpoint sits beside the GGUF, since that measured above prompt lookup on
// every prompt class, else prompt lookup, so speculation never needs a draft.
func Inkling2MetalSpecDefault(be compute.Backend) Inkling2SpecDefault {
	if !inkling2SpecCapable(be) {
		return Inkling2SpecDefault{}
	}
	return Inkling2SpecDefault{Depth: inkling2SpecMaxDraft, Draft: inkling2DraftAuto}
}

// inkling2SpecCapable reports whether be has every op family the resident
// verify chunk calls.
func inkling2SpecCapable(be compute.Backend) bool {
	_, dev := be.(compute.Inkling2Device)
	_, pre := be.(compute.Inkling2PrefillDevice)
	_, ver := be.(compute.Inkling2VerifyDevice)
	return dev && pre && ver
}

// inkling2SpecDepthFor is inkling2SpecDepth with m's default standing in for an
// unset or blank FAK_INKLING2_SPEC.
func inkling2SpecDepthFor(m *Inkling2LoopModel) int {
	return inkling2SpecDepthForFrom(m, os.LookupEnv)
}

func inkling2SpecDepthForFrom(m *Inkling2LoopModel, lookup func(string) (string, bool)) int {
	if inkling2SpecExplicit(lookup) || m == nil {
		return inkling2SpecDepthFrom(lookup)
	}
	return min(max(m.SpecDefault.Depth, 0), inkling2SpecMaxDraft)
}

// inkling2SpecDraftKindFor is inkling2SpecDraftKind with m's default draft
// standing in for an unset FAK_INKLING2_SPEC_DRAFT.
func inkling2SpecDraftKindFor(m *Inkling2LoopModel) (string, error) {
	return inkling2SpecDraftKindForFrom(m, os.LookupEnv)
}

func inkling2SpecDraftKindForFrom(m *Inkling2LoopModel, lookup func(string) (string, bool)) (string, error) {
	if v, set := inkling2Lookup(lookup, "FAK_INKLING2_SPEC_DRAFT"); (!set || strings.TrimSpace(v) == "") && m != nil && m.SpecDefault.Draft != "" {
		return m.SpecDefault.Draft, nil
	}
	return inkling2SpecDraftKindFrom(lookup)
}

// Inkling2SpecLabel names the greedy decode m runs on be, for a run banner:
// "spec=off" or "spec=<draft> depth=<n>". Sampled decode never speculates.
func Inkling2SpecLabel(m *Inkling2LoopModel, be compute.Backend) string {
	return Inkling2SpecLabelFor(m, be, os.LookupEnv)
}

// Inkling2SpecLabelFor is Inkling2SpecLabel evaluated against lookup.
func Inkling2SpecLabelFor(m *Inkling2LoopModel, be compute.Backend, lookup func(string) (string, bool)) string {
	depth := inkling2SpecDepthForFrom(m, lookup)
	if depth == 0 || !inkling2SpecCapable(be) || inkling2SpecAutoDisabledForKV(be, lookup) {
		return "spec=off"
	}
	kind, err := inkling2SpecDraftKindForFrom(m, lookup)
	if err != nil {
		return "spec=invalid"
	}
	mtpDir, _ := inkling2Lookup(lookup, "FAK_INKLING2_MTP_DIR")
	kind = inkling2ResolveSpecDraft(kind, strings.TrimSpace(mtpDir), inkling2DFlash2DirFor(m, be))
	return fmt.Sprintf("spec=%s depth=%d", kind, depth)
}
