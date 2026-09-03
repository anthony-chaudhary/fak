package ggufload

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// DraftKind identifies the speculative decoding mechanism of a companion GGUF sidecar.
type DraftKind string

const (
	DraftKindNone   DraftKind = ""
	DraftKindMtp    DraftKind = "mtp"
	DraftKindDflash DraftKind = "dflash"
)

func (k DraftKind) String() string {
	return string(k)
}

// Errors returned during companion draft discovery and pairing.
var (
	ErrNoCompanionFound        = errors.New("ggufload: no speculative companion candidate found")
	ErrAmbiguousDraftMechanism = errors.New("ggufload: ambiguous speculative draft mechanism: multiple mechanisms coexist")
)

// CompanionDraft represents a discovered GGUF companion file and its classification.
type CompanionDraft struct {
	Path      string    // Path or filename of the companion
	DraftKind DraftKind // Speculative decoding mechanism (DraftKindMtp or DraftKindDflash)
	Quant     string    // Extracted canonical quantization tag (e.g., "Q4_K_M", "Q4_0")
	Bits      float64   // Effective bits per weight
	Depth     int       // Shared directory depth with the target model
	Exact     bool      // True if candidate quant exactly matches target quant
	Distance  float64   // Absolute bit distance |Bits - TargetBits|
}

// CompanionOptions configures candidate filtering and ranking.
type CompanionOptions struct {
	TargetQuant   string    // Explicit quantization tag of the target model (if not extracted from path)
	PreferredKind DraftKind // If non-empty, restricts candidates to this mechanism (resolving ambiguity)
}

type quantDef struct {
	tag  string
	bits float64
}

// knownQuantTags lists canonical quantization tags and their representative bits per weight.
// Longer tags appear before shorter substrings to prevent prefix collision.
var knownQuantTags = []quantDef{
	// 32-bit
	{"FP32", 32.0},
	{"F32", 32.0},
	// 16-bit
	{"FP16", 16.0},
	{"BF16", 16.0},
	{"F16", 16.0},
	// 8-bit
	{"Q8_K", 8.0},
	{"Q8_1", 8.5},
	{"Q8_0", 8.0},
	// 6-bit
	{"Q6_K", 6.56},
	// 5-bit
	{"Q5_K_M", 5.5},
	{"Q5_K_S", 5.5},
	{"Q5_K", 5.5},
	{"Q5_1", 5.5},
	{"Q5_0", 5.0},
	// 4-bit
	{"Q4_K_M", 4.5},
	{"Q4_K_S", 4.5},
	{"Q4_K", 4.5},
	{"Q4_1", 4.5},
	{"Q4_0", 4.0},
	{"IQ4_NL", 4.5},
	{"IQ4_XS", 4.25},
	{"MXFP4", 4.0},
	// 3-bit
	{"Q3_K_XL", 3.7},
	{"Q3_K_L", 3.8},
	{"Q3_K_M", 3.4},
	{"Q3_K_S", 3.4},
	{"Q3_K", 3.4},
	{"Q3_1", 3.5},
	{"Q3_0", 3.0},
	{"IQ3_S", 3.4},
	{"IQ3_XXS", 3.06},
	// 2-bit
	{"IQ2_XXS", 2.06},
	{"IQ2_XS", 2.31},
	{"IQ2_S", 2.5},
	{"Q2_K", 2.5},
	{"Q2_0", 2.0},
	// 1-bit
	{"IQ1_M", 1.75},
	{"IQ1_S", 1.56},
	{"Q1_0", 1.0},
}

// hasBoundaryToken checks whether token appears within s surrounded by delimiter boundaries.
func hasBoundaryToken(s, token string) bool {
	idx := 0
	for {
		pos := strings.Index(s[idx:], token)
		if pos < 0 {
			return false
		}
		start := idx + pos
		end := start + len(token)
		idx = end

		if start > 0 {
			ch := s[start-1]
			if ch != '-' && ch != '_' && ch != '.' && ch != '/' && ch != '\\' && ch != ' ' {
				continue
			}
		}
		if end < len(s) {
			ch := s[end]
			if ch != '-' && ch != '_' && ch != '.' && ch != '/' && ch != '\\' && ch != ' ' {
				continue
			}
		}
		return true
	}
}

// DetectDraftKind inspects a path or filename and returns the recognized speculative draft mechanism.
func DetectDraftKind(path string) DraftKind {
	base := filepath.Base(path)
	lower := strings.ToLower(base)
	lower = strings.TrimSuffix(lower, ".gguf")

	hasMtp := hasBoundaryToken(lower, "mtp") || hasBoundaryToken(lower, "nextn")
	hasDflash := hasBoundaryToken(lower, "dflash")

	if hasMtp && hasDflash {
		return DraftKindNone
	}
	if hasMtp {
		return DraftKindMtp
	}
	if hasDflash {
		return DraftKindDflash
	}
	return DraftKindNone
}

// ExtractQuantTag extracts and canonicalizes the quantization tag from a path or filename.
func ExtractQuantTag(path string) string {
	base := filepath.Base(path)
	upper := strings.ToUpper(base)
	upper = strings.TrimSuffix(upper, ".GGUF")

	for _, qd := range knownQuantTags {
		if hasBoundaryToken(upper, qd.tag) {
			if qd.tag == "FP16" {
				return "F16"
			}
			if qd.tag == "FP32" {
				return "F32"
			}
			return qd.tag
		}
	}
	return ""
}

// QuantBits returns the representative bits per weight for a canonical quantization tag.
func QuantBits(quant string) (float64, bool) {
	upper := strings.ToUpper(strings.TrimSpace(quant))
	if upper == "FP16" {
		upper = "F16"
	}
	if upper == "FP32" {
		upper = "F32"
	}
	for _, qd := range knownQuantTags {
		if qd.tag == upper {
			return qd.bits, true
		}
	}
	return 0, false
}

// SharedDirectoryDepth calculates the number of matching directory segments between target and candidate.
func SharedDirectoryDepth(targetPath, candidatePath string) int {
	cleanTarget := filepath.Clean(targetPath)
	cleanCand := filepath.Clean(candidatePath)

	dirTarget := filepath.Dir(cleanTarget)
	dirCand := filepath.Dir(cleanCand)

	dirTarget = filepath.ToSlash(dirTarget)
	dirCand = filepath.ToSlash(dirCand)

	if dirTarget == "." || dirCand == "." {
		return 0
	}

	partsTarget := strings.Split(strings.Trim(dirTarget, "/"), "/")
	partsCand := strings.Split(strings.Trim(dirCand, "/"), "/")

	depth := 0
	minLen := len(partsTarget)
	if len(partsCand) < minLen {
		minLen = len(partsCand)
	}
	for i := 0; i < minLen; i++ {
		if partsTarget[i] == partsCand[i] && partsTarget[i] != "" && partsTarget[i] != "." {
			depth++
		} else {
			break
		}
	}
	return depth
}

// RankCompanionDrafts ranks candidate draft files for a target model using Lemonade's hf_variants ranking:
// 1. Speculative mechanism consistency (mixed mechanisms return ErrAmbiguousDraftMechanism)
// 2. Shared directory depth (higher shared depth preferred)
// 3. Exact quantization tag match
// 4. Minimum quant bit-distance |draft_bits - target_bits|
// 5. Deterministic lexicographical tie-break
func RankCompanionDrafts(targetPath string, candidatePaths []string, opts ...CompanionOptions) ([]CompanionDraft, error) {
	var opt CompanionOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	targetQuant := opt.TargetQuant
	if targetQuant == "" {
		targetQuant = ExtractQuantTag(targetPath)
	}
	var targetBits float64
	var hasTargetBits bool
	if targetQuant != "" {
		targetBits, hasTargetBits = QuantBits(targetQuant)
	}

	cleanTarget := filepath.Clean(targetPath)

	type parsedCand struct {
		path string
		kind DraftKind
	}
	var validDrafts []parsedCand
	mechanisms := make(map[DraftKind]struct{})

	for _, candPath := range candidatePaths {
		cleanCand := filepath.Clean(candPath)
		if cleanCand == cleanTarget {
			continue
		}
		kind := DetectDraftKind(candPath)
		if kind == DraftKindNone {
			continue
		}
		if opt.PreferredKind != "" && kind != opt.PreferredKind {
			continue
		}
		validDrafts = append(validDrafts, parsedCand{path: candPath, kind: kind})
		mechanisms[kind] = struct{}{}
	}

	if len(validDrafts) == 0 {
		return nil, ErrNoCompanionFound
	}

	if opt.PreferredKind == "" && len(mechanisms) > 1 {
		return nil, ErrAmbiguousDraftMechanism
	}

	var candidates []CompanionDraft
	for _, vd := range validDrafts {
		cQuant := ExtractQuantTag(vd.path)
		cBits, hasCBits := QuantBits(cQuant)

		depth := SharedDirectoryDepth(targetPath, vd.path)
		exact := (targetQuant != "" && strings.EqualFold(targetQuant, cQuant))

		var distance float64
		if hasTargetBits && hasCBits {
			distance = math.Abs(cBits - targetBits)
		} else {
			distance = math.MaxFloat64
		}

		candidates = append(candidates, CompanionDraft{
			Path:      vd.path,
			DraftKind: vd.kind,
			Quant:     cQuant,
			Bits:      cBits,
			Depth:     depth,
			Exact:     exact,
			Distance:  distance,
		})
	}

	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].Depth != candidates[j].Depth {
			return candidates[i].Depth > candidates[j].Depth
		}
		if candidates[i].Exact != candidates[j].Exact {
			return candidates[i].Exact
		}
		if math.Abs(candidates[i].Distance-candidates[j].Distance) > 1e-6 {
			return candidates[i].Distance < candidates[j].Distance
		}
		return filepath.Clean(candidates[i].Path) < filepath.Clean(candidates[j].Path)
	})

	return candidates, nil
}

// DiscoverCompanion discovers and returns the highest-ranked companion draft for targetPath.
func DiscoverCompanion(targetPath string, candidatePaths []string, opts ...CompanionOptions) (*CompanionDraft, error) {
	ranked, err := RankCompanionDrafts(targetPath, candidatePaths, opts...)
	if err != nil {
		return nil, err
	}
	if len(ranked) == 0 {
		return nil, ErrNoCompanionFound
	}
	return &ranked[0], nil
}

// DiscoverDirectoryCompanions scans the directory of targetPath and pairs the best companion draft.
func DiscoverDirectoryCompanions(targetPath string, opts ...CompanionOptions) (*CompanionDraft, error) {
	dir := filepath.Dir(targetPath)
	if dir == "" {
		dir = "."
	}

	var candidates []string
	targetDirClean := filepath.Clean(dir)
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			rel, relErr := filepath.Rel(targetDirClean, path)
			if relErr == nil && rel != "." && strings.Count(filepath.ToSlash(rel), "/") >= 2 {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.EqualFold(filepath.Ext(path), ".gguf") {
			candidates = append(candidates, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("ggufload: scanning directory %s: %w", dir, err)
	}

	return DiscoverCompanion(targetPath, candidates, opts...)
}
