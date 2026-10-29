// allocation_sidecar.go — parser and validator for RCO per-tensor allocation
// sidecars (.rco-allocation.txt), the text manifests the ISTA-DASLab GSQ/RCO
// mixed-precision GGUF releases ship under tensor-allocation/ (#14091).
//
// Provenance: convention adoption from the Apache-2.0 GSQ/RCO tooling, per the
// 2026-10-28 study pass of those artifacts (borrow R5). This is a clean-room
// reader for the published manifest format; no code is ported from the upstream
// tooling. Pinned grammar witness:
// https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/raw/main/tensor-allocation/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS.rco-allocation.txt
// (sha256 305e14351d8f116f411e0370a9a6f73f0cde2941b7a4650b0fb35be00356a91b).
//
// The manifest is a comment header (repository, shard name, source URLs, target
// bits-per-weight with an optional measured value, GGUF version/metadata counts,
// declared tensor count, quant-type histogram) followed by one
// "<GGUF tensor name>: <GGML type>" row per tensor, fixed non-RCO tensors
// included so the released GGUF is fully reproducible. Parsing is fail-closed:
// every structural defect refuses through a sentinel error usable with
// errors.Is, and ValidateAllocation cross-checks a parsed sidecar against a
// parsed GGUF tensor inventory reusing this package's own parse results.
package ggufload

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Sentinel errors for sidecar parsing and validation. Callers match these with
// errors.Is; the wrapped detail names the offending line, field, or tensor.
var (
	// ErrAllocationSidecarMalformed: a line or number in the file does not fit
	// the manifest grammar (bad row shape, duplicate tensor, trailing comments,
	// unparseable count).
	ErrAllocationSidecarMalformed = errors.New("gguf: rco allocation sidecar is malformed")
	// ErrAllocationSidecarMissingField: a required header field (target bpw,
	// tensor count, quant-type counts) is absent.
	ErrAllocationSidecarMissingField = errors.New("gguf: rco allocation sidecar is missing a required field")
	// ErrAllocationSidecarEmpty: the manifest lists no tensor rows.
	ErrAllocationSidecarEmpty = errors.New("gguf: rco allocation sidecar lists no tensors")
	// ErrAllocationSidecarUnknownType: a row or histogram entry names a type
	// tag this loader does not recognize.
	ErrAllocationSidecarUnknownType = errors.New("gguf: rco allocation sidecar names an unrecognized tensor type")
	// ErrAllocationDeclaredCountMismatch: the declared "Tensor count" disagrees
	// with the number of listed rows.
	ErrAllocationDeclaredCountMismatch = errors.New("gguf: rco allocation sidecar tensor count disagrees with its rows")
	// ErrAllocationCoverageMismatch: the sidecar and GGUF tensor sets disagree
	// in either direction (or the inventory itself duplicates a name).
	ErrAllocationCoverageMismatch = errors.New("gguf: rco allocation sidecar and GGUF tensor sets disagree")
	// ErrAllocationTypeMismatch: a shared tensor's sidecar type differs from
	// the GGUF encoding.
	ErrAllocationTypeMismatch = errors.New("gguf: rco allocation sidecar type disagrees with the GGUF tensor")
	// ErrAllocationHistogramDrift: the declared quant-type histogram differs
	// from the inventory's computed histogram.
	ErrAllocationHistogramDrift = errors.New("gguf: rco allocation sidecar histogram disagrees with the GGUF inventory")
	// ErrAllocationBPWDrift: the effective bits-per-weight is outside the
	// tolerance band around the declared target.
	ErrAllocationBPWDrift = errors.New("gguf: effective bits-per-weight is outside the sidecar target tolerance")
	// ErrAllocationUnsupportedType: an inventory tensor has no block geometry
	// (unknown future tag), so bits-per-weight cannot be accounted.
	ErrAllocationUnsupportedType = errors.New("gguf: tensor type has no block geometry for bits-per-weight accounting")
)

// AllocationErrorToken maps a sidecar error to its stable uppercase token for
// operator-facing surfaces (the fak-dev allocation-check verb prints it). An
// unrecognized error reports UNKNOWN.
func AllocationErrorToken(err error) string {
	switch {
	case errors.Is(err, ErrAllocationSidecarMalformed):
		return "ALLOCATION_SIDECAR_MALFORMED"
	case errors.Is(err, ErrAllocationSidecarMissingField):
		return "ALLOCATION_SIDECAR_MISSING_FIELD"
	case errors.Is(err, ErrAllocationSidecarEmpty):
		return "ALLOCATION_SIDECAR_EMPTY"
	case errors.Is(err, ErrAllocationSidecarUnknownType):
		return "ALLOCATION_SIDECAR_UNKNOWN_TYPE"
	case errors.Is(err, ErrAllocationDeclaredCountMismatch):
		return "ALLOCATION_DECLARED_COUNT_MISMATCH"
	case errors.Is(err, ErrAllocationCoverageMismatch):
		return "ALLOCATION_COVERAGE_MISMATCH"
	case errors.Is(err, ErrAllocationTypeMismatch):
		return "ALLOCATION_TYPE_MISMATCH"
	case errors.Is(err, ErrAllocationHistogramDrift):
		return "ALLOCATION_HISTOGRAM_DRIFT"
	case errors.Is(err, ErrAllocationBPWDrift):
		return "ALLOCATION_BPW_DRIFT"
	case errors.Is(err, ErrAllocationUnsupportedType):
		return "ALLOCATION_UNSUPPORTED_TYPE"
	default:
		return "UNKNOWN"
	}
}

// AllocationTensor is one manifest row: a GGUF tensor name and the quantization
// type the RCO search assigned it (fixed non-RCO tensors appear verbatim).
type AllocationTensor struct {
	Name string
	Type TensorType
}

// AllocationSidecar is a parsed .rco-allocation.txt manifest. Tensors keeps the
// file's row order; Histogram is the declared quant-type counts from the header
// (NOT recomputed from Tensors — a truncated excerpt legitimately disagrees,
// and ValidateAllocation is the seam that compares against an inventory).
type AllocationSidecar struct {
	Title           string
	Repository      string
	GGUFShard       string
	Sources         []string
	TargetBPW       float64
	TargetScope     string // raw label before the colon, e.g. "whole-file bpw"
	MeasuredBPW     *float64
	GGUFVersion     *uint64
	MetadataEntries *uint64
	TensorCount     uint64
	Histogram       map[TensorType]int
	Tensors         []AllocationTensor
}

const allocationTitleMarker = "# RCO per-tensor GGUF allocation"

// allocationMaxBytes bounds a manifest read: published manifests are tens of
// kilobytes; anything larger is not a sidecar and must not be buffered.
const allocationMaxBytes = 16 << 20

var (
	// allocationTargetRe matches the header's target-bits-per-weight line in
	// both observed scopes ("Target whole-file bpw: 3.00" and "Target bpw over
	// the backbone (...): 2.00 (measured: 2.0038)").
	allocationTargetRe = regexp.MustCompile(`^Target\s+(?P<scope>[^:]*bpw[^:]*?)\s*:\s*(?P<target>\d+(?:\.\d+)?)\s*(?:\(measured:\s*(?P<measured>\d+(?:\.\d+)?)\))?\s*$`)
	// allocationNameRe is the closed character set of GGUF tensor names.
	allocationNameRe = regexp.MustCompile(`^[A-Za-z0-9._]+$`)
	// allocationCountPairRe parses one "TYPE=n" histogram pair.
	allocationCountPairRe = regexp.MustCompile(`^(?P<type>[A-Za-z0-9_]+)=(?P<count>\d+)$`)
)

// allocationTypeNames maps every canonical GGML type name this loader knows to
// its TensorType. It is built from TensorType.String() so the accepted spelling
// can never drift from the names the rest of the package renders. Any other
// token refuses closed as ErrAllocationSidecarUnknownType.
var allocationTypeNames = func() map[string]TensorType {
	m := make(map[string]TensorType)
	for _, t := range []TensorType{
		TensorF32, TensorF16, TensorBF16,
		TensorQ4_0, TensorQ4_1, TensorQ5_0, TensorQ5_1, TensorQ8_0,
		TensorQ2_K, TensorQ3_K, TensorQ4_K, TensorQ5_K, TensorQ6_K,
		TensorIQ2_XXS, TensorIQ2_XS, TensorIQ1_S, TensorIQ2_S, TensorIQ1_M,
		TensorMXFP4, TensorIQ4_NL, TensorIQ3_S, TensorIQ4_XS, TensorIQ3_XXS,
		TensorQ2_0, TensorPQ2_0, TensorQ1_0, TensorHQQ4,
	} {
		m[t.String()] = t
	}
	return m
}()

// ParseAllocationSidecar parses a .rco-allocation.txt manifest from r. It is
// purely syntactic: it records what the file claims and refuses structural
// defects; semantic agreement with a GGUF inventory is ValidateAllocation's
// job. Required fields: the title marker, a target-bpw line, a tensor count,
// and a quant-type histogram. Everything else (repository, shard, sources,
// measured bpw, GGUF version, metadata entries) is captured when present.
func ParseAllocationSidecar(r io.Reader) (*AllocationSidecar, error) {
	data, err := io.ReadAll(io.LimitReader(r, allocationMaxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrAllocationSidecarMalformed, err)
	}
	if int64(len(data)) > allocationMaxBytes {
		return nil, fmt.Errorf("%w: manifest exceeds %d bytes", ErrAllocationSidecarMalformed, allocationMaxBytes)
	}
	sc := &AllocationSidecar{}
	lines := strings.Split(string(data), "\n")
	bodyStarted := false
	for i, raw := range lines {
		line := strings.TrimSuffix(raw, "\r")
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "#") {
			if bodyStarted {
				return nil, fmt.Errorf("%w: line %d: comment after tensor rows began", ErrAllocationSidecarMalformed, i+1)
			}
			if i == 0 || sc.Title == "" {
				if trimmed := strings.TrimSpace(line); trimmed != allocationTitleMarker {
					return nil, fmt.Errorf("%w: line %d: first line %q is not the RCO allocation title", ErrAllocationSidecarMalformed, i+1, trimmed)
				}
				sc.Title = allocationTitleMarker
				continue
			}
			if err := sc.parseHeaderLine(strings.TrimSpace(strings.TrimPrefix(line, "#")), i+1); err != nil {
				return nil, err
			}
			continue
		}
		bodyStarted = true
		row, err := parseAllocationRow(line, i+1)
		if err != nil {
			return nil, err
		}
		for _, prev := range sc.Tensors {
			if prev.Name == row.Name {
				return nil, fmt.Errorf("%w: line %d: tensor %q listed more than once", ErrAllocationSidecarMalformed, i+1, row.Name)
			}
		}
		sc.Tensors = append(sc.Tensors, row)
	}
	// Zero values double as absent markers: a real target bpw is strictly
	// positive, parseHistogram refuses an empty pair list, and a manifest with
	// zero declared tensors is not one — so no separate presence flags are
	// needed.
	switch {
	case sc.TargetBPW == 0:
		return nil, fmt.Errorf("%w: no target bits-per-weight line", ErrAllocationSidecarMissingField)
	case len(sc.Histogram) == 0:
		return nil, fmt.Errorf("%w: no quant-type counts line", ErrAllocationSidecarMissingField)
	case sc.TensorCount == 0:
		return nil, fmt.Errorf("%w: no tensor count line", ErrAllocationSidecarMissingField)
	case len(sc.Tensors) == 0:
		return nil, ErrAllocationSidecarEmpty
	}
	return sc, nil
}

// LoadAllocationSidecar reads and parses the sidecar at path.
func LoadAllocationSidecar(path string) (*AllocationSidecar, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ParseAllocationSidecar(f)
}

// parseHeaderLine handles one known header field. Unrecognized comment lines
// are prose notes (the published manifests carry free-form scope explanations)
// and are ignored; the REQUIRED fields are enforced by the post-parse checks.
func (sc *AllocationSidecar) parseHeaderLine(body string, lineNo int) error {
	if body == "" {
		return nil // bare "#" separator
	}
	if m := allocationTargetRe.FindStringSubmatch(body); m != nil {
		if sc.TargetBPW != 0 {
			return fmt.Errorf("%w: line %d: multiple target-bpw lines", ErrAllocationSidecarMalformed, lineNo)
		}
		v, err := strconv.ParseFloat(m[allocationTargetRe.SubexpIndex("target")], 64)
		if err != nil || v <= 0 {
			return fmt.Errorf("%w: line %d: bad target bpw %q", ErrAllocationSidecarMalformed, lineNo, m[2])
		}
		sc.TargetBPW = v
		sc.TargetScope = strings.TrimSpace(m[allocationTargetRe.SubexpIndex("scope")])
		if meas := m[allocationTargetRe.SubexpIndex("measured")]; meas != "" {
			mv, err := strconv.ParseFloat(meas, 64)
			if err != nil {
				return fmt.Errorf("%w: line %d: bad measured bpw %q", ErrAllocationSidecarMalformed, lineNo, meas)
			}
			sc.MeasuredBPW = &mv
		}
		return nil
	}
	key, value, ok := strings.Cut(body, ":")
	if !ok {
		return nil // prose note
	}
	key = strings.TrimSpace(key)
	value = strings.TrimSpace(value)
	switch key {
	case "Repository":
		sc.Repository = value
	case "GGUF":
		sc.GGUFShard = value
	case "Source":
		if value == "" {
			return fmt.Errorf("%w: line %d: empty source URL", ErrAllocationSidecarMalformed, lineNo)
		}
		sc.Sources = append(sc.Sources, value)
	case "GGUF version":
		n, err := strconv.ParseUint(value, 10, 64)
		if err != nil {
			return fmt.Errorf("%w: line %d: bad GGUF version %q", ErrAllocationSidecarMalformed, lineNo, value)
		}
		sc.GGUFVersion = &n
	case "GGUF metadata entries":
		n, err := strconv.ParseUint(value, 10, 64)
		if err != nil {
			return fmt.Errorf("%w: line %d: bad metadata entry count %q", ErrAllocationSidecarMalformed, lineNo, value)
		}
		sc.MetadataEntries = &n
	case "Tensor count":
		n, err := strconv.ParseUint(value, 10, 64)
		if err != nil {
			return fmt.Errorf("%w: line %d: bad tensor count %q", ErrAllocationSidecarMalformed, lineNo, value)
		}
		sc.TensorCount = n
	case "Quant-type counts":
		return sc.parseHistogram(value, lineNo)
	}
	return nil
}

func (sc *AllocationSidecar) parseHistogram(value string, lineNo int) error {
	pairs := strings.Split(value, ",")
	hist := make(map[TensorType]int, len(pairs))
	for _, p := range pairs {
		m := allocationCountPairRe.FindStringSubmatch(strings.TrimSpace(p))
		if m == nil {
			return fmt.Errorf("%w: line %d: bad histogram pair %q", ErrAllocationSidecarMalformed, lineNo, p)
		}
		typ, ok := allocationTypeNames[m[allocationCountPairRe.SubexpIndex("type")]]
		if !ok {
			return fmt.Errorf("%w: line %d: histogram type %q", ErrAllocationSidecarUnknownType, lineNo, m[1])
		}
		n, err := strconv.Atoi(m[allocationCountPairRe.SubexpIndex("count")])
		if err != nil || n < 0 {
			return fmt.Errorf("%w: line %d: bad histogram count %q", ErrAllocationSidecarMalformed, lineNo, p)
		}
		hist[typ] += n
	}
	if len(hist) == 0 {
		return fmt.Errorf("%w: line %d: empty quant-type counts", ErrAllocationSidecarMalformed, lineNo)
	}
	sc.Histogram = hist
	return nil
}

func parseAllocationRow(line string, lineNo int) (AllocationTensor, error) {
	idx := strings.Index(line, ":")
	if idx <= 0 || idx == len(line)-1 {
		return AllocationTensor{}, fmt.Errorf("%w: line %d: row %q is not \"<tensor>: <TYPE>\"", ErrAllocationSidecarMalformed, lineNo, line)
	}
	name := strings.TrimSpace(line[:idx])
	typName := strings.TrimSpace(line[idx+1:])
	if !allocationNameRe.MatchString(name) {
		return AllocationTensor{}, fmt.Errorf("%w: line %d: invalid tensor name %q", ErrAllocationSidecarMalformed, lineNo, name)
	}
	typ, ok := allocationTypeNames[typName]
	if !ok {
		return AllocationTensor{}, fmt.Errorf("%w: line %d: tensor %q type %q", ErrAllocationSidecarUnknownType, lineNo, name, typName)
	}
	return AllocationTensor{Name: name, Type: typ}, nil
}

// DefaultAllocationBPWTolerance is the default absolute bits-per-weight band
// around the declared target that ValidateAllocation accepts. Published
// manifests round their target to two decimals while the exact per-block
// geometry lands within ~0.005 bpw of it; the band is a sanity bound against a
// wrong-build/wrong-sidecar pairing, not a quality claim.
const DefaultAllocationBPWTolerance = 0.05

// AllocationValidationOptions tunes ValidateAllocation. Zero BPWTolerance
// selects DefaultAllocationBPWTolerance; ExcludeBpw, when non-nil, removes
// tensors from the bits-per-weight scope only (coverage, type and histogram
// checks always span the full inventory) — the seam for scoped targets such as
// "backbone, all tensors except the Engram table".
type AllocationValidationOptions struct {
	BPWTolerance float64
	ExcludeBpw   func(name string) bool
}

// EffectiveBpw computes the inventory's effective bits-per-weight:
// 8 * sum(ceil(numel/blockSize)*blockSize bytes) / sum(numel), using this
// package's own block-geometry table. exclude may remove tensors from the
// scope; a type without geometry refuses closed.
func EffectiveBpw(tensors []TensorInfo, exclude func(string) bool) (float64, error) {
	var payloadBytes, numel uint64
	for _, t := range tensors {
		if exclude != nil && exclude(t.Name) {
			continue
		}
		tr, ok := tensorTypeTraits(t.Type)
		if !ok {
			return 0, fmt.Errorf("%w: tensor %s has type %s", ErrAllocationUnsupportedType, t.Name, t.Type)
		}
		n := uint64(1)
		for _, d := range t.Dims {
			n *= d
		}
		blocks := (n + tr.BlockSize - 1) / tr.BlockSize
		payloadBytes += blocks * tr.TypeSize
		numel += n
	}
	if numel == 0 {
		return 0, fmt.Errorf("%w: no tensors contribute to the bits-per-weight scope", ErrAllocationUnsupportedType)
	}
	return float64(payloadBytes) * 8 / float64(numel), nil
}

// ValidateAllocation cross-checks a parsed sidecar against a parsed GGUF
// tensor inventory (this package's own File.Tensors — no duplicated header
// logic). Checks run in a fixed order and the FIRST failing class is returned,
// wrapped around its sentinel so callers can match with errors.Is:
//
//  1. declared tensor count vs listed rows;
//  2. coverage: the two name sets agree exactly (both directions);
//  3. per-tensor type agreement;
//  4. declared histogram vs computed histogram;
//  5. effective bits-per-weight within opts.BPWTolerance of the target.
func ValidateAllocation(sc *AllocationSidecar, tensors []TensorInfo, opts AllocationValidationOptions) error {
	if sc == nil || len(sc.Tensors) == 0 {
		return ErrAllocationSidecarEmpty
	}
	if int64(sc.TensorCount) != int64(len(sc.Tensors)) {
		return fmt.Errorf("%w: declared %d, listed %d", ErrAllocationDeclaredCountMismatch, sc.TensorCount, len(sc.Tensors))
	}
	ggufByName := make(map[string]TensorInfo, len(tensors))
	for _, t := range tensors {
		if _, dup := ggufByName[t.Name]; dup {
			return fmt.Errorf("%w: duplicate tensor %q in the GGUF inventory", ErrAllocationCoverageMismatch, t.Name)
		}
		ggufByName[t.Name] = t
	}
	sidecarSet := make(map[string]struct{}, len(sc.Tensors))
	var onlyInSidecar, onlyInGGUF []string
	for _, row := range sc.Tensors {
		sidecarSet[row.Name] = struct{}{}
		if _, ok := ggufByName[row.Name]; !ok {
			onlyInSidecar = append(onlyInSidecar, row.Name)
		}
	}
	for _, t := range tensors {
		if _, ok := sidecarSet[t.Name]; !ok {
			onlyInGGUF = append(onlyInGGUF, t.Name)
		}
	}
	if len(onlyInSidecar) > 0 || len(onlyInGGUF) > 0 {
		return fmt.Errorf("%w: %s", ErrAllocationCoverageMismatch, describeAllocationSets(onlyInSidecar, onlyInGGUF))
	}
	for _, row := range sc.Tensors {
		got := ggufByName[row.Name].Type
		if got != row.Type {
			return fmt.Errorf("%w: tensor %s sidecar=%s gguf=%s", ErrAllocationTypeMismatch, row.Name, row.Type, got)
		}
	}
	computed := make(map[TensorType]int)
	for _, t := range tensors {
		computed[t.Type]++
	}
	types := make([]TensorType, 0, len(computed)+len(sc.Histogram))
	seen := make(map[TensorType]struct{})
	for typ := range computed {
		types = append(types, typ)
		seen[typ] = struct{}{}
	}
	for typ := range sc.Histogram {
		if _, ok := seen[typ]; !ok {
			types = append(types, typ)
		}
	}
	sort.Slice(types, func(i, j int) bool { return types[i].String() < types[j].String() })
	for _, typ := range types {
		want, got := sc.Histogram[typ], computed[typ]
		if want != got {
			return fmt.Errorf("%w: type %s declared=%d inventory=%d", ErrAllocationHistogramDrift, typ, want, got)
		}
	}
	tol := opts.BPWTolerance
	if tol <= 0 {
		tol = DefaultAllocationBPWTolerance
	}
	bpw, err := EffectiveBpw(tensors, opts.ExcludeBpw)
	if err != nil {
		return err
	}
	if diff := math.Abs(bpw - sc.TargetBPW); diff > tol {
		return fmt.Errorf("%w: effective %.4f vs target %.4f (tolerance %.4f)", ErrAllocationBPWDrift, bpw, sc.TargetBPW, tol)
	}
	return nil
}

func describeAllocationSets(onlyInSidecar, onlyInGGUF []string) string {
	parts := make([]string, 0, 2)
	if len(onlyInSidecar) > 0 {
		parts = append(parts, fmt.Sprintf("%d in sidecar but not the GGUF (%s)", len(onlyInSidecar), truncateNames(onlyInSidecar)))
	}
	if len(onlyInGGUF) > 0 {
		parts = append(parts, fmt.Sprintf("%d in the GGUF but not the sidecar (%s)", len(onlyInGGUF), truncateNames(onlyInGGUF)))
	}
	return strings.Join(parts, "; ")
}

func truncateNames(names []string) string {
	const limit = 5
	if len(names) > limit {
		names = names[:limit]
	}
	return strings.Join(names, ", ")
}
