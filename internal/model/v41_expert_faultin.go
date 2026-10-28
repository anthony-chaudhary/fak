package model

import (
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41ExpertFaultIn binds immutable admitted checkpoint extents, not copied
// payloads. Its reader must remain immutable and open for the model-weight
// lifetime (bind the checkpoint's closer with SetWeightCloser). Registration
// does not qualify an artifact or lift the V4.1 loader/numerical admission gates.
type v41ExpertFaultIn struct {
	r                         io.ReaderAt
	weightOffset, scaleOffset int64
	weightBytes, scaleBytes   int
}

// installV41ExpertQuantSource registers one MXFP4 projection without payload I/O.
// stagingBytes is the explicit packed HOST staging cap shared by every source
// registered on this model; it excludes caller activations, backend allocations
// and descriptor metadata. The existing device-byte-in kernel is synchronous;
// only one source projection can borrow the staging slab at a time. This is a
// stream-through first-light seam, not an expert cache or device-memory planner.
func installV41ExpertQuantSource(weightName, scaleName string, we, se stEntry,
	r io.ReaderAt, dataBase, sourceBytes int64, stagingBytes int, m *Model) error {
	if m == nil || !m.Cfg.IsDeepSeekV41() {
		return v4QuantMetadataf("V4.1 source install: missing V4.1 model")
	}
	_, projection, err := parseV41ExpertQuantWeightName(weightName)
	if err != nil {
		return err
	}
	if scaleName != strings.TrimSuffix(weightName, ".weight")+".scale" || we.Dtype != "I8" || se.Dtype != "F8_E8M0" {
		return v4QuantMetadataf("V4.1 source install %s: mismatched scale identity or packed dtype", weightName)
	}
	spec, ok := selectV41ExpertQuantSpec(projection, we.Shape, se.Shape)
	if !ok {
		return v4QuantMetadataf("V4.1 source install %s: unsupported packed geometry", weightName)
	}
	wb, wok := checkedShapeProduct(spec.weightRows, spec.weightCols)
	sb, sok := checkedShapeProduct(spec.scaleRows, spec.scaleCols)
	unpacked, iok := checkedShapeProduct(spec.weightCols, 2)
	if !wok || !sok || !iok || wb > stagingBytes || sb > stagingBytes-wb || r == nil {
		return v4QuantMetadataf("V4.1 source install %s: invalid source or insufficient staging budget", weightName)
	}
	wo, err := v41ExpertSourceExtent(we, wb, dataBase, sourceBytes)
	if err != nil {
		return err
	}
	so, err := v41ExpertSourceExtent(se, sb, dataBase, sourceBytes)
	if err != nil {
		return err
	}
	if wo < so+int64(sb) && so < wo+int64(wb) {
		return v4QuantMetadataf("V4.1 source install %s: weight and scale extents overlap", weightName)
	}
	owner := m.ensureWeightCloser()
	owner.mu.Lock()
	defer owner.mu.Unlock()
	if owner.closing || owner.closed {
		return v4QuantMetadataf("V4.1 source install %s: model closing or closed", weightName)
	}
	store := owner.v41ExpertPacked
	if store == nil {
		store = &v41ExpertDeviceInstall{weights: map[string]*v41ExpertDeviceWeight{}}
		owner.v41ExpertPacked = store
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.weights[weightName] != nil {
		return v4QuantMetadataf("V4.1 source install %s: already installed", weightName)
	}
	if store.faultBudget != 0 && store.faultBudget != stagingBytes {
		return v4QuantMetadataf("V4.1 source install %s: staging budget differs from existing source", weightName)
	}
	if store.faultBudget == 0 {
		store.faultBudget = stagingBytes
	}
	store.weights[weightName] = &v41ExpertDeviceWeight{
		projection: projection, dtype: "I8", O: spec.weightRows, I: unpacked,
		source: &v41ExpertFaultIn{r: r, weightOffset: wo, scaleOffset: so, weightBytes: wb, scaleBytes: sb},
	}
	store.packedPreserved = true
	return nil
}

func v41ExpertSourceExtent(e stEntry, want int, base, size int64) (int64, error) {
	if base < 0 || size < base || len(e.DataOffsets) != 2 || e.DataOffsets[0] < 0 || e.DataOffsets[1] < e.DataOffsets[0] || e.DataOffsets[1]-e.DataOffsets[0] != want {
		return 0, v4QuantMetadataf("V4.1 source install: invalid tensor extent")
	}
	start, end := int64(e.DataOffsets[0]), int64(e.DataOffsets[1])
	if end > math.MaxInt64-base || base+end > size {
		return 0, v4QuantMetadataf("V4.1 source install: tensor extent exceeds immutable source")
	}
	return base + start, nil
}

func (m *Model) v41ExpertFaultMatmul(name string, store *v41ExpertDeviceInstall, w *v41ExpertDeviceWeight, be compute.Backend, x []float32) ([]float32, error) {
	// Refuse a new fault once close starts; an admitted call pins its ReaderAt
	// through the synchronous kernel, including every error/panic exit.
	if !m.holdModelWeights() {
		return nil, v4QuantMetadataf("V4.1 source matmul %s: model closing or closed", name)
	}
	defer m.releaseWeightSession()
	if _, ok := be.(compute.RoutedExpertDeviceKernel); !ok {
		return nil, compute.ErrExpertDeviceKernelAbsent
	}
	store.faultMu.Lock()
	defer store.faultMu.Unlock()
	if store.faultSlab == nil {
		store.faultSlab = make([]byte, store.faultBudget)
	}
	src := w.source
	weights := store.faultSlab[:src.weightBytes:src.weightBytes]
	scales := store.faultSlab[src.weightBytes : src.weightBytes+src.scaleBytes : src.weightBytes+src.scaleBytes]
	if err := v41ReadExpertExtent(src.r, weights, src.weightOffset); err != nil {
		return nil, fmt.Errorf("V4.1 source %s weight: %w", name, err)
	}
	if err := v41ReadExpertExtent(src.r, scales, src.scaleOffset); err != nil {
		return nil, fmt.Errorf("V4.1 source %s scales: %w", name, err)
	}
	for i, scale := range scales {
		if scale == 0xff {
			return nil, v4QuantMetadataf("V4.1 source %s scale byte %d is F8_E8M0 NaN", name, i)
		}
	}
	return compute.GemvMXFP4ExpertDevice(name, be, w.O, w.I, weights, scales, x)
}

func v41ReadExpertExtent(r io.ReaderAt, dst []byte, offset int64) error {
	n, err := r.ReadAt(dst, offset)
	if err != nil {
		return err
	}
	if n != len(dst) {
		return io.ErrUnexpectedEOF
	}
	return nil
}
