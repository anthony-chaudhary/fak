package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
)

const v41DSparkMaxHeaderBytes = 16 << 20

// V41DSparkPackedShard exposes only validated bundled-draft tensor spans. It
// owns a ReaderAt, never an expanded model or whole-shard buffer. A partial
// shard is useful for bounded intake but is not evidence of a complete drafter.
// Call Missing and combine the inventory across all pinned artifact shards
// before considering a later runtime load. Execution remains unsupported.
// The caller must keep the backing artifact immutable until Close; schema
// validation does not authenticate its bytes or attest a checkpoint digest.
type V41DSparkPackedShard struct {
	mu        sync.RWMutex
	closed    bool
	closeErr  error
	packed    *V41PackedShard
	inventory *V41DSparkInventory
	sources   map[string]string // canonical weight role -> exact original header key
}

// OpenV41DSparkPackedShard reads at most a 16 MiB safetensors header and no
// tensor payload. It preserves independent draft quantization and exact source
// keys. Non-draft tensors are not exposed; unknown/ambiguous mtp.* names refuse.
// Required weight/scale pairs must be colocated in this shard.
func OpenV41DSparkPackedShard(path string, cfg Config) (*V41DSparkPackedShard, error) {
	inv, err := DeepSeekV41DSparkInventory(cfg)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("%w: open: %v", ErrV41DSparkArtifact, err)
	}
	stat, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("%w: stat: %v", ErrV41DSparkArtifact, err)
	}
	sf, err := readV41DSparkHeader(f, stat.Size(), f)
	if err != nil { // readV41DSparkHeader closes f on failure.
		return nil, err
	}
	shard, err := bindV41DSparkPackedShard(path, inv, sf)
	if err != nil {
		_ = sf.Close()
		return nil, err
	}
	return shard, nil
}

// readV41DSparkHeader owns closer and reads the length prefix exactly once.
// The checked byte count is the count subsequently read and parsed: a mutable
// ReaderAt cannot replace a checked prefix before a second parser reads it.
func readV41DSparkHeader(r io.ReaderAt, size int64, closer io.Closer) (*safetensorsFile, error) {
	fail := func(err error) (*safetensorsFile, error) {
		if closer != nil {
			_ = closer.Close()
		}
		return nil, err
	}
	if size < 8 {
		return fail(fmt.Errorf("%w: missing safetensors header", ErrV41DSparkArtifact))
	}
	var length [8]byte
	n, err := r.ReadAt(length[:], 0)
	if err != nil || n != len(length) {
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return fail(fmt.Errorf("%w: header length: %w", ErrV41DSparkArtifact, err))
	}
	hlen := binary.LittleEndian.Uint64(length[:])
	if hlen > v41DSparkMaxHeaderBytes || hlen > uint64(size-8) {
		return fail(fmt.Errorf("%w: oversized or out-of-bounds safetensors header", ErrV41DSparkArtifact))
	}
	header := make([]byte, int(hlen))
	n, err = r.ReadAt(header, 8)
	if err != nil || n != len(header) {
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return fail(fmt.Errorf("%w: header bytes: %w", ErrV41DSparkArtifact, err))
	}
	var hdr map[string]json.RawMessage
	if err := unmarshalUniqueJSON(header, &hdr); err != nil {
		return fail(fmt.Errorf("%w: header json: %w", ErrV41DSparkArtifact, err))
	}
	return &safetensorsFile{r: r, closer: closer, hdr: hdr, dataBase: 8 + int64(hlen), size: size}, nil
}

// DeepSeek's convert.py explicitly excludes these tied target-owned aliases.
// Admit only the exact three-stage spellings, not an arbitrary mtp.* suffix.
func v41DSparkTargetOwnedAlias(name string) bool {
	switch name {
	case "mtp.0.embed.weight", "mtp.1.embed.weight", "mtp.2.embed.weight",
		"mtp.0.head.weight", "mtp.1.head.weight", "mtp.2.head.weight":
		return true
	}
	return false
}

func bindV41DSparkPackedShard(path string, inv *V41DSparkInventory, sf *safetensorsFile) (*V41DSparkPackedShard, error) {
	actual := &V41Inventory{byName: map[string]V41TensorRef{}, layers: 3}
	p := &V41PackedShard{path: path, inv: actual, file: sf, entries: map[string]stEntry{}, refs: map[string]V41PackedTensor{}}
	s := &V41DSparkPackedShard{packed: p, inventory: inv, sources: map[string]string{}}
	allSources := map[string]string{}
	type span struct {
		start, end int64
		name       string
	}
	var spans []span
	for _, name := range safetensorsTensorNames(sf.hdr) {
		// Bounds and non-overlap belong to the whole file, including tensors
		// intentionally not exposed by this draft-only reader.
		var entry stEntry
		if err := json.Unmarshal(sf.hdr[name], &entry); err != nil {
			return nil, fmt.Errorf("%w: %s header: %v", ErrV41DSparkArtifact, name, err)
		}
		start, end, err := safetensorsDataBounds(sf.dataBase, sf.size, entry)
		if err != nil {
			return nil, fmt.Errorf("%w: %s bounds: %v", ErrV41DSparkArtifact, name, err)
		}
		if start != end {
			spans = append(spans, span{start, end, name})
		}
		canonical, draft := v41DSparkCanonicalName(name)
		if !draft {
			continue
		}
		if previous, ok := allSources[canonical]; ok {
			return nil, fmt.Errorf("%w: ambiguous aliases %s and %s", ErrV41DSparkArtifact, previous, name)
		}
		allSources[canonical] = name
		if v41DSparkTargetOwnedAlias(canonical) {
			continue
		}
		base := canonical
		isScale := strings.HasSuffix(canonical, ".scale")
		if isScale {
			base = strings.TrimSuffix(canonical, ".scale") + ".weight"
		}
		spec, ok := inv.Lookup(base)
		if !ok || (isScale && len(spec.ScaleShape) == 0) {
			return nil, fmt.Errorf("%w: unsupported draft tensor %s", ErrV41DSparkArtifact, name)
		}
		shape, dtypes := spec.Shape, spec.AllowedDtypes
		if isScale {
			shape, dtypes = spec.ScaleShape, []string{"F8_E8M0"}
		}
		allowed := false
		for _, dtype := range dtypes {
			allowed = allowed || entry.Dtype == dtype
		}
		if !allowed || !sameShape(entry.Shape, shape) {
			return nil, fmt.Errorf("%w: %s storage %s%v, want %v%v", ErrV41DSparkArtifact, name, entry.Dtype, entry.Shape, dtypes, shape)
		}
		width := 1
		if entry.Dtype == "BF16" {
			width = 2
		} else if entry.Dtype == "F32" {
			width = 4
		}
		axes := append(append([]int(nil), shape...), width)
		n, ok := checkedShapeProduct(axes...)
		if !ok || int64(n) != end-start {
			return nil, fmt.Errorf("%w: %s payload size does not match storage shape", ErrV41DSparkArtifact, name)
		}
		p.entries[name] = entry
		if !isScale {
			s.sources[base] = name
		}
	}
	if len(s.sources) == 0 {
		return nil, fmt.Errorf("%w: shard has no bundled draft weights", ErrV41DSparkArtifact)
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i].start < spans[j].start })
	for i := 1; i < len(spans); i++ {
		if spans[i].start < spans[i-1].end {
			return nil, fmt.Errorf("%w: overlapping shard tensor spans %s and %s", ErrV41DSparkArtifact, spans[i-1].name, spans[i].name)
		}
	}
	for canonical, source := range allSources {
		if strings.HasSuffix(canonical, ".scale") {
			if _, ok := s.sources[strings.TrimSuffix(canonical, ".scale")+".weight"]; !ok {
				return nil, fmt.Errorf("%w: orphan scale %s", ErrV41DSparkArtifact, source)
			}
		}
	}
	for _, canonical := range append(inv.Names(), inv.optional...) {
		name, present := s.sources[canonical]
		if !present {
			continue
		}
		spec, _ := inv.Lookup(canonical)
		entry := p.entries[name]
		ref := V41TensorRef{Name: name, Class: spec.Class, Layer: spec.Stage,
			Dtype: entry.Dtype, Shape: spec.Shape, LogicalShape: spec.LogicalShape, PackedWeight: spec.PackedWeight}
		tensor := V41PackedTensor{Shaper: path, ScaleStart: -1, ScaleEnd: -1}
		tensor.Start, tensor.End, _ = safetensorsDataBounds(sf.dataBase, sf.size, entry)
		if len(spec.ScaleShape) > 0 {
			scale := strings.TrimSuffix(canonical, ".weight") + ".scale"
			scaleName, ok := allSources[scale]
			if !ok {
				return nil, fmt.Errorf("%w: %s missing scale %s", ErrV41DSparkArtifact, name, scale)
			}
			ref.ScaleName, ref.ScaleDtype, ref.ScaleShape = scaleName, "F8_E8M0", spec.ScaleShape
			tensor.ScaleStart, tensor.ScaleEnd, _ = safetensorsDataBounds(sf.dataBase, sf.size, p.entries[scaleName])
		}
		tensor.Ref = ref
		actual.byName[name] = ref
		actual.order = append(actual.order, ref)
		actual.names = append(actual.names, name)
		p.refs[name] = tensor
	}
	return s, nil
}

// Missing lists canonical draft weight roles absent from this one shard. Even
// an empty list does not attest model hash, baseline parity or runtime support.
func (s *V41DSparkPackedShard) Missing() []string {
	if s == nil || s.inventory == nil {
		return nil
	}
	var missing []string
	for _, name := range s.inventory.names {
		if _, ok := s.sources[name]; !ok {
			missing = append(missing, name)
		}
	}
	return missing
}

// Reference returns a canonical role's original file identity and packed span.
// The explicit export aliases accepted while opening are also accepted here.
func (s *V41DSparkPackedShard) Reference(name string) (V41PackedTensor, error) {
	if s == nil {
		return V41PackedTensor{}, fmt.Errorf("%w: shard is closed", ErrV41DSparkArtifact)
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.referenceLocked(name)
}

func (s *V41DSparkPackedShard) referenceLocked(name string) (V41PackedTensor, error) {
	if s.closed || s.packed == nil || s.packed.file == nil {
		return V41PackedTensor{}, fmt.Errorf("%w: shard is closed", ErrV41DSparkArtifact)
	}
	canonical, draft := v41DSparkCanonicalName(name)
	source, present := s.sources[canonical]
	if !draft || !present {
		return V41PackedTensor{}, fmt.Errorf("%w: %s is not a draft weight in this shard", ErrV41DSparkArtifact, name)
	}
	ref := s.packed.refs[source]
	ref.Ref.Shape = append([]int(nil), ref.Ref.Shape...)
	ref.Ref.LogicalShape = append([]int(nil), ref.Ref.LogicalShape...)
	ref.Ref.ScaleShape = append([]int(nil), ref.Ref.ScaleShape...)
	return ref, nil
}

// Read performs one bounded read, preserving packed E2M1/FP8 bytes unchanged.
// A strictly positive byte bound is mandatory; zero never means unlimited.
func (s *V41DSparkPackedShard) Read(name string, bound int64) ([]byte, error) {
	if s == nil {
		return nil, fmt.Errorf("%w: shard is closed", ErrV41DSparkArtifact)
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	ref, err := s.referenceLocked(name)
	if err != nil {
		return nil, err
	}
	if bound <= 0 || ref.ByteLen() > bound {
		return nil, fmt.Errorf("%w: %s payload exceeds positive read bound %d", ErrV41DSparkArtifact, name, bound)
	}
	return s.packed.Read(ref, bound)
}

// ReadScale returns raw E8M0 exponent bytes under an explicit positive bound.
func (s *V41DSparkPackedShard) ReadScale(name string, bound int64) ([]byte, error) {
	if s == nil {
		return nil, fmt.Errorf("%w: shard is closed", ErrV41DSparkArtifact)
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	ref, err := s.referenceLocked(name)
	if err != nil {
		return nil, err
	}
	if !ref.HasScale() || bound <= 0 || ref.ScaleEnd-ref.ScaleStart > bound {
		return nil, fmt.Errorf("%w: %s scale missing or exceeds positive read bound %d", ErrV41DSparkArtifact, name, bound)
	}
	return s.packed.ReadScale(ref, bound)
}

// Close waits for active payload reads, closes once, and retains the first
// close result for every later caller. No new read starts after closure.
func (s *V41DSparkPackedShard) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return s.closeErr
	}
	s.closed = true
	if s.packed != nil && s.packed.file != nil {
		s.closeErr = s.packed.Close()
		s.packed.file = nil
	}
	return s.closeErr
}
