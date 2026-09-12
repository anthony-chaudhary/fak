package model

// v41_weights.go — the bounded lazy packed-reference reader for the
// DeepSeek-V4.1-Flash text checkpoint (issue #12892, leaf of parent #12640).
//
// The full V4.1 checkpoint is ~510 GB across 48 shards (DeepSeekV41CheckpointBytes
// in v41_budget.go). A developer loop must be able to name and read ONE tensor
// without materializing a shard, a tensor family, or the checkpoint. This file
// provides that seam:
//
//   - V41PackedShard opens a single .safetensors shard and indexes ONLY the
//     tensor positions the exact V41Inventory (v41_inventory.go) admits. Opening
//     reads the header and validates bounds; it reads no tensor payload.
//   - Reference(name) returns a lazy V41PackedTensor: identity, dtype, shapes and
//     the exact byte span, with no payload. This is what a sibling Engram /
//     attention / decoder consumer binds against.
//   - Read(ref, bound) performs exactly one bounded ReadAt over the existing
//     safetensors ReaderAt/mmap seam and returns the PACKED bytes unchanged
//     (FP8 stays FP8, packed MXFP4 stays packed I8). There is no dequantization,
//     no f32 expansion and no whole-shard residency. The returned slice is owned
//     by the caller and is length-capped by `bound`.
//
// Fail-closed discipline (mirrors v4ExpertSource / v4ShardedExpertSource):
//
//   - a config that is not admitted V4.1 refuses before any file IO;
//   - a requested name absent from the inventory, or present in the inventory but
//     absent from this shard, refuses (shard confinement) with no payload read;
//   - a tensor whose on-disk dtype or shape disagrees with the exact inventory
//     refuses before allocation;
//   - a tensor with an exact-inventory scale sibling requires that sibling to be
//     present in the same shard with the exact scale dtype/shape;
//   - a read larger than the caller's byte bound refuses before allocation.
//
// Nothing here admits or implements the native forward: ErrV41NativeUnsupported
// still fences every execution seam, and this reader is never wired into a
// generic loader.

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// ErrV41PackedUnsupported is the typed refusal for a V4.1 packed-read request the
// shard cannot safely serve: an unadmitted config, an unknown name, a tensor
// outside this shard, a dtype/shape that disagrees with the exact inventory, a
// missing or malformed scale sibling, or a request over the caller's byte bound.
var ErrV41PackedUnsupported = errors.New("model: DeepSeek V4.1 packed tensor request is not served")

// V41PackedTensor is a lazy reference to one packed V4.1 weight in one shard.
// It carries no payload: Start/End are the half-open byte span within the shard's
// data region, exactly as read by the existing safetensors ReaderAt seam.
type V41PackedTensor struct {
	Ref        V41TensorRef
	Shaper     string // the shard path this reference was resolved against
	Start      int64  // first payload byte, absolute within the shard file
	End        int64  // one past the last payload byte
	ScaleStart int64  // -1 when the tensor has no scale sibling
	ScaleEnd   int64
}

// ByteLen returns the packed payload length of the weight tensor.
func (t V41PackedTensor) ByteLen() int64 { return t.End - t.Start }

// HasScale reports whether a resolved scale sibling span is attached.
func (t V41PackedTensor) HasScale() bool { return t.Ref.HasScale() && t.ScaleStart >= 0 }

// V41PackedShard is a header-only index of one V4.1 safetensors shard, resolved
// against an exact inventory. It holds the underlying file handle so it can serve
// bounded per-tensor reads; it never holds a whole-shard buffer beyond the mmap
// the base reader may already have.
type V41PackedShard struct {
	path    string
	inv     *V41Inventory
	file    *safetensorsFile
	entries map[string]stEntry
	refs    map[string]V41PackedTensor
}

// OpenV41PackedShard opens one shard and indexes the inventory-admitted tensors
// it contains. cfg must be admitted V4.1; a non-V4.1 config refuses before the
// file is opened. The caller must Close the shard.
func OpenV41PackedShard(path string, cfg Config) (*V41PackedShard, error) {
	if !cfg.IsDeepSeekV41() || cfg.DeepSeekV41 == nil {
		return nil, fmt.Errorf("%w: config is not admitted DeepSeek V4.1", ErrV41PackedUnsupported)
	}
	inv, err := DeepSeekV41Inventory(cfg)
	if err != nil {
		return nil, err
	}
	return openV41PackedShardWith(path, inv, openSafetensorsFile)
}

// openV41PackedShardWith is the test/injection seam so a caller can supply an
// alternate opener (e.g. a read-only ReaderAt) without touching the file system.
func openV41PackedShardWith(path string, inv *V41Inventory, open safetensorsFileOpener) (*V41PackedShard, error) {
	if inv == nil {
		return nil, fmt.Errorf("%w: nil inventory", ErrV41PackedUnsupported)
	}
	sf, err := open(path)
	if err != nil {
		return nil, err
	}
	s := &V41PackedShard{
		path:    path,
		inv:     inv,
		file:    sf,
		entries: map[string]stEntry{},
		refs:    map[string]V41PackedTensor{},
	}
	for name, raw := range sf.hdr {
		if name == "__metadata__" {
			continue
		}
		base := name
		if strings.HasSuffix(name, ".scale") {
			base = strings.TrimSuffix(name, ".scale") + ".weight"
		}
		ref, ok := inv.Lookup(base)
		if !ok {
			continue // not a V4.1 text tensor; ignore foreign tensors.
		}
		var entry stEntry
		if err := json.Unmarshal(raw, &entry); err != nil {
			_ = sf.Close()
			return nil, fmt.Errorf("%w: %s header: %v", ErrV41PackedUnsupported, name, err)
		}
		if err := s.validateEntry(name, ref, entry); err != nil {
			_ = sf.Close()
			return nil, err
		}
		s.entries[name] = entry
	}
	if err := s.buildRefs(); err != nil {
		_ = sf.Close()
		return nil, err
	}
	return s, nil
}

// validateEntry checks one header entry against the exact inventory before it is
// admitted to the index. No payload is read.
func (s *V41PackedShard) validateEntry(name string, ref V41TensorRef, entry stEntry) error {
	isScale := name == ref.ScaleName
	wantDtype := ref.Dtype
	wantShape := ref.Shape
	if isScale {
		wantDtype = ref.ScaleDtype
		wantShape = ref.ScaleShape
	}
	if entry.Dtype != wantDtype {
		return fmt.Errorf("%w: %s dtype %q, want %q", ErrV41PackedUnsupported, name, entry.Dtype, wantDtype)
	}
	if !sameShape(entry.Shape, wantShape) {
		return fmt.Errorf("%w: %s shape %v, want %v", ErrV41PackedUnsupported, name, entry.Shape, wantShape)
	}
	for _, dim := range entry.Shape {
		if dim <= 0 {
			return fmt.Errorf("%w: %s has non-positive shape %v", ErrV41PackedUnsupported, name, entry.Shape)
		}
	}
	if _, _, err := safetensorsDataBounds(s.file.dataBase, s.file.size, entry); err != nil {
		return fmt.Errorf("%w: %s: %v", ErrV41PackedUnsupported, name, err)
	}
	return nil
}

// buildRefs resolves each admitted weight into a lazy V41PackedTensor, pairing
// it with its scale sibling when the inventory requires one. A missing or
// malformed scale for an inventory-declared scaled tensor fails closed.
func (s *V41PackedShard) buildRefs() error {
	for name, entry := range s.entries {
		if strings.HasSuffix(name, ".scale") {
			continue
		}
		ref, ok := s.inv.Lookup(name)
		if !ok {
			continue
		}
		tensor := V41PackedTensor{Ref: ref, Shaper: s.path, ScaleStart: -1, ScaleEnd: -1}
		start, end, err := safetensorsDataBounds(s.file.dataBase, s.file.size, entry)
		if err != nil {
			return fmt.Errorf("%w: %s: %v", ErrV41PackedUnsupported, name, err)
		}
		tensor.Start, tensor.End = start, end
		if ref.HasScale() {
			scaleEntry, ok := s.entries[ref.ScaleName]
			if !ok {
				return fmt.Errorf("%w: %s is missing its scale sibling %s", ErrV41PackedUnsupported, name, ref.ScaleName)
			}
			ss, se, err := safetensorsDataBounds(s.file.dataBase, s.file.size, scaleEntry)
			if err != nil {
				return fmt.Errorf("%w: %s: %v", ErrV41PackedUnsupported, ref.ScaleName, err)
			}
			tensor.ScaleStart, tensor.ScaleEnd = ss, se
		}
		s.refs[name] = tensor
	}
	return nil
}

// Close releases the shard's file handle.
func (s *V41PackedShard) Close() error {
	if s == nil || s.file == nil {
		return nil
	}
	return s.file.Close()
}

// Len returns the number of inventory-admitted weight tensors in this shard.
func (s *V41PackedShard) Len() int {
	if s == nil {
		return 0
	}
	return len(s.refs)
}

// Reference returns the lazy packed reference for a weight name. It performs no
// payload read. A name absent from the inventory, or present in the inventory
// but not in this shard, refuses — the caller cannot silently address a tensor
// that lives in a different shard.
func (s *V41PackedShard) Reference(name string) (V41PackedTensor, error) {
	if s == nil {
		return V41PackedTensor{}, fmt.Errorf("%w: nil shard", ErrV41PackedUnsupported)
	}
	if _, ok := s.inv.Lookup(name); !ok {
		return V41PackedTensor{}, fmt.Errorf("%w: %s is not a V4.1 inventory tensor", ErrV41PackedUnsupported, name)
	}
	ref, ok := s.refs[name]
	if !ok {
		return V41PackedTensor{}, fmt.Errorf("%w: %s is not present in shard %s", ErrV41PackedUnsupported, name, s.path)
	}
	return ref, nil
}

// Read returns the packed payload for a resolved tensor, performing exactly one
// bounded read over the shard's ReaderAt. bound is the caller's allocation cap:
// a payload larger than bound refuses before any allocation. The returned bytes
// are the on-disk packed bytes, unchanged.
func (s *V41PackedShard) Read(t V41PackedTensor, bound int64) ([]byte, error) {
	if s == nil || s.file == nil {
		return nil, fmt.Errorf("%w: nil shard", ErrV41PackedUnsupported)
	}
	if bound > 0 && t.ByteLen() > bound {
		return nil, fmt.Errorf("%w: %s packed payload %d exceeds bound %d", ErrV41PackedUnsupported, t.Ref.Name, t.ByteLen(), bound)
	}
	entry, ok := s.entries[t.Ref.Name]
	if !ok {
		return nil, fmt.Errorf("%w: %s is not present in shard %s", ErrV41PackedUnsupported, t.Ref.Name, s.path)
	}
	start, end, err := safetensorsDataBounds(s.file.dataBase, s.file.size, entry)
	if err != nil {
		return nil, fmt.Errorf("%w: %s: %v", ErrV41PackedUnsupported, t.Ref.Name, err)
	}
	if start != t.Start || end != t.End {
		return nil, fmt.Errorf("%w: %s span moved %d..%d -> %d..%d", ErrV41PackedUnsupported, t.Ref.Name, t.Start, t.End, start, end)
	}
	payload, err := s.file.tensorBytes(entry)
	if err != nil {
		return nil, fmt.Errorf("read V4.1 packed tensor %s: %w", t.Ref.Name, err)
	}
	return payload, nil
}

// ReadScale returns the packed E8M0 scale payload for a reference that has one.
// It refuses when the tensor carries no scale or when the payload exceeds bound.
func (s *V41PackedShard) ReadScale(t V41PackedTensor, bound int64) ([]byte, error) {
	if s == nil || s.file == nil {
		return nil, fmt.Errorf("%w: nil shard", ErrV41PackedUnsupported)
	}
	if !t.HasScale() {
		return nil, fmt.Errorf("%w: %s has no scale sibling", ErrV41PackedUnsupported, t.Ref.Name)
	}
	if bound > 0 && t.ScaleEnd-t.ScaleStart > bound {
		return nil, fmt.Errorf("%w: %s scale payload %d exceeds bound %d", ErrV41PackedUnsupported, t.Ref.ScaleName, t.ScaleEnd-t.ScaleStart, bound)
	}
	entry, ok := s.entries[t.Ref.ScaleName]
	if !ok {
		return nil, fmt.Errorf("%w: %s is not present in shard %s", ErrV41PackedUnsupported, t.Ref.ScaleName, s.path)
	}
	payload, err := s.file.tensorBytes(entry)
	if err != nil {
		return nil, fmt.Errorf("read V4.1 scale %s: %w", t.Ref.ScaleName, err)
	}
	return payload, nil
}
