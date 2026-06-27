package ggufload

import (
	"encoding/json"
	"fmt"
	"sort"
)

// String renders the GGUF metadata ValueType as its spec name (e.g. "uint32",
// "string", "array"), falling back to "ValueType(n)" for an unrecognized tag. It
// is the metadata-export counterpart of TensorType.String (which names the tensor
// block layout); together they let a full export label every value by its on-disk
// GGUF type rather than a bare numeric tag.
func (t ValueType) String() string {
	switch t {
	case TypeUint8:
		return "uint8"
	case TypeInt8:
		return "int8"
	case TypeUint16:
		return "uint16"
	case TypeInt16:
		return "int16"
	case TypeUint32:
		return "uint32"
	case TypeInt32:
		return "int32"
	case TypeFloat32:
		return "float32"
	case TypeBool:
		return "bool"
	case TypeString:
		return "string"
	case TypeArray:
		return "array"
	case TypeUint64:
		return "uint64"
	case TypeInt64:
		return "int64"
	case TypeFloat64:
		return "float64"
	default:
		return fmt.Sprintf("ValueType(%d)", t)
	}
}

// MetaKV is one metadata key/value in a full export: the key, its GGUF type name,
// and the decoded value rendered losslessly — scalars as their natural Go type and
// arrays as a flat []any of the same, recursively. For an array, ElemType names the
// element type (the GGUF wire format tags an array with a single element type).
type MetaKV struct {
	Key      string `json:"key"`
	Type     string `json:"type"`
	ElemType string `json:"elem_type,omitempty"`
	Value    any    `json:"value"`
}

// TensorExport is one tensor's directory entry in a full export: its name, GGUF
// block type, dimensions, and in-data offset. (The absolute FileOffset is derived
// at load time and intentionally omitted — the export describes the checkpoint as
// authored, not where it landed in a particular reader's address space.)
type TensorExport struct {
	Name   string   `json:"name"`
	Type   string   `json:"type"`
	Dims   []uint64 `json:"dims"`
	Offset uint64   `json:"offset"`
}

// Metadata is a complete, machine-readable snapshot of a parsed GGUF header: its
// format version, data alignment, the offset where the tensor blob begins, every
// metadata key/value (no array truncation, keys sorted for a stable diff), and the
// tensor directory in file order. It marshals to JSON via MetadataJSON.
type Metadata struct {
	Version          uint32         `json:"version"`
	Alignment        uint64         `json:"alignment"`
	TensorDataOffset int64          `json:"tensor_data_offset"`
	Metadata         []MetaKV       `json:"metadata"`
	Tensors          []TensorExport `json:"tensors"`
}

// ExportMetadata builds the full, lossless metadata snapshot of the parsed file:
// every key/value (arrays kept in full, not summarized), every tensor directory
// entry, and the header scalars. Metadata keys are sorted so two exports of the
// same checkpoint are byte-identical; tensors keep their GGUF file order.
//
// Unlike the human-oriented ggufprobe dump (which summarizes a 248k-token vocab so
// it does not flood a terminal), this is the faithful export: it preserves every
// element of every array so the result round-trips the on-disk metadata.
func (f *File) ExportMetadata() Metadata {
	keys := make([]string, 0, len(f.Metadata))
	for k := range f.Metadata {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	kvs := make([]MetaKV, 0, len(keys))
	for _, k := range keys {
		v := f.Metadata[k]
		kv := MetaKV{Key: k, Type: v.Type.String(), Value: exportMetaValue(v)}
		if v.Type == TypeArray {
			kv.ElemType = arrayElemType(v).String()
		}
		kvs = append(kvs, kv)
	}

	tensors := make([]TensorExport, 0, len(f.Tensors))
	for _, t := range f.Tensors {
		tensors = append(tensors, TensorExport{
			Name:   t.Name,
			Type:   t.Type.String(),
			Dims:   t.Dims,
			Offset: t.Offset,
		})
	}

	return Metadata{
		Version:          f.Version,
		Alignment:        f.Alignment,
		TensorDataOffset: f.TensorDataOffset,
		Metadata:         kvs,
		Tensors:          tensors,
	}
}

// MetadataJSON renders ExportMetadata as indented JSON — the full-metadata-export
// surface for tooling that wants the complete header (every KV and tensor) in a
// machine-readable form rather than the summarized ggufprobe dump.
func (f *File) MetadataJSON() ([]byte, error) {
	return json.MarshalIndent(f.ExportMetadata(), "", "  ")
}

// arrayElemType reports the element ValueType of an array metadata value, or
// TypeArray itself when the value is empty or not an array (no element to name).
func arrayElemType(v Value) ValueType {
	if v.Type != TypeArray {
		return v.Type
	}
	items, ok := v.Value.([]Value)
	if !ok || len(items) == 0 {
		return TypeArray
	}
	return items[0].Type
}

// exportMetaValue renders a decoded GGUF Value as a JSON-marshalable Go value:
// scalars pass through as their natural type and an array becomes a flat []any of
// its recursively-rendered elements — preserving every element so the export is
// lossless.
func exportMetaValue(v Value) any {
	if v.Type != TypeArray {
		return v.Value
	}
	items, ok := v.Value.([]Value)
	if !ok {
		return v.Value
	}
	out := make([]any, len(items))
	for i, it := range items {
		out[i] = exportMetaValue(it)
	}
	return out
}
