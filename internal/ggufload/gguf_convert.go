package ggufload

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
)

const conversionHeaderLimit int64 = 64 << 20

// ConversionReport describes an opt-in local artifact conversion, not model
// quality or performance. The source and all noneligible payloads remain intact.
type ConversionReport struct {
	ConvertedTensors int    `json:"converted_tensors"`
	PreservedTensors int    `json:"preserved_tensors"`
	InputBytes       int64  `json:"input_bytes"`
	OutputBytes      int64  `json:"output_bytes"`
	Type             string `json:"type"`
}

// ConvertGGUF converts F32 matrices in one local GGUFv3 file to Q4_K, Q5_K or
// Q6_K. It never resolves aliases, downloads weights, discovers shards or installs
// caches. Output must not exist. Publication is a same-directory hard link after
// bounded validation/reload; unsupported filesystems fail without overwriting.
// Cancellation before publication removes only this invocation's temporary file.
func ConvertGGUF(ctx context.Context, input, output string, target TensorType) (report ConversionReport, err error) {
	fileType, ok := map[TensorType]uint32{TensorQ4_K: 14, TensorQ5_K: 16, TensorQ6_K: 18}[target]
	if !ok {
		return report, fmt.Errorf("gguf: conversion requires q4_k, q5_k or q6_k")
	}
	if input == "" || output == "" {
		return report, fmt.Errorf("gguf: input and output paths required")
	}
	if err := ctx.Err(); err != nil {
		return report, err
	}
	if _, err := os.Lstat(output); !errors.Is(err, os.ErrNotExist) {
		if err == nil {
			err = fmt.Errorf("gguf: output already exists")
		}
		return report, err
	}
	in, err := os.Open(input)
	if err != nil {
		return report, err
	}
	defer in.Close()
	stat, err := in.Stat()
	if err != nil {
		return report, err
	}
	if !stat.Mode().IsRegular() {
		return report, fmt.Errorf("gguf: input must be a regular local file")
	}
	var layout ggufHeaderLayout
	header, err := readConversionHeader(in, stat.Size(), &layout)
	if err != nil {
		return report, err
	}
	if err := validateConversionLayout(header, stat.Size()); err != nil {
		return report, err
	}
	for _, m := range layout.metadata {
		key := strings.ToLower(m.key)
		if strings.HasPrefix(key, "split.") {
			return report, fmt.Errorf("gguf: conversion does not support split metadata")
		}
		if key == "general.file_type" || key == "general.quantization_version" {
			if m.typ != TypeUint32 {
				return report, fmt.Errorf("gguf: conversion recipe scalar %s must be UINT32", m.key)
			}
		} else if strings.Contains(key, "quantiz") || strings.Contains(key, "imatrix") {
			return report, fmt.Errorf("gguf: conversion refuses stale recipe metadata %s", m.key)
		}
	}
	tensors := append([]TensorInfo(nil), header.Tensors...)
	var extent uint64
	for i := range tensors {
		t := &tensors[i]
		if conversionEligible(*t) {
			// The existing row reload consumer supplies this explicit memory ceiling.
			if t.Dims[0] > maxTensorRowF32Bytes/4 {
				return report, fmt.Errorf("gguf: conversion row exceeds bounded reload envelope")
			}
			t.Type = target
			report.ConvertedTensors++
		} else {
			// A MOSTLY_* scalar must not assert a recipe that these bytes do not
			// implement. Version one preserves vectors and already-target matrices,
			// but refuses heterogeneous higher-rank payloads rather than guessing.
			if len(t.Dims) >= 2 && (t.Type != target || t.Dims[0]%256 != 0) {
				return report, fmt.Errorf("gguf: preserved tensor %s prevents a truthful uniform K-quant recipe", t.Name)
			}
			report.PreservedTensors++
		}
		extent = alignOffset(extent, header.Alignment)
		t.Offset = extent
		size, err := tensorPayloadBytes(*t)
		if err != nil {
			return report, err
		}
		if size > math.MaxInt64 || extent > math.MaxInt64-size {
			return report, fmt.Errorf("gguf: converted payload extent overflow")
		}
		extent += size
	}
	if report.ConvertedTensors == 0 {
		return report, fmt.Errorf("gguf: no eligible F32 matrices with row width divisible by 256")
	}
	rawHeader, err := conversionHeader(in, layout, tensors, fileType, header.Alignment)
	if err != nil {
		return report, err
	}
	if extent > math.MaxInt64-uint64(len(rawHeader)) {
		return report, fmt.Errorf("gguf: converted file extent overflow")
	}
	tmp, err := os.CreateTemp(filepath.Dir(output), ".fak-quantize-*.gguf")
	if err != nil {
		return report, err
	}
	tmpPath := tmp.Name()
	defer func() { tmp.Close(); os.Remove(tmpPath) }()
	if _, err := tmp.Write(rawHeader); err != nil {
		return report, err
	}
	var f32 [256]float32
	var raw [1024]byte
	copyBuffer := make([]byte, 64<<10)
	position := uint64(len(rawHeader))
	for i, t := range tensors {
		if err := ctx.Err(); err != nil {
			return report, err
		}
		want := uint64(len(rawHeader)) + t.Offset
		if _, err := io.CopyN(tmp, zeroReader{}, int64(want-position)); err != nil {
			return report, err
		}
		source := header.Tensors[i]
		size, _ := tensorPayloadBytes(source)
		r := io.NewSectionReader(in, source.FileOffset, int64(size))
		if !conversionEligible(source) {
			if _, err := io.CopyBuffer(tmp, conversionContextReader{ctx, r}, copyBuffer); err != nil {
				return report, err
			}
		} else {
			for left := size; left > 0; left -= 1024 {
				if err := ctx.Err(); err != nil {
					return report, err
				}
				if _, err := io.ReadFull(r, raw[:]); err != nil {
					return report, err
				}
				for j := range f32 {
					f32[j] = math.Float32frombits(binary.LittleEndian.Uint32(raw[j*4:]))
				}
				encoded, err := quantizeKRow(f32[:], target)
				if err != nil {
					return report, fmt.Errorf("gguf: tensor %s: %w", source.Name, err)
				}
				if _, err := tmp.Write(encoded); err != nil {
					return report, err
				}
			}
		}
		n, _ := tensorPayloadBytes(t)
		position = want + n
	}
	if err := tmp.Sync(); err != nil {
		return report, err
	}
	if err := validateConvertedFile(ctx, tmp, tensors, header.Tensors); err != nil {
		return report, err
	}
	if err := tmp.Close(); err != nil {
		return report, err
	}
	if err := ctx.Err(); err != nil {
		return report, err
	}
	// Link is atomic with respect to destination existence. Rename would replace a
	// racing creator on Unix, violating the no-overwrite contract.
	if err := os.Link(tmpPath, output); err != nil {
		return report, fmt.Errorf("gguf: no-overwrite publication: %w", err)
	}
	report.InputBytes = stat.Size()
	report.OutputBytes = int64(position)
	report.Type = target.String()
	return report, nil
}

func conversionEligible(t TensorInfo) bool {
	return t.Type == TensorF32 && len(t.Dims) == 2 && t.Dims[0]%256 == 0
}

func readConversionHeader(r io.ReaderAt, size int64, layout *ggufHeaderLayout) (*File, error) {
	if size < 24 {
		return nil, fmt.Errorf("gguf: truncated conversion input")
	}
	limit := min(size, conversionHeaderLimit)
	return readHeaderLayout(io.NewSectionReader(r, 0, limit), limit, layout)
}

func validateConversionLayout(f *File, size int64) error {
	if f.Alignment > 1<<20 {
		return fmt.Errorf("gguf: conversion alignment exceeds 1 MiB")
	}
	if f.TensorDataOffset > size {
		return fmt.Errorf("gguf: tensor directory overruns input")
	}
	tensors := append([]TensorInfo(nil), f.Tensors...)
	sort.Slice(tensors, func(i, j int) bool { return tensors[i].FileOffset < tensors[j].FileOffset })
	seen := map[string]bool{}
	end := f.TensorDataOffset
	for _, t := range tensors {
		if t.Name == "" || seen[t.Name] {
			return fmt.Errorf("gguf: empty or duplicate tensor name %q", t.Name)
		}
		seen[t.Name] = true
		n, err := tensorPayloadBytes(t)
		if err != nil {
			return err
		}
		if t.FileOffset < end || t.FileOffset > size || n > uint64(size-t.FileOffset) {
			return fmt.Errorf("gguf: tensor %s overlaps or exceeds input", t.Name)
		}
		end = t.FileOffset + int64(n)
	}
	return nil
}

func conversionHeader(in io.ReaderAt, l ggufHeaderLayout, tensors []TensorInfo, fileType uint32, alignment uint64) ([]byte, error) {
	var b bytes.Buffer
	write := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { write(uint64(len(s))); b.WriteString(s) }
	seenType, seenVersion := false, false
	for _, m := range l.metadata {
		seenType = seenType || m.key == "general.file_type"
		seenVersion = seenVersion || m.key == "general.quantization_version"
	}
	count := uint64(len(l.metadata))
	if !seenType {
		count++
	}
	if !seenVersion {
		count++
	}
	b.WriteString(Magic)
	write(uint32(Version))
	write(uint64(len(tensors)))
	write(count)
	for _, m := range l.metadata {
		if m.key == "general.file_type" || m.key == "general.quantization_version" {
			if _, err := io.Copy(&b, io.NewSectionReader(in, m.start, m.value-m.start)); err != nil {
				return nil, err
			}
			value := fileType
			if m.key == "general.quantization_version" {
				value = 2
			}
			write(value)
		} else if _, err := io.Copy(&b, io.NewSectionReader(in, m.start, m.end-m.start)); err != nil {
			return nil, err
		}
	}
	if !seenType {
		str("general.file_type")
		write(uint32(TypeUint32))
		write(fileType)
	}
	if !seenVersion {
		str("general.quantization_version")
		write(uint32(TypeUint32))
		write(uint32(2))
	}
	for _, t := range tensors {
		str(t.Name)
		write(uint32(len(t.Dims)))
		for _, d := range t.Dims {
			write(d)
		}
		write(uint32(t.Type))
		write(t.Offset)
	}
	aligned := alignOffset(uint64(b.Len()), alignment)
	if aligned > uint64(conversionHeaderLimit) {
		return nil, fmt.Errorf("gguf: converted header exceeds 64 MiB")
	}
	b.Write(make([]byte, int(aligned)-b.Len()))
	return b.Bytes(), nil
}

func validateConvertedFile(ctx context.Context, f *os.File, want, original []TensorInfo) error {
	stat, err := f.Stat()
	if err != nil {
		return err
	}
	var layout ggufHeaderLayout
	header, err := readConversionHeader(f, stat.Size(), &layout)
	if err != nil {
		return err
	}
	if err := validateConversionLayout(header, stat.Size()); err != nil {
		return err
	}
	if len(header.Tensors) != len(want) {
		return fmt.Errorf("gguf: converted tensor count mismatch")
	}
	ws, err := NewWeightSource(header, f, stat.Size())
	if err != nil {
		return err
	}
	defer ws.Close()
	for i, t := range header.Tensors {
		if t.Name != want[i].Name || t.Type != want[i].Type || t.Offset != want[i].Offset || !slices.Equal(t.Dims, want[i].Dims) {
			return fmt.Errorf("gguf: converted tensor directory mismatch")
		}
		if !conversionEligible(original[i]) {
			continue
		}
		// Reload every converted row through the existing consumer, never an entire
		// model or tensor allocation. The input plan has already bounded row width.
		if t.Dims[0] > maxTensorRowF32Bytes/4 {
			return fmt.Errorf("gguf: reload row exceeds budget")
		}
		for row := uint64(0); row < t.Dims[1]; row++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			values, _, err := ws.TensorRowF32(t.Name, int(row))
			if err != nil {
				return err
			}
			for _, v := range values {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					return fmt.Errorf("gguf: converted nonfinite reconstruction")
				}
			}
		}
	}
	return nil
}

type conversionContextReader struct {
	ctx context.Context
	r   io.Reader
}

func (r conversionContextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}

type zeroReader struct{}

func (zeroReader) Read(p []byte) (int, error) { clear(p); return len(p), nil }
