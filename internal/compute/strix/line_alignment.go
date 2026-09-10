// Package strix implements 64-byte cache-line alignment, false-sharing barrier guards,
// and struct layout verification for coherent CPU-GPU MALL data structures on AMD Strix Halo.
// Placement: Gate 2 (Commercial Serving & Appliance Infrastructure, strictly private).

package strix

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"
	"unsafe"
)

var (
	// ErrMisalignedAddress indicates a memory pointer is not aligned to a 64-byte boundary.
	ErrMisalignedAddress = errors.New("address is not 64-byte aligned")

	// ErrFalseSharingDetected indicates concurrent CPU and GPU fields share the same 64-byte cache line.
	ErrFalseSharingDetected = errors.New("false sharing detected between cross-device fields in same cache line")

	// ErrNilPointer indicates an unexpected nil pointer argument.
	ErrNilPointer = errors.New("pointer cannot be nil")
)

var defaultReporter = NewCoherencyTelemetryReporter()

// DefaultReporter returns the global coherency telemetry reporter instance.
func DefaultReporter() *CoherencyTelemetryReporter {
	return defaultReporter
}

// IsAligned64 reports whether addr is aligned to a 64-byte cache line boundary.
func IsAligned64(addr uintptr) bool {
	return (addr & CacheLineMask) == 0
}

// AlignUp64 rounds addr up to the next 64-byte cache line boundary.
func AlignUp64(addr uintptr) uintptr {
	return (addr + CacheLineMask) &^ CacheLineMask
}

// AlignDown64 rounds addr down to the preceding 64-byte cache line boundary.
func AlignDown64(addr uintptr) uintptr {
	return addr &^ CacheLineMask
}

// OffsetToNextLine returns the byte distance from addr to the next 64-byte cache line boundary.
// If addr is already 64-byte aligned, it returns 0.
func OffsetToNextLine(addr uintptr) uintptr {
	return (CacheLineBytes - (addr & CacheLineMask)) & CacheLineMask
}

// AlignedBuffer manages a heap-allocated memory slice aligned to a 64-byte cache line boundary.
type AlignedBuffer struct {
	raw     []byte
	aligned []byte
	addr    uintptr
	size    int
}

// Address returns the 64-byte aligned memory address of the buffer.
func (b *AlignedBuffer) Address() uintptr {
	if b == nil {
		return 0
	}
	return b.addr
}

// Pointer returns an unsafe.Pointer to the 64-byte aligned memory buffer.
func (b *AlignedBuffer) Pointer() unsafe.Pointer {
	if b == nil || len(b.aligned) == 0 {
		return nil
	}
	return unsafe.Pointer(unsafe.SliceData(b.aligned))
}

// Bytes returns the aligned byte slice of the requested size.
func (b *AlignedBuffer) Bytes() []byte {
	if b == nil {
		return nil
	}
	return b.aligned
}

// Size returns the usable size of the aligned buffer in bytes.
func (b *AlignedBuffer) Size() int {
	if b == nil {
		return 0
	}
	return b.size
}

// IsAligned reports whether the buffer address is strictly 64-byte aligned.
func (b *AlignedBuffer) IsAligned() bool {
	if b == nil || b.addr == 0 {
		return false
	}
	return IsAligned64(b.addr)
}

// Release zeroes slice references to facilitate garbage collection.
func (b *AlignedBuffer) Release() {
	if b == nil {
		return
	}
	b.raw = nil
	b.aligned = nil
	b.addr = 0
	b.size = 0
}

// AllocateAligned allocates a contiguous byte slice of the given size guaranteed to start
// on a 64-byte cache line boundary. Returns ErrInvalidBufferSize if size <= 0.
func AllocateAligned(size int) (*AlignedBuffer, error) {
	if size <= 0 {
		return nil, ErrInvalidBufferSize
	}

	raw := make([]byte, size+CacheLineBytes)
	rawAddr := uintptr(unsafe.Pointer(unsafe.SliceData(raw)))
	offset := (CacheLineBytes - (rawAddr & CacheLineMask)) & CacheLineMask
	offsetInt := int(offset)
	aligned := raw[offsetInt : offsetInt+size]
	alignedAddr := rawAddr + offset

	return &AlignedBuffer{
		raw:     raw,
		aligned: aligned,
		addr:    alignedAddr,
		size:    size,
	}, nil
}

// AllocateAlignedStruct allocates memory for type T on a 64-byte cache line boundary.
func AllocateAlignedStruct[T any]() (*T, *AlignedBuffer, error) {
	var zero T
	size := int(unsafe.Sizeof(zero))
	if size == 0 {
		size = 1
	}
	buf, err := AllocateAligned(size)
	if err != nil {
		return nil, nil, err
	}
	ptr := (*T)(buf.Pointer())
	return ptr, buf, nil
}

// AuditStructLayout inspects struct fields via reflection to detect false-sharing hazards
// between CPU and GPU domains across 64-byte cache lines.
func AuditStructLayout(v any) (*AlignmentAuditReport, error) {
	if v == nil {
		return nil, ErrNilPointer
	}
	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("expected struct or pointer to struct, got %s", t.Kind())
	}

	structName := t.Name()
	if structName == "" {
		structName = t.String()
	}

	totalSize := t.Size()
	cacheLines := int((totalSize + uintptr(CacheLineBytes) - 1) / uintptr(CacheLineBytes))
	if totalSize == 0 {
		cacheLines = 0
	}

	fields := make([]FieldLayoutInfo, 0, t.NumField())
	hasAlign64Tag := false

	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		offset := f.Offset
		size := f.Type.Size()
		domain := f.Tag.Get("domain")
		role := f.Tag.Get("role")
		alignTag := f.Tag.Get("align")
		if alignTag == "64" {
			hasAlign64Tag = true
		}

		lineIdx := int(offset / uintptr(CacheLineBytes))
		fields = append(fields, FieldLayoutInfo{
			Name:           f.Name,
			Offset:         offset,
			Size:           size,
			Domain:         domain,
			Role:           role,
			CacheLineIndex: lineIdx,
		})
	}

	var violations []FalseSharingViolation

	// Domain classification helpers
	isCPUWrite := func(domain, role string) bool {
		d := strings.ToLower(domain)
		r := strings.ToLower(role)
		if strings.Contains(d, "cpu_write") || strings.Contains(d, "cpu_rw") {
			return true
		}
		if strings.Contains(r, "cpu_producer") && !strings.Contains(d, "read") {
			return true
		}
		return false
	}

	isGPURW := func(domain, role string) bool {
		d := strings.ToLower(domain)
		r := strings.ToLower(role)
		if strings.Contains(d, "gpu_read") || strings.Contains(d, "gpu_write") || strings.Contains(d, "gpu_rw") {
			return true
		}
		if strings.Contains(r, "gpu_consumer") {
			return true
		}
		return false
	}

	isGPUWrite := func(domain, role string) bool {
		d := strings.ToLower(domain)
		return strings.Contains(d, "gpu_write") || strings.Contains(d, "gpu_rw")
	}

	isCPURead := func(domain, role string) bool {
		d := strings.ToLower(domain)
		return strings.Contains(d, "cpu_read") || strings.Contains(d, "cpu_rw")
	}

	// Pairwise false-sharing analysis
	for i := 0; i < len(fields); i++ {
		fA := fields[i]
		if fA.Name == "_" || (fA.Domain == "" && fA.Role == "") {
			continue
		}

		// Check if individual field straddles a 64-byte boundary
		if fA.Size > 0 {
			endLine := int((fA.Offset + fA.Size - 1) / uintptr(CacheLineBytes))
			if fA.CacheLineIndex != endLine {
				violations = append(violations, FalseSharingViolation{
					StructName:     structName,
					FieldA:         fA.Name,
					FieldB:         fA.Name,
					OffsetA:        fA.Offset,
					OffsetB:        fA.Offset + fA.Size,
					CacheLineIndex: fA.CacheLineIndex,
					Description: fmt.Sprintf("field %s (offset %d, size %d) crosses 64-byte cache line boundary (%d -> %d)",
						fA.Name, fA.Offset, fA.Size, fA.CacheLineIndex, endLine),
				})
			}
		}

		for j := i + 1; j < len(fields); j++ {
			fB := fields[j]
			if fB.Name == "_" || (fB.Domain == "" && fB.Role == "") {
				continue
			}

			if fA.CacheLineIndex != fB.CacheLineIndex {
				continue
			}

			// In same cache line: check for cross-device read/write collision
			cpuWriteGpuRW := isCPUWrite(fA.Domain, fA.Role) && isGPURW(fB.Domain, fB.Role)
			gpuRWCpuWrite := isCPUWrite(fB.Domain, fB.Role) && isGPURW(fA.Domain, fA.Role)
			cpuReadGpuWrite := isCPURead(fA.Domain, fA.Role) && isGPUWrite(fB.Domain, fB.Role)
			gpuWriteCpuRead := isCPURead(fB.Domain, fB.Role) && isGPUWrite(fA.Domain, fA.Role)

			if cpuWriteGpuRW || gpuRWCpuWrite || cpuReadGpuWrite || gpuWriteCpuRead {
				violations = append(violations, FalseSharingViolation{
					StructName:     structName,
					FieldA:         fA.Name,
					FieldB:         fB.Name,
					OffsetA:        fA.Offset,
					OffsetB:        fB.Offset,
					CacheLineIndex: fA.CacheLineIndex,
					Description: fmt.Sprintf("false sharing between field %s (domain:%s, offset:%d) and field %s (domain:%s, offset:%d) on cache line %d",
						fA.Name, fA.Domain, fA.Offset, fB.Name, fB.Domain, fB.Offset, fA.CacheLineIndex),
				})
			}
		}
	}

	// Alignment packaging verification
	if hasAlign64Tag && (totalSize%uintptr(CacheLineBytes) != 0) {
		violations = append(violations, FalseSharingViolation{
			StructName:     structName,
			FieldA:         structName,
			FieldB:         structName,
			OffsetA:        0,
			OffsetB:        totalSize,
			CacheLineIndex: -1,
			Description: fmt.Sprintf("struct %s has align:\"64\" tag but total size %d is not a multiple of 64",
				structName, totalSize),
		})
	}

	report := &AlignmentAuditReport{
		StructName: structName,
		TotalSize:  totalSize,
		CacheLines: cacheLines,
		Passed:     len(violations) == 0,
		Violations: violations,
		Fields:     fields,
		Timestamp:  time.Now(),
	}

	return report, nil
}

// AssertNoFalseSharing runs an audit on v and returns ErrFalseSharingDetected if any violations are found.
func AssertNoFalseSharing(v any) error {
	report, err := AuditStructLayout(v)
	if err != nil {
		return err
	}
	if !report.Passed {
		var detail string
		if len(report.Violations) > 0 {
			detail = report.Violations[0].Description
		} else {
			detail = "alignment layout violation"
		}
		return fmt.Errorf("%w: struct %s (%s)", ErrFalseSharingDetected, report.StructName, detail)
	}
	return nil
}

// AuditAndProtect inspects v for false-sharing hazards. If violations are detected,
// it invokes fallbackFn (or SoftwareFallbackBarrier if nil) and records telemetry before returning the report.
func AuditAndProtect(v any, fallbackFn func()) *AlignmentAuditReport {
	report, err := AuditStructLayout(v)
	if err != nil {
		return nil
	}
	if !report.Passed {
		if fallbackFn != nil {
			fallbackFn()
		} else {
			SoftwareFallbackBarrier()
		}
		defaultReporter.RecordAuditReport(report)
		defaultReporter.RecordFabricReplay(uint64(len(report.Violations)), uint64(len(report.Violations)), 0, 1)
	} else {
		defaultReporter.RecordAuditReport(report)
		defaultReporter.RecordFabricReplay(0, 0, 1, 1)
	}
	return report
}
