package strix

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
	"unsafe"
)

// Constants for AMD Strix Halo GTT mmap memory loader.
const (
	// DefaultPageSize is the 4KB memory page alignment boundary.
	DefaultPageSize int64 = 4096

	// DefaultDRMRenderNode is the standard Linux AMDGPU DRM render node.
	DefaultDRMRenderNode = "/dev/dri/renderD128"

	// DRM_IOCTL_AMDGPU_GEM_USERPTR ioctl code on Linux x86_64: _IOWR('d', 0x51, 32)
	DRM_IOCTL_AMDGPU_GEM_USERPTR uintptr = 0xc0206451

	// AMDGPU GEM userptr flag constants.
	AMDGPU_GEM_USERPTR_READONLY uint32 = (1 << 0)
	AMDGPU_GEM_USERPTR_ANONONLY uint32 = (1 << 1)
	AMDGPU_GEM_USERPTR_VALIDATE uint32 = (1 << 2)
	AMDGPU_GEM_USERPTR_REGISTER uint32 = (1 << 3)

	// SafetensorsHeaderMaxBytes sets a conservative 100MB ceiling for header length.
	SafetensorsHeaderMaxBytes uint64 = 100 * 1024 * 1024

	// GGUFMagic is the four-byte file identifier "GGUF" (0x46554747).
	GGUFMagic uint32 = 0x46554747
)

var (
	// ErrUnalignedPointer is returned when an mmap pointer is not 4KB page aligned.
	ErrUnalignedPointer = errors.New("strix/loader: memory address is not 4KB page aligned")

	// ErrCorruptHeader is returned when safetensors or model header is malformed.
	ErrCorruptHeader = errors.New("strix/loader: corrupt model header")

	// ErrZeroLengthFile is returned when attempting to mmap an empty file.
	ErrZeroLengthFile = errors.New("strix/loader: cannot mmap zero-length file")

	// ErrOutOfBoundsOffset is returned when tensor data offset extends beyond file bounds.
	ErrOutOfBoundsOffset = errors.New("strix/loader: tensor data offset out of bounds")
)

// DRMAMDGPUGEMUserptrArgs represents Linux struct drm_amdgpu_gem_userptr (32 bytes).
type DRMAMDGPUGEMUserptrArgs struct {
	Addr   uint64 `json:"addr"`   // Virtual address of user memory
	Size   uint64 `json:"size"`   // Size in bytes (must be page aligned)
	Flags  uint32 `json:"flags"`  // AMDGPU_GEM_USERPTR_* flags
	Handle uint32 `json:"handle"` // Output: GEM object handle
}

// TensorDescriptor describes an individual zero-copy mapped tensor.
type TensorDescriptor struct {
	Name        string  `json:"name"`
	DType       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	StartOffset int64   `json:"start_offset"` // Absolute file offset
	EndOffset   int64   `json:"end_offset"`   // Absolute file offset
	SizeBytes   int64   `json:"size_bytes"`
	HostAddr    uintptr `json:"host_addr"`  // Host virtual address in mmap
	DevPtr      uintptr `json:"dev_ptr"`    // UMA device pointer (HostAddr on Strix APU)
	GEMHandle   uint32  `json:"gem_handle"` // AMDGPU GEM object handle
}

// ModelLoadTelemetry captures line-rate and latency metrics for model loading.
type ModelLoadTelemetry struct {
	FilePath        string        `json:"file_path"`
	FileSizeBytes   int64         `json:"file_size_bytes"`
	MappedBytes     int64         `json:"mapped_bytes"`
	LoadDuration    time.Duration `json:"duration_ns"`
	ThroughputMBps  float64       `json:"throughput_mbps"`
	PageAligned     bool          `json:"page_aligned"`
	HugepageAligned bool          `json:"hugepage_aligned"`
	NoCoWActive     bool          `json:"nocow_active"`
	ZeroCopy        bool          `json:"zero_copy"`
	GEMRegistered   bool          `json:"gem_registered"`
	GEMHandle       uint32        `json:"gem_handle"`
	FallbackActive  bool          `json:"fallback_active"`
	FallbackReason  string        `json:"fallback_reason,omitempty"`
	TensorCount     int           `json:"tensor_count"`
	HeaderSizeBytes uint64        `json:"header_size_bytes"`
	ModelFormat     string        `json:"model_format"` // "safetensors", "gguf", or "binary"
}

// MappedModel represents a loaded model weight file mapped into memory.
type MappedModel struct {
	mu         sync.RWMutex
	Path       string                       `json:"path"`
	Size       int64                        `json:"size"`
	BaseAddr   uintptr                      `json:"base_addr"`
	RawBytes   []byte                       `json:"-"`
	GEMHandle  uint32                       `json:"gem_handle"`
	DRMFd      int                          `json:"-"`
	Tensors    map[string]*TensorDescriptor `json:"tensors"`
	Metadata   map[string]string            `json:"metadata"`
	Telemetry  ModelLoadTelemetry           `json:"telemetry"`
	unmapFn    func() error                 `json:"-"`
	closeDRMFn func() error                 `json:"-"`
	closed     bool
}

// Close unmaps memory and releases DRM GEM handles.
func (m *MappedModel) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true

	var errs []error
	if m.unmapFn != nil {
		if err := m.unmapFn(); err != nil {
			errs = append(errs, err)
		}
	}
	if m.closeDRMFn != nil {
		if err := m.closeDRMFn(); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("close mapped model %s: %v", m.Path, errs)
	}
	return nil
}

// IsClosed reports whether the mapped model has been released.
func (m *MappedModel) IsClosed() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.closed
}

// GetTensor retrieves a tensor descriptor by name.
func (m *MappedModel) GetTensor(name string) (*TensorDescriptor, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	t, ok := m.Tensors[name]
	return t, ok
}

// MmapLoaderConfig configures the model weight loader.
type MmapLoaderConfig struct {
	DRMRenderNode    string
	EnforceNoCoW     bool
	EnforceAlignment bool
	RequireHugepages bool
	PreferUserptrGEM bool
}

// DefaultMmapLoaderConfig returns standard defaults for AMD Strix Halo.
func DefaultMmapLoaderConfig() MmapLoaderConfig {
	return MmapLoaderConfig{
		DRMRenderNode:    DefaultDRMRenderNode,
		EnforceNoCoW:     true,
		EnforceAlignment: true,
		RequireHugepages: false,
		PreferUserptrGEM: true,
	}
}

// MmapModelLoader implements direct zero-copy mmap weight loading from Btrfs NoCoW
// storage into AMDGPU GTT address space via DRM_IOCTL_AMDGPU_GEM_USERPTR.
type MmapModelLoader struct {
	mu            sync.RWMutex
	cfg           MmapLoaderConfig
	nocowEnforcer *BtrfsNoCoWEnforcer
	sysMmap       func(f *os.File, size int64) ([]byte, uintptr, func() error, error)
	sysUserptr    func(drmFd uintptr, addr uint64, size uint64, flags uint32) (uint32, error)
	sysOpenDRM    func(path string) (int, error)
}

// NewMmapModelLoader constructs an mmap model loader with default or custom config.
func NewMmapModelLoader(cfg MmapLoaderConfig) *MmapModelLoader {
	if cfg.DRMRenderNode == "" {
		cfg.DRMRenderNode = DefaultDRMRenderNode
	}
	return &MmapModelLoader{
		cfg:           cfg,
		nocowEnforcer: NewBtrfsNoCoWEnforcer(),
		sysMmap:       platformMmap,
		sysUserptr:    platformGEMUserptr,
		sysOpenDRM:    platformOpenDRM,
	}
}

// LoadModel maps a model weight file directly into host and GTT address space.
func (l *MmapModelLoader) LoadModel(ctx context.Context, modelPath string) (*MappedModel, error) {
	start := time.Now()
	cleanPath := filepath.Clean(modelPath)

	fi, err := os.Stat(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("stat model file %s: %w", cleanPath, err)
	}
	if fi.IsDir() {
		return nil, fmt.Errorf("model path %s is a directory, want file", cleanPath)
	}
	fileSize := fi.Size()
	if fileSize == 0 {
		return nil, ErrZeroLengthFile
	}

	telemetry := ModelLoadTelemetry{
		FilePath:      cleanPath,
		FileSizeBytes: fileSize,
		MappedBytes:   fileSize,
	}

	// 1. Check Btrfs NoCoW status on file / parent directory.
	noCoWActive := false
	if l.nocowEnforcer != nil {
		active, nErr := l.nocowEnforcer.CheckNoCoW(ctx, cleanPath)
		if nErr == nil && active {
			noCoWActive = true
		} else if l.cfg.EnforceNoCoW {
			// Check parent directory
			parentDir := filepath.Dir(cleanPath)
			parentActive, pErr := l.nocowEnforcer.CheckNoCoW(ctx, parentDir)
			if pErr == nil && parentActive {
				noCoWActive = true
			}
		}
	}
	telemetry.NoCoWActive = noCoWActive

	// 2. Open file read-only
	f, err := os.OpenFile(cleanPath, os.O_RDONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("open model file %s: %w", cleanPath, err)
	}
	defer f.Close()

	// 3. Perform memory mapping
	var (
		rawBytes       []byte
		baseAddr       uintptr
		unmapFn        func() error
		zeroCopy       bool
		fallbackActive bool
		fallbackReason string
	)

	mmapFn := l.sysMmap
	if mmapFn == nil {
		mmapFn = platformMmap
	}

	rawBytes, baseAddr, unmapFn, err = mmapFn(f, fileSize)
	if err != nil {
		// Fallback to buffered read
		fallbackActive = true
		fallbackReason = fmt.Sprintf("mmap failed: %v, falling back to buffered heap read", err)
		buf, bErr := readBufferedFile(f, fileSize)
		if bErr != nil {
			return nil, fmt.Errorf("both mmap and buffered read failed: %w", bErr)
		}
		rawBytes = buf
		baseAddr = uintptr(unsafe.Pointer(&rawBytes[0]))
		unmapFn = func() error {
			rawBytes = nil
			return nil
		}
		zeroCopy = false
	} else {
		zeroCopy = true
	}

	// 4. Verify 4KB page alignment
	isPageAligned := (baseAddr % uintptr(DefaultPageSize)) == 0
	isHugepageAligned := (baseAddr % uintptr(HugepageSize2MB)) == 0
	telemetry.PageAligned = isPageAligned
	telemetry.HugepageAligned = isHugepageAligned

	if l.cfg.EnforceAlignment && !isPageAligned {
		_ = unmapFn()
		return nil, fmt.Errorf("%w: base address 0x%x is not aligned to %d bytes", ErrUnalignedPointer, baseAddr, DefaultPageSize)
	}

	// 5. Register with AMDGPU MMU via DRM_IOCTL_AMDGPU_GEM_USERPTR
	var (
		gemHandle     uint32
		gemRegistered bool
		drmFd         int = -1
		closeDRMFn    func() error
	)

	if zeroCopy && l.cfg.PreferUserptrGEM {
		openDRM := l.sysOpenDRM
		if openDRM == nil {
			openDRM = platformOpenDRM
		}
		userptrFn := l.sysUserptr
		if userptrFn == nil {
			userptrFn = platformGEMUserptr
		}

		fd, dErr := openDRM(l.cfg.DRMRenderNode)
		if dErr == nil && fd >= 0 {
			drmFd = fd
			closeDRMFn = func() error {
				return closeDRMHandle(drmFd, gemHandle)
			}

			// Align size to page boundary for ioctl
			alignedSize := (uint64(fileSize) + uint64(DefaultPageSize) - 1) & ^uint64(DefaultPageSize-1)
			flags := AMDGPU_GEM_USERPTR_READONLY | AMDGPU_GEM_USERPTR_REGISTER

			handle, uErr := userptrFn(uintptr(drmFd), uint64(baseAddr), alignedSize, flags)
			if uErr == nil && handle > 0 {
				gemHandle = handle
				gemRegistered = true
			} else {
				if !fallbackActive {
					fallbackReason = fmt.Sprintf("GEM userptr registration failed: %v (operating in pure UMA host mmap)", uErr)
				}
			}
		} else {
			if !fallbackActive {
				fallbackReason = fmt.Sprintf("DRM render node %s unavailable: %v", l.cfg.DRMRenderNode, dErr)
			}
		}
	}

	telemetry.ZeroCopy = zeroCopy
	telemetry.GEMRegistered = gemRegistered
	telemetry.GEMHandle = gemHandle
	telemetry.FallbackActive = fallbackActive
	telemetry.FallbackReason = fallbackReason

	// 6. Parse Header (Safetensors or GGUF)
	tensors, metadata, headerSize, format, parseErr := parseModelHeader(rawBytes, fileSize, baseAddr, gemHandle)
	if parseErr != nil {
		// Even if header parsing fails on raw weights, we can treat it as a monolithic binary block
		format = "binary"
		tensors = map[string]*TensorDescriptor{
			"raw_weights": {
				Name:        "raw_weights",
				DType:       "RAW",
				Shape:       []int64{fileSize},
				StartOffset: 0,
				EndOffset:   fileSize,
				SizeBytes:   fileSize,
				HostAddr:    baseAddr,
				DevPtr:      baseAddr,
				GEMHandle:   gemHandle,
			},
		}
	}

	telemetry.TensorCount = len(tensors)
	telemetry.HeaderSizeBytes = headerSize
	telemetry.ModelFormat = format

	loadDuration := time.Since(start)
	telemetry.LoadDuration = loadDuration
	if loadDuration > 0 {
		mb := float64(fileSize) / (1024 * 1024)
		telemetry.ThroughputMBps = mb / loadDuration.Seconds()
	}

	return &MappedModel{
		Path:       cleanPath,
		Size:       fileSize,
		BaseAddr:   baseAddr,
		RawBytes:   rawBytes,
		GEMHandle:  gemHandle,
		DRMFd:      drmFd,
		Tensors:    tensors,
		Metadata:   metadata,
		Telemetry:  telemetry,
		unmapFn:    unmapFn,
		closeDRMFn: closeDRMFn,
	}, nil
}

// parseModelHeader inspects raw mapped bytes to parse safetensors or GGUF structures.
func parseModelHeader(data []byte, fileSize int64, baseAddr uintptr, gemHandle uint32) (map[string]*TensorDescriptor, map[string]string, uint64, string, error) {
	if len(data) < 8 {
		return nil, nil, 0, "", ErrCorruptHeader
	}

	// 1. Check for GGUF magic (0x46554747 = "GGUF")
	magic := binary.LittleEndian.Uint32(data[:4])
	if magic == GGUFMagic {
		version := binary.LittleEndian.Uint32(data[4:8])
		metadata := map[string]string{
			"format":  "gguf",
			"version": fmt.Sprintf("%d", version),
		}
		tensors := map[string]*TensorDescriptor{
			"gguf_payload": {
				Name:        "gguf_payload",
				DType:       "GGUF",
				Shape:       []int64{fileSize},
				StartOffset: 0,
				EndOffset:   fileSize,
				SizeBytes:   fileSize,
				HostAddr:    baseAddr,
				DevPtr:      baseAddr,
				GEMHandle:   gemHandle,
			},
		}
		return tensors, metadata, 8, "gguf", nil
	}

	// 2. Check for Safetensors: 8-byte header length followed by JSON
	headerLen := binary.LittleEndian.Uint64(data[:8])
	if headerLen == 0 || headerLen > SafetensorsHeaderMaxBytes || int64(8+headerLen) > fileSize {
		return nil, nil, 0, "", ErrCorruptHeader
	}

	headerJSON := data[8 : 8+headerLen]
	var parsedHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &parsedHeader); err != nil {
		return nil, nil, 0, "", fmt.Errorf("%w: invalid JSON: %v", ErrCorruptHeader, err)
	}

	metadata := make(map[string]string)
	tensors := make(map[string]*TensorDescriptor)
	dataPayloadStart := int64(8 + headerLen)

	for key, rawVal := range parsedHeader {
		if key == "__metadata__" {
			var metaMap map[string]interface{}
			if err := json.Unmarshal(rawVal, &metaMap); err == nil {
				for mk, mv := range metaMap {
					metadata[mk] = fmt.Sprintf("%v", mv)
				}
			}
			continue
		}

		var tInfo struct {
			DType       string   `json:"dtype"`
			Shape       []int64  `json:"shape"`
			DataOffsets [2]int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(rawVal, &tInfo); err != nil {
			continue // skip unparseable entry
		}

		startOff := dataPayloadStart + tInfo.DataOffsets[0]
		endOff := dataPayloadStart + tInfo.DataOffsets[1]
		if startOff < 0 || endOff > fileSize || startOff > endOff {
			return nil, nil, 0, "", fmt.Errorf("%w: tensor %s offsets [%d, %d] outside file size %d",
				ErrOutOfBoundsOffset, key, startOff, endOff, fileSize)
		}

		sizeBytes := endOff - startOff
		tensorHostAddr := baseAddr + uintptr(startOff)

		tensors[key] = &TensorDescriptor{
			Name:        key,
			DType:       tInfo.DType,
			Shape:       tInfo.Shape,
			StartOffset: startOff,
			EndOffset:   endOff,
			SizeBytes:   sizeBytes,
			HostAddr:    tensorHostAddr,
			DevPtr:      tensorHostAddr, // UMA 1:1 address equivalence
			GEMHandle:   gemHandle,
		}
	}

	return tensors, metadata, headerLen, "safetensors", nil
}

// readBufferedFile reads an entire file into a heap-allocated buffer for fallback.
func readBufferedFile(r io.Reader, size int64) ([]byte, error) {
	buf := make([]byte, size)
	_, err := io.ReadFull(r, buf)
	if err != nil {
		return nil, err
	}
	return buf, nil
}
