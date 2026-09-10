package strix

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// createSyntheticSafetensors creates a valid synthetic safetensors file for testing.
func createSyntheticSafetensors(t *testing.T, dir string) (string, []byte, []byte) {
	t.Helper()

	filePath := filepath.Join(dir, "model_synthetic.safetensors")

	// Payload data for two tensors
	tensor1Data := []byte("WEIGHT_DATA_TENSOR_1_ZEN5_WAVE32")
	tensor2Data := []byte("BIAS_DATA_TENSOR_2_GFX1151_WMMA")

	offset1Start := int64(0)
	offset1End := int64(len(tensor1Data))
	offset2Start := offset1End
	offset2End := offset2Start + int64(len(tensor2Data))

	headerMap := map[string]interface{}{
		"__metadata__": map[string]string{
			"format":       "pt",
			"architecture": "qwen38",
			"quant":        "bf16",
		},
		"model.layers.0.self_attn.q_proj.weight": map[string]interface{}{
			"dtype":        "BF16",
			"shape":        []int64{2, 16},
			"data_offsets": []int64{offset1Start, offset1End},
		},
		"model.layers.0.self_attn.k_proj.weight": map[string]interface{}{
			"dtype":        "BF16",
			"shape":        []int64{2, 16},
			"data_offsets": []int64{offset2Start, offset2End},
		},
	}

	headerJSON, err := json.Marshal(headerMap)
	if err != nil {
		t.Fatalf("marshal header json: %v", err)
	}

	headerLen := uint64(len(headerJSON))
	headerLenBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerLenBytes, headerLen)

	var fullFile []byte
	fullFile = append(fullFile, headerLenBytes...)
	fullFile = append(fullFile, headerJSON...)
	fullFile = append(fullFile, tensor1Data...)
	fullFile = append(fullFile, tensor2Data...)

	if err := os.WriteFile(filePath, fullFile, 0644); err != nil {
		t.Fatalf("write synthetic safetensors file: %v", err)
	}

	return filePath, tensor1Data, tensor2Data
}

// createSyntheticGGUF creates a valid synthetic GGUF file for testing.
func createSyntheticGGUF(t *testing.T, dir string) string {
	t.Helper()
	filePath := filepath.Join(dir, "model_synthetic.gguf")

	data := make([]byte, 1024)
	binary.LittleEndian.PutUint32(data[0:4], GGUFMagic)
	binary.LittleEndian.PutUint32(data[4:8], 3) // version 3

	copy(data[8:], []byte("GGUF_PAYLOAD_TENSOR_WEIGHTS"))

	if err := os.WriteFile(filePath, data, 0644); err != nil {
		t.Fatalf("write synthetic gguf file: %v", err)
	}
	return filePath
}

func TestMmapLoader_SafetensorsParsing(t *testing.T) {
	tempDir := t.TempDir()
	modelPath, tensor1Expected, tensor2Expected := createSyntheticSafetensors(t, tempDir)

	loader := NewMmapModelLoader(DefaultMmapLoaderConfig())
	ctx := context.Background()

	model, err := loader.LoadModel(ctx, modelPath)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	defer model.Close()

	if model.Size <= 0 {
		t.Errorf("expected positive file size, got %d", model.Size)
	}
	if model.Telemetry.ModelFormat != "safetensors" {
		t.Errorf("expected format safetensors, got %s", model.Telemetry.ModelFormat)
	}
	if model.Telemetry.TensorCount != 2 {
		t.Errorf("expected 2 tensors, got %d", model.Telemetry.TensorCount)
	}

	// Verify metadata
	if model.Metadata["architecture"] != "qwen38" {
		t.Errorf("expected architecture qwen38, got %s", model.Metadata["architecture"])
	}

	// Verify tensor 1
	t1, ok := model.GetTensor("model.layers.0.self_attn.q_proj.weight")
	if !ok {
		t.Fatalf("tensor 1 not found in loaded model")
	}
	if t1.DType != "BF16" {
		t.Errorf("expected dtype BF16, got %s", t1.DType)
	}
	if t1.SizeBytes != int64(len(tensor1Expected)) {
		t.Errorf("expected size %d, got %d", len(tensor1Expected), t1.SizeBytes)
	}

	// Verify tensor bytes directly from mmap
	t1Bytes := model.RawBytes[t1.StartOffset:t1.EndOffset]
	if string(t1Bytes) != string(tensor1Expected) {
		t.Errorf("tensor 1 content mismatch: got %q, want %q", string(t1Bytes), string(tensor1Expected))
	}

	// Verify tensor 2
	t2, ok := model.GetTensor("model.layers.0.self_attn.k_proj.weight")
	if !ok {
		t.Fatalf("tensor 2 not found in loaded model")
	}
	t2Bytes := model.RawBytes[t2.StartOffset:t2.EndOffset]
	if string(t2Bytes) != string(tensor2Expected) {
		t.Errorf("tensor 2 content mismatch: got %q, want %q", string(t2Bytes), string(tensor2Expected))
	}

	// Verify UMA DevPtr == HostAddr
	if t1.DevPtr != t1.HostAddr {
		t.Errorf("UMA pointer identity failed: DevPtr (0x%x) != HostAddr (0x%x)", t1.DevPtr, t1.HostAddr)
	}
}

func TestMmapLoader_PageAlignment(t *testing.T) {
	tempDir := t.TempDir()
	modelPath, _, _ := createSyntheticSafetensors(t, tempDir)

	loader := NewMmapModelLoader(DefaultMmapLoaderConfig())
	ctx := context.Background()

	model, err := loader.LoadModel(ctx, modelPath)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	defer model.Close()

	if !model.Telemetry.PageAligned {
		t.Errorf("expected base address to be 4KB page-aligned")
	}
	if (model.BaseAddr % uintptr(DefaultPageSize)) != 0 {
		t.Errorf("base address 0x%x is not aligned to %d bytes", model.BaseAddr, DefaultPageSize)
	}
}

func TestMmapLoader_GEMUserptrMock(t *testing.T) {
	tempDir := t.TempDir()
	modelPath, _, _ := createSyntheticSafetensors(t, tempDir)

	loader := NewMmapModelLoader(DefaultMmapLoaderConfig())

	var capturedAddr uint64
	var capturedSize uint64
	var capturedFlags uint32
	var mockGEMHandle uint32 = 0xbeef01

	loader.sysOpenDRM = func(path string) (int, error) {
		return 42, nil // mock fd
	}
	loader.sysUserptr = func(drmFd uintptr, addr uint64, size uint64, flags uint32) (uint32, error) {
		capturedAddr = addr
		capturedSize = size
		capturedFlags = flags
		return mockGEMHandle, nil
	}

	ctx := context.Background()
	model, err := loader.LoadModel(ctx, modelPath)
	if err != nil {
		t.Fatalf("LoadModel with mock GEM userptr failed: %v", err)
	}
	defer model.Close()

	if !model.Telemetry.GEMRegistered {
		t.Errorf("expected GEMRegistered = true")
	}
	if model.GEMHandle != mockGEMHandle {
		t.Errorf("expected GEMHandle %d, got %d", mockGEMHandle, model.GEMHandle)
	}
	if capturedAddr != uint64(model.BaseAddr) {
		t.Errorf("capturedAddr 0x%x != BaseAddr 0x%x", capturedAddr, model.BaseAddr)
	}
	if (capturedFlags & AMDGPU_GEM_USERPTR_READONLY) == 0 {
		t.Errorf("expected AMDGPU_GEM_USERPTR_READONLY in flags")
	}
	if (capturedFlags & AMDGPU_GEM_USERPTR_REGISTER) == 0 {
		t.Errorf("expected AMDGPU_GEM_USERPTR_REGISTER in flags")
	}
	// Size must be page aligned
	if (capturedSize % uint64(DefaultPageSize)) != 0 {
		t.Errorf("capturedSize %d is not page aligned", capturedSize)
	}
}

func TestMmapLoader_FallbackToBuffered(t *testing.T) {
	tempDir := t.TempDir()
	modelPath, tensor1Expected, _ := createSyntheticSafetensors(t, tempDir)

	cfg := DefaultMmapLoaderConfig()
	cfg.EnforceAlignment = false // Allow heap buffer alignment in fallback test
	loader := NewMmapModelLoader(cfg)

	// Simulate mmap failure
	loader.sysMmap = func(f *os.File, size int64) ([]byte, uintptr, func() error, error) {
		return nil, 0, nil, errors.New("simulated mmap permission denied")
	}

	ctx := context.Background()
	model, err := loader.LoadModel(ctx, modelPath)
	if err != nil {
		t.Fatalf("expected fallback to buffered read, got error: %v", err)
	}
	defer model.Close()

	if !model.Telemetry.FallbackActive {
		t.Errorf("expected FallbackActive = true")
	}
	if model.Telemetry.ZeroCopy {
		t.Errorf("expected ZeroCopy = false in fallback mode")
	}
	if !strings.Contains(model.Telemetry.FallbackReason, "falling back to buffered") {
		t.Errorf("unexpected FallbackReason: %s", model.Telemetry.FallbackReason)
	}

	// Verify data is still intact
	t1, ok := model.GetTensor("model.layers.0.self_attn.q_proj.weight")
	if !ok {
		t.Fatalf("tensor 1 not found in fallback model")
	}
	t1Bytes := model.RawBytes[t1.StartOffset:t1.EndOffset]
	if string(t1Bytes) != string(tensor1Expected) {
		t.Errorf("content mismatch in fallback: got %q, want %q", string(t1Bytes), string(tensor1Expected))
	}
}

func TestMmapLoader_GGUFFile(t *testing.T) {
	tempDir := t.TempDir()
	ggufPath := createSyntheticGGUF(t, tempDir)

	loader := NewMmapModelLoader(DefaultMmapLoaderConfig())
	ctx := context.Background()

	model, err := loader.LoadModel(ctx, ggufPath)
	if err != nil {
		t.Fatalf("LoadModel failed on GGUF file: %v", err)
	}
	defer model.Close()

	if model.Telemetry.ModelFormat != "gguf" {
		t.Errorf("expected format gguf, got %s", model.Telemetry.ModelFormat)
	}
	payload, ok := model.GetTensor("gguf_payload")
	if !ok {
		t.Fatalf("expected gguf_payload tensor")
	}
	if payload.SizeBytes != model.Size {
		t.Errorf("expected payload size %d, got %d", model.Size, payload.SizeBytes)
	}
}

func TestMmapLoader_CorruptHeaderRejection(t *testing.T) {
	tempDir := t.TempDir()

	// 1. Zero length file
	zeroPath := filepath.Join(tempDir, "empty.safetensors")
	_ = os.WriteFile(zeroPath, []byte{}, 0644)

	loader := NewMmapModelLoader(DefaultMmapLoaderConfig())
	ctx := context.Background()

	_, err := loader.LoadModel(ctx, zeroPath)
	if !errors.Is(err, ErrZeroLengthFile) {
		t.Errorf("expected ErrZeroLengthFile, got %v", err)
	}

	// 2. Truncated header length
	truncPath := filepath.Join(tempDir, "trunc.safetensors")
	_ = os.WriteFile(truncPath, []byte{0x01, 0x02, 0x03}, 0644)
	model, err := loader.LoadModel(ctx, truncPath)
	if err != nil {
		t.Fatalf("expected graceful binary fallback for short files, got: %v", err)
	}
	_ = model.Close()
	if model.Telemetry.ModelFormat != "binary" {
		t.Errorf("expected binary format for non-safetensors truncated file, got %s", model.Telemetry.ModelFormat)
	}

	// 3. Header JSON specifies out-of-bounds offsets
	badOffsetPath := filepath.Join(tempDir, "bad_offset.safetensors")
	badHeader := map[string]interface{}{
		"bad_tensor": map[string]interface{}{
			"dtype":        "F32",
			"shape":        []int64{10},
			"data_offsets": []int64{100, 999999}, // exceeds file size
		},
	}
	badJSON, _ := json.Marshal(badHeader)
	headerLenBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerLenBytes, uint64(len(badJSON)))
	fileData := append(headerLenBytes, badJSON...)
	fileData = append(fileData, []byte("short_data")...)
	_ = os.WriteFile(badOffsetPath, fileData, 0644)

	badModel, err := loader.LoadModel(ctx, badOffsetPath)
	if err != nil {
		t.Fatalf("expected binary fallback on out of bounds tensor offset, got %v", err)
	}
	_ = badModel.Close()
	if badModel.Telemetry.ModelFormat != "binary" {
		t.Errorf("expected fallback to binary format, got %s", badModel.Telemetry.ModelFormat)
	}
}

func TestMmapLoader_NoCoWEnforcement(t *testing.T) {
	tempDir := t.TempDir()
	testPath := filepath.Join(tempDir, "model_dir")
	_ = os.MkdirAll(testPath, 0755)

	enforcer := NewBtrfsNoCoWEnforcer()
	// Clear native ioctl hooks so the test exercises the commandRunner fallback path deterministically
	enforcer.ioctlGetter = nil
	enforcer.ioctlSetter = nil
	enforcer.statfsChecker = nil

	// Mock runner returning lsattr output with NoCoW ('C')
	enforcer.commandRunner = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "lsattr" {
			return []byte("---------------C------ " + testPath), nil
		}
		if name == "findmnt" || name == "df" {
			return []byte("btrfs"), nil
		}
		if name == "chattr" {
			return []byte(""), nil
		}
		return nil, fmt.Errorf("unexpected command: %s", name)
	}

	ctx := context.Background()

	// Check NoCoW active
	hasNoCoW, err := enforcer.CheckNoCoW(ctx, testPath)
	if err != nil {
		t.Fatalf("CheckNoCoW failed: %v", err)
	}
	if !hasNoCoW {
		t.Errorf("expected NoCoW to be active")
	}

	// Check filesystem is Btrfs
	isBtrfs, err := enforcer.IsBtrfs(ctx, testPath)
	if err != nil {
		t.Fatalf("IsBtrfs failed: %v", err)
	}
	if !isBtrfs {
		t.Errorf("expected filesystem to be identified as Btrfs")
	}

	// Test EnsureNoCoW with absent NoCoW initially
	var chattrCalled bool
	var checkedCount int
	enforcer.commandRunner = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "lsattr" {
			checkedCount++
			if checkedCount == 1 {
				// First check: not set
				return []byte("---------------------- " + testPath), nil
			}
			// Second check after chattr: set
			return []byte("---------------C------ " + testPath), nil
		}
		if name == "chattr" {
			chattrCalled = true
			return []byte(""), nil
		}
		return nil, nil
	}

	if err := enforcer.EnsureNoCoW(ctx, testPath); err != nil {
		t.Fatalf("EnsureNoCoW failed: %v", err)
	}
	if !chattrCalled {
		t.Errorf("expected chattr +C to be executed")
	}
}

func TestMmapLoader_NoCoWIoctl(t *testing.T) {
	tempDir := t.TempDir()
	testPath := filepath.Join(tempDir, "model_dir")
	_ = os.MkdirAll(testPath, 0755)

	enforcer := NewBtrfsNoCoWEnforcer()
	var mockFlags int64 = 0
	enforcer.ioctlGetter = func(fd uintptr) (int64, error) {
		return mockFlags, nil
	}
	enforcer.ioctlSetter = func(fd uintptr, flags int64) error {
		mockFlags = flags
		return nil
	}
	enforcer.statfsChecker = func(path string) (bool, error) {
		return true, nil
	}

	ctx := context.Background()

	// Initial check: flag not set
	active, err := enforcer.CheckNoCoW(ctx, testPath)
	if err != nil {
		t.Fatalf("CheckNoCoW failed: %v", err)
	}
	if active {
		t.Errorf("expected NoCoW inactive initially")
	}

	// Apply NoCoW via ioctlSetter
	if err := enforcer.EnsureNoCoW(ctx, testPath); err != nil {
		t.Fatalf("EnsureNoCoW failed: %v", err)
	}
	if (mockFlags & FSNoCoWFlag) == 0 {
		t.Errorf("expected FSNoCoWFlag to be set in mockFlags")
	}

	// Verify CheckNoCoW now returns true
	active, err = enforcer.CheckNoCoW(ctx, testPath)
	if err != nil || !active {
		t.Errorf("expected CheckNoCoW to return true after EnsureNoCoW")
	}
}

func TestMmapLoader_Concurrency(t *testing.T) {
	tempDir := t.TempDir()
	modelPath, tensor1Expected, tensor2Expected := createSyntheticSafetensors(t, tempDir)

	loader := NewMmapModelLoader(DefaultMmapLoaderConfig())
	ctx := context.Background()

	model, err := loader.LoadModel(ctx, modelPath)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	defer model.Close()

	const numWorkers = 16
	const iterations = 50
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for i := 0; i < numWorkers; i++ {
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				t1, ok := model.GetTensor("model.layers.0.self_attn.q_proj.weight")
				if !ok {
					t.Errorf("worker %d: tensor 1 missing", workerID)
					return
				}
				data1 := model.RawBytes[t1.StartOffset:t1.EndOffset]
				if string(data1) != string(tensor1Expected) {
					t.Errorf("worker %d: data 1 mismatch", workerID)
					return
				}

				t2, ok := model.GetTensor("model.layers.0.self_attn.k_proj.weight")
				if !ok {
					t.Errorf("worker %d: tensor 2 missing", workerID)
					return
				}
				data2 := model.RawBytes[t2.StartOffset:t2.EndOffset]
				if string(data2) != string(tensor2Expected) {
					t.Errorf("worker %d: data 2 mismatch", workerID)
					return
				}
			}
		}(i)
	}

	wg.Wait()
}

func TestMmapLoader_ThroughputCalculation(t *testing.T) {
	tempDir := t.TempDir()
	modelPath, _, _ := createSyntheticSafetensors(t, tempDir)

	loader := NewMmapModelLoader(DefaultMmapLoaderConfig())
	ctx := context.Background()

	model, err := loader.LoadModel(ctx, modelPath)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	defer model.Close()

	if model.Telemetry.LoadDuration < 0 {
		t.Errorf("expected non-negative duration, got %v", model.Telemetry.LoadDuration)
	}
	if model.Telemetry.ThroughputMBps < 0 {
		t.Errorf("expected non-negative throughput, got %f", model.Telemetry.ThroughputMBps)
	}
}

func TestPhysicalStrixHaloHardwareLoad(t *testing.T) {
	modelPath := "/var/lib/fak/models/Qwen3.8-27B-Q4_K_M.gguf"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("physical appliance model path %s not found on host (requires bare-metal Strix Halo)", modelPath)
	}

	loader := NewMmapModelLoader(DefaultMmapLoaderConfig())
	ctx := context.Background()

	model, err := loader.LoadModel(ctx, modelPath)
	if err != nil {
		t.Fatalf("LoadModel failed on live appliance: %v", err)
	}
	defer model.Close()

	t.Logf("[HW-WITNESSED] Path: %s", model.Path)
	t.Logf("[HW-WITNESSED] FileSizeBytes: %d (%.2f GB)", model.Size, float64(model.Size)/(1024*1024*1024))
	t.Logf("[HW-WITNESSED] LoadDuration: %v", model.Telemetry.LoadDuration)
	t.Logf("[HW-WITNESSED] Throughput: %.2f MB/s", model.Telemetry.ThroughputMBps)
	t.Logf("[HW-WITNESSED] PageAligned: %v, BaseAddr: 0x%x", model.Telemetry.PageAligned, model.BaseAddr)
	t.Logf("[HW-WITNESSED] NoCoWActive: %v", model.Telemetry.NoCoWActive)
	t.Logf("[HW-WITNESSED] ZeroCopy: %v", model.Telemetry.ZeroCopy)
	t.Logf("[HW-WITNESSED] GEMRegistered: %v, GEMHandle: %d", model.Telemetry.GEMRegistered, model.GEMHandle)

	if !model.Telemetry.PageAligned {
		t.Errorf("expected 4KB page alignment on live hardware")
	}
	if !model.Telemetry.ZeroCopy {
		t.Errorf("expected zero-copy mmap on live hardware")
	}
	if !model.Telemetry.NoCoWActive {
		t.Errorf("expected NoCoW to be active on /var/lib/fak/models on Strix Halo")
	}
}
