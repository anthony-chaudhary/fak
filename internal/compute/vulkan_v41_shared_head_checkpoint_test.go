//go:build wip_v41_shared_head_qualification && vulkan && (windows || linux) && cgo

package compute_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// This hardware-only collector reads the pinned head directly from its GGUF
// shard and compares MatMul with BatchedMatMul for every survivor count 1..4.
// The input corpus is synthetic BF16. A pass is checkpoint-head component
// evidence, never model/cohort correctness, serving activation, or throughput.
// Vulkan package init may initialize a device before this test or its opt-in
// guard runs. Building/executing this target needs separate approval.
// The provenance gate streams all 43,072,989,056 shard bytes before upload;
// the head used below is captured from that same SHA-256 pass, not reread.
// fak-test:runtime integration est=2m lane=optin
func TestVulkanV41SharedHeadCheckpointParity(t *testing.T) {
	if os.Getenv("FAK_V41_SHARED_HEAD_QUALIFY") != "1" {
		t.Skip("hardware-only shared-head qualification requires explicit opt-in")
	}
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" || os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("start the test process with device-required and native dispatch profiling enabled")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 5*time.Minute {
		t.Fatal("run this hardware-only collector with a test timeout no greater than five minutes")
	}
	const out, in, maxRows, payloadBytes = 129280, 5120, 4, 542976000
	be, ok := compute.Lookup("vulkan")
	if !ok {
		t.Fatal("required Vulkan backend is not registered")
	}
	probe, ok := be.(interface {
		VulkanQ6KSharedHeadQualificationPreflight(compute.Dtype, int, int, int) (compute.VulkanQ6KSharedHeadQualification, error)
		VulkanDebugResetDispatchProfile()
		VulkanDebugDispatchProfileSnapshot() compute.VulkanDispatchProfile
	})
	if !ok {
		t.Fatal("private qualification preflight is unavailable")
	}
	device, err := probe.VulkanQ6KSharedHeadQualificationPreflight(compute.Q6_K, out, in, maxRows)
	if err != nil {
		t.Fatal(err)
	}
	expectedDevice, err := strconv.ParseUint(os.Getenv("FAK_V41_SHARED_HEAD_EXPECT_DEVICE_ID"), 0, 32)
	if err != nil || uint32(expectedDevice) != device.DeviceID {
		t.Fatalf("require the numeric device ID from the independently verified Halo receipt; observed %#x", device.DeviceID)
	}
	spirvDir := os.Getenv("FAK_VULKAN_SPIRV")
	if !filepath.IsAbs(spirvDir) {
		t.Fatal("require an explicit absolute immutable FAK_VULKAN_SPIRV directory")
	}
	shaderPath := filepath.Join(spirvDir, "q6k_matmul.spv")
	shaderHash := sharedHeadFileSHA(t, shaderPath)
	sharedHeadRequireSHA(t, "FAK_V41_SHARED_HEAD_SPIRV_SHA256", shaderHash)
	manifestHash := sharedHeadFileSHA(t, os.Getenv("FAK_V41_SHARED_HEAD_SOURCE_MANIFEST"))
	sharedHeadRequireSHA(t, "FAK_V41_SHARED_HEAD_SOURCE_MANIFEST_SHA256", manifestHash)
	path := os.Getenv("FAK_V41_SHARED_HEAD_GGUF")
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil || !stat.Mode().IsRegular() {
		t.Fatalf("regular checkpoint shard required: %v", err)
	}
	const shardBytes, headerBytes = int64(43072989056), 5772832
	const shardSHA = "0bcee934bd4e8350c54410681d9300b0321a300fcb1df1437a20703394d08ee0"
	const headerSHA = "1841deee0aa1d6cd3ffe3db924e94acbfc49a06749e4ac5bf3cfaed7dcc64862"
	if stat.Size() != shardBytes {
		t.Fatalf("pinned shard length=%d, got %d", shardBytes, stat.Size())
	}
	whole := sha256.New()
	stream := io.TeeReader(f, whole)
	headerRaw := make([]byte, headerBytes)
	if _, err := io.ReadFull(stream, headerRaw); err != nil {
		t.Fatal(err)
	}
	headerSum := sha256.Sum256(headerRaw)
	headerHash := hex.EncodeToString(headerSum[:])
	if headerHash != headerSHA {
		t.Fatalf("checkpoint header hash=%s, want %s", headerHash, headerSHA)
	}
	header, err := ggufload.ReadSize(bytes.NewReader(headerRaw), shardBytes)
	if err != nil {
		t.Fatal(err)
	}
	var head *ggufload.TensorInfo
	for i := range header.Tensors {
		if header.Tensors[i].Name == "output.weight" {
			if head != nil {
				t.Fatal("duplicate output.weight entries")
			}
			head = &header.Tensors[i]
		}
	}
	if head == nil || head.Type != ggufload.TensorQ6_K || len(head.Dims) != 2 || head.Dims[0] != in || head.Dims[1] != out {
		t.Fatalf("required output.weight GGUF Q6_K dimensions [5120,129280], got %+v", head)
	}
	if header.TensorDataOffset != headerBytes || head.Offset != 0 || head.FileOffset != headerBytes {
		t.Fatal("output.weight does not start at the pinned shard tensor-data offset")
	}
	raw := make([]byte, payloadBytes)
	if _, err := io.ReadFull(stream, raw); err != nil {
		t.Fatal(err)
	}
	tail, err := io.Copy(io.Discard, stream)
	if err != nil || int64(headerBytes+payloadBytes)+tail != shardBytes {
		t.Fatalf("checkpoint shard changed or read failed: tail=%d error=%v", tail, err)
	}
	if got := hex.EncodeToString(whole.Sum(nil)); got != shardSHA {
		t.Fatalf("full checkpoint shard hash=%s, want %s", got, shardSHA)
	}
	sum := sha256.Sum256(raw)
	headHash := hex.EncodeToString(sum[:])
	host, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("CPU host tensor wrapper backend is unavailable")
	}
	w := be.Upload(compute.NewQ6K(host, []int{out, in}, raw), compute.Q6_K)
	weightOwned := true
	defer func() {
		if weightOwned {
			be.Free(w)
		}
	}()
	x := make([]float32, maxRows*in)
	for row := 0; row < maxRows; row++ {
		for col := 0; col < in; col++ {
			h := uint32(col+row*in)*2654435761 + uint32(row+1)*40503
			h ^= h >> 15
			h *= 2246822519
			h ^= h >> 13
			v := float32(int32(h>>8)-(1<<23)) / float32(1<<23)
			bits := math.Float32bits(v)
			x[row*in+col] = math.Float32frombits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
		}
	}
	oracle := make([]float32, 0, maxRows*out)
	profiles := make([]compute.VulkanDispatchProfile, 0, maxRows*2)
	project := func(data []float32, rows int, scalar bool) []float32 {
		t.Helper()
		shape := []int{rows, in}
		if scalar {
			shape = []int{in}
		}
		input := be.Upload(compute.NewF32(host, shape, data), compute.F32)
		defer be.Free(input)
		probe.VulkanDebugResetDispatchProfile()
		var output compute.Tensor
		if scalar {
			output = be.MatMul(w, input)
		} else {
			output = be.BatchedMatMul(w, input, rows)
		}
		defer be.Free(output)
		got := append([]float32(nil), be.Read(output)...)
		p := probe.VulkanDebugDispatchProfileSnapshot()
		if p.ComputeDispatches != 1 || p.OtherComputeDispatches != 1 || p.OtherMatmulDispatches != 1 {
			t.Fatalf("rows=%d scalar=%v: require exactly one native Q6_K dispatch, got %+v", rows, scalar, p)
		}
		profiles = append(profiles, p)
		if len(got) != rows*out {
			t.Fatalf("rows=%d output length=%d", rows, len(got))
		}
		for i, value := range got {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("nonfinite output at %d", i)
			}
		}
		return got
	}
	for row := 0; row < maxRows; row++ {
		oracle = append(oracle, project(x[row*in:(row+1)*in], 1, true)...)
	}
	retainsF32 := false
	for _, v := range oracle {
		retainsF32 = retainsF32 || math.Float32bits(v)&0xffff != 0
	}
	if !retainsF32 {
		t.Fatal("input corpus failed to distinguish F32 logits from rounded BF16")
	}
	for rows := 1; rows <= maxRows; rows++ {
		got := project(x[:rows*in], rows, false)
		for i, value := range got {
			if math.Float32bits(value) != math.Float32bits(oracle[i]) {
				t.Fatalf("rows=%d row=%d output=%d panel=%#08x scalar=%#08x", rows, i/out, i%out, math.Float32bits(value), math.Float32bits(oracle[i]))
			}
		}
	}
	// Emit no success receipt until the shared weight's release also returns.
	// Drop ownership first so a selected release failure is never retried.
	weightOwned = false
	be.Free(w)
	if sharedHeadFileSHA(t, shaderPath) != shaderHash {
		t.Fatal("shader file changed during collection")
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	inputBytes := make([]byte, len(x)*4)
	for i, v := range x {
		binary.LittleEndian.PutUint32(inputBytes[i*4:], math.Float32bits(v))
	}
	inputSum := sha256.Sum256(inputBytes)
	receipt := map[string]any{
		"schema": "fak-v41-q6k-shared-head-component/1", "engine": "fak-native", "status": "component-pass",
		"scope":  "checkpoint head bytes with synthetic BF16 input; no model/cohort/serving/performance qualification",
		"device": device, "tensor": "output.weight", "shape": []int{out, in}, "rows": []int{1, 2, 3, 4},
		"checkpoint_header_sha256": headerHash, "tensor_sha256": headHash, "shader_file_sha256": shaderHash,
		"checkpoint_shard_sha256": shardSHA, "checkpoint_revision": "58d8ac86298fdf85a2440defee08b1abcad32e45", "checkpoint_shard_bytes_hashed": shardBytes,
		"test_binary_sha256": sharedHeadFileSHA(t, exe), "input_f32_sha256": hex.EncodeToString(inputSum[:]),
		"source_manifest_sha256": manifestHash,
		"dispatches":             profiles, "capability_provider_enabled": false,
	}
	encoded, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("shared-head component receipt: %s", encoded)
}

func sharedHeadRequireSHA(t *testing.T, key, got string) {
	t.Helper()
	want := os.Getenv(key)
	decoded, err := hex.DecodeString(want)
	if err != nil || len(decoded) != sha256.Size || want != got {
		t.Fatalf("%s must be the independently pinned SHA-256; observed %s", key, got)
	}
}

func sharedHeadReaderSHA(t *testing.T, r io.Reader) string {
	t.Helper()
	h := sha256.New()
	if _, err := io.Copy(h, r); err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func sharedHeadFileSHA(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	return sharedHeadReaderSHA(t, f)
}
