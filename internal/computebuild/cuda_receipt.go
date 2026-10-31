package computebuild

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const CUDANativeBuildReceiptSchema = "fak.cuda-native-build-receipt.v1"
const DefaultCUDANativeBuildReceiptPath = ".fak/cuda-native-build-receipt.json"

type cudaReceiptPathContext struct{}
type cudaReceiptStartContext struct{}

// CUDANativeReceiptPath retains independent evidence beside custom build receipts.
func CUDANativeReceiptPath(root, computeReceipt string) string {
	if computeReceipt == "" || filepath.Clean(computeReceipt) == filepath.Clean(filepath.Join(root, DefaultCUDABuildReceiptPath)) {
		return filepath.Join(root, DefaultCUDANativeBuildReceiptPath)
	}
	return strings.TrimSuffix(computeReceipt, filepath.Ext(computeReceipt)) + ".native.json"
}

type CUDAObjectBuildReceipt struct {
	Source              string   `json:"source"`
	Object              string   `json:"object"`
	Action              string   `json:"action"`
	MissReason          string   `json:"miss_reason,omitempty"`
	InputSHA256         string   `json:"input_sha256,omitempty"`
	SourceClosureSHA256 string   `json:"source_closure_sha256,omitempty"`
	OutputSHA256        string   `json:"output_sha256,omitempty"`
	EffectiveFlags      []string `json:"effective_flags"`
	DependencyScanMS    int64    `json:"dependency_scan_ms"`
	ToolchainIdentityMS int64    `json:"toolchain_identity_ms"`
	CompileMS           int64    `json:"compile_ms"`
}

// CUDANativeBuildReceipt records native build work; it makes no hardware claim.
type CUDANativeBuildReceipt struct {
	Schema                        string                   `json:"schema"`
	Outcome                       string                   `json:"outcome"`
	Mode                          string                   `json:"mode"`
	Architecture                  string                   `json:"architecture"`
	NCCL                          bool                     `json:"nccl"`
	ToolchainSHA256               string                   `json:"toolchain_sha256,omitempty"`
	SourceClosureSHA256           string                   `json:"source_closure_sha256"`
	SourceCommit                  string                   `json:"source_commit,omitempty"`
	SourceTree                    string                   `json:"source_tree,omitempty"`
	SourceDirty                   bool                     `json:"source_dirty"`
	SourceKnown                   bool                     `json:"source_known"`
	HostOS                        string                   `json:"host_os"`
	HostArch                      string                   `json:"host_arch"`
	Hostname                      string                   `json:"hostname"`
	StartedAt                     time.Time                `json:"started_at"`
	FinishedAt                    time.Time                `json:"finished_at"`
	ElapsedMS                     int64                    `json:"elapsed_ms"`
	NativeElapsedMS               int64                    `json:"native_elapsed_ms"`
	ReceiptPath                   string                   `json:"receipt_path"`
	ArchiveSHA256                 string                   `json:"archive_sha256,omitempty"`
	ComputeReceiptSHA256          string                   `json:"compute_receipt_sha256,omitempty"`
	ComputeReceiptCanonicalSHA256 string                   `json:"compute_receipt_canonical_sha256,omitempty"`
	Executed                      int                      `json:"executed"`
	Reused                        int                      `json:"reused"`
	Objects                       []CUDAObjectBuildReceipt `json:"objects"`

	V41CompressorPending bool `json:"v41_compressor_pending,omitempty"` // Selection only, not device qualification.
	Q4KQKVQualification  bool `json:"q4k_qkv_qualification,omitempty"`
	Q4KQKVAsync          bool `json:"q4k_qkv_async,omitempty"`

	V41RoutedSwiGLU    bool                        `json:"v41_routed_swiglu,omitempty"`    // Build selection, not hardware acceptance.
	V41EngramGate      bool                        `json:"v41_engram_gate,omitempty"`      // Build selection, not hardware acceptance.
	V41CompressorPool  bool                        `json:"v41_compressor_pool,omitempty"`  // Build selection, not hardware acceptance.
	V41Compressor      bool                        `json:"v41_compressor,omitempty"`       // Build selection, not hardware acceptance.
	V41Router          bool                        `json:"v41_router,omitempty"`           // Build selection, not hardware acceptance.
	V41TailRoPE        bool                        `json:"v41_tail_rope,omitempty"`        // Ordinary candidate selection, not exact arithmetic qualification.
	V41BF16Publish     bool                        `json:"v41_bf16_publish,omitempty"`     // Build selection, not native/device qualification.
	V41MHC             bool                        `json:"v41_mhc,omitempty"`              // Witness build selection; public capability remains false.
	V41SharedAttention bool                        `json:"v41_shared_attention,omitempty"` // Build selection, not hardware acceptance.
	V41Defaults        []string                    `json:"v41_defaults,omitempty"`         // Tags selected by the sm_120 default, not by an explicit flag.
	DeepGEMMWKV        *CUDADeepGEMMWKVReceipt     `json:"deepgemm_wkv,omitempty"`
	FlashMLADecode     *CUDAFlashMLADecodeReceipt  `json:"flashmla_decode,omitempty"`
	FlashMLAPrefill    *CUDAFlashMLAPrefillReceipt `json:"flashmla_prefill,omitempty"`
}

func newCUDANativeReceipt(ctx context.Context, root, arch string, nccl bool) *CUDANativeBuildReceipt {
	host, _ := os.Hostname()
	r := &CUDANativeBuildReceipt{Schema: CUDANativeBuildReceiptSchema, Outcome: "failed", Mode: "not_started", Architecture: arch, NCCL: nccl, HostOS: runtime.GOOS, HostArch: runtime.GOARCH, Hostname: host, StartedAt: time.Now().UTC()}
	if plan, ok := ctx.Value(cudaFlashMLADecodeKey{}).(*cudaFlashMLAOptions); ok {
		candidate := plan.receipt
		r.FlashMLADecode = &candidate
	}
	if plan, ok := ctx.Value(cudaFlashMLAKey{}).(*cudaFlashMLAOptions); ok {
		candidate := plan.receipt
		r.FlashMLAPrefill = &candidate
	}
	if plan, ok := ctx.Value(cudaDeepGEMMKey{}).(*cudaDeepGEMMOptions); ok {
		candidate := plan.receipt
		r.DeepGEMMWKV = &candidate
	}
	p, _ := ctx.Value(cudaAsyncQualificationKey{}).(cudaAsyncQualificationProfile)
	r.V41CompressorPending, r.Q4KQKVQualification, r.Q4KQKVAsync = p.CompressorPending, p.Q4KQKV, p.Q4KQKVAsync
	r.V41RoutedSwiGLU, _ = ctx.Value(cudaV41RoutedSwiGLUKey{}).(bool)
	r.V41EngramGate, _ = ctx.Value(cudaV41EngramGateKey{}).(bool)
	r.V41CompressorPool, _ = ctx.Value(cudaV41CompressorPoolKey{}).(bool)
	r.V41Compressor, _ = ctx.Value(cudaV41CompressorKey{}).(bool)
	r.V41Router, _ = ctx.Value(cudaV41RouterKey{}).(bool)
	r.V41TailRoPE, _ = ctx.Value(cudaV41TailRoPEKey{}).(bool)
	r.V41BF16Publish, _ = ctx.Value(cudaV41BF16PublishKey{}).(bool)
	r.V41MHC, _ = ctx.Value(cudaV41MHCKey{}).(bool)
	r.V41SharedAttention, _ = ctx.Value(cudaV41SharedAttentionKey{}).(bool)
	r.V41Defaults = cudaV41DefaultsFromContext(ctx)
	r.ReceiptPath, _ = ctx.Value(cudaReceiptPathContext{}).(string)
	if r.ReceiptPath == "" {
		r.ReceiptPath = CUDANativeReceiptPath(root, "")
	}
	if started, ok := ctx.Value(cudaReceiptStartContext{}).(time.Time); ok {
		r.StartedAt = started
	}
	git := func(args ...string) (string, bool) {
		cmd := exec.CommandContext(ctx, "git", append([]string{"-C", root}, args...)...)
		out, err := cmd.Output()
		return strings.TrimSpace(string(out)), err == nil
	}
	var commitOK, treeOK, statusOK bool
	top, topOK := git("rev-parse", "--show-toplevel")
	rootAbs, rootErr := filepath.Abs(root)
	if !topOK || rootErr != nil || filepath.Clean(top) != filepath.Clean(rootAbs) {
		return r
	}
	r.SourceCommit, commitOK = git("rev-parse", "HEAD")
	r.SourceTree, treeOK = git("rev-parse", "HEAD^{tree}")
	status, statusOK := git("status", "--porcelain", "--untracked-files=normal")
	r.SourceDirty = status != ""
	r.SourceKnown = commitOK && treeOK && statusOK
	return r
}

func (r *CUDANativeBuildReceipt) finish(root string, result error) error {
	r.Outcome = "failed"
	if result == nil {
		r.Outcome = "success"
	}
	inputs := map[string]string{}
	for _, object := range r.Objects {
		if object.Action == "reused" {
			r.Reused++
		} else {
			r.Executed++
		}
		inputs[object.Source] = object.SourceClosureSHA256
	}
	r.SourceClosureSHA256 = cudaJSONDigest(inputs)
	if len(inputs) == 0 {
		r.SourceClosureSHA256 = ""
	}
	for _, digest := range inputs {
		if digest == "" {
			r.SourceClosureSHA256 = ""
		}
	}
	r.Mode = "cold"
	if r.Reused > 0 {
		r.Mode = "mixed"
		if r.Executed == 0 {
			r.Mode = "warm"
		}
	}
	r.FinishedAt = time.Now().UTC()
	r.ElapsedMS = r.FinishedAt.Sub(r.StartedAt).Milliseconds()
	r.NativeElapsedMS = r.ElapsedMS
	return cudaWriteJSON(r.ReceiptPath, r)
}

func finishCUDANativeArchiveReceipt(root, computeReceipt string, result error) error {
	path := CUDANativeReceiptPath(root, computeReceipt)
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var r CUDANativeBuildReceipt
	if err := json.Unmarshal(raw, &r); err != nil {
		return err
	}
	r.Outcome = "failed"
	if result == nil {
		r.Outcome = "success"
		r.ArchiveSHA256, err = cudaArchiveSHA256(filepath.Join(root, "internal", "compute", "libfakcuda.a"))
		if err != nil {
			return err
		}
	}
	if computeReceipt != "" {
		r.ComputeReceiptSHA256, err = cudaFileDigest(computeReceipt)
		if err != nil {
			return err
		}
		raw, err := os.ReadFile(computeReceipt)
		if err != nil {
			return err
		}
		var value any
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.UseNumber()
		if err := decoder.Decode(&value); err != nil {
			return err
		}
		r.ComputeReceiptCanonicalSHA256 = cudaJSONDigest(value)
		var main ComputeBuildReceipt
		if err := json.Unmarshal(raw, &main); err != nil {
			return err
		}
		r.NativeElapsedMS = 0
		for _, phase := range main.Phases {
			if phase.Name == "compile_kernels" || phase.Name == "archive" {
				r.NativeElapsedMS += phase.ElapsedMS
			}
		}
	}
	r.FinishedAt = time.Now().UTC()
	r.ElapsedMS = r.FinishedAt.Sub(r.StartedAt).Milliseconds()
	return cudaWriteJSON(path, &r)
}
