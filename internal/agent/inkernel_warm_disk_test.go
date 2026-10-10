package agent

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/l3kv"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const warmDiskProcessResultMarker = "FAK_WARM_DISK_PROCESS_RESULT="

type warmDiskProcessResult struct {
	Phase        string `json:"phase"`
	ErrorCode    string `json:"error_code,omitempty"`
	DiskOutcome  string `json:"disk_outcome,omitempty"`
	WriteBytes   int64  `json:"write_bytes,omitempty"`
	ReadBytes    int64  `json:"read_bytes,omitempty"`
	Prefilled    int    `json:"prefilled_tokens,omitempty"`
	Stable       int    `json:"stable_tokens,omitempty"`
	Matched      int    `json:"matched_tokens,omitempty"`
	Generated    []int  `json:"generated_tokens,omitempty"`
	LogitsDigest string `json:"logits_digest,omitempty"`
}

// fak-test:runtime slow est=10s lane=serial
func TestInKernelWarmPrefixDiskAcrossProcesses(t *testing.T) {
	cacheRoot := t.TempDir()
	run := func(phase string) warmDiskProcessResult {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestInKernelWarmPrefixDiskProcessHelper$", "-test.count=1")
		cmd.Env = warmDiskProcessEnv(os.Environ(), cacheRoot, phase)
		output, err := cmd.CombinedOutput()
		if ctx.Err() != nil {
			t.Fatalf("%s child timed out", phase)
		}
		if err != nil {
			t.Fatalf("%s child failed: %v output=%q", phase, err, string(output))
		}
		for _, line := range bytes.Split(output, []byte{'\n'}) {
			line = bytes.TrimSpace(line)
			if !bytes.HasPrefix(line, []byte(warmDiskProcessResultMarker)) {
				continue
			}
			var result warmDiskProcessResult
			if err := json.Unmarshal(bytes.TrimPrefix(line, []byte(warmDiskProcessResultMarker)), &result); err != nil {
				t.Fatalf("%s child result decode: %v", phase, err)
			}
			return result
		}
		t.Fatalf("%s child emitted no structured result", phase)
		return warmDiskProcessResult{}
	}

	cold := run("cold")
	warm := run("warm")
	if cold.ErrorCode != "" || warm.ErrorCode != "" {
		t.Fatalf("child error codes cold=%q warm=%q", cold.ErrorCode, warm.ErrorCode)
	}
	if cold.DiskOutcome != string(WarmDiskPersisted) || cold.WriteBytes <= 0 || cold.Prefilled != cold.Stable {
		t.Fatalf("cold outcome=%s write=%d prefilled=%d stable=%d, want persisted positive full prefill", cold.DiskOutcome, cold.WriteBytes, cold.Prefilled, cold.Stable)
	}
	if warm.DiskOutcome != string(WarmDiskRestored) || warm.ReadBytes <= 0 || warm.Prefilled != 0 {
		t.Fatalf("warm outcome=%s read=%d prefilled=%d, want restored positive zero prefill", warm.DiskOutcome, warm.ReadBytes, warm.Prefilled)
	}
	if cold.Matched < cold.Stable || warm.Matched < warm.Stable {
		t.Fatalf("demand matched cold=%d/%d warm=%d/%d, want stable reuse", cold.Matched, cold.Stable, warm.Matched, warm.Stable)
	}
	if !eqInts(cold.Generated, warm.Generated) || cold.LogitsDigest == "" || cold.LogitsDigest != warm.LogitsDigest {
		t.Fatalf("cross-process parity tokens=%t logits=%t", eqInts(cold.Generated, warm.Generated), cold.LogitsDigest == warm.LogitsDigest)
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestInKernelWarmPrefixDiskProcessHelper(t *testing.T) {
	phase := os.Getenv("FAK_TEST_WARM_DISK_PROCESS")
	if phase == "" {
		t.Skip("subprocess helper")
	}
	result := runWarmDiskProcessHelper(t, phase)
	payload, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("marshal structured result: %v", err)
	}
	fmt.Printf("%s%s\n", warmDiskProcessResultMarker, payload)
}

func runWarmDiskProcessHelper(t *testing.T, phase string) warmDiskProcessResult {
	t.Helper()
	result := warmDiskProcessResult{Phase: phase}
	ctx := context.Background()
	in := warmFixtureInputs()
	p := NewInKernelPlanner(model.NewSynthetic(warmCfg()), loadProbeTok(t), "synthetic-process-disk", false, nil, false)
	p.quant = false
	p.SetWarmDiskModelDigest("sha256:synthetic-process-disk")
	spec, err := p.DeriveWarmPrefix("tenant-process", "agent-1", in)
	if err != nil {
		result.ErrorCode = "derive_failed"
		return result
	}
	p.SetWarmPrefixInputs(in)
	receipt, err := p.WarmPrefix(WithPrefixCacheIdentity(ctx, spec.Scope.Tenant, spec.Scope.Agent), spec)
	if err != nil || !receipt.Usable() || receipt.Disk == nil {
		result.ErrorCode = "warm_failed"
		return result
	}
	result.DiskOutcome = string(receipt.Disk.Outcome)
	result.WriteBytes = receipt.Disk.WriteBytes
	result.ReadBytes = receipt.Disk.ReadBytes
	result.Prefilled = receipt.PrefilledTokens
	result.Stable = spec.StableTokens

	messages := make([]Message, 0, len(in.SystemBlocks)+1)
	messages = append(messages, Message{Role: RoleSystem, Content: string(in.Instructions)})
	for _, block := range in.SystemBlocks {
		messages = append(messages, Message{Role: RoleSystem, Content: string(block)})
	}
	encoded, err := p.EncodePrompt(ctx, messages, in.Tools)
	if err != nil {
		result.ErrorCode = "encode_failed"
		return result
	}
	kv, logits, matched, _, err := p.scopedTree.Lookup(spec.Scope, encoded.TokenIDs)
	if err != nil || kv == nil || matched < spec.StableTokens || len(logits) == 0 {
		result.ErrorCode = "readback_failed"
		return result
	}
	result.LogitsDigest = warmDiskLogitsDigest(logits)
	demand := append(append([]int(nil), encoded.TokenIDs...), synthIDs(warmCfg().VocabSize, 5, 7719)...)
	_, _, _, result.Matched, _, _, _, _, err = p.generateReusedContextWithBias(
		WithPrefixCacheIdentity(ctx, spec.Scope.Tenant, spec.Scope.Agent), demand,
		4, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
			result.Generated = append(result.Generated, id)
			return false
		})
	if err != nil {
		result.ErrorCode = "demand_failed"
	}
	return result
}

func warmDiskLogitsDigest(logits []float32) string {
	wire := make([]byte, len(logits)*4)
	for i, value := range logits {
		binary.LittleEndian.PutUint32(wire[i*4:], math.Float32bits(value))
	}
	sum := sha256.Sum256(wire)
	return hex.EncodeToString(sum[:])
}

func warmDiskProcessEnv(base []string, cacheRoot, phase string) []string {
	overrides := map[string]string{
		"FAK_TEST_WARM_DISK_PROCESS": phase,
		"FAK_L3_KVBACKEND":           "",
		"FAK_BLOB_HTTP_URL":          "",
		"FAK_BLOB_HTTP_TOKEN":        "",
		"FAK_INKERNEL_RADIX":         "on",
		"LOCALAPPDATA":               cacheRoot,
		"XDG_CACHE_HOME":             cacheRoot,
		"HOME":                       cacheRoot,
	}
	env := make([]string, 0, len(base)+len(overrides))
	for _, entry := range base {
		key, _, ok := strings.Cut(entry, "=")
		if _, replaced := overrides[strings.ToUpper(key)]; ok && replaced {
			continue
		}
		env = append(env, entry)
	}
	for key, value := range overrides {
		env = append(env, key+"="+value)
	}
	return env
}

// fak-test:runtime fast est=3s lane=default
// TestInKernelWarmPrefixDiskPersistence is the restart witness for the default local
// SSD prefix tier. It uses the tiny native CPU fixture: every hit restores real session
// state and the following demand runs the production reuse path, without GPU hardware.
func TestInKernelWarmPrefixDiskPersistence(t *testing.T) {
	ctx := context.Background()
	in := warmFixtureInputs()

	newPlanner := func(t *testing.T, dir, modelID string, disabled bool) *InKernelPlanner {
		t.Helper()
		p := NewInKernelPlanner(model.NewSynthetic(warmCfg()), loadProbeTok(t), modelID, false, nil, false)
		p.quant = false
		p.SetWarmDiskConfig(WarmDiskConfig{Dir: dir, Disabled: disabled, MaxEntries: 8, MaxBytes: 8 << 20})
		p.SetWarmDiskModelDigest("sha256:test-" + modelID)
		return p
	}

	derive := func(t *testing.T, p *InKernelPlanner, tenant, agent string, inputs WarmPrefixInputs) WarmPrefixSpec {
		t.Helper()
		spec, err := p.DeriveWarmPrefix(tenant, agent, inputs)
		if err != nil {
			t.Fatalf("DeriveWarmPrefix: %v", err)
		}
		p.SetWarmPrefixInputs(inputs)
		return spec
	}

	stableTokens := func(t *testing.T, p *InKernelPlanner, inputs WarmPrefixInputs) []int {
		t.Helper()
		messages := make([]Message, 0, len(inputs.SystemBlocks)+1)
		if len(inputs.Instructions) > 0 {
			messages = append(messages, Message{Role: RoleSystem, Content: string(inputs.Instructions)})
		}
		for _, block := range inputs.SystemBlocks {
			if len(block) > 0 {
				messages = append(messages, Message{Role: RoleSystem, Content: string(block)})
			}
		}
		enc, err := p.EncodePrompt(ctx, messages, inputs.Tools)
		if err != nil {
			t.Fatalf("encode stable tokens: %v", err)
		}
		return enc.TokenIDs
	}

	warm := func(t *testing.T, p *InKernelPlanner, spec WarmPrefixSpec) WarmReceipt {
		t.Helper()
		receipt, err := p.WarmPrefix(WithPrefixCacheIdentity(ctx, spec.Scope.Tenant, spec.Scope.Agent), spec)
		if err != nil || !receipt.Usable() {
			t.Fatalf("WarmPrefix: err=%v status=%s reason=%s restored=%d requested=%d", err, receipt.Status, receipt.Reason, receipt.RestoredTokens, receipt.RequestedTokens)
		}
		return receipt
	}

	t.Run("record budget follows host headroom and explicit bounds", func(t *testing.T) {
		tests := []struct {
			name      string
			requested int64
			maxBytes  int64
			freeBytes int64
			known     bool
			want      int64
		}{
			{name: "unknown memory fallback", maxBytes: 8 << 30, want: 512 << 20},
			{name: "low memory eighth", maxBytes: 8 << 30, freeBytes: 800 << 20, known: true, want: 100 << 20},
			{name: "high memory capped", maxBytes: 8 << 30, freeBytes: 32 << 30, known: true, want: 2 << 30},
			{name: "known zero remains bounded", maxBytes: 8 << 30, known: true, want: 1},
			{name: "explicit record below headroom", requested: 64 << 20, maxBytes: 8 << 30, freeBytes: 32 << 30, known: true, want: 64 << 20},
			{name: "explicit record cannot exceed headroom", requested: 4 << 30, maxBytes: 8 << 30, freeBytes: 32 << 30, known: true, want: 2 << 30},
			{name: "aggregate budget bounds explicit record", requested: 64 << 20, maxBytes: 32 << 20, freeBytes: 32 << 30, known: true, want: 32 << 20},
			{name: "aggregate budget bounds derived default", maxBytes: 128 << 20, freeBytes: 32 << 30, known: true, want: 128 << 20},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				if got := resolveWarmDiskRecordBudget(tc.requested, tc.maxBytes, tc.freeBytes, tc.known); got != tc.want {
					t.Fatalf("record budget=%d, want %d", got, tc.want)
				}
			})
		}
	})

	t.Run("startup capture is armed only when persistence can consume it", func(t *testing.T) {
		tests := []struct {
			name        string
			disabled    bool
			setIdentity bool
			wantOutcome WarmDiskOutcome
		}{
			{name: "explicit off", disabled: true, setIdentity: true, wantOutcome: WarmDiskDisabled},
			{name: "model identity unavailable", wantOutcome: WarmDiskOutcomeUnsupported},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				cfg := warmCfg()
				cfg.EnableResidualHook = true
				m := model.NewSynthetic(cfg)
				armedDuringPrefill := false
				var p *InKernelPlanner
				m.SetResidualHook(func(int, []float32) {
					if p != nil && p.warmDiskCapture {
						armedDuringPrefill = true
					}
				})
				p = NewInKernelPlanner(m, loadProbeTok(t), "capture-gate", false, nil, false)
				p.quant = false
				p.SetWarmDiskConfig(WarmDiskConfig{Dir: t.TempDir(), Disabled: tc.disabled, MaxEntries: 1, MaxBytes: 8 << 20, MaxRecordBytes: 8 << 20})
				if tc.setIdentity {
					p.SetWarmDiskModelDigest("sha256:capture-gate")
				}
				receipt := warm(t, p, derive(t, p, "tenant-capture", "agent-1", in))
				if receipt.Disk == nil || receipt.Disk.Outcome != tc.wantOutcome {
					t.Fatalf("disk outcome=%s, want %s", warmDiskOutcome(receipt.Disk), tc.wantOutcome)
				}
				if armedDuringPrefill {
					t.Fatal("ineligible persistence armed startup capture during prefill")
				}
				if p.warmDiskCapture || p.warmDiskCaptured != nil {
					t.Fatal("transient capture state survived WarmPrefix")
				}
			})
		}

		p := newPlanner(t, t.TempDir(), "capture-budget", false)
		session := p.m.NewSession()
		defer session.Close()
		tokens := []int{3, 7, 11}
		logits := session.Prefill(tokens)
		p.beginWarmDiskCapture(1)
		p.captureWarmDiskCandidate(ctx, session, tokens, logits)
		if p.warmDiskCaptured != nil {
			t.Fatal("over-budget candidate cloned a startup snapshot")
		}
		p.endWarmDiskCapture()
	})

	t.Run("fresh planner restores the complete prefix before demand", func(t *testing.T) {
		dir := t.TempDir()
		first := newPlanner(t, dir, "synthetic-disk", false)
		specA := derive(t, first, "tenant-a", "agent-1", in)
		written := warm(t, first, specA)
		if written.Disk == nil || written.Disk.Outcome != WarmDiskPersisted || written.Disk.WriteBytes <= 0 {
			t.Fatalf("first startup disk outcome=%s write_bytes=%d, want persisted positive bytes", warmDiskOutcome(written.Disk), warmDiskWriteBytes(written.Disk))
		}
		if written.PrefilledTokens != specA.StableTokens {
			t.Fatalf("cold startup prefilled %d tokens, want full stable prefix %d", written.PrefilledTokens, specA.StableTokens)
		}

		// A distinct planner models a process restart: its radix tree begins empty and
		// shares only the durable directory and stable model identity with process A.
		second := newPlanner(t, dir, "synthetic-disk", false)
		specB := derive(t, second, "tenant-a", "agent-1", in)
		if specB.Identity != specA.Identity {
			t.Fatalf("stable descriptor changed across restart: %q != %q", specB.Identity, specA.Identity)
		}
		tokens := stableTokens(t, second, in)
		if matched, err := second.scopedTree.MatchLen(specB.Scope, tokens); err != nil || matched != 0 {
			t.Fatalf("fresh planner precondition matched=%d err=%v, want empty tree", matched, err)
		}

		restored := warm(t, second, specB)
		if restored.Disk == nil || restored.Disk.Outcome != WarmDiskRestored ||
			restored.Disk.Tier != WarmDiskTierLocalSSD || restored.Disk.ReadBytes <= 0 || restored.Disk.WriteBytes != 0 {
			t.Fatalf("restart disk outcome=%s tier=%s read=%d write=%d, want restored local SSD read", warmDiskOutcome(restored.Disk), warmDiskReceiptTier(restored.Disk), warmDiskReadBytes(restored.Disk), warmDiskWriteBytes(restored.Disk))
		}
		if restored.RestoredTokens < specB.StableTokens || restored.SourceTier != WarmDiskSnapshotTier {
			t.Fatalf("restart restored=%d stable=%d source=%s, want full local-SSD restore", restored.RestoredTokens, specB.StableTokens, restored.SourceTier)
		}
		if restored.PrefilledTokens != 0 {
			t.Fatalf("restart disk hit still prefilled %d tokens, want 0", restored.PrefilledTokens)
		}

		// The next demand extends the restored stable state. The production demand path
		// must report a prefix hit and preserve cold-reference logits/token parity.
		demand := append(append([]int(nil), tokens...), synthIDs(warmCfg().VocabSize, 5, 9917)...)
		var got []int
		_, _, _, matched, _, _, _, _, err := second.generateReusedContextWithBias(
			WithPrefixCacheIdentity(ctx, specB.Scope.Tenant, specB.Scope.Agent), demand,
			4, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
				got = append(got, id)
				return false
			})
		if err != nil {
			t.Fatalf("restored demand: %v", err)
		}
		cold := newPlanner(t, t.TempDir(), "synthetic-disk", true)
		cold.SetWarmPrefixInputs(in)
		var want []int
		_, _, _, coldMatched, _, _, _, _, err := cold.generateReusedContextWithBias(
			WithPrefixCacheIdentity(ctx, specB.Scope.Tenant, specB.Scope.Agent), demand,
			4, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
				want = append(want, id)
				return false
			})
		if err != nil {
			t.Fatalf("cold reference demand: %v", err)
		}
		if matched < specB.StableTokens || coldMatched != 0 {
			t.Fatalf("demand reuse matched=%d cold=%d, want >=%d/0", matched, coldMatched, specB.StableTokens)
		}
		if !eqInts(got, want) {
			t.Fatalf("disk-restored demand changed output: restored=%v cold=%v", got, want)
		}
	})

	t.Run("different model bytes under the same model ID cannot reuse an old record", func(t *testing.T) {
		dir := t.TempDir()
		basePlanner := newPlanner(t, dir, "synthetic-isolation", false)
		base := derive(t, basePlanner, "tenant-a", "agent-1", in)
		warm(t, basePlanner, base)

		changedWeights := newPlanner(t, dir, "synthetic-isolation", false)
		changedWeights.SetWarmDiskModelDigest("sha256:different-loaded-model-bytes")
		receipt := warm(t, changedWeights, derive(t, changedWeights, "tenant-a", "agent-1", in))
		if receipt.Disk == nil || receipt.Disk.RestoreOutcome != WarmDiskOutcomeMiss || receipt.Disk.RestoreReason != "model_identity_changed" {
			t.Fatalf("different model bytes disk receipt=%+v, want restore miss/model_identity_changed", receipt.Disk)
		}
		if receipt.PrefilledTokens != receipt.RequestedTokens {
			t.Fatalf("refused restore prefilled=%d, want cold %d", receipt.PrefilledTokens, receipt.RequestedTokens)
		}

		fresh := newPlanner(t, t.TempDir(), "synthetic-isolation", false)
		freshReceipt := warm(t, fresh, derive(t, fresh, "tenant-a", "agent-1", in))
		if freshReceipt.Disk == nil || freshReceipt.Disk.RestoreReason != "absent" {
			t.Fatalf("never-persisted disk receipt=%+v, want restore reason absent", freshReceipt.Disk)
		}
	})

	t.Run("qwen35 device policy restores through host disk without prefix prefill", func(t *testing.T) {
		dir := t.TempDir()
		cfg := tinyHybridCfg()
		cfg.AttnOutputGate = false
		cfg.VocabSize = warmCfg().VocabSize
		newHybrid := func(t *testing.T, disabled bool) *InKernelPlanner {
			t.Helper()
			backend := &countingBackend{Backend: compute.Default(), deviceMemory: true}
			p := NewInKernelPlannerWithConfig(model.NewSynthetic(cfg), loadProbeTok(t), "qwen35-host-disk", false, backend, false, InKernelPlannerConfig{
				RequireDeviceExecution: true,
				DenseGPULayers:         cfg.NumLayers,
			})
			p.quant = false
			p.SetWarmDiskConfig(WarmDiskConfig{Dir: dir, Disabled: disabled, MaxEntries: 2, MaxBytes: 8 << 20})
			p.SetWarmDiskModelDigest("sha256:qwen35-host-v1")
			return p
		}

		first := newHybrid(t, false)
		firstSpec := derive(t, first, "tenant-qwen", "agent-1", in)
		written := warm(t, first, firstSpec)
		if written.Disk == nil || written.Disk.Outcome != WarmDiskPersisted {
			t.Fatalf("Qwen cold startup disk outcome=%s, want persisted", warmDiskOutcome(written.Disk))
		}

		second := newHybrid(t, false)
		secondSpec := derive(t, second, "tenant-qwen", "agent-1", in)
		restored := warm(t, second, secondSpec)
		if restored.Disk == nil || restored.Disk.Outcome != WarmDiskRestored || restored.PrefilledTokens != 0 {
			t.Fatalf("Qwen restart disk=%s prefilled=%d, want restored/0", warmDiskOutcome(restored.Disk), restored.PrefilledTokens)
		}
		if runtime := second.RuntimeConfig(); !runtime.RequireDeviceExecution || runtime.DenseGPULayers != cfg.NumLayers {
			t.Fatalf("Qwen restored runtime device_only=%t gpu_layers=%d, want true/%d", runtime.RequireDeviceExecution, runtime.DenseGPULayers, cfg.NumLayers)
		}

		tokens := stableTokens(t, second, in)
		demand := append(append([]int(nil), tokens...), synthIDs(cfg.VocabSize, 3, 8107)...)
		var got []int
		_, _, _, matched, _, _, _, _, err := second.generateReusedContextWithBias(
			WithPrefixCacheIdentity(ctx, secondSpec.Scope.Tenant, secondSpec.Scope.Agent), demand,
			2, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool { got = append(got, id); return false })
		if err != nil {
			t.Fatalf("Qwen restored suffix: %v", err)
		}
		cold := newHybrid(t, true)
		cold.SetWarmPrefixInputs(in)
		var want []int
		_, _, _, coldMatched, _, _, _, _, err := cold.generateReusedContextWithBias(
			WithPrefixCacheIdentity(ctx, secondSpec.Scope.Tenant, secondSpec.Scope.Agent), demand,
			2, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool { want = append(want, id); return false })
		if err != nil {
			t.Fatalf("Qwen cold suffix: %v", err)
		}
		if matched < secondSpec.StableTokens || coldMatched != 0 || !eqInts(got, want) {
			t.Fatalf("Qwen suffix matched=%d cold=%d parity=%t, want >=%d/0/true", matched, coldMatched, eqInts(got, want), secondSpec.StableTokens)
		}
	})

	t.Run("corrupt and unwritable stores fall back cold without blocking startup", func(t *testing.T) {
		dir := t.TempDir()
		first := newPlanner(t, dir, "synthetic-fallback", false)
		spec := derive(t, first, "tenant-a", "agent-1", in)
		warm(t, first, spec)
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				if err := os.WriteFile(filepath.Join(dir, entry.Name()), []byte("corrupt"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
		}
		corrupt := newPlanner(t, dir, "synthetic-fallback", false)
		corruptReceipt := warm(t, corrupt, derive(t, corrupt, "tenant-a", "agent-1", in))
		if corruptReceipt.Disk != nil && corruptReceipt.Disk.Outcome == WarmDiskRestored {
			t.Fatalf("corrupt record disk outcome=%s, want non-restored", corruptReceipt.Disk.Outcome)
		}
		if corruptReceipt.Disk == nil || corruptReceipt.Disk.RestoreOutcome != WarmDiskOutcomeFault || corruptReceipt.Disk.RestoreReason != "read_failed" {
			t.Fatalf("corrupt restore outcome=%s reason=%s, want fault/read_failed", warmDiskRestoreOutcome(corruptReceipt.Disk), warmDiskRestoreReason(corruptReceipt.Disk))
		}

		parent := filepath.Join(t.TempDir(), "regular-file")
		if err := os.WriteFile(parent, []byte("not a directory"), 0o600); err != nil {
			t.Fatal(err)
		}
		unwritable := newPlanner(t, filepath.Join(parent, "cache"), "synthetic-unwritable", false)
		unwritableReceipt := warm(t, unwritable, derive(t, unwritable, "tenant-a", "agent-1", in))
		if unwritableReceipt.Disk == nil || unwritableReceipt.Disk.Outcome != WarmDiskFault {
			t.Fatalf("unwritable store outcome=%s, want nonfatal fault", warmDiskOutcome(unwritableReceipt.Disk))
		}
	})

	t.Run("entry and byte budgets bound the store", func(t *testing.T) {
		dir := t.TempDir()
		p := newPlanner(t, dir, "synthetic-bounded", false)
		p.SetWarmDiskConfig(WarmDiskConfig{Dir: dir, MaxEntries: 1, MaxBytes: 8 << 20})
		warm(t, p, derive(t, p, "tenant-a", "agent-1", in))
		changed := in
		changed.Instructions = []byte("A different stable startup prefix.")
		warm(t, p, derive(t, p, "tenant-a", "agent-1", changed))
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		files := 0
		for _, entry := range entries {
			if !entry.IsDir() {
				files++
			}
		}
		if files > 1 {
			t.Fatalf("entry budget retained %d records, want <= 1", files)
		}

		tiny := newPlanner(t, t.TempDir(), "synthetic-byte-bound", false)
		tiny.SetWarmDiskConfig(WarmDiskConfig{Dir: t.TempDir(), MaxEntries: 1, MaxBytes: 1})
		tinyReceipt := warm(t, tiny, derive(t, tiny, "tenant-a", "agent-1", in))
		if tinyReceipt.Disk == nil || tinyReceipt.Disk.Outcome != WarmDiskOutcomeUnsupported || tinyReceipt.Disk.Reason != "complete_snapshot_unavailable" {
			t.Fatalf("byte bound outcome=%s reason=%s, want unsupported/complete_snapshot_unavailable", warmDiskOutcome(tinyReceipt.Disk), warmDiskReason(tinyReceipt.Disk))
		}
	})

	t.Run("unset uses user cache and explicit off disables persistence", func(t *testing.T) {
		cacheRoot := t.TempDir()
		t.Setenv("FAK_L3_KVBACKEND", "")
		t.Setenv("XDG_CACHE_HOME", cacheRoot)
		t.Setenv("LOCALAPPDATA", cacheRoot)
		t.Setenv("HOME", cacheRoot)
		def := NewInKernelPlanner(model.NewSynthetic(warmCfg()), loadProbeTok(t), "synthetic-default", false, nil, false)
		def.quant = false
		def.SetWarmDiskModelDigest("sha256:test-synthetic-default")
		defaultReceipt := warm(t, def, derive(t, def, "tenant-a", "agent-1", in))
		if defaultReceipt.Disk == nil || defaultReceipt.Disk.Outcome != WarmDiskPersisted || defaultReceipt.Disk.WriteBytes <= 0 {
			t.Fatalf("unset default outcome=%s write_bytes=%d, want persisted positive bytes", warmDiskOutcome(defaultReceipt.Disk), warmDiskWriteBytes(defaultReceipt.Disk))
		}

		t.Setenv("FAK_L3_KVBACKEND", "off")
		t.Setenv("FAK_BLOB_HTTP_URL", "https://blob.invalid.test")
		if store, configured, err := l3kv.ConfiguredRemoteStore(); err != nil || configured || store != nil {
			t.Fatalf("explicit off remote store=%T configured=%t err=%v, want nil/false/nil", store, configured, err)
		}
		off := NewInKernelPlanner(model.NewSynthetic(warmCfg()), loadProbeTok(t), "synthetic-off", false, nil, false)
		off.quant = false
		off.SetWarmDiskModelDigest("sha256:test-synthetic-off")
		offReceipt := warm(t, off, derive(t, off, "tenant-a", "agent-1", in))
		if offReceipt.Disk == nil || offReceipt.Disk.Outcome != WarmDiskDisabled || offReceipt.Disk.ReadBytes != 0 || offReceipt.Disk.WriteBytes != 0 {
			t.Fatalf("explicit off outcome=%s read=%d write=%d, want disabled zero bytes", warmDiskOutcome(offReceipt.Disk), warmDiskReadBytes(offReceipt.Disk), warmDiskWriteBytes(offReceipt.Disk))
		}
	})
}

func warmDiskOutcome(d *WarmDiskReceipt) WarmDiskOutcome {
	if d == nil {
		return ""
	}
	return d.Outcome
}

func warmDiskRestoreOutcome(d *WarmDiskReceipt) WarmDiskOutcome {
	if d == nil {
		return ""
	}
	return d.RestoreOutcome
}

func warmDiskReceiptTier(d *WarmDiskReceipt) string {
	if d == nil {
		return ""
	}
	return string(d.Tier)
}

func warmDiskReason(d *WarmDiskReceipt) string {
	if d == nil {
		return ""
	}
	return d.Reason
}

func warmDiskRestoreReason(d *WarmDiskReceipt) string {
	if d == nil {
		return ""
	}
	return d.RestoreReason
}

func warmDiskReadBytes(d *WarmDiskReceipt) int64 {
	if d == nil {
		return 0
	}
	return d.ReadBytes
}

func warmDiskWriteBytes(d *WarmDiskReceipt) int64 {
	if d == nil {
		return 0
	}
	return d.WriteBytes
}
