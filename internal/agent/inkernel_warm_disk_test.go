package agent

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
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
		// Records now live inside the 16-hex identity subdirs; top-level files are
		// only clock/word bookkeeping whose corruption degrades LRU ordering, so the
		// record corruption must descend into the identity dirs.
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range entries {
			if !entry.IsDir() || !isLearnedIdentityName(entry.Name()) {
				continue
			}
			sub, err := os.ReadDir(filepath.Join(dir, entry.Name()))
			if err != nil {
				t.Fatal(err)
			}
			for _, se := range sub {
				if se.IsDir() {
					continue
				}
				if err := os.WriteFile(filepath.Join(dir, entry.Name(), se.Name()), []byte("corrupt"), 0o600); err != nil {
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
		// Record files live inside the 16-hex identity subdirs; count them there.
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		files := 0
		for _, entry := range entries {
			if !entry.IsDir() || !isLearnedIdentityName(entry.Name()) {
				continue
			}
			sub, err := os.ReadDir(filepath.Join(dir, entry.Name()))
			if err != nil {
				t.Fatal(err)
			}
			for _, se := range sub {
				if !se.IsDir() && isLearnedRecordKey(se.Name()) {
					files++
				}
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

// fak-test:runtime fast est=1s lane=default
// TestLearnedPrefixDiskIndex witnesses the identity-scoped learned shared-prefix
// disk index: per-scope identity dirs, the JSON state index, the pending-write
// marker with crash reconciliation and concurrent-writer deferral, corrupt-index
// rebuild, stray-record reaping, the per-identity entry cap, the cross-identity
// byte-cap sweep, and legacy flat-record fallback.
func TestLearnedPrefixDiskIndex(t *testing.T) {
	ctx := context.Background()
	in := warmFixtureInputs()
	const modelDigest = "sha256:learned-disk-index"

	newPlanner := func(t *testing.T, dir string, cfg WarmDiskConfig) *InKernelPlanner {
		t.Helper()
		p := NewInKernelPlanner(model.NewSynthetic(warmCfg()), loadProbeTok(t), "learned-disk-index", false, nil, false)
		p.quant = false
		p.SetWarmDiskConfig(cfg)
		p.SetWarmDiskModelDigest(modelDigest)
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

	warm := func(t *testing.T, p *InKernelPlanner, spec WarmPrefixSpec) WarmReceipt {
		t.Helper()
		receipt, err := p.WarmPrefix(WithPrefixCacheIdentity(ctx, spec.Scope.Tenant, spec.Scope.Agent), spec)
		if err != nil || !receipt.Usable() {
			t.Fatalf("WarmPrefix: err=%v status=%s reason=%s restored=%d requested=%d", err, receipt.Status, receipt.Reason, receipt.RestoredTokens, receipt.RequestedTokens)
		}
		return receipt
	}

	openResolved := func(t *testing.T, dir string, spec WarmPrefixSpec) *learnedPrefixStore {
		t.Helper()
		resolved, err := resolveWarmDiskConfig(WarmDiskConfig{Dir: dir})
		if err != nil {
			t.Fatalf("resolveWarmDiskConfig: %v", err)
		}
		store, err := openLearnedPrefixStore(ctx, resolved, spec.Scope)
		if err != nil {
			t.Fatalf("openLearnedPrefixStore: %v", err)
		}
		return store
	}

	readIndex := func(t *testing.T, dir, identity string) learnedIndex {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(dir, identity, learnedIndexFile))
		if err != nil {
			t.Fatalf("read index: %v", err)
		}
		var idx learnedIndex
		if err := json.Unmarshal(data, &idx); err != nil {
			t.Fatalf("decode index: %v", err)
		}
		return idx
	}

	recordSize := func(t *testing.T, dir, identity, key string) int64 {
		t.Helper()
		info, err := os.Stat(filepath.Join(dir, identity, key))
		if err != nil {
			t.Fatalf("stat record %s: %v", key, err)
		}
		return info.Size()
	}

	buildPendingMarker := func(t *testing.T, key string) []byte {
		t.Helper()
		buf := make([]byte, 0, 8+len(key)+sha256.Size)
		buf = append(buf, learnedPendingMagic...)
		var head [4]byte
		binary.LittleEndian.PutUint16(head[:2], learnedPendingVersion)
		binary.LittleEndian.PutUint16(head[2:4], uint16(len(key)))
		buf = append(buf, head[:]...)
		buf = append(buf, key...)
		sum := sha256.Sum256(buf)
		return append(buf, sum[:]...)
	}

	stdCfg := func(dir string) WarmDiskConfig { return WarmDiskConfig{Dir: dir, MaxEntries: 8, MaxBytes: 8 << 20} }

	t.Run("identity scoping isolates tenants", func(t *testing.T) {
		dir := t.TempDir()
		a := newPlanner(t, dir, stdCfg(dir))
		specA := derive(t, a, "tenant-a", "agent-1", in)
		ra := warm(t, a, specA)
		if ra.Disk == nil || ra.Disk.Outcome != WarmDiskPersisted {
			t.Fatalf("tenant A outcome=%s reason=%s, want persisted", warmDiskOutcome(ra.Disk), warmDiskReason(ra.Disk))
		}
		b := newPlanner(t, dir, stdCfg(dir))
		specB := derive(t, b, "tenant-b", "agent-1", in)
		rb := warm(t, b, specB)
		if rb.Disk == nil || rb.Disk.Outcome != WarmDiskPersisted {
			t.Fatalf("tenant B outcome=%s reason=%s, want persisted", warmDiskOutcome(rb.Disk), warmDiskReason(rb.Disk))
		}

		idA, idB := learnedPrefixIdentity(specA.Scope), learnedPrefixIdentity(specB.Scope)
		if idA == idB {
			t.Fatalf("identities %q/%q, want distinct names", idA, idB)
		}
		for _, id := range []string{idA, idB} {
			decoded, err := hex.DecodeString(id)
			if err != nil || len(decoded) != binary.Size(uint64(0)) || id != strings.ToLower(id) {
				t.Fatalf("identity %q: decoded=%x err=%v, want a 64-bit lowercase hex name", id, decoded, err)
			}
		}
		keyA := warmDiskKey(specA, modelDigest, a.warmDiskExecutionIdentity())
		keyB := warmDiskKey(specB, modelDigest, b.warmDiskExecutionIdentity())
		if keyA == keyB {
			t.Fatal("distinct tenant scopes derived the same record key")
		}
		for _, tc := range []struct{ id, keep, other string }{
			{id: idA, keep: keyA, other: keyB},
			{id: idB, keep: keyB, other: keyA},
		} {
			entries, err := os.ReadDir(filepath.Join(dir, tc.id))
			if err != nil {
				t.Fatalf("identity dir %s: %v", tc.id, err)
			}
			records := 0
			for _, e := range entries {
				if !e.IsDir() && isLearnedRecordKey(e.Name()) {
					records++
				}
			}
			if records != 1 {
				t.Fatalf("identity %s holds %d records, want 1", tc.id, records)
			}
			if _, err := os.Stat(filepath.Join(dir, tc.id, learnedIndexFile)); err != nil {
				t.Fatalf("identity %s index missing: %v", tc.id, err)
			}
			if _, err := os.Stat(filepath.Join(dir, tc.id, tc.other)); !os.IsNotExist(err) {
				t.Fatalf("identity %s holds the other tenant's record %s", tc.id, tc.other)
			}
		}

		// A restart under tenant A restores its own record...
		freshA := newPlanner(t, dir, stdCfg(dir))
		raHit := warm(t, freshA, derive(t, freshA, "tenant-a", "agent-1", in))
		if raHit.Disk == nil || raHit.Disk.RestoreOutcome != WarmDiskOutcomeRestored {
			t.Fatalf("tenant A restart restore outcome=%s reason=%s, want restored", warmDiskRestoreOutcome(raHit.Disk), warmDiskRestoreReason(raHit.Disk))
		}
		// ...while a scope that never persisted stays a clean miss.
		freshC := newPlanner(t, dir, stdCfg(dir))
		rcMiss := warm(t, freshC, derive(t, freshC, "tenant-c", "agent-1", in))
		if rcMiss.Disk == nil || rcMiss.Disk.RestoreOutcome != WarmDiskOutcomeMiss || rcMiss.Disk.RestoreReason != "absent" {
			t.Fatalf("tenant C restore outcome=%s reason=%s, want miss/absent", warmDiskRestoreOutcome(rcMiss.Disk), warmDiskRestoreReason(rcMiss.Disk))
		}
	})

	t.Run("index lists states and pending is cleared after commit", func(t *testing.T) {
		dir := t.TempDir()
		p := newPlanner(t, dir, stdCfg(dir))
		spec := derive(t, p, "tenant-a", "agent-1", in)
		r := warm(t, p, spec)
		if r.Disk == nil || r.Disk.Outcome != WarmDiskPersisted || r.Disk.WriteBytes <= 0 {
			t.Fatalf("outcome=%s write=%d, want persisted positive bytes", warmDiskOutcome(r.Disk), warmDiskWriteBytes(r.Disk))
		}
		id := learnedPrefixIdentity(spec.Scope)
		key := warmDiskKey(spec, modelDigest, p.warmDiskExecutionIdentity())
		idx := readIndex(t, dir, id)
		if idx.Version != learnedIndexVersion || len(idx.States) != 1 {
			t.Fatalf("index version=%d states=%d, want v%d one state", idx.Version, len(idx.States), learnedIndexVersion)
		}
		st := idx.States[0]
		if st.Key != key || st.Bytes != l3kv.RecordBytes(r.Disk.WriteBytes) || st.Used < 1 {
			t.Fatalf("state key=%s bytes=%d used=%d, want %s/%d/used>=1", st.Key, st.Bytes, st.Used, key, l3kv.RecordBytes(r.Disk.WriteBytes))
		}
		if _, err := os.Stat(filepath.Join(dir, id, learnedPendingFile)); !os.IsNotExist(err) {
			t.Fatalf("pending marker survived commit: %v", err)
		}
		readWord := func(path string) uint64 {
			t.Helper()
			b, err := os.ReadFile(path)
			if err != nil || len(b) != binary.Size(uint64(0)) {
				t.Fatalf("word file %s: err=%v len=%d, want a uint64 encoding", path, err, len(b))
			}
			return binary.LittleEndian.Uint64(b)
		}
		usedPath := filepath.Join(dir, id, learnedUsedFile)
		clockPath := filepath.Join(dir, learnedClockFile)
		clock := readWord(clockPath)
		if used := readWord(usedPath); clock == 0 || used != clock {
			t.Fatalf("clock=%d used=%d, want matching nonzero open stamps", clock, used)
		}

		// Cross a byte boundary so the reopen witness also detects wrong endianness.
		const previousClock uint64 = 0xff
		if err := os.WriteFile(clockPath, binary.LittleEndian.AppendUint64(nil, previousClock), 0o600); err != nil {
			t.Fatal(err)
		}
		openResolved(t, dir, spec)
		for _, path := range []string{clockPath, usedPath} {
			if got := readWord(path); got != previousClock+1 {
				t.Fatalf("reopened word %s=%d, want %d", path, got, previousClock+1)
			}
		}
	})

	t.Run("pending write defers a second persist and reconciles on open", func(t *testing.T) {
		dir := t.TempDir()
		p := newPlanner(t, dir, stdCfg(dir))
		spec := derive(t, p, "tenant-a", "agent-1", in)
		warm(t, p, spec)
		id := learnedPrefixIdentity(spec.Scope)

		// While a valid marker claims the slot, a second in-flight writer defers.
		s := openResolved(t, dir, spec)
		k := strings.Repeat("ab", 32)
		if err := s.beginWrite(k); err != nil {
			t.Fatalf("beginWrite: %v", err)
		}
		if err := s.beginWrite(strings.Repeat("cd", 32)); !errors.Is(err, errLearnedPending) {
			t.Fatalf("second beginWrite err=%v, want errLearnedPending", err)
		}

		// Crash recovery: reopening removes the stale marker (no matching record)
		// and leaves no phantom index entry.
		openResolved(t, dir, spec)
		if _, err := os.Stat(learnedPendingMarkerPath(dir, id)); !os.IsNotExist(err) {
			t.Fatalf("stale pending marker survived reopen: %v", err)
		}
		for _, st := range readIndex(t, dir, id).States {
			if st.Key == k {
				t.Fatalf("phantom index entry for pending key %s", k)
			}
		}

		// A marker whose record DID land is adopted into the index instead.
		k3 := strings.Repeat("ef", 32)
		rec, err := l3kv.NewDiskStore(filepath.Join(dir, id))
		if err != nil {
			t.Fatal(err)
		}
		if err := rec.Put(ctx, k3, []byte("adopted payload")); err != nil {
			t.Fatal(err)
		}
		// Plant the marker from raw bytes so the reconciliation path is witnessed
		// against the on-disk format itself.
		if err := os.WriteFile(learnedPendingMarkerPath(dir, id), buildPendingMarker(t, k3), 0o600); err != nil {
			t.Fatal(err)
		}
		openResolved(t, dir, spec)
		if _, err := os.Stat(learnedPendingMarkerPath(dir, id)); !os.IsNotExist(err) {
			t.Fatalf("pending marker for a landed record survived reopen: %v", err)
		}
		adopted := false
		for _, st := range readIndex(t, dir, id).States {
			if st.Key == k3 {
				adopted = true
				if st.Bytes != recordSize(t, dir, id, k3) {
					t.Fatalf("adopted bytes=%d, want %d", st.Bytes, recordSize(t, dir, id, k3))
				}
			}
		}
		if !adopted {
			t.Fatal("landed pending-marked record was not adopted into the index")
		}
	})

	t.Run("corrupt index rebuilds from record files", func(t *testing.T) {
		dir := t.TempDir()
		p := newPlanner(t, dir, stdCfg(dir))
		spec := derive(t, p, "tenant-a", "agent-1", in)
		warm(t, p, spec)
		id := learnedPrefixIdentity(spec.Scope)
		key := warmDiskKey(spec, modelDigest, p.warmDiskExecutionIdentity())
		if err := os.WriteFile(filepath.Join(dir, id, learnedIndexFile), []byte("{not-json"), 0o600); err != nil {
			t.Fatal(err)
		}
		fresh := newPlanner(t, dir, stdCfg(dir))
		rf := warm(t, fresh, derive(t, fresh, "tenant-a", "agent-1", in))
		if rf.Disk == nil || rf.Disk.RestoreOutcome != WarmDiskOutcomeRestored {
			t.Fatalf("corrupt-index restore outcome=%s reason=%s, want restored", warmDiskRestoreOutcome(rf.Disk), warmDiskRestoreReason(rf.Disk))
		}
		idx := readIndex(t, dir, id)
		if idx.Version != learnedIndexVersion || len(idx.States) != 1 || idx.States[0].Key != key {
			t.Fatalf("rebuilt index=%+v, want one state %s", idx, key)
		}
	})

	t.Run("unindexed record files are removed on open", func(t *testing.T) {
		dir := t.TempDir()
		p := newPlanner(t, dir, stdCfg(dir))
		spec := derive(t, p, "tenant-a", "agent-1", in)
		warm(t, p, spec)
		id := learnedPrefixIdentity(spec.Scope)
		key := warmDiskKey(spec, modelDigest, p.warmDiskExecutionIdentity())
		stray := strings.Repeat("cd", 32)
		if stray == key {
			t.Fatal("fixture collision: stray key equals the persisted key")
		}
		rec, err := l3kv.NewDiskStore(filepath.Join(dir, id))
		if err != nil {
			t.Fatal(err)
		}
		if err := rec.Put(ctx, stray, []byte("stray payload")); err != nil {
			t.Fatal(err)
		}
		openResolved(t, dir, spec)
		if _, err := os.Stat(filepath.Join(dir, id, stray)); !os.IsNotExist(err) {
			t.Fatalf("unindexed record %s survived open: %v", stray, err)
		}
		if _, err := os.Stat(filepath.Join(dir, id, key)); err != nil {
			t.Fatalf("indexed record %s lost on open: %v", key, err)
		}
	})

	t.Run("LRU eviction keeps newest within entry cap", func(t *testing.T) {
		dir := t.TempDir()
		p := newPlanner(t, dir, WarmDiskConfig{Dir: dir, MaxEntries: 2, MaxBytes: 8 << 20})
		var keys []string
		var lastSpec WarmPrefixSpec
		for i, instr := range []string{"First stable startup prefix.", "Second stable startup prefix.", "Third stable startup prefix."} {
			changed := in
			changed.Instructions = []byte(instr)
			spec := derive(t, p, "tenant-a", "agent-1", changed)
			r := warm(t, p, spec)
			if r.Disk == nil || r.Disk.Outcome != WarmDiskPersisted {
				t.Fatalf("persist %d outcome=%s reason=%s, want persisted", i, warmDiskOutcome(r.Disk), warmDiskReason(r.Disk))
			}
			keys = append(keys, warmDiskKey(spec, modelDigest, p.warmDiskExecutionIdentity()))
			lastSpec = spec
		}
		id := learnedPrefixIdentity(lastSpec.Scope)
		entries, err := os.ReadDir(filepath.Join(dir, id))
		if err != nil {
			t.Fatal(err)
		}
		records := map[string]bool{}
		for _, e := range entries {
			if !e.IsDir() && isLearnedRecordKey(e.Name()) {
				records[e.Name()] = true
			}
		}
		if len(records) != 2 {
			t.Fatalf("identity holds %d records, want 2", len(records))
		}
		if records[keys[0]] {
			t.Fatalf("oldest record %s survived the entry cap", keys[0])
		}
		if !records[keys[1]] || !records[keys[2]] {
			t.Fatalf("records present=%v, want the newest two %s and %s", records, keys[1], keys[2])
		}
		if got := len(readIndex(t, dir, id).States); got != 2 {
			t.Fatalf("index lists %d states, want 2", got)
		}
	})

	t.Run("cross-identity sweep reclaims oldest identity first", func(t *testing.T) {
		// Measure the fixture record size so the shared cap fits exactly two.
		scratch := t.TempDir()
		ps := newPlanner(t, scratch, stdCfg(scratch))
		sspec := derive(t, ps, "tenant-m", "agent-1", in)
		sr := warm(t, ps, sspec)
		if sr.Disk == nil || sr.Disk.Outcome != WarmDiskPersisted {
			t.Fatalf("measure persist outcome=%s reason=%s, want persisted", warmDiskOutcome(sr.Disk), warmDiskReason(sr.Disk))
		}
		rec := recordSize(t, scratch, learnedPrefixIdentity(sspec.Scope), warmDiskKey(sspec, modelDigest, ps.warmDiskExecutionIdentity()))
		capBytes := 2*rec + rec/2

		dir := t.TempDir()
		a := newPlanner(t, dir, stdCfg(dir))
		specA := derive(t, a, "tenant-a", "agent-1", in)
		if r := warm(t, a, specA); r.Disk == nil || r.Disk.Outcome != WarmDiskPersisted {
			t.Fatalf("tenant A outcome=%s reason=%s, want persisted", warmDiskOutcome(r.Disk), warmDiskReason(r.Disk))
		}
		keyA := warmDiskKey(specA, modelDigest, a.warmDiskExecutionIdentity())
		idA := learnedPrefixIdentity(specA.Scope)
		// Age A's open stamp so the sweep sees it as the least-recently-opened.
		if err := writeWordFileAtomic(filepath.Join(dir, idA, learnedUsedFile), 1); err != nil {
			t.Fatal(err)
		}
		b := newPlanner(t, dir, stdCfg(dir))
		specB1 := derive(t, b, "tenant-b", "agent-1", in)
		if r := warm(t, b, specB1); r.Disk == nil || r.Disk.Outcome != WarmDiskPersisted {
			t.Fatalf("tenant B outcome=%s reason=%s, want persisted", warmDiskOutcome(r.Disk), warmDiskReason(r.Disk))
		}
		keyB1 := warmDiskKey(specB1, modelDigest, b.warmDiskExecutionIdentity())
		idB := learnedPrefixIdentity(specB1.Scope)
		got, ok := readWordFile(filepath.Join(dir, idB, learnedUsedFile))
		if !ok || got <= 1 {
			t.Fatalf("tenant B used word=%d ok=%v, want a valid word newer than tenant A's forced 1", got, ok)
		}
		// The cross-identity sweep reclaims the least-recently-opened identity
		// first, leaving the newer identity intact.
		if err := learnedRootSweep(ctx, dir, "", "", rec, capBytes); err != nil {
			t.Fatalf("learnedRootSweep: %v", err)
		}
		if _, err := os.Stat(filepath.Join(dir, idA, keyA)); !os.IsNotExist(err) {
			t.Fatalf("oldest identity record %s was not swept first: %v", keyA, err)
		}
		if _, err := os.Stat(filepath.Join(dir, idB, keyB1)); err != nil {
			t.Fatalf("newer identity record %s missing after sweep: %v", keyB1, err)
		}
		// Once the older identity is empty, the sweep continues into the newest.
		if err := learnedRootSweep(ctx, dir, "", "", rec, rec); err != nil {
			t.Fatalf("second learnedRootSweep: %v", err)
		}
		if _, err := os.Stat(filepath.Join(dir, idB, keyB1)); !os.IsNotExist(err) {
			t.Fatalf("record %s survived the second sweep: %v", keyB1, err)
		}

		// keepKey inside keepIdentity is never evicted: with only keepKey's record
		// left under full pressure, the sweep must refuse rather than evict it.
		if r := warm(t, b, specB1); r.Disk == nil || r.Disk.Outcome != WarmDiskOutcomePersisted {
			t.Fatalf("re-persist B outcome=%s reason=%s, want persisted", warmDiskOutcome(r.Disk), warmDiskReason(r.Disk))
		}
		if err := learnedRootSweep(ctx, dir, idB, keyB1, rec, 0); !errors.Is(err, errLearnedSweepExhausted) {
			t.Fatalf("keepKey sweep err=%v, want errLearnedSweepExhausted", err)
		}
		if _, err := os.Stat(filepath.Join(dir, idB, keyB1)); err != nil {
			t.Fatalf("keepKey record %s evicted despite protection: %v", keyB1, err)
		}
	})

	t.Run("legacy flat records remain readable and are reclaimed first", func(t *testing.T) {
		dir := t.TempDir()
		p := newPlanner(t, dir, stdCfg(dir))
		spec := derive(t, p, "tenant-a", "agent-1", in)
		r := warm(t, p, spec)
		if r.Disk == nil || r.Disk.Outcome != WarmDiskPersisted {
			t.Fatalf("outcome=%s reason=%s, want persisted", warmDiskOutcome(r.Disk), warmDiskReason(r.Disk))
		}
		id := learnedPrefixIdentity(spec.Scope)
		key := warmDiskKey(spec, modelDigest, p.warmDiskExecutionIdentity())
		rec := recordSize(t, dir, id, key)
		// Move the record to the pre-upgrade flat location directly under the root.
		data, err := os.ReadFile(filepath.Join(dir, id, key))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(filepath.Join(dir, id, key)); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, key), data, 0o600); err != nil {
			t.Fatal(err)
		}

		// A fresh planner restores through the legacy fallback path.
		fresh := newPlanner(t, dir, stdCfg(dir))
		rf := warm(t, fresh, derive(t, fresh, "tenant-a", "agent-1", in))
		if rf.Disk == nil || rf.Disk.RestoreOutcome != WarmDiskOutcomeRestored || rf.Disk.ReadBytes <= 0 {
			t.Fatalf("legacy fallback restore outcome=%s reason=%s read=%d, want restored positive read", warmDiskRestoreOutcome(rf.Disk), warmDiskRestoreReason(rf.Disk), warmDiskReadBytes(rf.Disk))
		}

		// Under cap pressure the sweep counts the legacy record and reclaims it.
		if err := learnedRootSweep(ctx, dir, "", "", rec, rec+rec/2); err != nil {
			t.Fatalf("learnedRootSweep: %v", err)
		}
		if _, err := os.Stat(filepath.Join(dir, key)); !os.IsNotExist(err) {
			t.Fatalf("legacy record %s survived the sweep: %v", key, err)
		}
		// The store stays usable for fresh persists after the sweep.
		second := in
		second.Instructions = []byte("Second stable startup prefix.")
		p2 := newPlanner(t, dir, stdCfg(dir))
		spec2 := derive(t, p2, "tenant-a", "agent-1", second)
		r2 := warm(t, p2, spec2)
		if r2.Disk == nil || r2.Disk.Outcome != WarmDiskPersisted {
			t.Fatalf("post-sweep persist outcome=%s reason=%s, want persisted", warmDiskOutcome(r2.Disk), warmDiskReason(r2.Disk))
		}
		key2 := warmDiskKey(spec2, modelDigest, p2.warmDiskExecutionIdentity())
		if _, err := os.Stat(filepath.Join(dir, id, key2)); err != nil {
			t.Fatalf("new record %s missing after sweep: %v", key2, err)
		}
	})

	t.Run("cross-instance commits merge indexes", func(t *testing.T) {
		dir := t.TempDir()
		p := newPlanner(t, dir, stdCfg(dir))
		spec := derive(t, p, "tenant-a", "agent-1", in)
		r := warm(t, p, spec)
		if r.Disk == nil || r.Disk.Outcome != WarmDiskPersisted {
			t.Fatalf("persist outcome=%s reason=%s, want persisted", warmDiskOutcome(r.Disk), warmDiskReason(r.Disk))
		}
		id := learnedPrefixIdentity(spec.Scope)
		resolved, err := resolveWarmDiskConfig(stdCfg(dir))
		if err != nil {
			t.Fatalf("resolveWarmDiskConfig: %v", err)
		}
		// Two instances over the same identity dir, both opened BEFORE either
		// commits: each holds a stale index view, so the second commit must
		// merge with the on-disk index instead of clobbering the first entry.
		s1, err := openLearnedPrefixStore(ctx, resolved, spec.Scope)
		if err != nil {
			t.Fatalf("openLearnedPrefixStore #1: %v", err)
		}
		s2, err := openLearnedPrefixStore(ctx, resolved, spec.Scope)
		if err != nil {
			t.Fatalf("openLearnedPrefixStore #2: %v", err)
		}
		commitThrough := func(s *learnedPrefixStore, key, body string) {
			t.Helper()
			if err := s.beginWrite(key); err != nil {
				t.Fatalf("beginWrite %s: %v", key, err)
			}
			if err := s.Put(ctx, key, []byte(body)); err != nil {
				t.Fatalf("Put %s: %v", key, err)
			}
			if err := s.commitWrite(ctx, key, int64(len(body))); err != nil {
				t.Fatalf("commitWrite %s: %v", key, err)
			}
		}
		key1, key2 := strings.Repeat("bb", 16), strings.Repeat("cc", 16)
		commitThrough(s1, key1, "record-one")
		commitThrough(s2, key2, "record-two")
		idx := readIndex(t, dir, id)
		if len(idx.States) != 3 {
			t.Fatalf("index states=%d, want 3 (planner persist + two cross-instance commits)", len(idx.States))
		}
		for _, key := range []string{key1, key2} {
			found := false
			for _, st := range idx.States {
				if st.Key == key {
					found = true
				}
			}
			if !found {
				t.Fatalf("key %s missing from merged index", key)
			}
		}
		// A fresh open must keep both records indexed, not reap them as strays.
		s3, err := openLearnedPrefixStore(ctx, resolved, spec.Scope)
		if err != nil {
			t.Fatalf("reopen: %v", err)
		}
		for i, key := range []string{key1, key2} {
			payload, ok, err := s3.lookupRecord(ctx, key, 1<<20)
			if err != nil || !ok || string(payload) != []string{"record-one", "record-two"}[i] {
				t.Fatalf("lookup %s: ok=%v err=%v, want the staged body", key, ok, err)
			}
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
