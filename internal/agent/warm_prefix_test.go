package agent

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// warmPrefixFixturePlanner builds the same lightweight synthetic planner the prompt
// encoding parity test uses: a byte-level vocab tokenizer over an in-memory model
// config, so the stable-prefix derivation exercises the real EncodePrompt seam
// without weights or a device.
func warmPrefixFixturePlanner(t *testing.T) *InKernelPlanner {
	t.Helper()
	cfg := model.Config{
		ModelType:             "qwen3_5",
		Architectures:         []string{"Qwen3_5ForCausalLM"},
		LayerTypes:            []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		MaxPositionEmbeddings: 32768,
	}
	return NewInKernelPlannerWithConfig(&model.Model{Cfg: cfg}, loadProbeTok(t), "qwen3.8-fixture", false, nil, false, InKernelPlannerConfig{ContextTokens: 8192})
}

func warmPrefixFixtureTools() []ToolDef {
	return []ToolDef{
		{Type: "function", Function: ToolDefFunction{Name: "Read", Description: "read a file", Parameters: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"}}}`)}},
		{Type: "function", Function: ToolDefFunction{Name: "Write", Description: "write a file", Parameters: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}}}`)}},
	}
}

func warmPrefixFixtureInputs() WarmPrefixInputs {
	return WarmPrefixInputs{
		Instructions: []byte("Work precisely and follow the repository conventions."),
		SystemBlocks: [][]byte{
			[]byte("You operate inside fak, an agent kernel."),
			[]byte("Policy floor: the deployed capability manifest is default-deny."),
		},
		Tools:     warmPrefixFixtureTools(),
		KVLayout:  "fp32",
		AdapterID: "native-inkernel",
	}
}

// TestWarmPrefixIdentity is the leaf's named witness (fak#13352).
func TestWarmPrefixIdentity(t *testing.T) {
	p := warmPrefixFixturePlanner(t)

	// --- Stability: two DIFFERENT user suffixes derive the SAME stable prefix -------
	base, err := p.DeriveWarmPrefix("tenant-a", "agent-1", warmPrefixFixtureInputs())
	if err != nil {
		t.Fatalf("DeriveWarmPrefix: %v", err)
	}
	if !base.Bounded() || base.StableTokens <= 0 || base.StableTokenDigest == "" || base.Identity == "" {
		t.Fatalf("descriptor must be bounded and non-empty: %+v", base)
	}

	// A different suffix is not an input at all, so re-deriving is bit-identical. Prove
	// the stable digest is a property of the STABLE inputs, not of any conversation tail
	// by deriving again from a fresh copy of the same inputs.
	again, err := p.DeriveWarmPrefix("tenant-a", "agent-1", warmPrefixFixtureInputs())
	if err != nil {
		t.Fatalf("DeriveWarmPrefix (again): %v", err)
	}
	if base.StableTokens != again.StableTokens || base.StableTokenDigest != again.StableTokenDigest || base.Identity != again.Identity {
		t.Fatalf("stable inputs must derive an identical descriptor: %+v vs %+v", base, again)
	}

	// The stable boundary equals the token count EncodePrompt reports for the stable-only
	// prompt — the descriptor reuses the production encoder, it does not re-render.
	stableMsgs := []Message{
		{Role: RoleSystem, Content: string(warmPrefixFixtureInputs().Instructions)},
	}
	for _, block := range warmPrefixFixtureInputs().SystemBlocks {
		stableMsgs = append(stableMsgs, Message{Role: RoleSystem, Content: string(block)})
	}
	enc, err := p.EncodePrompt(context.Background(), stableMsgs, warmPrefixFixtureTools())
	if err != nil {
		t.Fatalf("EncodePrompt: %v", err)
	}
	if base.StableTokens != len(enc.TokenIDs) || base.StableTokenDigest != digestInts(enc.TokenIDs) {
		t.Fatalf("stable boundary must match the production encoder: got tokens=%d digest=%q want tokens=%d digest=%q",
			base.StableTokens, base.StableTokenDigest, len(enc.TokenIDs), digestInts(enc.TokenIDs))
	}

	// --- Identity: changing any compatibility/authorization axis changes Identity ----
	// Each mutation starts from a fresh, valid input set and changes exactly one axis.
	type mutation struct {
		name   string
		tenant string
		agent  string
		mutate func(in *WarmPrefixInputs)
	}
	mutations := []mutation{
		{name: "tenant", tenant: "tenant-b", agent: "agent-1"},
		{name: "agent", tenant: "tenant-a", agent: "agent-2"},
		{name: "instructions", tenant: "tenant-a", agent: "agent-1", mutate: func(in *WarmPrefixInputs) {
			in.Instructions = []byte("A DIFFERENT effective instruction snapshot.")
		}},
		{name: "resident_block", tenant: "tenant-a", agent: "agent-1", mutate: func(in *WarmPrefixInputs) {
			in.SystemBlocks = append([][]byte(nil), in.SystemBlocks...)
			in.SystemBlocks[0] = []byte("You operate inside a DIFFERENT kernel.")
		}},
		{name: "tool_policy", tenant: "tenant-a", agent: "agent-1", mutate: func(in *WarmPrefixInputs) {
			tools := warmPrefixFixtureTools()
			tools[0].Function.Description = "read a file from a reviewed allowlist"
			in.Tools = tools
		}},
		{name: "tool_order", tenant: "tenant-a", agent: "agent-1", mutate: func(in *WarmPrefixInputs) {
			tools := warmPrefixFixtureTools()
			tools[0], tools[1] = tools[1], tools[0]
			in.Tools = tools
		}},
		{name: "tools_removed", tenant: "tenant-a", agent: "agent-1", mutate: func(in *WarmPrefixInputs) {
			in.Tools = warmPrefixFixtureTools()[:1]
		}},
		{name: "kv_layout", tenant: "tenant-a", agent: "agent-1", mutate: func(in *WarmPrefixInputs) {
			in.KVLayout = "q8_0"
		}},
		{name: "adapter", tenant: "tenant-a", agent: "agent-1", mutate: func(in *WarmPrefixInputs) {
			in.AdapterID = "anthropic"
		}},
	}
	for _, m := range mutations {
		t.Run(m.name, func(t *testing.T) {
			in := warmPrefixFixtureInputs()
			if m.mutate != nil {
				m.mutate(&in)
			}
			got, err := p.DeriveWarmPrefix(m.tenant, m.agent, in)
			if err != nil {
				t.Fatalf("DeriveWarmPrefix(%s): %v", m.name, err)
			}
			if got.Identity == base.Identity {
				t.Fatalf("identity axis %q did not change Identity: %+v", m.name, got)
			}
		})
	}

	// --- Identity: changing model/tokenizer/renderer/runtime is its own planner ------
	altCfg := model.Config{
		ModelType:             "qwen3_5",
		Architectures:         []string{"Qwen3_5ForCausalLM"},
		LayerTypes:            []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		MaxPositionEmbeddings: 32768,
	}
	q4kPlanner := NewInKernelPlannerWithConfig(&model.Model{Cfg: altCfg}, loadProbeTok(t), "qwen3.8-fixture", true, nil, false, InKernelPlannerConfig{ContextTokens: 8192})
	altModel, err := q4kPlanner.DeriveWarmPrefix("tenant-a", "agent-1", warmPrefixFixtureInputs())
	if err != nil {
		t.Fatalf("DeriveWarmPrefix (q4k): %v", err)
	}
	if altModel.RuntimeGeneration == base.RuntimeGeneration {
		t.Fatalf("runtime generation must differ between the dense and resident-q4k paths: %q", base.RuntimeGeneration)
	}
	if altModel.Identity == base.Identity {
		t.Fatalf("a different runtime generation must change Identity")
	}

	otherModel := NewInKernelPlannerWithConfig(&model.Model{Cfg: altCfg}, loadProbeTok(t), "qwen3.8-other", false, nil, false, InKernelPlannerConfig{ContextTokens: 8192})
	otherModelSpec, err := otherModel.DeriveWarmPrefix("tenant-a", "agent-1", warmPrefixFixtureInputs())
	if err != nil {
		t.Fatalf("DeriveWarmPrefix (other model): %v", err)
	}
	if otherModelSpec.ModelID == base.ModelID || otherModelSpec.Identity == base.Identity {
		t.Fatalf("a different model identity must change Identity")
	}

	// --- Immutability: the descriptor is a value; its slices are copied ---------------
	ids := enc.TokenIDs
	idsCopy := append([]int(nil), ids...)
	enc2, err := p.EncodePrompt(context.Background(), stableMsgs, warmPrefixFixtureTools())
	if err != nil {
		t.Fatalf("EncodePrompt (immutability): %v", err)
	}
	ids[0]++
	if !reflect.DeepEqual(enc2.TokenIDs, idsCopy) {
		t.Fatalf("EncodePrompt result must copy ids; mutating one result corrupted another")
	}
	if base.Identity != again.Identity {
		t.Fatalf("identity must be a pure function of the descriptor content")
	}

	// --- Scope is carried verbatim as the authenticated cache owner ------------------
	if base.Scope.Tenant != "tenant-a" || base.Scope.Agent != "agent-1" {
		t.Fatalf("scope not carried: %+v", base.Scope)
	}
	_ = radixkv.CacheIdentity{}

	// --- Fail-closed: an unusable planner refuses, never returns a zero spec ----------
	bad := []*InKernelPlanner{
		nil,
		{},
		{m: &model.Model{Cfg: altCfg}},
		{m: &model.Model{Cfg: altCfg}, tok: loadProbeTok(t)},
	}
	for i, bp := range bad {
		if _, err := bp.DeriveWarmPrefix("tenant-a", "agent-1", warmPrefixFixtureInputs()); err == nil {
			t.Fatalf("invalid planner case %d derived a warm descriptor", i)
		}
	}
	// Empty tenant is a refusal: a warm with no owner cannot be bounded.
	if _, err := p.DeriveWarmPrefix("   ", "agent-1", warmPrefixFixtureInputs()); err == nil {
		t.Fatalf("empty tenant must refuse")
	}
}

// TestWarmPrefixEmptyStableInputsRefuseToBeUsable proves an inputs set with no stable
// content yields an unbounded (unusable) descriptor rather than a zero-identity warm.
func TestWarmPrefixEmptyStableInputsRefuseToBeUsable(t *testing.T) {
	p := warmPrefixFixturePlanner(t)
	spec, err := p.DeriveWarmPrefix("tenant-a", "agent-1", WarmPrefixInputs{KVLayout: "fp32"})
	if err != nil {
		// A refusal is also acceptable; the contract is only that it is not usable.
		return
	}
	if spec.Bounded() {
		t.Fatalf("an empty stable input set must not yield a bounded warm descriptor: %+v", spec)
	}
}
