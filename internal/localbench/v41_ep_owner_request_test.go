package localbench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func v41EPOwnerRequestFixture() V41EPOwnerRequest {
	root, err := filepath.Abs("models")
	if err != nil {
		panic(err)
	}
	req := V41EPOwnerRequest{
		Schema: V41EPOwnerRequestSchema, RunID: "bounded-test", WorldSize: 3,
		Shards: []ggufload.ShardIdentity{
			{Path: filepath.Join(root, "checkpoint-00001-of-00002.gguf"), Size: 64, RawFileSHA256: strings.Repeat("a", 64)},
			{Path: filepath.Join(root, "checkpoint-00002-of-00002.gguf"), Size: 128, RawFileSHA256: strings.Repeat("b", 64)},
		},
		SourceSHA256: strings.Repeat("c", 64), CoordinatorAddress: "127.0.0.1:19341",
		Owners: []V41EPOwnerSelection{
			{Rank: 0, Backend: "vulkan", ExpectedDevice: "device 0"},
			{Rank: 1, Backend: "vulkan", ExpectedDevice: "device 1"},
			{Rank: 2, Backend: "vulkan", ExpectedDevice: "device 2"},
		},
		PromptTokenIDs: []int{0, 1}, ContextLimit: 4, GeneratedTokenLimit: 2, Repetitions: 1,
		Budgets:    V41EPOwnerBudgets{JoinNS: time.Second, LoadNS: 5 * time.Second, ForwardNS: 5 * time.Second, CleanupNS: time.Second, TotalNS: 20 * time.Second},
		Supervisor: V41EPOwnerSupervisorContract{Kind: "external-process-group", ContractID: "external-test-contract", TerminateNS: 20 * time.Second, KillGraceNS: time.Second},
	}
	v41EPOwnerPinFixtureManifest(&req)
	return req
}

func v41EPOwnerPinFixtureManifest(req *V41EPOwnerRequest) {
	// Synthetic identities only; this helper never opens or hashes model files.
	raw, _ := json.Marshal(req.Shards)
	digest := sha256.Sum256(raw)
	req.ExpectedShardSetSHA256 = hex.EncodeToString(digest[:])
}

// fak-test:runtime fast est=20ms
func TestV41EPOwnerRequestContract(t *testing.T) {
	t.Parallel()
	req := v41EPOwnerRequestFixture()
	raw, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeV41EPOwnerRequest(strings.NewReader(string(raw)))
	if err != nil || !reflect.DeepEqual(decoded, req) {
		t.Fatalf("valid request changed: err=%v got=%+v", err, decoded)
	}
	digest, err := V41EPOwnerRequestDigest(req)
	want := sha256.Sum256(raw)
	if err != nil || digest != hex.EncodeToString(want[:]) {
		t.Fatalf("request agreement digest=%s err=%v", digest, err)
	}

	for name, mutate := range map[string]func(*V41EPOwnerRequest){
		"schema":            func(r *V41EPOwnerRequest) { r.Schema = "other" },
		"run ID absent":     func(r *V41EPOwnerRequest) { r.RunID = "" },
		"run ID cap":        func(r *V41EPOwnerRequest) { r.RunID = strings.Repeat("x", V41EPOwnerMaxRunIDBytes+1) },
		"missing rank":      func(r *V41EPOwnerRequest) { r.Owners = r.Owners[:2] },
		"duplicate rank":    func(r *V41EPOwnerRequest) { r.Owners[2].Rank = 1 },
		"rank order":        func(r *V41EPOwnerRequest) { r.Owners[0], r.Owners[1] = r.Owners[1], r.Owners[0] },
		"world size":        func(r *V41EPOwnerRequest) { r.WorldSize = 4 },
		"repetition":        func(r *V41EPOwnerRequest) { r.Repetitions = 2 },
		"empty backend":     func(r *V41EPOwnerRequest) { r.Owners[0].Backend = "" },
		"automatic backend": func(r *V41EPOwnerRequest) { r.Owners[0].Backend = "auto" },
		"reference backend": func(r *V41EPOwnerRequest) { r.Owners[0].Backend = "cpu-ref" },
		"backend alias":     func(r *V41EPOwnerRequest) { r.Owners[0].Backend = "Vulkan" },
		"backend cap":       func(r *V41EPOwnerRequest) { r.Owners[0].Backend = strings.Repeat("x", V41EPOwnerMaxBackendBytes+1) },
		"missing device":    func(r *V41EPOwnerRequest) { r.Owners[0].ExpectedDevice = "" },
		"device cap": func(r *V41EPOwnerRequest) {
			r.Owners[0].ExpectedDevice = strings.Repeat("x", V41EPOwnerMaxDeviceBytes+1)
		},
		"binary git SHA":            func(r *V41EPOwnerRequest) { r.Owners[0].ExpectedExecutableSHA256 = strings.Repeat("d", 40) },
		"source git SHA":            func(r *V41EPOwnerRequest) { r.SourceSHA256 = strings.Repeat("d", 40) },
		"source uppercase":          func(r *V41EPOwnerRequest) { r.SourceSHA256 = strings.Repeat("D", 64) },
		"source nonhex":             func(r *V41EPOwnerRequest) { r.SourceSHA256 = strings.Repeat("z", 64) },
		"address missing host":      func(r *V41EPOwnerRequest) { r.CoordinatorAddress = ":19341" },
		"address implicit port":     func(r *V41EPOwnerRequest) { r.CoordinatorAddress = "127.0.0.1:0" },
		"address remapped port":     func(r *V41EPOwnerRequest) { r.CoordinatorAddress = "127.0.0.1:019341" },
		"address cap":               func(r *V41EPOwnerRequest) { r.CoordinatorAddress = strings.Repeat("x", V41EPOwnerMaxAddressBytes+1) },
		"shards absent":             func(r *V41EPOwnerRequest) { r.Shards = nil },
		"shard count cap":           func(r *V41EPOwnerRequest) { r.Shards = make([]ggufload.ShardIdentity, V41EPOwnerMaxShards+1) },
		"relative path":             func(r *V41EPOwnerRequest) { r.Shards[0].Path = "model.gguf" },
		"unclean path":              func(r *V41EPOwnerRequest) { r.Shards[0].Path = "/models/../checkpoint.gguf" },
		"path cap":                  func(r *V41EPOwnerRequest) { r.Shards[0].Path = "/" + strings.Repeat("x", V41EPOwnerMaxPathBytes) },
		"invalid UTF8 path":         func(r *V41EPOwnerRequest) { r.Shards[0].Path = "/models/\xff" },
		"duplicate path":            func(r *V41EPOwnerRequest) { r.Shards[1].Path = r.Shards[0].Path },
		"path remapping":            func(r *V41EPOwnerRequest) { r.Shards[0].Path = "/other/checkpoint-00001-of-00002.gguf" },
		"shard order":               func(r *V41EPOwnerRequest) { r.Shards[0], r.Shards[1] = r.Shards[1], r.Shards[0] },
		"shard size":                func(r *V41EPOwnerRequest) { r.Shards[0].Size = 0 },
		"shard size overflow":       func(r *V41EPOwnerRequest) { r.Shards[0].Size = int64(^uint64(0) >> 1) },
		"file digest":               func(r *V41EPOwnerRequest) { r.Shards[0].RawFileSHA256 = "bad" },
		"set digest":                func(r *V41EPOwnerRequest) { r.ExpectedShardSetSHA256 = strings.Repeat("0", 64) },
		"empty prompt":              func(r *V41EPOwnerRequest) { r.PromptTokenIDs = nil },
		"prompt cap":                func(r *V41EPOwnerRequest) { r.PromptTokenIDs = make([]int, V41EPOwnerMaxPromptTokens+1) },
		"negative token":            func(r *V41EPOwnerRequest) { r.PromptTokenIDs[0] = -1 },
		"token wire overflow":       func(r *V41EPOwnerRequest) { id := int64(1) << 31; r.PromptTokenIDs[0] = int(id) },
		"zero generated":            func(r *V41EPOwnerRequest) { r.GeneratedTokenLimit = 0 },
		"generated cap":             func(r *V41EPOwnerRequest) { r.GeneratedTokenLimit = V41EPOwnerMaxGeneratedTokens + 1 },
		"context cap":               func(r *V41EPOwnerRequest) { r.ContextLimit = V41EPOwnerMaxContextTokens + 1 },
		"context sum overflow":      func(r *V41EPOwnerRequest) { r.GeneratedTokenLimit = int(^uint(0) >> 1) },
		"context too small":         func(r *V41EPOwnerRequest) { r.ContextLimit = 3 },
		"zero join":                 func(r *V41EPOwnerRequest) { r.Budgets.JoinNS = 0 },
		"join cap":                  func(r *V41EPOwnerRequest) { r.Budgets.JoinNS = V41EPOwnerMaxJoinTime + 1 },
		"load cap":                  func(r *V41EPOwnerRequest) { r.Budgets.LoadNS = V41EPOwnerMaxLoadTime + 1 },
		"forward cap":               func(r *V41EPOwnerRequest) { r.Budgets.ForwardNS = V41EPOwnerMaxForwardTime + 1 },
		"cleanup cap":               func(r *V41EPOwnerRequest) { r.Budgets.CleanupNS = V41EPOwnerMaxCleanupTime + 1 },
		"total cap":                 func(r *V41EPOwnerRequest) { r.Budgets.TotalNS = V41EPOwnerMaxTotalTime + 1 },
		"duration sum overflow":     func(r *V41EPOwnerRequest) { r.Budgets.LoadNS = time.Duration(1<<63 - 1) },
		"phase sum exceeds total":   func(r *V41EPOwnerRequest) { r.Budgets.TotalNS = 10 * time.Second },
		"supervisor absent":         func(r *V41EPOwnerRequest) { r.Supervisor.Kind = "" },
		"self declared supervision": func(r *V41EPOwnerRequest) { r.Supervisor.Kind = "verified" },
		"supervisor too short":      func(r *V41EPOwnerRequest) { r.Supervisor.TerminateNS = time.Second },
		"supervisor unbounded":      func(r *V41EPOwnerRequest) { r.Supervisor.TerminateNS = time.Duration(1<<63 - 1) },
		"kill grace cap":            func(r *V41EPOwnerRequest) { r.Supervisor.KillGraceNS = V41EPOwnerMaxKillGrace + 1 },
	} {
		t.Run(name, func(t *testing.T) {
			r := v41EPOwnerRequestFixture()
			mutate(&r)
			if err := r.Validate(); err == nil {
				t.Fatal("invalid request accepted")
			}
			if digest, err := V41EPOwnerRequestDigest(r); err == nil || digest != "" {
				t.Fatalf("invalid request gained agreement digest: %s %v", digest, err)
			}
		})
	}

	for name, malformed := range map[string]string{
		"duplicate key":       strings.Replace(string(raw), `"run_id":`, `"run_id":"shadow","run_id":`, 1),
		"escaped duplicate":   strings.Replace(string(raw), `"run_id":`, `"\u0072un_id":"shadow","run_id":`, 1),
		"nested duplicate":    strings.Replace(string(raw), `"rank":0`, `"rank":0,"rank":0`, 1),
		"null prompt token":   strings.Replace(string(raw), `"prompt_token_ids":[0,1]`, `"prompt_token_ids":[null,1]`, 1),
		"null rank zero":      strings.Replace(string(raw), `"rank":0`, `"rank":null`, 1),
		"missing rank zero":   strings.Replace(string(raw), `"rank":0,`, ``, 1),
		"null EOS policy":     strings.Replace(string(raw), `"ignore_eos":false`, `"ignore_eos":null`, 1),
		"missing EOS policy":  strings.Replace(string(raw), `"ignore_eos":false,`, ``, 1),
		"uppercase alias":     strings.Replace(string(raw), `"run_id":`, `"RUN_ID":`, 1),
		"unknown field":       strings.Replace(string(raw), `"run_id":`, `"extra":true,"run_id":`, 1),
		"trailing document":   string(raw) + `{}`,
		"byte cap":            strings.Repeat(" ", V41EPOwnerMaxRequestBytes+1),
		"depth cap":           strings.Repeat("[", V41EPOwnerMaxJSONDepth+2) + "0" + strings.Repeat("]", V41EPOwnerMaxJSONDepth+2),
		"fractional duration": strings.Replace(string(raw), `"join_ns":1000000000`, `"join_ns":1.5`, 1),
		"invalid JSON UTF8":   strings.Replace(string(raw), "bounded-test", "bounded-\xff", 1),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeV41EPOwnerRequest(strings.NewReader(malformed)); err == nil {
				t.Fatal("malformed JSON request accepted")
			}
		})
	}

	t.Run("explicit upper budgets", func(t *testing.T) {
		r := v41EPOwnerRequestFixture()
		r.Budgets = V41EPOwnerBudgets{JoinNS: V41EPOwnerMaxJoinTime, LoadNS: V41EPOwnerMaxLoadTime, ForwardNS: V41EPOwnerMaxForwardTime, CleanupNS: V41EPOwnerMaxCleanupTime, TotalNS: V41EPOwnerMaxTotalTime}
		r.Supervisor.TerminateNS, r.Supervisor.KillGraceNS = V41EPOwnerMaxTotalTime, V41EPOwnerMaxKillGrace
		r.PromptTokenIDs = make([]int, V41EPOwnerMaxPromptTokens)
		r.ContextLimit, r.GeneratedTokenLimit = V41EPOwnerMaxContextTokens, V41EPOwnerMaxGeneratedTokens
		if err := r.Validate(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("manifest bytes retained", func(t *testing.T) {
		r := v41EPOwnerRequestFixture()
		r.Shards[0].RawFileSHA256 = strings.ToUpper(r.Shards[0].RawFileSHA256)
		v41EPOwnerPinFixtureManifest(&r)
		before, _ := json.Marshal(r.Shards)
		if err := r.Validate(); err != nil {
			t.Fatal(err)
		}
		after, _ := json.Marshal(r.Shards)
		if string(before) != string(after) {
			t.Fatal("manifest normalized during validation")
		}
		changed, _ := V41EPOwnerRequestDigest(r)
		if changed == digest {
			t.Fatal("different exact manifest retained prior request agreement")
		}
	})
}

// fak-test:runtime fast est=1ms
func TestV41EPOwnerReceiptUnknownObservations(t *testing.T) {
	t.Parallel()
	r := V41EPOwnerReceipt{Schema: V41EPOwnerReceiptSchema, Status: "incomplete", LastPhase: "forward"}
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"artifact_authenticated", "executable_sha256", "backend_identity", "owner_rounds", "socket_io", "supervisor_evidence_sha256", "prefill_tokens", "prefill_ns", "completed_step_calls", "decode_loop_ns", "execution_joined", "cleanup_completed"} {
		if !strings.Contains(string(raw), fmt.Sprintf(`"%s":null`, field)) {
			t.Fatalf("unavailable %s did not remain explicit null: %s", field, raw)
		}
	}
	zero := 0
	zeroNS := int64(0)
	r.RootWork.CompletedStepCalls, r.RootWork.DecodeLoopNS = &zero, &zeroNS
	r.OwnerRounds, r.SocketIO = &V41EPOwnerRoundCounters{}, &V41EPOwnerSocketCounters{}
	raw, err = json.Marshal(r)
	if err != nil || !strings.Contains(string(raw), `"completed_step_calls":0`) || !strings.Contains(string(raw), `"decode_loop_ns":0`) || !strings.Contains(string(raw), `"read_bytes":0`) {
		t.Fatalf("observed zero conflated with unavailable: %s %v", raw, err)
	}
}
