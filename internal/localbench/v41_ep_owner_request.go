package localbench

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/strictjson"
)

const (
	V41EPOwnerRequestSchema = "fak.v41-ep-owner.request/v1"
	V41EPOwnerReceiptSchema = "fak.v41-ep-owner.receipt/v1"

	// These are admission ceilings, not default workloads or execution grants.
	V41EPOwnerMaxRequestBytes    = 1 << 20
	V41EPOwnerMaxJSONDepth       = 32
	V41EPOwnerMaxShards          = 256
	V41EPOwnerMaxPathBytes       = 4096
	V41EPOwnerMaxRunIDBytes      = 128
	V41EPOwnerMaxBackendBytes    = 64
	V41EPOwnerMaxDeviceBytes     = 512
	V41EPOwnerMaxAddressBytes    = 512
	V41EPOwnerMaxPromptTokens    = 32768
	V41EPOwnerMaxGeneratedTokens = 256
	V41EPOwnerMaxContextTokens   = 65536
	V41EPOwnerMaxJoinTime        = 5 * time.Minute
	V41EPOwnerMaxLoadTime        = 2 * time.Hour
	V41EPOwnerMaxForwardTime     = 30 * time.Minute
	V41EPOwnerMaxCleanupTime     = 2 * time.Minute
	V41EPOwnerMaxTotalTime       = 3 * time.Hour
	V41EPOwnerMaxKillGrace       = 30 * time.Second
)

// V41EPOwnerRequest is one common request shared by three separately launched
// owner processes. It cannot launch them or grant hardware/network authority.
// Shards and ExpectedShardSetSHA256 must be independently trusted before use.
// SourceSHA256 and the request digest establish agreement, not attestation.
type V41EPOwnerRequest struct {
	Schema                 string                       `json:"schema"`
	RunID                  string                       `json:"run_id"`
	WorldSize              int                          `json:"world_size"`
	Shards                 []ggufload.ShardIdentity     `json:"shards"`
	ExpectedShardSetSHA256 string                       `json:"expected_shard_set_sha256"`
	SourceSHA256           string                       `json:"source_sha256"`
	CoordinatorAddress     string                       `json:"coordinator_address"`
	Owners                 []V41EPOwnerSelection        `json:"owners"`
	PromptTokenIDs         []int                        `json:"prompt_token_ids"`
	ContextLimit           int                          `json:"context_limit"`
	GeneratedTokenLimit    int                          `json:"generated_token_limit"`
	Repetitions            int                          `json:"repetitions"`
	IgnoreEOS              bool                         `json:"ignore_eos"`
	Budgets                V41EPOwnerBudgets            `json:"budgets"`
	Supervisor             V41EPOwnerSupervisorContract `json:"supervisor"`
}

type V41EPOwnerSelection struct {
	Rank                     int    `json:"rank"`
	Backend                  string `json:"backend"`
	ExpectedDevice           string `json:"expected_device"`
	ExpectedExecutableSHA256 string `json:"expected_executable_sha256,omitempty"`
}

type V41EPOwnerBudgets struct {
	JoinNS    time.Duration `json:"join_ns"`
	LoadNS    time.Duration `json:"load_ns"`
	ForwardNS time.Duration `json:"forward_ns"`
	CleanupNS time.Duration `json:"cleanup_ns"`
	TotalNS   time.Duration `json:"total_ns"`
}

// V41EPOwnerSupervisorContract describes required external enforcement. Parsing
// this declaration cannot establish that a supervisor exists or enforced it.
// The separately reviewed process-group harness supplies actual run evidence.
type V41EPOwnerSupervisorContract struct {
	Kind        string        `json:"kind"`
	ContractID  string        `json:"contract_id"`
	TerminateNS time.Duration `json:"terminate_ns"`
	KillGraceNS time.Duration `json:"kill_grace_ns"`
}

// DecodeV41EPOwnerRequest reads one bounded strict JSON request without opening
// checkpoint paths, resolving a backend, or contacting any endpoint.
func DecodeV41EPOwnerRequest(reader io.Reader) (V41EPOwnerRequest, error) {
	if reader == nil {
		return V41EPOwnerRequest{}, errors.New("v41 EP owner: missing request reader")
	}
	raw, err := io.ReadAll(io.LimitReader(reader, V41EPOwnerMaxRequestBytes+1))
	if err != nil {
		return V41EPOwnerRequest{}, err
	}
	if len(raw) > V41EPOwnerMaxRequestBytes {
		return V41EPOwnerRequest{}, errors.New("v41 EP owner: request exceeds byte limit")
	}
	if !utf8.Valid(raw) {
		return V41EPOwnerRequest{}, errors.New("v41 EP owner: request is not valid UTF-8")
	}
	if err := v41EPOwnerUniqueJSON(raw); err != nil {
		return V41EPOwnerRequest{}, err
	}
	var req V41EPOwnerRequest
	if err := strictjson.Decode(raw, &req, "v41 EP owner: trailing JSON document"); err != nil {
		return V41EPOwnerRequest{}, err
	}
	if err := req.Validate(); err != nil {
		return V41EPOwnerRequest{}, err
	}
	return req, nil
}

// Validate checks only input structure, bounds and declared identity agreement.
// The authenticated loader must still validate shard layout and actual contents;
// runtime admission must still verify the selected backend/device and supervisor.
func (req V41EPOwnerRequest) Validate() error {
	fail := func(reason string) error { return fmt.Errorf("v41 EP owner: %s", reason) }
	if req.Schema != V41EPOwnerRequestSchema || !v41EPOwnerText(req.RunID, V41EPOwnerMaxRunIDBytes) {
		return fail("invalid schema or run ID")
	}
	if req.WorldSize != 3 || len(req.Owners) != 3 || req.Repetitions != 1 {
		return fail("exactly three owners and one repetition are required")
	}
	if !v41EPOwnerSHA256(req.SourceSHA256, true) || !v41EPOwnerSHA256(req.ExpectedShardSetSHA256, true) {
		return fail("source and shard-set agreement require lowercase SHA-256")
	}
	if !v41EPOwnerText(req.CoordinatorAddress, V41EPOwnerMaxAddressBytes) {
		return fail("invalid coordinator address")
	}
	host, portText, err := net.SplitHostPort(req.CoordinatorAddress)
	port, portErr := strconv.ParseUint(portText, 10, 16)
	if err != nil || host == "" || strings.TrimSpace(host) != host || strings.ContainsAny(host, " \t") || portErr != nil || port == 0 || strconv.FormatUint(port, 10) != portText {
		return fail("coordinator address requires explicit host and canonical port")
	}
	for rank, owner := range req.Owners {
		if owner.Rank != rank {
			return fail("owners must appear exactly once in rank order 0, 1, 2")
		}
		if !v41EPOwnerText(owner.Backend, V41EPOwnerMaxBackendBytes) || strings.ContainsAny(owner.Backend, " \t") || owner.Backend != strings.ToLower(owner.Backend) {
			return fail("each owner requires an explicit canonical backend")
		}
		switch owner.Backend {
		case "auto", "default", "legacy", "cpu", "cpu-ref":
			return fail("automatic, legacy and reference backends are forbidden")
		}
		if !v41EPOwnerText(owner.ExpectedDevice, V41EPOwnerMaxDeviceBytes) {
			return fail("each owner requires an expected device identity")
		}
		if owner.ExpectedExecutableSHA256 != "" && !v41EPOwnerSHA256(owner.ExpectedExecutableSHA256, true) {
			return fail("expected executable identity must be lowercase SHA-256")
		}
	}
	if len(req.Shards) == 0 || len(req.Shards) > V41EPOwnerMaxShards {
		return fail("invalid shard count")
	}
	paths := make(map[string]bool, len(req.Shards))
	var artifactBytes int64
	for _, shard := range req.Shards {
		if !v41EPOwnerText(shard.Path, V41EPOwnerMaxPathBytes) || !filepath.IsAbs(shard.Path) || filepath.Clean(shard.Path) != shard.Path || paths[shard.Path] {
			return fail("shard paths must be distinct, absolute and already clean")
		}
		paths[shard.Path] = true
		if shard.Size <= 0 || shard.Size > int64(^uint64(0)>>1)-artifactBytes || !v41EPOwnerSHA256(shard.RawFileSHA256, false) {
			return fail("invalid or overflowing shard size or file SHA-256")
		}
		artifactBytes += shard.Size
	}
	// Preserve the loader's exact canonical representation, including shard
	// order, absolute paths, field order and original raw-file digest casing.
	manifest, err := json.Marshal(req.Shards)
	if err != nil {
		return err
	}
	manifestDigest := sha256.Sum256(manifest)
	if hex.EncodeToString(manifestDigest[:]) != req.ExpectedShardSetSHA256 {
		return fail("ordered manifest does not match independently trusted set SHA-256")
	}
	if len(req.PromptTokenIDs) == 0 || len(req.PromptTokenIDs) > V41EPOwnerMaxPromptTokens ||
		req.GeneratedTokenLimit <= 0 || req.GeneratedTokenLimit > V41EPOwnerMaxGeneratedTokens ||
		req.ContextLimit <= 0 || req.ContextLimit > V41EPOwnerMaxContextTokens ||
		len(req.PromptTokenIDs) > req.ContextLimit || req.GeneratedTokenLimit > req.ContextLimit-len(req.PromptTokenIDs) {
		return fail("invalid prompt, context or generated-token bounds")
	}
	for _, token := range req.PromptTokenIDs {
		if token < 0 || uint64(token) > uint64(^uint32(0)>>1) {
			return fail("prompt token IDs must fit nonnegative int32")
		}
	}
	if req.Budgets.TotalNS <= 0 || req.Budgets.TotalNS > V41EPOwnerMaxTotalTime {
		return fail("invalid total duration")
	}
	remaining := req.Budgets.TotalNS
	for _, phase := range []struct{ value, cap time.Duration }{
		{req.Budgets.JoinNS, V41EPOwnerMaxJoinTime}, {req.Budgets.LoadNS, V41EPOwnerMaxLoadTime},
		{req.Budgets.ForwardNS, V41EPOwnerMaxForwardTime}, {req.Budgets.CleanupNS, V41EPOwnerMaxCleanupTime},
	} {
		if phase.value <= 0 || phase.value > phase.cap || phase.value > remaining {
			return fail("phase durations must be positive, capped and fit total duration")
		}
		remaining -= phase.value
	}
	supervisor := req.Supervisor
	if supervisor.Kind != "external-process-group" || !v41EPOwnerText(supervisor.ContractID, V41EPOwnerMaxRunIDBytes) ||
		supervisor.TerminateNS < req.Budgets.TotalNS || supervisor.TerminateNS > V41EPOwnerMaxTotalTime ||
		supervisor.KillGraceNS <= 0 || supervisor.KillGraceNS > V41EPOwnerMaxKillGrace {
		return fail("a bounded external process-group supervisor contract is required")
	}
	raw, err := json.Marshal(req)
	if err != nil || len(raw) > V41EPOwnerMaxRequestBytes {
		return fail("canonical request exceeds byte limit")
	}
	return nil
}

// V41EPOwnerRequestDigest returns canonical request agreement only. It does not
// authenticate files, the running source/binary, hardware, or external supervision.
func V41EPOwnerRequestDigest(req V41EPOwnerRequest) (string, error) {
	if err := req.Validate(); err != nil {
		return "", err
	}
	raw, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(raw)
	return hex.EncodeToString(digest[:]), nil
}

func v41EPOwnerText(value string, maxBytes int) bool {
	if len(value) == 0 || len(value) > maxBytes || !utf8.ValidString(value) || strings.TrimSpace(value) != value {
		return false
	}
	for _, r := range value {
		if r < 0x20 || r == 0x7f {
			return false
		}
	}
	return true
}

func v41EPOwnerSHA256(value string, lowercase bool) bool {
	if len(value) != sha256.Size*2 || lowercase && value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

// Reject ambiguous duplicate keys (including escaped spellings) before decoding
// structs. Lowercase keys also exclude encoding/json's case-insensitive aliases.
func v41EPOwnerUniqueJSON(raw []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value func(int, string) error
	value = func(depth int, field string) error {
		if depth > V41EPOwnerMaxJSONDepth {
			return errors.New("v41 EP owner: JSON nesting exceeds limit")
		}
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		if token == nil {
			return errors.New("v41 EP owner: null request values are forbidden")
		}
		switch token {
		case json.Delim('{'):
			seen := map[string]bool{}
			for decoder.More() {
				key, err := decoder.Token()
				if err != nil {
					return err
				}
				name, ok := key.(string)
				if !ok || name != strings.ToLower(name) || seen[name] {
					return errors.New("v41 EP owner: duplicate or noncanonical JSON key")
				}
				seen[name] = true
				if err := value(depth+1, name); err != nil {
					return err
				}
			}
			if depth == 0 && !seen["ignore_eos"] || field == "owners" && !seen["rank"] {
				return errors.New("v41 EP owner: explicit ignore_eos and owner ranks are required")
			}
			_, err = decoder.Token()
		case json.Delim('['):
			for decoder.More() {
				if err := value(depth+1, field); err != nil {
					return err
				}
			}
			_, err = decoder.Token()
		}
		return err
	}
	return value(0, "")
}

// V41EPOwnerReceipt is an observation container, not a receipt attestation or a
// proof of supervisor enforcement. Unavailable observations are explicit nulls.
// A failed rawdecode run has no trusted partial phase counts in the first adapter.
type V41EPOwnerReceipt struct {
	Schema                   string                          `json:"schema"`
	RunID                    string                          `json:"run_id"`
	RequestSHA256            string                          `json:"request_sha256"`
	Rank                     int                             `json:"rank"`
	WorldSize                int                             `json:"world_size"`
	Status                   string                          `json:"status"`
	LastPhase                string                          `json:"last_phase"`
	StartedAt                time.Time                       `json:"started_at"`
	FinishedAt               *time.Time                      `json:"finished_at"`
	Shards                   []ggufload.ShardIdentity        `json:"shards"`
	ExpectedShardSetSHA256   string                          `json:"expected_shard_set_sha256"`
	ArtifactAuthenticated    *bool                           `json:"artifact_authenticated"`
	SourceSHA256             string                          `json:"source_sha256"`
	ExecutableSHA256         *string                         `json:"executable_sha256"`
	BackendIdentity          *compute.BackendRuntimeIdentity `json:"backend_identity"`
	Phases                   V41EPOwnerPhaseTimes            `json:"phases"`
	RootWork                 V41EPOwnerRootWork              `json:"root_work"`
	OwnerRounds              *V41EPOwnerRoundCounters        `json:"owner_rounds"`
	SocketIO                 *V41EPOwnerSocketCounters       `json:"socket_io"`
	ExecutionJoined          *bool                           `json:"execution_joined"`
	ShutdownCompleted        *bool                           `json:"shutdown_completed"`
	CleanupCompleted         *bool                           `json:"cleanup_completed"`
	SupervisorEvidenceSHA256 *string                         `json:"supervisor_evidence_sha256"`
	Failure                  string                          `json:"failure,omitempty"`
	CleanupFailures          []string                        `json:"cleanup_failures,omitempty"`
}

// V41EPOwnerPhaseTimes contains only elapsed phases actually observed by a runner.
type V41EPOwnerPhaseTimes struct {
	JoinNS    *int64 `json:"join_ns"`
	LoadNS    *int64 `json:"load_ns"`
	ForwardNS *int64 `json:"forward_ns"`
	CleanupNS *int64 `json:"cleanup_ns"`
	TotalNS   *int64 `json:"total_ns"`
}

// V41EPOwnerRootWork distinguishes Prefill's first token from completed Step calls.
// DecodeLoopNS includes sampling/bookkeeping; it is not a GPU-kernel-only duration.
// Cold KV means a fresh Session after load, not cold storage or filesystem caches.
type V41EPOwnerRootWork struct {
	PrefillTokens      *int   `json:"prefill_tokens"`
	PrefillNS          *int64 `json:"prefill_ns"`
	CompletedStepCalls *int   `json:"completed_step_calls"`
	DecodeLoopNS       *int64 `json:"decode_loop_ns"`
	SessionSetupNS     *int64 `json:"session_setup_ns"`
	FirstSampleNS      *int64 `json:"first_sample_ns"`
	SessionTeardownNS  *int64 `json:"session_teardown_ns"`
	GeneratedTokenIDs  []int  `json:"generated_token_ids"`
	EOSStopped         *bool  `json:"eos_stopped"`
}

// V41EPOwnerRoundCounters preserves existing Model transport-stat meanings:
// completed owner rounds only, excluding decode frames, partial I/O and TCP/IP.
type V41EPOwnerRoundCounters struct {
	CompletedGathers   uint64 `json:"completed_gathers"`
	SentFrameBytes     uint64 `json:"sent_frame_bytes"`
	ReceivedFrameBytes uint64 `json:"received_frame_bytes"`
	GatherNanoseconds  uint64 `json:"gather_nanoseconds"`
}

// V41EPOwnerSocketCounters covers actual positive net.Conn Read/Write byte counts,
// including preamble/join/decode/failed partial I/O/shutdown, excluding TCP/IP.
type V41EPOwnerSocketCounters struct {
	ReadBytes    uint64 `json:"read_bytes"`
	WrittenBytes uint64 `json:"written_bytes"`
	ReadErrors   uint64 `json:"read_errors"`
	WriteErrors  uint64 `json:"write_errors"`
}
