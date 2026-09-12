package agentbench

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const comparisonArmSchema = "fak.agentbench.comparison-arm.v1"

type comparisonArmArtifact struct {
	Schema, ArmID, ManifestSHA256, ModelID, TokenizerSHA256, RendererSHA256 string
	WeightSHA256, Quantization, SamplingSHA256, OutputPolicySHA256          string
	ToolContractSHA256, TaskWitnessSHA256, ResourceSHA256, LaunchSHA256     string
	CachePolicySHA256, LoadScheduleSHA256                                   string
	ContextTokens, QualifiedConcurrency                                     int
	Replicates                                                              []comparisonReplicateRef
}
type comparisonReplicateRef struct {
	Pair                   int
	Seed                   uint64
	Order                  int
	Receipt, ReceiptSHA256 string
	ContextManifestSHA256  string
}

func readComparisonArm(dir string) (ComparisonInput, comparisonArmArtifact, error) {
	root, err := canonicalComparisonDirectory(dir)
	if err != nil {
		return ComparisonInput{}, comparisonArmArtifact{}, err
	}
	var arm comparisonArmArtifact
	if err := readJSON(filepath.Join(root, "comparison-arm.json"), &arm); err != nil {
		return ComparisonInput{}, arm, err
	}
	if err := validateComparisonArmIdentity(arm); err != nil {
		return ComparisonInput{}, arm, err
	}
	in := ComparisonInput{ArmID: arm.ArmID, ManifestDigest: arm.ManifestSHA256, ModelID: arm.ModelID, TokenizerID: arm.TokenizerSHA256, RendererID: arm.RendererSHA256, LoadDigest: arm.LoadScheduleSHA256, CachePreconditionDigest: arm.CachePolicySHA256, QualifiedConcurrency: arm.QualifiedConcurrency}
	var evidenceErr error
	for index, ref := range arm.Replicates {
		replicate := ComparisonReplicate{Pair: ref.Pair, Seed: ref.Seed, Order: ref.Order, EvidenceStatus: "incomplete", SharedC2: normalQualification{Reason: "shared C2 evidence is incomplete"}}
		path, err := containedComparisonPath(root, ref.Receipt)
		if err != nil {
			return in, arm, err
		}
		body, err := os.ReadFile(path)
		if err != nil {
			return in, arm, err
		}
		if digest(body) != strings.ToLower(ref.ReceiptSHA256) {
			return in, arm, fmt.Errorf("pair %d receipt digest mismatch", ref.Pair)
		}
		var outer receipt
		if err := json.Unmarshal(body, &outer); err != nil {
			return in, arm, err
		}
		if outer.Schema != "fak.agentbench.receipt.v1" || outer.Algorithm != "sha256" {
			replicate.EvidenceReason = "invalid normal outer receipt"
			in.Replicates = append(in.Replicates, replicate)
			evidenceErr = fmt.Errorf("%w: %s", errComparisonInconclusive, replicate.EvidenceReason)
			continue
		}
		if err := verifyArtifacts(filepath.Dir(path), outer.ArtifactSHA256); err != nil {
			return in, arm, err
		}
		contextDigest, err := validateContextManifest(filepath.Join(filepath.Dir(path), "normal-context-manifest.json"), arm)
		if err != nil {
			if errors.Is(err, errComparisonInconclusive) {
				replicate.EvidenceReason = err.Error()
				in.Replicates = append(in.Replicates, replicate)
				evidenceErr = err
				continue
			}
			return in, arm, err
		}
		arm.Replicates[index].ContextManifestSHA256 = contextDigest
		var run normalRunReceipt
		if err := readJSON(filepath.Join(filepath.Dir(path), "summary.json"), &run); err != nil {
			return in, arm, err
		}
		if run.Schema != "fak.agentbench.normal-run.v1" || run.ResolvedModel != arm.ModelID || !run.ServiceQualification.Qualified || arm.QualifiedConcurrency != 2 {
			replicate.EvidenceReason = "shared C2 evidence is incomplete"
			in.Replicates = append(in.Replicates, replicate)
			evidenceErr = fmt.Errorf("%w: %s", errComparisonInconclusive, replicate.EvidenceReason)
			continue
		}
		metrics, bootstrap, bootstrapKnown, err := verifySteadyEvents(filepath.Join(filepath.Dir(path), "normal-events.jsonl"))
		if err != nil {
			if errors.Is(err, errComparisonInconclusive) {
				replicate.EvidenceReason = err.Error()
				in.Replicates = append(in.Replicates, replicate)
				evidenceErr = err
				continue
			}
			return in, arm, err
		}
		accepted, taskWitness, err := verifyTaskReceipt(filepath.Join(filepath.Dir(path), "tasks-c2", "receipt.json"))
		if err != nil {
			if errors.Is(err, errComparisonInconclusive) {
				replicate.EvidenceReason = err.Error()
				in.Replicates = append(in.Replicates, replicate)
				evidenceErr = err
				continue
			}
			return in, arm, err
		}
		if taskWitness != arm.TaskWitnessSHA256 {
			replicate.EvidenceReason = "task definition witness mismatch"
			in.Replicates = append(in.Replicates, replicate)
			evidenceErr = fmt.Errorf("%w: %s", errComparisonInconclusive, replicate.EvidenceReason)
			continue
		}
		replicate = ComparisonReplicate{Pair: ref.Pair, Seed: ref.Seed, Order: ref.Order, SharedC2: run.ServiceQualification, OwnCapacity: run.ServiceQualification, SharedC2TasksAccepted: accepted, OwnCapacityTasksAccepted: accepted, BootstrapMillis: bootstrap, BootstrapKnown: bootstrapKnown, EvidenceStatus: "verified-selected-c2-only", EvidenceReason: "shared and own capacity are the same selected C2 cohort", MetricMillis: metrics}
		if !bootstrapKnown {
			replicate.EvidenceReason += "; startup bootstrap was not requested or measured"
		}
		in.Replicates = append(in.Replicates, replicate)
	}
	return in, arm, evidenceErr
}

func validateContextManifest(path string, arm comparisonArmArtifact) (string, error) {
	body, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	contextDigest := digest(body)
	var corpus normalCorpus
	if err := json.Unmarshal(body, &corpus); err != nil {
		return "", err
	}
	checked := 0
	check := func(request referenceRequest, encoding referenceEncoding) error {
		if encoding.ModelID != arm.ModelID || encoding.TokenizerID != arm.TokenizerSHA256 || encoding.RendererID != arm.RendererSHA256 || encoding.ContextWindowTokens != arm.ContextTokens || encoding.ReservedOutputTokens != request.OutputTokens {
			return fmt.Errorf("%w: used reference encoding identity mismatch", errComparisonInconclusive)
		}
		tools, err := json.Marshal(request.Tools)
		if err != nil {
			return err
		}
		if digest(tools) != arm.ToolContractSHA256 {
			return fmt.Errorf("%w: used tool contract digest mismatch", errComparisonInconclusive)
		}
		checked++
		return nil
	}
	for _, session := range corpus.Sessions {
		for _, turn := range session.Turns {
			if err := check(turn.Request, turn.Encoding); err != nil {
				return "", err
			}
		}
	}
	for _, precondition := range corpus.Preconditions {
		if err := check(precondition.Request, precondition.Encoding); err != nil {
			return "", err
		}
	}
	if checked == 0 {
		return "", fmt.Errorf("%w: context manifest contains no used encodings", errComparisonInconclusive)
	}
	return contextDigest, nil
}

func verifyArtifacts(root string, artifacts map[string]string) error {
	for _, name := range []string{"manifest.json", "normal-context-manifest.json", "normal-events.jsonl", "tasks-c2/receipt.json", "summary.json"} {
		if artifacts[name] == "" {
			return fmt.Errorf("missing artifact %s", name)
		}
	}
	for name, want := range artifacts {
		path, err := containedComparisonPath(root, name)
		if err != nil {
			return err
		}
		body, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if digest(body) != strings.ToLower(want) {
			return fmt.Errorf("artifact %s digest mismatch", name)
		}
	}
	return nil
}

func verifySteadyEvents(path string) (map[string]int64, int64, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, false, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	found := 0
	bootstrap, bootstrapRows := int64(0), 0
	var durations []int64
	for s.Scan() {
		var row struct {
			Phase         string `json:"phase"`
			Stage         string `json:"stage"`
			EventType     string `json:"event_type"`
			Rung          string `json:"rung"`
			Status        string `json:"status"`
			Concurrency   int    `json:"concurrency"`
			ScoredRequest bool   `json:"scored_request"`
		}
		if json.Unmarshal(s.Bytes(), &row) == nil {
			if row.EventType == "terminal" && row.Phase == "precondition" && row.Rung == "bootstrap-c2" && row.Status == "completed" && !row.ScoredRequest {
				var event lifecycleEvent
				if json.Unmarshal(s.Bytes(), &event) == nil && event.Event == "bootstrap" && event.DurationMilliseconds >= 0 && event.StartedAt != "" && event.CompletedAt != "" {
					bootstrap, bootstrapRows = event.DurationMilliseconds, bootstrapRows+1
				}
			}
			terminal := row.Stage == "terminal" || row.EventType == "terminal"
			steady := row.Rung == "steady" || row.Rung == "steady-c2"
			if terminal && steady && row.Phase == "steady" && row.Status == "completed" && (row.Concurrency == 0 || row.Concurrency == 2) && row.ScoredRequest {
				found++
				var event lifecycleEvent
				if json.Unmarshal(s.Bytes(), &event) == nil {
					durations = append(durations, event.DurationMilliseconds)
				}
			}
		}
	}
	if err := s.Err(); err != nil {
		return nil, 0, false, err
	}
	if found != 96 {
		return nil, 0, false, fmt.Errorf("%w: shared-C2 steady terminals=%d want 96", errComparisonInconclusive, found)
	}
	sort.Slice(durations, func(i, j int) bool { return durations[i] < durations[j] })
	return map[string]int64{"completion_p90": durations[(len(durations)*9-1)/10]}, bootstrap, bootstrapRows == 1, nil
}

func verifyTaskReceipt(path string) (bool, string, error) {
	var wire struct {
		ConcurrencyQualified bool `json:"concurrency_qualified"`
		Tasks                []struct {
			ID, Family       string
			DefinitionSHA256 string `json:"definition_sha256"`
			Accepted, Passed bool
			ExternalPassed   bool `json:"external_passed"`
			Controls         struct {
				BrokenDigest string `json:"broken_digest"`
				FixedDigest  string `json:"fixed_digest"`
			} `json:"controls"`
			Model struct {
				Requests            int    `json:"requests"`
				UpstreamRequests    int    `json:"upstream_requests"`
				SuccessfulRequests  int    `json:"successful_requests"`
				ModelIdentityStatus string `json:"model_identity_status"`
				EventsDigest        string `json:"events_digest"`
			} `json:"model_observation"`
		} `json:"tasks"`
	}
	if err := readJSON(path, &wire); err != nil {
		return false, "", err
	}
	if !wire.ConcurrencyQualified || len(wire.Tasks) != 6 {
		return false, "", fmt.Errorf("%w: C2 task cohort is incomplete", errComparisonInconclusive)
	}
	type definition struct{ ID, DefinitionSHA256 string }
	definitions := make([]definition, 0, len(wire.Tasks))
	for _, task := range wire.Tasks {
		if !task.Accepted || !task.Passed || !task.ExternalPassed || task.Model.Requests == 0 || task.Model.Requests != task.Model.UpstreamRequests || task.Model.UpstreamRequests != task.Model.SuccessfulRequests || task.Model.ModelIdentityStatus != "matched" || task.Model.EventsDigest == "" {
			return false, "", fmt.Errorf("%w: C2 task cohort has rejected or failed attempts", errComparisonInconclusive)
		}
		if task.ID == "" || task.Family == "" || task.DefinitionSHA256 == "" {
			return false, "", fmt.Errorf("%w: task definition witness is incomplete", errComparisonInconclusive)
		}
		definitions = append(definitions, definition{task.ID, task.DefinitionSHA256})
	}
	sort.Slice(definitions, func(i, j int) bool { return definitions[i].ID < definitions[j].ID })
	body, err := json.Marshal(definitions)
	if err != nil {
		return false, "", err
	}
	return true, digest(body), nil
}

func digest(body []byte) string { sum := sha256.Sum256(body); return hex.EncodeToString(sum[:]) }
func readJSON(path string, out any) error {
	body, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(body, out)
}
