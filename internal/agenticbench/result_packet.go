package agenticbench

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const (
	ResultPacketSchema     = "fak.agentic-benchmark-result-packet.v1"
	DefaultResultPacketDir = "experiments/agent-live/agentic-benchmark-result-packets"
)

var requiredMetricCategories = []string{
	"task_success",
	"safe_success",
	"cost_or_token_budget",
	"latency",
	"policy_events",
	"evidence_completeness",
}

type ResultPacketStatus struct {
	Path               string   `json:"path"`
	Issue              int      `json:"issue"`
	Packet             string   `json:"packet,omitempty"`
	Status             string   `json:"status"`
	ResultClaimAllowed bool     `json:"result_claim_allowed"`
	Gate               string   `json:"gate"`
	Detail             string   `json:"detail"`
	Missing            []string `json:"missing,omitempty"`
	Artifacts          []string `json:"artifacts,omitempty"`
}

func scanResultPackets(root string) []ResultPacketStatus {
	dir := filepath.Join(root, filepath.FromSlash(DefaultResultPacketDir))
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return []ResultPacketStatus{{
			Path:   filepath.ToSlash(DefaultResultPacketDir),
			Status: "UNREADABLE_RESULT_PACKET_DIR",
			Gate:   "FAIL",
			Detail: err.Error(),
		}}
	}
	var packets []ResultPacketStatus
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		rel := filepath.ToSlash(filepath.Join(DefaultResultPacketDir, entry.Name()))
		path := filepath.Join(root, filepath.FromSlash(rel))
		doc, err := readJSON(path)
		if err != nil {
			packets = append(packets, ResultPacketStatus{
				Path:   rel,
				Status: "UNREADABLE_RESULT_PACKET",
				Gate:   "FAIL",
				Detail: err.Error(),
			})
			continue
		}
		packets = append(packets, checkResultPacket(root, rel, doc))
	}
	sort.Slice(packets, func(i, j int) bool { return packets[i].Path < packets[j].Path })
	return packets
}

func checkResultPacket(root, rel string, doc map[string]any) ResultPacketStatus {
	packet := ResultPacketStatus{
		Path:               rel,
		Issue:              intValue(doc["issue"]),
		Packet:             str(doc, "packet"),
		Status:             str(doc, "status"),
		ResultClaimAllowed: boolv(doc, "result_claim_allowed"),
		Artifacts:          stringSlice(doc, "artifacts"),
	}
	var missing []string
	if str(doc, "schema") != ResultPacketSchema {
		missing = append(missing, "schema must be "+ResultPacketSchema)
	}
	if !liveResultIssue(packet.Issue) {
		missing = append(missing, "issue must be one of 870, 871, 872, 873, 874, or 875")
	}
	if packet.Status != "PASS_RESULT" {
		missing = append(missing, "status must be PASS_RESULT")
	}
	if !packet.ResultClaimAllowed {
		missing = append(missing, "result_claim_allowed must be true")
	}
	for _, gate := range []string{"benchmark_native", "same_task_ids", "same_model", "same_budget"} {
		if !boolv(doc, gate) {
			missing = append(missing, gate+" must be true")
		}
	}
	if !nestedBool(doc, "official_grader", "available") {
		missing = append(missing, "official_grader.available must be true")
	}
	if !hasArmRole(doc, "raw") {
		missing = append(missing, "arms must include a raw role")
	}
	if !hasArmRole(doc, "fak") {
		missing = append(missing, "arms must include a fak role")
	}
	metrics := mapValue(doc["metric_categories"])
	for _, category := range requiredMetricCategories {
		if !boolv(metrics, category) {
			missing = append(missing, "metric_categories."+category+" must be true")
		}
	}
	for _, artifact := range packet.Artifacts {
		if !artifactExists(root, artifact) {
			missing = append(missing, "artifact missing: "+artifact)
		}
	}
	if len(packet.Artifacts) == 0 {
		missing = append(missing, "artifacts must list the checked-in raw/fak evidence")
	}
	if len(missing) > 0 {
		packet.Gate = "FAIL"
		packet.Status = firstNonEmpty(packet.Status, "INCOMPLETE_RESULT_PACKET")
		packet.Detail = fmt.Sprintf("result packet incomplete: %s", strings.Join(missing, "; "))
		packet.Missing = missing
		packet.ResultClaimAllowed = false
		return packet
	}
	packet.Gate = "PASS_RESULT"
	packet.Detail = fmt.Sprintf("result packet for #%d passes raw/fak external harness evidence gate", packet.Issue)
	return packet
}

func liveResultIssue(issue int) bool {
	return issue >= 870 && issue <= 875
}

func hasArmRole(doc map[string]any, role string) bool {
	for _, raw := range anySlice(doc["arms"]) {
		arm := mapValue(raw)
		if str(arm, "role") == role {
			return true
		}
		name := strings.ToLower(str(arm, "name"))
		if strings.HasPrefix(name, role+"-") || strings.Contains(name, "-"+role) {
			return true
		}
	}
	return false
}

func artifactExists(root, artifact string) bool {
	if artifact == "" || filepath.IsAbs(artifact) {
		return false
	}
	rel := filepath.Clean(filepath.FromSlash(artifact))
	if strings.HasPrefix(rel, ".."+string(filepath.Separator)) || rel == ".." {
		return false
	}
	info, err := os.Stat(filepath.Join(root, rel))
	return err == nil && !info.IsDir()
}

func mapValue(raw any) map[string]any {
	m, _ := raw.(map[string]any)
	if m == nil {
		return map[string]any{}
	}
	return m
}

func intValue(raw any) int {
	switch v := raw.(type) {
	case int:
		return v
	case float64:
		return int(v)
	case jsonNumber:
		i, _ := strconv.Atoi(v.String())
		return i
	default:
		return 0
	}
}

type jsonNumber interface {
	String() string
}

func nestedBool(m map[string]any, keys ...string) bool {
	cur := any(m)
	for _, key := range keys {
		next, ok := cur.(map[string]any)
		if !ok {
			return false
		}
		cur = next[key]
	}
	v, _ := cur.(bool)
	return v
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
