package supportgraph

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

const WitnessSchema = "fak-support-witness/1"

type Witness struct {
	Schema         string    `json:"schema"`
	ID             string    `json:"id"`
	Tuple          Tuple     `json:"tuple"`
	State          State     `json:"state"`
	Tier           Tier      `json:"tier"`
	Authority      string    `json:"authority"`
	ArtifactDigest string    `json:"artifact_digest"`
	Environment    string    `json:"environment"`
	Reproduce      string    `json:"reproduce"`
	ObservedAt     time.Time `json:"observed_at"`
	Expires        time.Time `json:"expires"`
	Required       []string  `json:"required_baseline,omitempty"`
	Recommended    []string  `json:"recommended_baseline,omitempty"`
	Fallback       string    `json:"fallback,omitempty"`
	Penalty        string    `json:"penalty,omitempty"`
	SourceCommit   string    `json:"source_commit"`
	PayloadSHA256  string    `json:"payload_sha256"`
}

type IngestResult struct {
	Graph      Graph  `json:"graph"`
	Inserted   bool   `json:"inserted"`
	StaleEdges int    `json:"stale_edges"`
	WitnessID  string `json:"witness_id"`
}

func ParseWitness(raw []byte) (Witness, error) {
	var witness Witness
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&witness); err != nil {
		return Witness{}, fmt.Errorf("decode witness: %w", err)
	}
	if err := ValidateWitness(witness); err != nil {
		return Witness{}, err
	}
	return witness, nil
}

func ValidateWitness(w Witness) error {
	if w.Schema != WitnessSchema || w.ID == "" || w.Authority == "" || w.SourceCommit == "" {
		return fmt.Errorf("witness schema, id, authority, and source_commit are required")
	}
	if w.State != Supported && w.State != Unsupported {
		return fmt.Errorf("witness state must be supported or unsupported")
	}
	if w.Tier != Observed && w.Tier != Witnessed {
		return fmt.Errorf("witness tier must be observed or witnessed")
	}
	if !strings.HasPrefix(w.ArtifactDigest, "sha256:") || len(strings.TrimPrefix(w.ArtifactDigest, "sha256:")) != 64 {
		return fmt.Errorf("artifact_digest must be sha256:<64 hex>")
	}
	if _, err := hex.DecodeString(strings.TrimPrefix(w.ArtifactDigest, "sha256:")); err != nil {
		return fmt.Errorf("artifact_digest: %w", err)
	}
	if w.Environment == "" || w.Reproduce == "" || w.ObservedAt.IsZero() || w.Expires.IsZero() || !w.Expires.After(w.ObservedAt) {
		return fmt.Errorf("environment, reproduce, and a valid observation window are required")
	}
	if w.Tuple.Artifact == "" || w.Tuple.Backend == "" || w.Tuple.Runtime == "" || w.Tuple.Hardware == "" {
		return fmt.Errorf("artifact, backend, runtime, and hardware tuple fields are required")
	}
	if w.PayloadSHA256 != WitnessDigest(w) {
		return fmt.Errorf("payload_sha256 mismatch")
	}
	return nil
}

func WitnessDigest(w Witness) string {
	w.PayloadSHA256 = ""
	raw, _ := json.Marshal(w)
	digest := sha256.Sum256(raw)
	return hex.EncodeToString(digest[:])
}

func Ingest(graph Graph, witness Witness) (IngestResult, error) {
	if err := ValidateWitness(witness); err != nil {
		return IngestResult{}, err
	}
	result := IngestResult{Graph: graph, Inserted: true, WitnessID: witness.ID}
	for edgeIndex := range result.Graph.Edges {
		edge := &result.Graph.Edges[edgeIndex]
		for evidenceIndex := range edge.Evidence {
			if edge.Evidence[evidenceIndex].ID == witness.ID {
				return IngestResult{Graph: graph, WitnessID: witness.ID}, nil
			}
		}
		if sameArtifactPath(edge.Tuple, witness.Tuple) && edge.Tuple != witness.Tuple {
			for evidenceIndex := range edge.Evidence {
				evidence := &edge.Evidence[evidenceIndex]
				if evidence.Expires.IsZero() || evidence.Expires.After(witness.ObservedAt) {
					evidence.Expires = witness.ObservedAt
					result.StaleEdges++
				}
			}
		}
	}
	evidence := Evidence{ID: witness.ID, State: witness.State, Tier: witness.Tier, Authority: witness.Authority, Source: "witness:" + witness.ID, Expires: witness.Expires, Fallback: witness.Fallback, Penalty: witness.Penalty, ArtifactDigest: witness.ArtifactDigest, Environment: witness.Environment, Reproduce: witness.Reproduce}
	for edgeIndex := range result.Graph.Edges {
		if result.Graph.Edges[edgeIndex].Tuple == witness.Tuple {
			result.Graph.Edges[edgeIndex].Evidence = append(result.Graph.Edges[edgeIndex].Evidence, evidence)
			return result, nil
		}
	}
	result.Graph.Edges = append(result.Graph.Edges, Edge{Tuple: witness.Tuple, Required: sortedCopy(witness.Required), Recommended: sortedCopy(witness.Recommended), Evidence: []Evidence{evidence}})
	sort.Slice(result.Graph.Edges, func(i, j int) bool {
		return tupleKey(result.Graph.Edges[i].Tuple) < tupleKey(result.Graph.Edges[j].Tuple)
	})
	return result, nil
}

func sameArtifactPath(left, right Tuple) bool {
	return left.Artifact == right.Artifact && left.Architecture == right.Architecture && left.Quant == right.Quant && left.Layout == right.Layout && left.Backend == right.Backend && left.Kernel == right.Kernel
}

func tupleKey(tuple Tuple) string {
	return tuple.Artifact + "\x00" + tuple.Architecture + "\x00" + tuple.Quant + "\x00" + tuple.Layout + "\x00" + tuple.Backend + "\x00" + tuple.Kernel + "\x00" + tuple.Runtime + "\x00" + tuple.Hardware
}

func sortedCopy(values []string) []string {
	out := append([]string(nil), values...)
	sort.Strings(out)
	return out
}
