package trajectory

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

const ContinuationBundleSchema = "fak-portable-session/1alpha1"
const ContinuationReceiptSchema = "fak-session-lineage/1alpha1"

type PortableArtifact struct {
	Name      string `json:"name"`
	Digest    string `json:"digest"`
	MediaType string `json:"media_type,omitempty"`
}
type ContinuationBundle struct {
	Schema           string             `json:"schema"`
	SessionID        string             `json:"session_id"`
	ParentSessionID  string             `json:"parent_session_id,omitempty"`
	TrajectoryDigest string             `json:"trajectory_digest"`
	CheckpointDigest string             `json:"checkpoint_digest"`
	Artifacts        []PortableArtifact `json:"artifacts,omitempty"`
	Adapters         map[string]string  `json:"adapters,omitempty"`
	Loss             []string           `json:"loss,omitempty"`
	CreatedAt        time.Time          `json:"created_at"`
	ContentAddress   string             `json:"content_address"`
}
type ContinuationReceipt struct {
	Schema           string `json:"schema"`
	Operation        string `json:"operation"`
	InputIdentity    string `json:"input_identity"`
	OutputIdentity   string `json:"output_identity"`
	ManifestAddress  string `json:"manifest_address"`
	TrajectoryDigest string `json:"trajectory_digest"`
	CheckpointDigest string `json:"checkpoint_digest"`
}

func BuildContinuationBundle(sessionID, parent string, events []Event, checkpoint []byte, artifacts []PortableArtifact, adapters map[string]string, loss []string, created time.Time) (ContinuationBundle, error) {
	if strings.TrimSpace(sessionID) == "" || created.IsZero() {
		return ContinuationBundle{}, errors.New("continuation bundle requires session_id and created_at")
	}
	if len(events) == 0 || len(checkpoint) == 0 {
		return ContinuationBundle{}, errors.New("continuation bundle requires trajectory and checkpoint state")
	}
	encoded, err := EncodeEvents(events)
	if err != nil {
		return ContinuationBundle{}, err
	}
	for i := range artifacts {
		if artifacts[i].Name == "" || !strings.HasPrefix(artifacts[i].Digest, "sha256:") {
			return ContinuationBundle{}, fmt.Errorf("artifact %d lacks name or digest", i)
		}
	}
	artifacts = append([]PortableArtifact(nil), artifacts...)
	sort.Slice(artifacts, func(i, j int) bool { return artifacts[i].Name < artifacts[j].Name })
	loss = append([]string(nil), loss...)
	sort.Strings(loss)
	bundle := ContinuationBundle{Schema: ContinuationBundleSchema, SessionID: sessionID, ParentSessionID: parent, TrajectoryDigest: digestBytes(encoded), CheckpointDigest: digestBytes(checkpoint), Artifacts: artifacts, Adapters: cloneStrings(adapters), Loss: loss, CreatedAt: created.UTC()}
	bundle.ContentAddress = continuationAddress(bundle)
	return bundle, bundle.Verify()
}
func (b ContinuationBundle) Verify() error {
	if b.Schema != ContinuationBundleSchema || b.SessionID == "" || b.TrajectoryDigest == "" || b.CheckpointDigest == "" || b.CreatedAt.IsZero() || b.ContentAddress == "" {
		return errors.New("incomplete continuation bundle")
	}
	if continuationAddress(b) != b.ContentAddress {
		return errors.New("continuation bundle content address mismatch")
	}
	return nil
}
func VerifyContinuation(b ContinuationBundle, operation, resultID string, events []Event, checkpoint []byte) (ContinuationReceipt, error) {
	if err := b.Verify(); err != nil {
		return ContinuationReceipt{}, err
	}
	if operation != "resume" && operation != "fork" {
		return ContinuationReceipt{}, fmt.Errorf("unsupported continuation %q", operation)
	}
	if resultID == "" {
		return ContinuationReceipt{}, errors.New("continuation requires result identity")
	}
	if operation == "resume" && resultID != b.SessionID {
		return ContinuationReceipt{}, errors.New("resume must retain identity")
	}
	if operation == "fork" && resultID == b.SessionID {
		return ContinuationReceipt{}, errors.New("fork must create identity")
	}
	encoded, err := EncodeEvents(events)
	if err != nil {
		return ContinuationReceipt{}, err
	}
	if digestBytes(encoded) != b.TrajectoryDigest || digestBytes(checkpoint) != b.CheckpointDigest {
		return ContinuationReceipt{}, errors.New("continuation state mismatch")
	}
	return ContinuationReceipt{Schema: ContinuationReceiptSchema, Operation: operation, InputIdentity: b.SessionID, OutputIdentity: resultID, ManifestAddress: b.ContentAddress, TrajectoryDigest: b.TrajectoryDigest, CheckpointDigest: b.CheckpointDigest}, nil
}
func continuationAddress(b ContinuationBundle) string {
	b.ContentAddress = ""
	encoded, _ := json.Marshal(b)
	return digestBytes(encoded)
}
