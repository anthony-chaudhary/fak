package localapphelper

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/localappcontract"
)

// NativeClient composes the app helper with an already-running fak-native gateway.
// The private helper authenticates apps; the gateway endpoint stays loopback/private.
type NativeClient struct {
	Endpoint string
	Model    string
	Artifact string
	Revision string
	Client   *http.Client
}

func (e NativeClient) Execute(ctx context.Context, task TaskRequest) (TaskResult, error) {
	if e.Endpoint == "" || e.Model == "" || e.Artifact == "" || e.Revision == "" {
		return TaskResult{}, errors.New("localapphelper: incomplete native gateway identity")
	}
	client := e.Client
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	started := time.Now()
	output, err := agent.CompleteOpenAIChat(ctx, client, e.Endpoint, e.Model, []agent.Message{
		{Role: "system", Content: "Return only valid JSON for the requested app task."},
		{Role: "user", Content: string(task.Payload)},
	})
	if err != nil {
		return TaskResult{}, fmt.Errorf("localapphelper native gateway: %w", err)
	}
	sum := sha256.Sum256([]byte(output))
	elapsed := time.Since(started).Milliseconds()
	receipt := localappcontract.Receipt{Schema: localappcontract.Schema, TaskID: task.TaskID, Engine: "fak-native", Location: "local-app-helper", Revision: e.Revision, AdmittedEnvelope: map[string]int64{}, ObservedEnvelope: map[string]int64{"wall_ms": elapsed, "output_bytes": int64(len(output))}, Quality: "captured", Attempts: 1, Authority: "authenticated-app-scope", Terminal: localappcontract.Completed, Reason: "artifact=" + e.Artifact + ";fallback=none;output_sha256=" + hex.EncodeToString(sum[:])}
	events := []localappcontract.Event{{Schema: localappcontract.Schema, Sequence: 1, TaskID: task.TaskID, Kind: "ready"}, {Schema: localappcontract.Schema, Sequence: 2, TaskID: task.TaskID, Kind: "output", Reason: output}, {Schema: localappcontract.Schema, Sequence: 3, TaskID: task.TaskID, Kind: "completed"}}
	return TaskResult{Events: events, Receipt: receipt}, nil
}
