package localapphelper

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

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

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func (e NativeClient) Execute(ctx context.Context, task TaskRequest) (TaskResult, error) {
	if e.Endpoint == "" || e.Model == "" || e.Artifact == "" || e.Revision == "" {
		return TaskResult{}, errors.New("localapphelper: incomplete native gateway identity")
	}
	payload := map[string]any{"model": e.Model, "stream": false, "temperature": 0, "messages": []map[string]string{{"role": "system", "content": "Return only valid JSON for the requested app task."}, {"role": "user", "content": string(task.Payload)}}}
	raw, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(e.Endpoint, "/")+"/v1/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return TaskResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	client := e.Client
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	started := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return TaskResult{}, fmt.Errorf("localapphelper native gateway: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return TaskResult{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return TaskResult{}, fmt.Errorf("localapphelper native gateway status %d", resp.StatusCode)
	}
	var chat chatResponse
	if json.Unmarshal(body, &chat) != nil || len(chat.Choices) == 0 || strings.TrimSpace(chat.Choices[0].Message.Content) == "" {
		return TaskResult{}, errors.New("localapphelper: invalid native gateway response")
	}
	output := chat.Choices[0].Message.Content
	sum := sha256.Sum256([]byte(output))
	elapsed := time.Since(started).Milliseconds()
	receipt := localappcontract.Receipt{Schema: localappcontract.Schema, TaskID: task.TaskID, Engine: "fak-native", Location: "local-app-helper", Revision: e.Revision, AdmittedEnvelope: map[string]int64{}, ObservedEnvelope: map[string]int64{"wall_ms": elapsed, "output_bytes": int64(len(output))}, Quality: "captured", Attempts: 1, Authority: "authenticated-app-scope", Terminal: localappcontract.Completed, Reason: "artifact=" + e.Artifact + ";fallback=none;output_sha256=" + hex.EncodeToString(sum[:])}
	events := []localappcontract.Event{{Schema: localappcontract.Schema, Sequence: 1, TaskID: task.TaskID, Kind: "ready"}, {Schema: localappcontract.Schema, Sequence: 2, TaskID: task.TaskID, Kind: "output", Reason: output}, {Schema: localappcontract.Schema, Sequence: 3, TaskID: task.TaskID, Kind: "completed"}}
	return TaskResult{Events: events, Receipt: receipt}, nil
}
