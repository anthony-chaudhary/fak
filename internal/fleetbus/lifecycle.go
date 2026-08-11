package fleetbus

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const LifecycleSchema = "fak-fleet-lifecycle/1"

type LifecycleAction string

const (
	LifecyclePrepare    LifecycleAction = "prepare"
	LifecyclePause      LifecycleAction = "pause"
	LifecycleCheckpoint LifecycleAction = "checkpoint"
	LifecycleRestore    LifecycleAction = "restore"
	LifecycleResume     LifecycleAction = "resume"
	LifecycleStop       LifecycleAction = "stop"
	LifecycleCancel     LifecycleAction = "cancel"
	LifecycleStatus     LifecycleAction = "status"
)

type LifecycleRequest struct {
	Schema            string          `json:"schema"`
	TransactionID     string          `json:"transaction_id"`
	ForestID          string          `json:"forest_id"`
	MemberID          string          `json:"member_id,omitempty"`
	Generation        uint64          `json:"generation"`
	Action            LifecycleAction `json:"action"`
	Deadline          time.Time       `json:"deadline"`
	Capability        string          `json:"capability"`
	CausalPredecessor string          `json:"causal_predecessor,omitempty"`
	IdempotencyKey    string          `json:"idempotency_key"`
	Authority         string          `json:"authority"`
}

type LifecycleAckState string

const (
	AckAccepted         LifecycleAckState = "accepted"
	AckCompleted        LifecycleAckState = "completed"
	LifecycleAckRefused LifecycleAckState = "refused"
	AckUnsupported      LifecycleAckState = "unsupported"
)

type LifecycleAck struct {
	Schema        string            `json:"schema"`
	TransactionID string            `json:"transaction_id"`
	ForestID      string            `json:"forest_id"`
	MemberID      string            `json:"member_id"`
	Generation    uint64            `json:"generation"`
	State         LifecycleAckState `json:"state"`
	Reason        string            `json:"reason,omitempty"`
	CheckpointRef string            `json:"checkpoint_ref,omitempty"`
	ReadbackRef   string            `json:"readback_ref,omitempty"`
	At            time.Time         `json:"at"`
}

type LifecycleGate struct {
	Authority  string
	Capability string
	Generation uint64
	Seen       map[string]string
}

func (g *LifecycleGate) Validate(req LifecycleRequest, now time.Time) error {
	if req.Schema != LifecycleSchema || req.TransactionID == "" || req.ForestID == "" || req.IdempotencyKey == "" || req.Authority == "" || req.Capability == "" || req.Generation == 0 || req.Deadline.IsZero() || !validLifecycleAction(req.Action) {
		return errors.New("malformed lifecycle envelope")
	}
	if now.UTC().After(req.Deadline.UTC()) {
		return errors.New("lifecycle envelope expired")
	}
	if req.Authority != g.Authority || req.Capability != g.Capability {
		return errors.New("unauthorized lifecycle envelope")
	}
	if req.Generation != g.Generation {
		return errors.New("wrong forest generation")
	}
	if g.Seen == nil {
		g.Seen = map[string]string{}
	}
	d := lifecycleDigest(req)
	if prior := g.Seen[req.IdempotencyKey]; prior != "" && prior != d {
		return errors.New("idempotency key replayed with different request")
	}
	g.Seen[req.IdempotencyKey] = d
	return nil
}
func validLifecycleAction(a LifecycleAction) bool {
	switch a {
	case LifecyclePrepare, LifecyclePause, LifecycleCheckpoint, LifecycleRestore, LifecycleResume, LifecycleStop, LifecycleCancel, LifecycleStatus:
		return true
	}
	return false
}
func lifecycleDigest(v any) string {
	b, _ := json.Marshal(v)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

type LifecycleDirBus struct{ Root string }

func (b LifecycleDirBus) Broadcast(req LifecycleRequest, members []string) error {
	if len(members) == 0 {
		return errors.New("no lifecycle members")
	}
	data, err := json.MarshalIndent(req, "", "  ")
	if err != nil {
		return err
	}
	for _, m := range members {
		m = strings.TrimSpace(m)
		if m == "" {
			return errors.New("empty lifecycle member")
		}
		dir := filepath.Join(b.Root, "lifecycle", "requests", safeFleetName(req.TransactionID))
		if err := os.MkdirAll(dir, 0700); err != nil {
			return err
		}
		if err := writeDurable(filepath.Join(dir, safeFleetName(m)+".json"), append(data, '\n')); err != nil {
			return fmt.Errorf("persist lifecycle request for %s: %w", m, err)
		}
	}
	return nil
}
func (b LifecycleDirBus) WriteAck(ack LifecycleAck) error {
	if ack.Schema != LifecycleSchema || ack.TransactionID == "" || ack.MemberID == "" || ack.ForestID == "" || ack.Generation == 0 || ack.At.IsZero() {
		return errors.New("malformed lifecycle ack")
	}
	switch ack.State {
	case AckAccepted, AckCompleted, LifecycleAckRefused, AckUnsupported:
	default:
		return errors.New("invalid lifecycle ack state")
	}
	data, err := json.MarshalIndent(ack, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Join(b.Root, "lifecycle", "acks", safeFleetName(ack.TransactionID))
	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}
	return writeDurable(filepath.Join(dir, safeFleetName(ack.MemberID)+".json"), append(data, '\n'))
}
func (b LifecycleDirBus) ReadAcks(transaction string) ([]LifecycleAck, error) {
	dir := filepath.Join(b.Root, "lifecycle", "acks", safeFleetName(transaction))
	ents, err := os.ReadDir(dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []LifecycleAck
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, err
		}
		var a LifecycleAck
		if json.Unmarshal(raw, &a) != nil {
			return nil, errors.New("malformed durable lifecycle ack")
		}
		out = append(out, a)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].MemberID < out[j].MemberID })
	return out, nil
}
func writeDurable(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0600); err != nil {
		return err
	}
	f, err := os.OpenFile(tmp, os.O_RDWR, 0600)
	if err != nil {
		return err
	}
	if err = f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
func safeFleetName(v string) string {
	v = strings.TrimSpace(v)
	v = strings.ReplaceAll(v, "/", "_")
	v = strings.ReplaceAll(v, "\\", "_")
	v = strings.ReplaceAll(v, "..", "_")
	return v
}
