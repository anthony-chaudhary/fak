package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentqueue"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

const (
	dispatchAgentQueueBindingSchema = "fak.agentqueue.dispatch.v1"
	dispatchAgentQueueBindingSuffix = ".agentqueue"
	dispatchAgentQueueLandSuffix    = ".agentqueue-land.json"
)

// dispatchAgentQueueBinding is written before the wrapper starts. Its exact log
// stem and nonce prevent a later issue-level witness from completing a different
// attempt. A missing or torn binding leaves the attempt held for reconciliation.
type dispatchAgentQueueBinding struct {
	Schema       string `json:"schema"`
	StatePath    string `json:"state_path"`
	AttemptID    string `json:"attempt_id"`
	Nonce        string `json:"nonce"`
	Issue        int    `json:"issue"`
	Lane         string `json:"lane"`
	Stem         string `json:"stem"`
	WorktreePath string `json:"worktree_path,omitempty"`
}

func writeDispatchAgentQueueBinding(stem, statePath, attemptID, nonce string, issue int, lane, worktreePath string) error {
	if !filepath.IsAbs(stem) || !filepath.IsAbs(statePath) || issue <= 0 || attemptID == "" || nonce == "" || lane == "" {
		return errors.New("agentqueue dispatch binding requires absolute stem/state and complete launch identity")
	}
	binding := dispatchAgentQueueBinding{
		Schema: dispatchAgentQueueBindingSchema, StatePath: filepath.Clean(statePath),
		AttemptID: attemptID, Nonce: nonce, Issue: issue, Lane: lane, Stem: filepath.Clean(stem),
	}
	if worktreePath != "" {
		if !filepath.IsAbs(worktreePath) || !workerworktree.IsWorkerWorktree(worktreePath) {
			return errors.New("agentqueue dispatch binding requires a managed worktree path")
		}
		binding.WorktreePath = filepath.Clean(worktreePath)
	}
	raw, err := json.Marshal(binding)
	if err != nil {
		return err
	}
	path := stem + dispatchAgentQueueBindingSuffix
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("create agentqueue dispatch binding: %w", err)
	}
	defer f.Close()
	if _, err = f.Write(append(raw, '\n')); err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		_ = os.Remove(path)
		return fmt.Errorf("persist agentqueue dispatch binding: %w", err)
	}
	return nil
}

// resolveDispatchAgentQueueWitness reads the durable witness back before it
// changes queue state. A prior green witness is safe to replay after a crash
// between writing .witness and updating the queue snapshot.
func resolveDispatchAgentQueueWitness(stem string, rec dispatchtick.WitnessRecord, _ map[string]any) error {
	path := stem + dispatchAgentQueueBindingSuffix
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil // older, non-queue worker
	}
	if err != nil {
		return fmt.Errorf("read agentqueue dispatch binding: %w", err)
	}
	var binding dispatchAgentQueueBinding
	if err := json.Unmarshal(raw, &binding); err != nil {
		return fmt.Errorf("decode agentqueue dispatch binding: %w", err)
	}
	if binding.Schema != dispatchAgentQueueBindingSchema || !filepath.IsAbs(binding.StatePath) ||
		binding.AttemptID == "" || binding.Nonce == "" || binding.Issue <= 0 || binding.Lane == "" ||
		!filepath.IsAbs(binding.Stem) || filepath.Clean(binding.Stem) != filepath.Clean(stem) ||
		(binding.WorktreePath != "" && (!filepath.IsAbs(binding.WorktreePath) || !workerworktree.IsWorkerWorktree(binding.WorktreePath))) {
		return errors.New("agentqueue dispatch binding is incomplete or belongs to another log")
	}
	if rec.Issue != binding.Issue || rec.Log != filepath.Base(stem)+".log" {
		return errors.New("agentqueue dispatch witness does not match bound issue and log")
	}
	// Historical logs are scanned on every tick. Avoid reacquiring the queue
	// snapshot lock once a successful CAS has a durable local completion marker.
	donePath := stem + dispatchAgentQueueBindingSuffix + ".done"
	if _, err := os.Stat(donePath); err == nil {
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("stat agentqueue completion marker: %w", err)
	}
	pid, ok := readPID(stem + ".pid")
	if !ok {
		pid, ok = registeredDispatchAgentQueuePID(binding)
		if !ok {
			return errors.New("agentqueue dispatch witness has no durable wrapper PID")
		}
	}
	var durable struct {
		Issue        int                    `json:"issue"`
		Log          string                 `json:"log"`
		SHA          string                 `json:"sha"`
		Claim        string                 `json:"claim"`
		Verdict      string                 `json:"verdict"`
		Witness      string                 `json:"witness"`
		TestClaim    string                 `json:"test_claim"`
		WorktreeLand *workerworktree.Result `json:"worktree_land"`
	}
	witnessRaw, err := os.ReadFile(stem + dispatchtick.WitnessSidecarSuffix)
	if err != nil {
		return fmt.Errorf("read durable agentqueue work witness: %w", err)
	}
	if err := json.Unmarshal(witnessRaw, &durable); err != nil {
		return fmt.Errorf("decode durable agentqueue work witness: %w", err)
	}
	if durable.Issue != rec.Issue || durable.Log != rec.Log || durable.SHA != rec.SHA ||
		durable.Claim != rec.Claim || durable.Verdict != rec.Verdict ||
		durable.Witness != rec.Witness || durable.TestClaim != rec.TestClaim {
		return errors.New("agentqueue work witness changed before queue resolution")
	}
	if durable.Claim != dispatchtick.ClaimWitnessed || !dispatchtick.CommitWitnessed(durable.Verdict, durable.Witness) ||
		durable.TestClaim != dispatchtick.ClaimTestGreen || strings.TrimSpace(durable.SHA) == "" {
		return nil // no terminal green proof; the attempt remains held
	}
	if binding.WorktreePath != "" {
		land := durable.WorktreeLand
		if land == nil || !land.OK || !land.Committed || land.CommitSHA == "" || !strings.EqualFold(land.CommitSHA, durable.SHA) || land.DroppedOutOfLane != 0 ||
			(land.Path != "" && filepath.Clean(land.Path) != binding.WorktreePath) {
			return nil // a commit audit cannot override an unproven worktree land
		}
	}
	proof := agentqueue.WorkWitnessProof{
		Issue: durable.Issue, PID: pid, LogStem: filepath.Clean(stem), SHA: durable.SHA,
		Claim: durable.Claim, Verdict: durable.Verdict, Witness: durable.Witness, TestClaim: durable.TestClaim,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resolved, err := agentqueue.FileStore(binding.StatePath).ResolveWitnessHeld(ctx, binding.AttemptID, binding.Nonce, proof)
	if err != nil {
		return fmt.Errorf("resolve agentqueue held witness: %w", err)
	}
	doneRaw, err := json.Marshal(map[string]string{"attempt_id": binding.AttemptID, "nonce": binding.Nonce, "witness_digest": resolved.WitnessDigest})
	if err != nil {
		return err
	}
	if err := writeFileAtomic(donePath, append(doneRaw, '\n'), 0o600); err != nil {
		return fmt.Errorf("persist agentqueue completion marker: %w", err)
	}
	return nil
}

// readDispatchAgentQueuePID recovers the exact wrapper identity from the queue
// snapshot if the short-lived launcher's post-Start .pid write was interrupted.
func readDispatchAgentQueuePID(stem string) (int, bool) {
	raw, err := os.ReadFile(stem + dispatchAgentQueueBindingSuffix)
	if err != nil {
		return 0, false
	}
	var binding dispatchAgentQueueBinding
	if json.Unmarshal(raw, &binding) != nil || binding.Schema != dispatchAgentQueueBindingSchema ||
		!filepath.IsAbs(binding.StatePath) || !filepath.IsAbs(binding.Stem) ||
		filepath.Clean(binding.Stem) != filepath.Clean(stem) {
		return 0, false
	}
	return registeredDispatchAgentQueuePID(binding)
}

func registeredDispatchAgentQueuePID(binding dispatchAgentQueueBinding) (int, bool) {
	snapshot, err := agentqueue.FileStore(binding.StatePath).Load()
	if err != nil {
		return 0, false
	}
	for _, attempt := range snapshot.Attempts {
		if attempt.ID != binding.AttemptID || attempt.Nonce != binding.Nonce || attempt.PID <= 0 || attempt.StartedAt.IsZero() {
			continue
		}
		for _, intent := range snapshot.Intents {
			if intent.ID == attempt.IntentID && intent.Launch.Issue == binding.Issue && intent.Launch.Lane == binding.Lane {
				return attempt.PID, true
			}
		}
	}
	return 0, false
}
