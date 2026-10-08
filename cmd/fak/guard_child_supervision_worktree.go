package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

const guardOrchestrationWorktreeLifecycleSchema = "fak.orchestration_worktree_lifecycle.v1"

type guardOrchestrationWorktreeMetadata struct {
	Root        string
	Path        string
	BaseSHA     string
	Trees       []string
	ReceiptPath string
}

type guardOrchestrationWorktreeLifecycleReceipt struct {
	Schema      string                `json:"schema"`
	Status      string                `json:"status"`
	Code        string                `json:"code,omitempty"`
	Reason      string                `json:"reason,omitempty"`
	Root        string                `json:"root"`
	Worktree    string                `json:"worktree"`
	BaseSHA     string                `json:"base_sha"`
	Trees       []string              `json:"trees"`
	Land        workerworktree.Result `json:"land,omitempty"`
	Reap        workerworktree.Result `json:"reap,omitempty"`
	CompletedAt time.Time             `json:"completed_at"`
}

var guardOrchestrationWorktreeLand = func(meta guardOrchestrationWorktreeMetadata) workerworktree.Result {
	return landWorkerWorktreeVerified(meta.Root, meta.Path, meta.BaseSHA, meta.Trees, nil)
}

var guardOrchestrationWorktreeOwnerProcessLive = workerworktree.OwnerProcessLive

var guardOrchestrationWorktreeReap = func(root, path string) workerworktree.Result {
	return workerworktree.ReapChecked(root, path, "", nil)
}

func guardOrchestrationWorktreeMetadataFromEnv() (guardOrchestrationWorktreeMetadata, bool, error) {
	path := strings.TrimSpace(os.Getenv(orchestrationWorktreePathEnv))
	if path == "" {
		return guardOrchestrationWorktreeMetadata{}, false, nil
	}
	meta := guardOrchestrationWorktreeMetadata{
		Root:        filepath.Clean(strings.TrimSpace(os.Getenv(orchestrationWorktreeRootEnv))),
		Path:        filepath.Clean(path),
		BaseSHA:     strings.TrimSpace(os.Getenv(orchestrationWorktreeBaseSHAEnv)),
		ReceiptPath: filepath.Clean(strings.TrimSpace(os.Getenv(orchestrationWorktreeLifecycleReceiptEnv))),
	}
	if err := json.Unmarshal([]byte(os.Getenv(orchestrationWorktreeTreeEnv)), &meta.Trees); err != nil {
		return meta, true, fmt.Errorf("ORCHESTRATION_WORKTREE_METADATA_INVALID: decode declared tree: %w", err)
	}
	if meta.Root == "." || meta.Path == "." || meta.ReceiptPath == "." || meta.BaseSHA == "" || len(meta.Trees) == 0 ||
		!filepath.IsAbs(meta.Root) || !filepath.IsAbs(meta.Path) || !filepath.IsAbs(meta.ReceiptPath) || !workerworktree.IsWorkerWorktree(meta.Path) {
		return meta, true, fmt.Errorf("ORCHESTRATION_WORKTREE_METADATA_INVALID: root, managed path, base SHA, declared tree, and receipt are required")
	}
	for _, tree := range meta.Trees {
		clean := strings.TrimSuffix(filepath.ToSlash(strings.TrimSpace(tree)), "/**")
		if clean == "" || filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, "../") {
			return meta, true, fmt.Errorf("ORCHESTRATION_WORKTREE_METADATA_INVALID: unsafe declared tree %q", tree)
		}
	}
	return meta, true, nil
}

func writeGuardOrchestrationWorktreeReceipt(meta guardOrchestrationWorktreeMetadata, status, reason string, land, reap workerworktree.Result) error {
	receipt := guardOrchestrationWorktreeLifecycleReceipt{
		Schema: guardOrchestrationWorktreeLifecycleSchema, Status: status, Reason: strings.TrimSpace(reason),
		Root: meta.Root, Worktree: meta.Path, BaseSHA: meta.BaseSHA, Trees: append([]string(nil), meta.Trees...),
		Land: land, Reap: reap, CompletedAt: time.Now().UTC(),
	}
	if prefix, _, ok := strings.Cut(receipt.Reason, ":"); ok && strings.HasPrefix(prefix, "ORCHESTRATION_WORKTREE_") {
		receipt.Code = prefix
	}
	raw, err := json.Marshal(receipt)
	if err != nil {
		return err
	}
	return writeFileAtomic(meta.ReceiptPath, append(raw, '\n'), 0o600)
}

func guardFinalizeOrchestrationWorktree(runErr error, childClean bool, gatewayErr error) error {
	meta, present, err := guardOrchestrationWorktreeMetadataFromEnv()
	if !present {
		return nil
	}
	if err != nil {
		return err
	}
	owned, known := guardOrchestrationWorktreeOwnerProcessLive(meta.Path, func(pid int) bool { return pid == os.Getpid() })
	if !known || !owned {
		return fmt.Errorf("ORCHESTRATION_WORKTREE_OWNER_REFUSED: managed worktree owner stamp is missing or does not name this guard process; worktree preserved at %s", meta.Path)
	}
	if runErr != nil || !childClean {
		reason := "ORCHESTRATION_WORKTREE_CHILD_FAILED: managed worktree preserved; retry or reconcile with fak worktree worker land"
		if runErr != nil {
			reason += ": " + runErr.Error()
		}
		return writeGuardOrchestrationWorktreeReceipt(meta, "preserved", reason, workerworktree.Result{}, workerworktree.Result{})
	}
	if gatewayErr != nil && !errors.Is(gatewayErr, http.ErrServerClosed) && !errors.Is(gatewayErr, context.Canceled) {
		reason := "ORCHESTRATION_WORKTREE_GATEWAY_FAILED: managed worktree preserved; restore the guard gateway before landing: " + gatewayErr.Error()
		return writeGuardOrchestrationWorktreeReceipt(meta, "preserved", reason, workerworktree.Result{}, workerworktree.Result{})
	}
	land := guardOrchestrationWorktreeLand(meta)
	if !land.OK {
		reason := "ORCHESTRATION_WORKTREE_LAND_FAILED: managed worktree preserved; reconcile with fak worktree worker land"
		if land.Reason != "" {
			reason += ": " + land.Reason
		}
		if err := writeGuardOrchestrationWorktreeReceipt(meta, "preserved", reason, land, workerworktree.Result{}); err != nil {
			return fmt.Errorf("%s; write lifecycle receipt: %w", reason, err)
		}
		return errors.New(reason)
	}
	if land.DroppedOutOfLane > 0 {
		reason := fmt.Sprintf("ORCHESTRATION_WORKTREE_POLICY_VIOLATION: land dropped %d out-of-lane path(s); worktree preserved at %s", land.DroppedOutOfLane, meta.Path)
		if err := writeGuardOrchestrationWorktreeReceipt(meta, "landed_preserved", reason, land, workerworktree.Result{}); err != nil {
			return fmt.Errorf("%s; write lifecycle receipt: %w", reason, err)
		}
		return errors.New(reason)
	}
	reap := guardOrchestrationWorktreeReap(meta.Root, meta.Path)
	if !reap.OK {
		reason := "ORCHESTRATION_WORKTREE_REAP_FAILED: land succeeded but managed worktree remains; inspect with fak worktree worker list --json"
		if err := writeGuardOrchestrationWorktreeReceipt(meta, "landed_preserved", reason, land, reap); err != nil {
			return fmt.Errorf("%s; write lifecycle receipt: %w", reason, err)
		}
		return errors.New(reason)
	}
	if err := writeGuardOrchestrationWorktreeReceipt(meta, "landed_reaped", "", land, reap); err != nil {
		return fmt.Errorf("ORCHESTRATION_WORKTREE_RECEIPT_FAILED: %w", err)
	}
	return nil
}
