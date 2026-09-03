package main

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"
)

const tuiGuardGrantTTL = 15 * time.Minute

// tuiGuardGrantReceipt identifies the single overlay mutation owned by one
// confirmed denial-row action. Undo uses the installed expiry as a compare token
// so it cannot remove a later replacement of the same exact grant.
type tuiGuardGrantReceipt struct {
	OverlayPath string `json:"overlay_path"`
	Tool        string `json:"tool"`
	ExpiresAt   string `json:"expires_at,omitempty"`
	Added       bool   `json:"added"`
}

func applyTUIGuardGrant(overlayPath, tool string, now time.Time) (tuiGuardGrantReceipt, error) {
	tool = strings.TrimSpace(tool)
	receipt := tuiGuardGrantReceipt{OverlayPath: overlayPath, Tool: tool}
	if strings.TrimSpace(overlayPath) == "" {
		return receipt, errors.New("TUI guard grant overlay path is empty")
	}
	if !exactLaunchToolName.MatchString(tool) || strings.Contains(tool, "..") {
		return receipt, errors.New("TUI guard grant requires one literal tool name without wildcard, pattern, or traversal syntax")
	}

	overlay, err := loadGuardAllowOverlay(overlayPath)
	if err != nil {
		return receipt, err
	}
	if slices.Contains(overlay.Allow, tool) {
		return receipt, nil
	}
	if slices.Contains(overlay.AllowPrefix, tool) {
		return receipt, fmt.Errorf("TUI guard grant %q conflicts with an existing prefix entry", tool)
	}

	overlay.Allow = append(overlay.Allow, tool)
	receipt.ExpiresAt = guardAllowStampExpiry(&overlay, []string{tool}, tuiGuardGrantTTL, now)
	receipt.Added = true
	if err := saveGuardAllowOverlay(overlayPath, overlay); err != nil {
		return tuiGuardGrantReceipt{}, err
	}
	return receipt, nil
}

func undoTUIGuardGrant(receipt tuiGuardGrantReceipt) error {
	if !receipt.Added {
		return nil
	}
	if strings.TrimSpace(receipt.OverlayPath) == "" || strings.TrimSpace(receipt.Tool) == "" || strings.TrimSpace(receipt.ExpiresAt) == "" {
		return errors.New("TUI guard grant receipt does not identify an owned overlay mutation")
	}

	overlay, err := loadGuardAllowOverlay(receipt.OverlayPath)
	if err != nil {
		return err
	}
	if !slices.Contains(overlay.Allow, receipt.Tool) {
		return nil
	}
	if overlay.Expiry[receipt.Tool] != receipt.ExpiresAt {
		return fmt.Errorf("TUI guard grant %q changed after this action; refusing stale undo", receipt.Tool)
	}

	guardAllowRemove(&overlay, []string{receipt.Tool})
	return saveGuardAllowOverlay(receipt.OverlayPath, overlay)
}
