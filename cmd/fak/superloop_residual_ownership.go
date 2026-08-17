package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func attributeSuperloopResidual(root string, paths []string) (class, owner string) {
	self := strings.TrimSpace(os.Getenv("FAK_SESSION_ID"))
	rows, err := leaseref.NewInDir(root).ClassifyLive(context.Background(), self, time.Now())
	if err != nil {
		rows = nil
	}
	oldest := time.Duration(0)
	for _, path := range paths {
		if info, err := os.Stat(filepath.Join(root, filepath.FromSlash(path))); err == nil {
			if age := time.Since(info.ModTime()); age > oldest {
				oldest = age
			}
		}
	}
	return classifySuperloopResidualOwnership(paths, rows, oldest)
}

func classifySuperloopResidualOwnership(paths []string, rows []leaseref.ClassifiedLease, oldest time.Duration) (class, owner string) {
	allScratch := len(paths) > 0
	for _, path := range paths {
		if !strings.HasPrefix(filepath.ToSlash(strings.TrimSpace(path)), "_scratch/") {
			allScratch = false
		}
	}
	if allScratch {
		return "SCRATCH_REAP", ""
	}
	for _, row := range rows {
		if !dispatchorder.TreesOverlap(paths, row.Record.TreeGlobs) {
			continue
		}
		switch row.Liveness {
		case leaseref.LivenessSelf:
			return "OWNED_RECONCILE", row.Record.SessionID
		case leaseref.LivenessPeerLive:
			return "PEER_ACTIVE", row.Record.SessionID
		case leaseref.LivenessPeerDead:
			return "ABANDONED_RECOVER", row.Record.SessionID
		default:
			return "PEER_ACTIVE", row.Record.SessionID
		}
	}
	if oldest >= time.Hour {
		return "ABANDONED_RECOVER", ""
	}
	return "PEER_ACTIVE", ""
}

func ownerSuffix(owner string) string {
	if strings.TrimSpace(owner) == "" {
		return ""
	}
	return " (owner " + owner + ")"
}
