package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	prepushSuccessReuseTTL    = 24 * time.Hour
	prepushSuccessMaxFiles    = 64
	prepushGateContract       = "immutable-range-affected-importer-build/v2"
	prepushLegacyGateContract = "unqualified-legacy-build/v1"
)

type prepushSuccessReceipt struct {
	Schema       string            `json:"schema"`
	Tip          string            `json:"tip"`
	GateContract string            `json:"gate_contract"`
	CompletedAt  time.Time         `json:"completed_at"`
	BaseSha      string            `json:"base_sha,omitempty"`
	Environment  string            `json:"environment,omitempty"`
	Result       *trunkBuildResult `json:"result,omitempty"`
}

var (
	prepushSuccessReceiptMu sync.Mutex
	prepushSuccessCommonDir = discoverGitCommonDir
	prepushSuccessSleep     = time.Sleep
	prepushExecutableOnce   sync.Once
	prepushExecutableDigest string
)

func prepushSuccessReceiptPath(root, tip string) string {
	commonDir := prepushSuccessCommonDir(root)
	if commonDir == "" || tip == "" {
		return ""
	}
	return filepath.Join(commonDir, "fak-prepush-success", tip+".json")
}

func prepushSuccessReusable(root, tip string, now time.Time) bool {
	path := prepushSuccessReceiptPath(root, tip)
	if path == "" || tip == "" {
		return false
	}
	prepushSuccessReceiptMu.Lock()
	defer prepushSuccessReceiptMu.Unlock()
	b, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	var receipt prepushSuccessReceipt
	if json.Unmarshal(b, &receipt) != nil || receipt.Schema != "fak-prepush-success/2" || receipt.Tip != tip || receipt.GateContract != prepushLegacyGateContract {
		return false
	}
	age := now.Sub(receipt.CompletedAt)
	return age >= 0 && age <= prepushSuccessReuseTTL
}

func prunePrepushSuccessReceipts(dir string, now time.Time) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	type candidate struct {
		path string
		mod  time.Time
	}
	keep := make([]candidate, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if now.Sub(info.ModTime()) > prepushSuccessReuseTTL {
			_ = os.Remove(path)
			continue
		}
		keep = append(keep, candidate{path: path, mod: info.ModTime()})
	}
	sort.Slice(keep, func(i, j int) bool { return keep[i].mod.After(keep[j].mod) })
	if len(keep) <= prepushSuccessMaxFiles {
		return
	}
	for _, row := range keep[prepushSuccessMaxFiles:] {
		_ = os.Remove(row.path)
	}
}

func recordPrepushSuccess(root, tip string, now time.Time) {
	// Legacy producer: retained for commit-tree evidence, but without range and
	// build coverage it is never accepted by loadPrepushBuild.
	writePrepushSuccess(root, prepushSuccessReceipt{Schema: "fak-prepush-success/2", Tip: tip, GateContract: prepushLegacyGateContract, CompletedAt: now.UTC()}, now)
}

func writePrepushSuccess(root string, receipt prepushSuccessReceipt, now time.Time) {
	path := prepushSuccessReceiptPath(root, receipt.Tip)
	if path == "" || receipt.Tip == "" {
		return
	}
	prepushSuccessReceiptMu.Lock()
	defer prepushSuccessReceiptMu.Unlock()
	if os.MkdirAll(filepath.Dir(path), 0o755) != nil {
		return
	}
	prunePrepushSuccessReceipts(filepath.Dir(path), now)
	b, err := json.Marshal(receipt)
	if err != nil {
		return
	}
	tmp := fmt.Sprintf("%s.%d.tmp", path, os.Getpid())
	if os.WriteFile(tmp, b, 0o644) == nil {
		_ = os.Rename(tmp, path)
	}
	_ = os.Remove(tmp)
}

func prepushCommitPathsCovered(root, tip string) bool {
	out, err := gitOut(root, "diff-tree", "--no-commit-id", "--name-only", "-r", tip)
	if err != nil {
		return false
	}
	for _, line := range strings.Split(out, "\n") {
		p := filepath.ToSlash(strings.TrimSpace(line))
		if p == "" {
			continue
		}
		if !strings.HasSuffix(p, ".go") && p != "go.mod" && p != "go.sum" {
			return false
		}
	}
	return true
}

func recordPrepushSuccessForTree(root, tree string, now time.Time) {
	if strings.TrimSpace(tree) == "" {
		return
	}
	recordPrepushSuccess(root, "tree-"+tree, now)
}

func prepushTreeSuccessReusable(root, tree string, now time.Time) bool {
	return strings.TrimSpace(tree) != "" && prepushSuccessReusable(root, "tree-"+tree, now)
}
