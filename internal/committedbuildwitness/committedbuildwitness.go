// Package committedbuildwitness shares successful immutable-HEAD build evidence
// between repository gates and dispatch preflight.
package committedbuildwitness

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	schema = "fak-committed-build-success/1"
	TTL    = 24 * time.Hour
)

type receipt struct {
	Schema      string    `json:"schema"`
	Head        string    `json:"head"`
	Producer    string    `json:"producer"`
	CompletedAt time.Time `json:"completed_at"`
}

// Record persists a successful committed-tree build witness for head. Failures
// are best-effort: an unwritable cache makes the next consumer rebuild.
func Record(root, head, producer string, now time.Time) {
	path := witnessPath(root, head)
	if path == "" || strings.TrimSpace(producer) == "" {
		return
	}
	if os.MkdirAll(filepath.Dir(path), 0o755) != nil {
		return
	}
	body, err := json.Marshal(receipt{Schema: schema, Head: head, Producer: producer, CompletedAt: now.UTC()})
	if err != nil {
		return
	}
	tmp := fmt.Sprintf("%s.%d.tmp", path, os.Getpid())
	if os.WriteFile(tmp, body, 0o644) == nil {
		_ = os.Rename(tmp, path)
	}
	_ = os.Remove(tmp)
}

// Fresh reports whether a successful committed-tree gate witnessed this exact
// immutable HEAD recently enough to reuse.
func Fresh(root, head string, now time.Time) bool {
	path := witnessPath(root, head)
	if path == "" {
		return false
	}
	body, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	var got receipt
	if json.Unmarshal(body, &got) != nil || got.Schema != schema || got.Head != head || strings.TrimSpace(got.Producer) == "" {
		return false
	}
	age := now.Sub(got.CompletedAt)
	return age >= 0 && age <= TTL
}

func witnessPath(root, head string) string {
	head = strings.TrimSpace(head)
	if root == "" || head == "" || strings.ContainsAny(head, `/\\`) {
		return ""
	}
	cmd := windowgate.Command("git", "-C", root, "rev-parse", "--git-common-dir")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	common := strings.TrimSpace(string(out))
	if common == "" {
		return ""
	}
	if !filepath.IsAbs(common) {
		common = filepath.Join(root, common)
	}
	return filepath.Join(filepath.Clean(common), "fak-committed-build-success", head+".json")
}
