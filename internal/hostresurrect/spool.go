package hostresurrect

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func Enqueue(dir string, req Request) (string, error) {
	encoded, err := EncodeRequest(req)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	name := safeName(req.EventID) + "-" + safeName(req.Session) + ".request"
	final := filepath.Join(dir, name)
	tmp, err := os.CreateTemp(dir, ".relaunch-*.tmp")
	if err != nil {
		return "", err
	}
	tmpName := tmp.Name()
	ok := false
	defer func() {
		_ = tmp.Close()
		if !ok {
			_ = os.Remove(tmpName)
		}
	}()
	if err := tmp.Chmod(0o600); err != nil {
		return "", err
	}
	if _, err := tmp.WriteString(encoded); err != nil {
		return "", err
	}
	if err := tmp.Sync(); err != nil {
		return "", err
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if err := os.Rename(tmpName, final); err != nil {
		return "", err
	}
	ok = true
	return final, nil
}

func Pending(dir string) ([]string, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.request"))
	if err != nil {
		return nil, err
	}
	sort.Strings(matches)
	return matches, nil
}

func ReadQueued(path string) (Request, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Request{}, err
	}
	return DecodeRequest(strings.TrimSpace(string(b)))
}

func CompleteQueued(path string) error {
	if filepath.Ext(path) != ".request" {
		return fmt.Errorf("not a relaunch request: %s", path)
	}
	return os.Rename(path, strings.TrimSuffix(path, ".request")+".done")
}

func safeName(s string) string {
	var b strings.Builder
	for _, r := range s {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			b.WriteRune(r)
		}
	}
	if b.Len() == 0 {
		return "unknown"
	}
	return b.String()
}
