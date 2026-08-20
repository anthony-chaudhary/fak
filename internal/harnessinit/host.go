package harnessinit

import (
	"encoding/json"
	"fmt"
	"strings"
)

const (
	HostManifestPath = "product.json"
	HostLockPath     = "product.lock.json"
)

type hostArtifacts struct {
	host     string
	manifest string
	lock     string
}

func normalizeHostArtifacts(host, manifest, lock string) (hostArtifacts, error) {
	host = strings.ToLower(strings.TrimSpace(host))
	if host == "" {
		if strings.TrimSpace(manifest) != "" || strings.TrimSpace(lock) != "" {
			return hostArtifacts{}, fmt.Errorf("--host is required when host artifacts are supplied")
		}
		return hostArtifacts{}, nil
	}
	if host != "codex" && host != "claude" {
		return hostArtifacts{}, fmt.Errorf("unsupported --host %q (want codex|claude)", host)
	}
	if strings.TrimSpace(manifest) == "" || !json.Valid([]byte(manifest)) {
		return hostArtifacts{}, fmt.Errorf("host %q requires a valid resolved manifest", host)
	}
	if strings.TrimSpace(lock) == "" || !json.Valid([]byte(lock)) {
		return hostArtifacts{}, fmt.Errorf("host %q requires a valid resolved lock", host)
	}
	return hostArtifacts{host: host, manifest: manifest, lock: lock}, nil
}
