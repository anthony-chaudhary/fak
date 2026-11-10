// Package machineid is a hostname-independent machine identity. The hostname
// drifts with the network (one Mac appeared under five host strings), so a
// lock or record that must say "this machine" carries this id instead.
package machineid

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"
)

var (
	once   sync.Once
	cached string
	// source is the seam tests replace.
	source         = Source
	platformUUIDRE = regexp.MustCompile(`"IOPlatformUUID"\s*=\s*"([0-9A-Fa-f-]+)"`)
)

// ID is this machine's hashed identity, resolved once, or "" when none can be
// read.
func ID() string {
	once.Do(func() { cached = Hash(source()) })
	return cached
}

// Hash is 16 hex of sha256 over the trimmed, uppercased raw platform id, or ""
// for an empty one.
func Hash(raw string) string {
	raw = strings.ToUpper(strings.TrimSpace(raw))
	if raw == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(sum[:8])
}

// Source is the raw platform id: IOPlatformUUID on macOS, the systemd machine
// id on Linux, else "".
func Source() string {
	switch runtime.GOOS {
	case "darwin":
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		out, err := exec.CommandContext(ctx, "/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice").Output()
		if err != nil {
			return ""
		}
		if m := platformUUIDRE.FindSubmatch(out); len(m) == 2 {
			return string(m[1])
		}
	case "linux":
		for _, p := range []string{"/etc/machine-id", "/var/lib/dbus/machine-id"} {
			if b, err := os.ReadFile(p); err == nil && strings.TrimSpace(string(b)) != "" {
				return string(b)
			}
		}
	}
	return ""
}
