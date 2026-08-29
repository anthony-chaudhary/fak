package model

import (
	"encoding/json"
	"os"
	"strings"
	"sync"
	"time"
)

// PrefixProfileEvent attributes prefix-cache ownership and transfer work. Events
// are emitted only when SetPrefixProfilePath names a JSONL file, keeping the normal
// decode path free of clocks and I/O.
type PrefixProfileEvent struct {
	Schema        string    `json:"schema"`
	At            time.Time `json:"at"`
	Operation     string    `json:"operation"`
	Component     string    `json:"component"`
	DurationNS    int64     `json:"duration_ns"`
	HostBytes     int64     `json:"host_bytes"`
	DeviceBytes   int64     `json:"device_bytes"`
	TransferBytes int64     `json:"transfer_bytes"`
	MetadataBytes int64     `json:"metadata_bytes"`
	Tokens        int       `json:"tokens"`
	Backend       string    `json:"backend,omitempty"`
}

var prefixProfile = struct {
	sync.Mutex
	path string
}{}

// SetPrefixProfilePath explicitly configures the optional prefix-profile JSONL sink.
// An empty path disables profiling. The setting is process-wide because prefix snapshots
// are shared across model sessions; callers must configure it before serving requests.
func SetPrefixProfilePath(path string) {
	prefixProfile.Lock()
	prefixProfile.path = strings.TrimSpace(path)
	prefixProfile.Unlock()
}

func prefixProfileStart() time.Time {
	if prefixProfile.path == "" {
		return time.Time{}
	}
	return time.Now()
}

func emitPrefixProfile(start time.Time, operation, component string, p *PrefixSnapshot, h *HostPrefixSnapshot) {
	if start.IsZero() {
		return
	}
	e := PrefixProfileEvent{Schema: "fak.prefix-profile/1", At: time.Now().UTC(), Operation: operation, Component: component, DurationNS: time.Since(start).Nanoseconds()}
	if p != nil {
		e.HostBytes, e.DeviceBytes = p.ResidencyBytes()
		e.MetadataBytes = p.TokenLineageMetadataBytes()
		e.Tokens = p.Tokens
		if p.Backend != nil {
			e.Backend = p.Backend.Name()
		}
	}
	if h != nil {
		e.HostBytes = h.ResidentBytes()
		e.TransferBytes = h.TransferBytes()
		e.MetadataBytes = h.TokenLineageMetadataBytes()
		e.Tokens = h.Tokens()
		if h.backend != nil {
			e.Backend = h.backend.Name()
		}
	}
	b, err := json.Marshal(e)
	if err != nil {
		return
	}
	prefixProfile.Lock()
	defer prefixProfile.Unlock()
	f, err := os.OpenFile(prefixProfile.path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return
	}
	_, _ = f.Write(append(b, '\n'))
	_ = f.Close()
}
