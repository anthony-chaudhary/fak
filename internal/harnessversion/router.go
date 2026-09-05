package harnessversion

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"sort"
	"strings"
	"sync"
)

// HeaderHarnessVersion is the HTTP header inspected for explicit wire version negotiation.
const HeaderHarnessVersion = "X-Fak-Harness-Version"

// HarnessVersion represents a version identifier for a sub-harness runtime.
type HarnessVersion string

const (
	VersionV1 HarnessVersion = "v1"
	VersionV2 HarnessVersion = "v2"
)

// VersionDescriptor specifies configuration, traffic weight, and status for a harness version.
type VersionDescriptor struct {
	Version  string            `json:"version"`
	Weight   int               `json:"weight"`
	Active   bool              `json:"active"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// StickySessionRouter manages version registration, sticky session pinning,
// and canary traffic distribution across sub-harness runtime versions.
type StickySessionRouter struct {
	mu             sync.RWMutex
	versions       map[string]VersionDescriptor
	pinnedSessions map[string]string
	defaultVersion string
	randFunc       func(n int) int
}

// NewStickySessionRouter creates an initialized StickySessionRouter.
func NewStickySessionRouter() *StickySessionRouter {
	return &StickySessionRouter{
		versions:       make(map[string]VersionDescriptor),
		pinnedSessions: make(map[string]string),
	}
}

func (s *StickySessionRouter) initLocked() {
	if s.versions == nil {
		s.versions = make(map[string]VersionDescriptor)
	}
	if s.pinnedSessions == nil {
		s.pinnedSessions = make(map[string]string)
	}
}

// Register adds or updates a harness version descriptor in the router.
func (s *StickySessionRouter) Register(descriptor VersionDescriptor) error {
	trimmedVersion := strings.TrimSpace(descriptor.Version)
	if trimmedVersion == "" {
		return errors.New("version descriptor version cannot be empty")
	}
	if descriptor.Weight < 0 {
		return fmt.Errorf("version descriptor weight cannot be negative: %d", descriptor.Weight)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.initLocked()

	var metaCopy map[string]string
	if descriptor.Metadata != nil {
		metaCopy = make(map[string]string, len(descriptor.Metadata))
		for k, v := range descriptor.Metadata {
			metaCopy[k] = v
		}
	}

	s.versions[trimmedVersion] = VersionDescriptor{
		Version:  trimmedVersion,
		Weight:   descriptor.Weight,
		Active:   descriptor.Active,
		Metadata: metaCopy,
	}

	if descriptor.Active {
		isExplicitDefault := metaCopy["default"] == "true" || metaCopy["stable"] == "true"
		if isExplicitDefault || s.defaultVersion == "" {
			s.defaultVersion = trimmedVersion
		}
	} else if s.defaultVersion == trimmedVersion {
		s.defaultVersion = ""
	}

	return nil
}

// SetDefaultVersion explicitly configures the default stable version.
func (s *StickySessionRouter) SetDefaultVersion(version string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initLocked()

	desc, ok := s.versions[version]
	if !ok {
		return fmt.Errorf("cannot set unregistered version %q as default", version)
	}
	if !desc.Active {
		return fmt.Errorf("cannot set inactive version %q as default", version)
	}
	s.defaultVersion = version
	return nil
}

// DefaultVersion returns the currently configured or resolved default stable version.
func (s *StickySessionRouter) DefaultVersion() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.getDefaultVersionLocked()
}

func (s *StickySessionRouter) getDefaultVersionLocked() string {
	if s.defaultVersion != "" {
		if desc, ok := s.versions[s.defaultVersion]; ok && desc.Active {
			return s.defaultVersion
		}
	}

	var bestVer string
	var maxWeight = -1
	for ver, desc := range s.versions {
		if !desc.Active {
			continue
		}
		if desc.Weight > maxWeight || (desc.Weight == maxWeight && (bestVer == "" || ver < bestVer)) {
			bestVer = ver
			maxWeight = desc.Weight
		}
	}
	return bestVer
}

// SetRandFunc allows injecting a deterministic RNG for testing canary splits.
func (s *StickySessionRouter) SetRandFunc(fn func(n int) int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.randFunc = fn
}

// Negotiate inspects wire signals (X-Fak-Harness-Version header or path parameter)
// and returns the negotiated version. If wire signals were provided but are invalid
// or unregistered, it fails closed to the default stable version. If no wire signals
// are provided, it returns an empty string to defer to canary selection.
func (s *StickySessionRouter) Negotiate(headerVal string, pathParam string) string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	cleanHeader := strings.TrimSpace(headerVal)
	cleanPath := strings.TrimPrefix(strings.TrimSpace(pathParam), "/")
	if idx := strings.Index(cleanPath, "/"); idx != -1 {
		candidate := cleanPath[:idx]
		if desc, ok := s.versions[candidate]; ok && desc.Active {
			cleanPath = candidate
		}
	}

	if cleanHeader == "" && cleanPath == "" {
		return ""
	}

	if cleanHeader != "" {
		if desc, ok := s.versions[cleanHeader]; ok && desc.Active {
			return desc.Version
		}
	} else if cleanPath != "" {
		if desc, ok := s.versions[cleanPath]; ok && desc.Active {
			return desc.Version
		}
	}

	return s.getDefaultVersionLocked()
}

// Route resolves the harness version for a session and request signals.
// It returns (selectedVersion, wasPinned).
// 1. If sessionID is already pinned, returns the pinned version with wasPinned = true.
// 2. Else if header or path requests a valid registered version, pins and returns that with wasPinned = false.
// 3. Else performs canary selection across active versions according to their weights, pins to sessionID, and returns wasPinned = false.
func (s *StickySessionRouter) Route(sessionID string, headerVal string, pathParam string) (string, bool) {
	if sessionID != "" {
		s.mu.RLock()
		if s.pinnedSessions != nil {
			if pinned, ok := s.pinnedSessions[sessionID]; ok {
				s.mu.RUnlock()
				return pinned, true
			}
		}
		s.mu.RUnlock()
	}

	cleanHeader := strings.TrimSpace(headerVal)
	cleanPath := strings.TrimSpace(pathParam)
	hasWireSignal := cleanHeader != "" || cleanPath != ""

	var selected string
	if hasWireSignal {
		selected = s.Negotiate(headerVal, pathParam)
	}

	if selected == "" {
		selected = s.selectCanary(sessionID)
	}

	if selected == "" {
		return "", false
	}

	if sessionID != "" {
		s.mu.Lock()
		s.initLocked()
		if pinned, ok := s.pinnedSessions[sessionID]; ok {
			s.mu.Unlock()
			return pinned, true
		}
		s.pinnedSessions[sessionID] = selected
		s.mu.Unlock()
	}

	return selected, false
}

func (s *StickySessionRouter) selectCanary(sessionID string) string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.versions) == 0 {
		return ""
	}

	active := make([]VersionDescriptor, 0, len(s.versions))
	totalWeight := 0
	for _, desc := range s.versions {
		if desc.Active {
			active = append(active, desc)
			if desc.Weight > 0 {
				totalWeight += desc.Weight
			}
		}
	}

	if len(active) == 0 {
		return ""
	}

	sort.Slice(active, func(i, j int) bool {
		return active[i].Version < active[j].Version
	})

	if totalWeight <= 0 {
		return s.getDefaultVersionLocked()
	}

	var roll int
	if s.randFunc != nil {
		roll = s.randFunc(totalWeight)
	} else {
		roll = rand.IntN(totalWeight)
	}

	if roll < 0 {
		roll = 0
	} else if roll >= totalWeight {
		roll = totalWeight - 1
	}

	curr := 0
	for _, desc := range active {
		if desc.Weight <= 0 {
			continue
		}
		curr += desc.Weight
		if roll < curr {
			return desc.Version
		}
	}

	return active[0].Version
}

// ReleaseSession unpins a session from the sticky session table.
func (s *StickySessionRouter) ReleaseSession(sessionID string) {
	if sessionID == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.pinnedSessions != nil {
		delete(s.pinnedSessions, sessionID)
	}
}

// GetPinnedVersion returns the pinned version for a session if one exists.
func (s *StickySessionRouter) GetPinnedVersion(sessionID string) (string, bool) {
	if sessionID == "" {
		return "", false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.pinnedSessions == nil {
		return "", false
	}
	v, ok := s.pinnedSessions[sessionID]
	return v, ok
}

// ActiveVersions returns all registered versions currently active.
func (s *StickySessionRouter) ActiveVersions() []VersionDescriptor {
	s.mu.RLock()
	defer s.mu.RUnlock()

	active := make([]VersionDescriptor, 0, len(s.versions))
	for _, desc := range s.versions {
		if desc.Active {
			var metaCopy map[string]string
			if desc.Metadata != nil {
				metaCopy = make(map[string]string, len(desc.Metadata))
				for k, v := range desc.Metadata {
					metaCopy[k] = v
				}
			}
			active = append(active, VersionDescriptor{
				Version:  desc.Version,
				Weight:   desc.Weight,
				Active:   desc.Active,
				Metadata: metaCopy,
			})
		}
	}
	sort.Slice(active, func(i, j int) bool {
		return active[i].Version < active[j].Version
	})
	return active
}
