package ops

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

// FleetOpsSchemaV1 is the canonical schema tag for fleet ops manifests (#12513).
const FleetOpsSchemaV1 = "fak.fleet.ops/v1"

// Typed errors returned by fleet ops manifest validation.
var (
	ErrSchemaMismatch     = errors.New("invalid or missing schema: expected " + FleetOpsSchemaV1)
	ErrEmptyNodeID        = errors.New("node id cannot be empty: NODE_ID_EMPTY")
	ErrDuplicateNodeID    = errors.New("duplicate node id: NODE_ALREADY_EXISTS")
	ErrEmptyNodeHost      = errors.New("node host cannot be empty")
	ErrInvalidOSTarget    = errors.New("invalid OS target: INVALID_OS_TARGET")
	ErrEmptyTaskID        = errors.New("task id cannot be empty")
	ErrUnknownNode        = errors.New("task references unknown node: UNKNOWN_NODE")
	ErrInvalidInterval    = errors.New("task interval must be positive: INVALID_INTERVAL")
	ErrInvalidTimeout     = errors.New("task timeout must be positive: INVALID_TIMEOUT")
	ErrFleetLaneCollision = errors.New("task lane collision detected: FLEET_LANE_COLLISION")
)

// validOSTargets defines the supported operating systems for fleet nodes.
var validOSTargets = map[string]bool{
	"linux":   true,
	"windows": true,
	"darwin":  true,
}

// Duration is a wrapper around time.Duration that supports both string ("15m", "30s") and numeric (seconds) JSON unmarshaling.
type Duration time.Duration

// Duration returns the underlying time.Duration.
func (d Duration) Duration() time.Duration {
	return time.Duration(d)
}

// String returns the formatted duration string.
func (d Duration) String() string {
	return time.Duration(d).String()
}

// MarshalJSON serializes the duration as a formatted string.
func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(time.Duration(d).String())
}

// UnmarshalJSON parses duration from either string or numeric seconds representation.
func (d *Duration) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		dur, parseErr := time.ParseDuration(s)
		if parseErr != nil {
			return parseErr
		}
		*d = Duration(dur)
		return nil
	}

	var n float64
	if err := json.Unmarshal(b, &n); err == nil {
		*d = Duration(time.Duration(n * float64(time.Second)))
		return nil
	}

	return errors.New("invalid duration format: expected string (e.g. '15m') or number of seconds")
}

// RunWindowSpec defines an optional execution time window for a scheduled task.
type RunWindowSpec struct {
	Start string `json:"start,omitempty"` // e.g. "02:00" (HH:MM 24h)
	End   string `json:"end,omitempty"`   // e.g. "06:00" (HH:MM 24h)
	Cron  string `json:"cron,omitempty"`
}

// Overlaps reports whether two run windows may overlap in time.
// If either window is nil or unbounded, they are assumed to overlap (fail-closed).
func (w *RunWindowSpec) Overlaps(other *RunWindowSpec) bool {
	if w == nil || other == nil {
		return true
	}
	if w.Start == "" || w.End == "" || other.Start == "" || other.End == "" {
		return true
	}
	startA, err1 := parseHourMinute(w.Start)
	endA, err2 := parseHourMinute(w.End)
	startB, err3 := parseHourMinute(other.Start)
	endB, err4 := parseHourMinute(other.End)
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
		return true
	}
	if endA <= startA {
		endA += 24 * 60
	}
	if endB <= startB {
		endB += 24 * 60
	}
	return !(endA <= startB || endB <= startA)
}

func parseHourMinute(s string) (int, error) {
	var h, m int
	if _, err := fmt.Sscanf(s, "%d:%d", &h, &m); err != nil {
		return 0, err
	}
	if h < 0 || h > 23 || m < 0 || m > 59 {
		return 0, fmt.Errorf("time out of bounds: %s", s)
	}
	return h*60 + m, nil
}

// FleetNodeSpec describes a single physical or virtual machine in the ops fleet.
type FleetNodeSpec struct {
	ID           string   `json:"id"`
	Host         string   `json:"host"`
	Role         string   `json:"role,omitempty"`
	OS           string   `json:"os"`
	Lanes        []string `json:"lanes,omitempty"`
	Capabilities []string `json:"capabilities,omitempty"`
	Tags         []string `json:"tags,omitempty"`
}

// FleetScheduleTaskSpec describes an autonomous task scheduled on a fleet node.
type FleetScheduleTaskSpec struct {
	ID        string         `json:"id"`
	Name      string         `json:"name,omitempty"`
	NodeID    string         `json:"node_id"`
	Command   string         `json:"command,omitempty"`
	Interval  Duration       `json:"interval"`
	Timeout   Duration       `json:"timeout,omitempty"`
	LaneScope []string       `json:"lane_scope,omitempty"`
	Active    bool           `json:"active,omitempty"`
	RunWindow *RunWindowSpec `json:"run_window,omitempty"`
}

// FleetOpsManifest is the declarative manifest for multi-node operations (fak.fleet.ops/v1).
type FleetOpsManifest struct {
	Schema    string                  `json:"schema"`
	ClusterID string                  `json:"cluster_id,omitempty"`
	Version   string                  `json:"version,omitempty"`
	Nodes     []FleetNodeSpec         `json:"nodes"`
	Tasks     []FleetScheduleTaskSpec `json:"tasks"`
}

// NewFleetOpsManifest initializes an empty FleetOpsManifest with canonical schema.
func NewFleetOpsManifest(clusterID string) *FleetOpsManifest {
	return &FleetOpsManifest{
		Schema:    FleetOpsSchemaV1,
		ClusterID: clusterID,
		Nodes:     make([]FleetNodeSpec, 0),
		Tasks:     make([]FleetScheduleTaskSpec, 0),
	}
}

// AddNode adds a node to the manifest, rejecting duplicates with NODE_ALREADY_EXISTS.
func (m *FleetOpsManifest) AddNode(node FleetNodeSpec) error {
	if strings.TrimSpace(node.ID) == "" {
		return ErrEmptyNodeID
	}
	for _, n := range m.Nodes {
		if strings.EqualFold(n.ID, node.ID) {
			return fmt.Errorf("%w: node %q already exists in manifest", ErrDuplicateNodeID, node.ID)
		}
	}
	if node.OS != "" && !validOSTargets[strings.ToLower(node.OS)] {
		return fmt.Errorf("%w: %q (expected linux, windows, or darwin)", ErrInvalidOSTarget, node.OS)
	}
	m.Nodes = append(m.Nodes, node)
	return nil
}

// IngestNodeCapabilities updates capabilities for an existing node in the manifest.
func (m *FleetOpsManifest) IngestNodeCapabilities(nodeID string, capabilities []string) error {
	for i, n := range m.Nodes {
		if strings.EqualFold(n.ID, nodeID) {
			m.Nodes[i].Capabilities = append([]string(nil), capabilities...)
			return nil
		}
	}
	return fmt.Errorf("%w: %q", ErrUnknownNode, nodeID)
}

// Validate checks structural integrity and invariants of the manifest.
func (m *FleetOpsManifest) Validate() error {
	if m.Schema != FleetOpsSchemaV1 {
		return fmt.Errorf("%w (got %q)", ErrSchemaMismatch, m.Schema)
	}

	nodeIDs := make(map[string]bool)
	for _, node := range m.Nodes {
		if strings.TrimSpace(node.ID) == "" {
			return ErrEmptyNodeID
		}
		idLower := strings.ToLower(node.ID)
		if nodeIDs[idLower] {
			return fmt.Errorf("%w: %s", ErrDuplicateNodeID, node.ID)
		}
		nodeIDs[idLower] = true

		if strings.TrimSpace(node.Host) == "" {
			return fmt.Errorf("%w for node %q", ErrEmptyNodeHost, node.ID)
		}

		if node.OS != "" && !validOSTargets[strings.ToLower(node.OS)] {
			return fmt.Errorf("%w: %q on node %q", ErrInvalidOSTarget, node.OS, node.ID)
		}
	}

	for _, task := range m.Tasks {
		if strings.TrimSpace(task.ID) == "" {
			return ErrEmptyTaskID
		}
		if !nodeIDs[strings.ToLower(task.NodeID)] {
			return fmt.Errorf("%w: task %q references missing node %q", ErrUnknownNode, task.ID, task.NodeID)
		}
		if task.Interval.Duration() <= 0 {
			return fmt.Errorf("%w on task %q (%v)", ErrInvalidInterval, task.ID, task.Interval)
		}
		if task.Timeout.Duration() < 0 {
			return fmt.Errorf("%w on task %q (%v)", ErrInvalidTimeout, task.ID, task.Timeout)
		}
	}

	return m.ValidateDisjointLanes()
}

// ValidateDisjointLanes checks that tasks with overlapping run windows do not collide on lane scopes.
func (m *FleetOpsManifest) ValidateDisjointLanes() error {
	for i := 0; i < len(m.Tasks); i++ {
		taskA := m.Tasks[i]
		if len(taskA.LaneScope) == 0 {
			continue
		}
		for j := i + 1; j < len(m.Tasks); j++ {
			taskB := m.Tasks[j]
			if len(taskB.LaneScope) == 0 {
				continue
			}
			if taskA.RunWindow.Overlaps(taskB.RunWindow) {
				for _, laneA := range taskA.LaneScope {
					for _, laneB := range taskB.LaneScope {
						if laneadmit.LanesConflict(laneA, laneB) {
							return fmt.Errorf("%w: task %q and task %q conflict on lane %q vs %q",
								ErrFleetLaneCollision, taskA.ID, taskB.ID, laneA, laneB)
						}
					}
				}
			}
		}
	}
	return nil
}

// NodeProbeResult holds discovered host capabilities from a remote probe.
type NodeProbeResult struct {
	Reachable    bool     `json:"reachable"`
	OS           string   `json:"os"`
	Silicon      string   `json:"silicon,omitempty"`
	UmaGB        int      `json:"uma_gb,omitempty"`
	Engine       string   `json:"engine,omitempty"`
	Capabilities []string `json:"capabilities,omitempty"`
}

// NodeProber checks reachability and capabilities on a remote host.
type NodeProber interface {
	ProbeNode(ctx context.Context, host string) (*NodeProbeResult, error)
}

// LoadFleetOpsManifest reads and unmarshals a FleetOpsManifest from a file.
func LoadFleetOpsManifest(path string) (*FleetOpsManifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseFleetOpsManifest(data)
}

// ParseFleetOpsManifest unmarshals and parses a manifest from JSON bytes.
func ParseFleetOpsManifest(data []byte) (*FleetOpsManifest, error) {
	var m FleetOpsManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// Save writes the manifest formatted as JSON to path.
func (m *FleetOpsManifest) Save(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

// DefaultFleetOpsManifestPath returns the canonical location for fleet-ops.json.
func DefaultFleetOpsManifestPath(repoRoot string) string {
	if repoRoot != "" {
		return filepath.Join(repoRoot, ".fak", "fleet-ops.json")
	}
	return filepath.Join(".fak", "fleet-ops.json")
}
