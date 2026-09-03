package localbench

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"sort"
	"strings"
	"sync"
	"time"
)

type ModerationState string

const (
	ModerationPending     ModerationState = "pending"
	ModerationAccepted    ModerationState = "accepted"
	ModerationRejected    ModerationState = "rejected"
	ModerationQuarantined ModerationState = "quarantined"
	ModerationWithdrawn   ModerationState = "withdrawn"
)

// ModerationEvent records one auditable moderation transition.
type ModerationEvent struct {
	Timestamp string          `json:"timestamp"`
	Actor     string          `json:"actor"`
	FromState ModerationState `json:"from_state"`
	ToState   ModerationState `json:"to_state"`
	Reason    string          `json:"reason"`
}

// ScoreboardEntry represents a stored community benchmark receipt submission.
type ScoreboardEntry struct {
	ID          string            `json:"id"`
	State       ModerationState   `json:"state"`
	SubmittedAt string            `json:"submitted_at"`
	Receipt     Receipt           `json:"receipt"`
	Attestation *Attestation      `json:"attestation,omitempty"`
	TrustStatus TrustStatus       `json:"trust_status"`
	History     []ModerationEvent `json:"history"`
	Tags        []string          `json:"tags,omitempty"`
}

// PublicProjection is the privacy-safe, HTML-escaped public view of a benchmark run.
// It deliberately omits raw child output, private paths, secrets, and machine IDs.
type PublicProjection struct {
	ID                  string          `json:"id"`
	State               ModerationState `json:"state"`
	Benchmark           string          `json:"benchmark"`
	Workload            string          `json:"workload"`
	Engine              string          `json:"engine"`
	Backend             string          `json:"backend"`
	ModelName           string          `json:"model_name"`
	QuantFormat         string          `json:"quant_format"`
	QualityPass         bool            `json:"quality_pass"`
	QualityEvalKind     string          `json:"quality_eval_kind"`
	HardwareOS          string          `json:"hardware_os"`
	HardwareArch        string          `json:"hardware_arch"`
	HardwareCPU         string          `json:"hardware_cpu"`
	HardwareMemoryBytes uint64          `json:"hardware_memory_bytes"`
	Accelerators        []Accelerator   `json:"accelerators,omitempty"`
	DurationMS          int64           `json:"duration_ms"`
	ExitStatus          int             `json:"exit_status"`
	FakVersion          string          `json:"fak_version"`
	FakRevision         string          `json:"fak_revision,omitempty"`
	TrustStatus         TrustStatus     `json:"trust_status"`
	Attested            bool            `json:"attested"`
	SubmittedAt         string          `json:"submitted_at"`
}

// FilterCriteria defines filtering across declared compatible envelopes.
type FilterCriteria struct {
	State        ModerationState
	Benchmark    string
	Engine       string
	Backend      string
	HardwareOS   string
	HardwareArch string
	ModelName    string
}

// Scoreboard manages community benchmark receipt intake, moderation,
// deduplication, and privacy-safe public projections.
type Scoreboard struct {
	mu              sync.RWMutex
	maxPayloadBytes int64
	entries         map[string]*ScoreboardEntry
	trustStore      *TrustStore
}

func NewScoreboard(trustStore *TrustStore) *Scoreboard {
	return &Scoreboard{
		maxPayloadBytes: 1024 * 1024, // 1 MiB default limit
		entries:         make(map[string]*ScoreboardEntry),
		trustStore:      trustStore,
	}
}

func (sb *Scoreboard) SetMaxPayloadBytes(n int64) {
	sb.mu.Lock()
	defer sb.mu.Unlock()
	sb.maxPayloadBytes = n
}

// ComputeDedupKey generates a deterministic deduplication ID for a benchmark receipt.
func ComputeDedupKey(r Receipt) string {
	raw := fmt.Sprintf("dedup:v1:%s:%s:%s", r.Integrity.SHA256, r.Benchmark, r.Engine)
	sum := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(sum[:])
}

// Intake processes and validates raw JSON representing either a v1 receipt or
// an attestation envelope. Re-submitting an identical receipt is idempotent.
func (sb *Scoreboard) Intake(data []byte, submittedAt time.Time) (*ScoreboardEntry, bool, error) {
	sb.mu.Lock()
	defer sb.mu.Unlock()

	if int64(len(data)) > sb.maxPayloadBytes {
		return nil, false, fmt.Errorf("payload size %d exceeds limit %d bytes", len(data), sb.maxPayloadBytes)
	}

	var probe struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, false, fmt.Errorf("malformed JSON payload: %w", err)
	}

	var receipt Receipt
	var attestation *Attestation
	trustStatus := TrustUnsigned

	switch probe.Schema {
	case receiptSchema:
		dec := json.NewDecoder(bytes.NewReader(data))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&receipt); err != nil {
			return nil, false, fmt.Errorf("decoding receipt: %w", err)
		}
		var trailing any
		if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
			return nil, false, errors.New("receipt contains trailing JSON data")
		}
		if err := verify(receipt); err != nil {
			return nil, false, fmt.Errorf("server-side receipt integrity verification failed: %w", err)
		}

	case AttestationSchema:
		var env AttestationEnvelope
		dec := json.NewDecoder(bytes.NewReader(data))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&env); err != nil {
			return nil, false, fmt.Errorf("decoding attestation envelope: %w", err)
		}
		var trailing any
		if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
			return nil, false, errors.New("envelope contains trailing JSON data")
		}
		status, err := VerifyAttestation(env, sb.trustStore, submittedAt)
		if err != nil && status == TrustInvalid {
			return nil, false, fmt.Errorf("attestation verification failed: %w", err)
		}
		receipt = env.Receipt
		attCopy := env.Attestation
		attestation = &attCopy
		trustStatus = status

	default:
		return nil, false, fmt.Errorf("unsupported receipt schema %q", probe.Schema)
	}

	id := ComputeDedupKey(receipt)

	// Deduplication: re-submitting an existing receipt is idempotent.
	if existing, ok := sb.entries[id]; ok {
		return existing, true, nil
	}

	if submittedAt.IsZero() {
		submittedAt = time.Now().UTC()
	}

	entry := &ScoreboardEntry{
		ID:          id,
		State:       ModerationPending,
		SubmittedAt: submittedAt.UTC().Format(time.RFC3339),
		Receipt:     receipt,
		Attestation: attestation,
		TrustStatus: trustStatus,
		History: []ModerationEvent{
			{
				Timestamp: submittedAt.UTC().Format(time.RFC3339),
				Actor:     "system/intake",
				FromState: "",
				ToState:   ModerationPending,
				Reason:    "initial submission accepted into pending queue",
			},
		},
	}

	sb.entries[id] = entry
	return entry, false, nil
}

// Moderate executes a state transition with an auditable reason and actor trail.
func (sb *Scoreboard) Moderate(id string, toState ModerationState, actor, reason string, at time.Time) (*ScoreboardEntry, error) {
	sb.mu.Lock()
	defer sb.mu.Unlock()

	entry, ok := sb.entries[id]
	if !ok {
		return nil, fmt.Errorf("entry %q not found", id)
	}
	if strings.TrimSpace(actor) == "" {
		return nil, errors.New("moderation actor cannot be empty")
	}
	if strings.TrimSpace(reason) == "" {
		return nil, errors.New("moderation reason cannot be empty")
	}
	if entry.State == toState {
		return nil, fmt.Errorf("entry %q is already in state %q", id, toState)
	}

	// Validate permitted state transitions
	if err := validateTransition(entry.State, toState); err != nil {
		return nil, err
	}

	if at.IsZero() {
		at = time.Now().UTC()
	}

	event := ModerationEvent{
		Timestamp: at.UTC().Format(time.RFC3339),
		Actor:     actor,
		FromState: entry.State,
		ToState:   toState,
		Reason:    reason,
	}

	entry.State = toState
	entry.History = append(entry.History, event)
	return entry, nil
}

func validateTransition(from, to ModerationState) error {
	switch from {
	case ModerationPending:
		if to == ModerationAccepted || to == ModerationRejected || to == ModerationQuarantined || to == ModerationWithdrawn {
			return nil
		}
	case ModerationAccepted:
		if to == ModerationQuarantined || to == ModerationWithdrawn || to == ModerationRejected {
			return nil
		}
	case ModerationQuarantined:
		if to == ModerationAccepted || to == ModerationRejected || to == ModerationWithdrawn {
			return nil
		}
	case ModerationRejected:
		if to == ModerationPending || to == ModerationQuarantined || to == ModerationWithdrawn {
			return nil
		}
	case ModerationWithdrawn:
		if to == ModerationPending || to == ModerationQuarantined {
			return nil
		}
	}
	return fmt.Errorf("invalid moderation transition from %q to %q", from, to)
}

// GetEntry retrieves the stored entry by ID.
func (sb *Scoreboard) GetEntry(id string) (*ScoreboardEntry, error) {
	sb.mu.RLock()
	defer sb.mu.RUnlock()

	entry, ok := sb.entries[id]
	if !ok {
		return nil, fmt.Errorf("entry %q not found", id)
	}
	return entry, nil
}

// Project produces a privacy-safe, HTML-escaped public view of an entry.
func (sb *Scoreboard) Project(id string) (*PublicProjection, error) {
	sb.mu.RLock()
	defer sb.mu.RUnlock()

	entry, ok := sb.entries[id]
	if !ok {
		return nil, fmt.Errorf("entry %q not found", id)
	}

	return projectEntry(entry), nil
}

// ListProjections returns all public projections matching the provided filter criteria.
func (sb *Scoreboard) ListProjections(filter FilterCriteria) []PublicProjection {
	sb.mu.RLock()
	defer sb.mu.RUnlock()

	var results []PublicProjection
	for _, entry := range sb.entries {
		p := projectEntry(entry)

		if filter.State != "" && p.State != filter.State {
			continue
		}
		if filter.Benchmark != "" && !strings.EqualFold(p.Benchmark, filter.Benchmark) {
			continue
		}
		if filter.Engine != "" && !strings.EqualFold(p.Engine, filter.Engine) {
			continue
		}
		if filter.Backend != "" && !strings.EqualFold(p.Backend, filter.Backend) {
			continue
		}
		if filter.HardwareOS != "" && !strings.EqualFold(p.HardwareOS, filter.HardwareOS) {
			continue
		}
		if filter.HardwareArch != "" && !strings.EqualFold(p.HardwareArch, filter.HardwareArch) {
			continue
		}
		if filter.ModelName != "" && !strings.EqualFold(p.ModelName, filter.ModelName) {
			continue
		}

		results = append(results, *p)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].SubmittedAt > results[j].SubmittedAt
	})
	return results
}

func projectEntry(entry *ScoreboardEntry) *PublicProjection {
	r := entry.Receipt

	workload := "unspecified"
	backend := "unspecified"
	modelName := "unspecified"
	quantFormat := "unspecified"
	qualityPass := false
	qualityEval := "unspecified"
	attested := false

	if entry.Attestation != nil {
		attested = true
		b := entry.Attestation.Bindings
		if b.Benchmark.Workload != "" {
			workload = b.Benchmark.Workload
		}
		if b.Execution.Backend != "" {
			backend = b.Execution.Backend
		}
		if b.ModelArtifact.Name != "" {
			modelName = b.ModelArtifact.Name
		}
		if b.Quantization.Format != "" {
			quantFormat = b.Quantization.Format
		}
		qualityPass = b.Quality.Passed
		if b.Quality.EvalKind != "" {
			qualityEval = b.Quality.EvalKind
		}
	} else if len(r.Hardware.Accelerators) > 0 {
		backend = r.Hardware.Accelerators[0].Backend
	}

	// Sanitize text fields against HTML/script injection
	proj := &PublicProjection{
		ID:                  html.EscapeString(entry.ID),
		State:               entry.State,
		Benchmark:           html.EscapeString(displayUnset(r.Benchmark)),
		Workload:            html.EscapeString(workload),
		Engine:              html.EscapeString(displayUnset(r.Engine)),
		Backend:             html.EscapeString(backend),
		ModelName:           html.EscapeString(modelName),
		QuantFormat:         html.EscapeString(quantFormat),
		QualityPass:         qualityPass,
		QualityEvalKind:     html.EscapeString(qualityEval),
		HardwareOS:          html.EscapeString(r.Hardware.OS),
		HardwareArch:        html.EscapeString(r.Hardware.Arch),
		HardwareCPU:         html.EscapeString(r.Hardware.CPU),
		HardwareMemoryBytes: r.Hardware.MemoryBytes,
		Accelerators:        r.Hardware.Accelerators,
		DurationMS:          r.DurationMS,
		ExitStatus:          r.ExitStatus,
		FakVersion:          html.EscapeString(r.Provenance.FakVersion),
		FakRevision:         html.EscapeString(r.Provenance.FakRevision),
		TrustStatus:         entry.TrustStatus,
		Attested:            attested,
		SubmittedAt:         entry.SubmittedAt,
	}

	return proj
}
