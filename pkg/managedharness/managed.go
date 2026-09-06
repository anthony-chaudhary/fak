// Package managedharness implements a local immutable-generation lifecycle for
// named harness products.
package managedharness

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Schema identifies the managed harness record and receipt specification version.
const Schema = "fak.managed-harness/v1"

// ProductID uniquely identifies a harness product line across variants and versions.
type ProductID string

// ReleaseID identifies an immutable release bundle by its content digest.
type ReleaseID string

// InstallationID identifies a named local deployment instance of a harness product.
type InstallationID string

// GenerationID identifies an activated generation deployment instance for an installation.
type GenerationID string

// PinKind defines the retention policy or lifecycle reason that prevents a generation
// from being reclaimed during garbage collection.
type PinKind string

const (
	// PinStagedCandidate retains a newly staged candidate generation undergoing qualification.
	PinStagedCandidate PinKind = "staged_candidate"
	// PinOpenSession retains a generation currently referenced by an active runtime or agent session.
	PinOpenSession PinKind = "open_session"
	// PinCheckpoint retains a generation marked as an explicit recovery checkpoint.
	PinCheckpoint PinKind = "checkpoint"
)

// GenerationPin records a retention lease that protects a specific generation from GC.
type GenerationPin struct {
	Kind       PinKind      `json:"kind"`
	Reference  string       `json:"reference,omitempty"`
	Generation GenerationID `json:"generation"`
}

// Product describes a harness product's identity, variant, compatibility contract,
// capabilities, and architectural layers.
type Product struct {
	ID            ProductID `json:"id"`
	Variant       string    `json:"variant"`
	Compatibility string    `json:"compatibility"`
	Capabilities  []string  `json:"capabilities"`
	Layers        []string  `json:"layers"`
}

// Provenance records the source repository, revision identifier, and builder
// identity that produced a release.
type Provenance struct{ Source, Revision, Builder string }

// Release is an immutable product release descriptor with its cryptographic digest
// and build provenance.
type Release struct {
	ID         ReleaseID  `json:"id"`
	Product    Product    `json:"product"`
	Digest     string     `json:"digest"`
	Provenance Provenance `json:"provenance"`
}

// Generation represents an activated deployment instance of a release within an
// installation, recording when it became active.
type Generation struct {
	ID          GenerationID `json:"id"`
	Release     Release      `json:"release"`
	ActivatedAt time.Time    `json:"activated_at"`
}

// Installation tracks the current desired release, effective generation,
// rollback target (LastKnownGood), historical generation list, and active retention
// pins for a named product installation.
type Installation struct {
	ID            InstallationID  `json:"id"`
	Desired       ReleaseID       `json:"desired"`
	Effective     GenerationID    `json:"effective"`
	LastKnownGood GenerationID    `json:"last_known_good"`
	Generations   []GenerationID  `json:"generations"`
	Pins          []GenerationPin `json:"pins,omitempty"`
}

// Receipt records the outcome, status, and state transitions of a lifecycle
// operation on an installation or release.
type Receipt struct {
	Schema        string         `json:"schema"`
	Installation  InstallationID `json:"installation"`
	Action        string         `json:"action"`
	Desired       ReleaseID      `json:"desired,omitempty"`
	Before        GenerationID   `json:"before,omitempty"`
	After         GenerationID   `json:"after,omitempty"`
	LastKnownGood GenerationID   `json:"last_known_good,omitempty"`
	PinKind       PinKind        `json:"pin_kind,omitempty"`
	Reference     string         `json:"reference,omitempty"`
	Status        string         `json:"status"`
	Capabilities  []string       `json:"capabilities,omitempty"`
	Output        string         `json:"output,omitempty"`
	Reason        string         `json:"reason,omitempty"`
}

// GCReceipt records the outcome, retained and reclaimed generation lists, and
// disk space reclaimed by a garbage collection operation.
type GCReceipt struct {
	Schema         string         `json:"schema"`
	Installation   InstallationID `json:"installation"`
	Action         string         `json:"action"`
	Status         string         `json:"status"`
	Retained       []GenerationID `json:"retained"`
	Reclaimed      []GenerationID `json:"reclaimed"`
	Failed         []GenerationID `json:"failed,omitempty"`
	RetainedBytes  int64          `json:"retained_bytes"`
	ReclaimedBytes int64          `json:"reclaimed_bytes"`
	Reason         string         `json:"reason,omitempty"`
}

// Bundle pairs an immutable Release descriptor with its serialized JSON payload.
type Bundle struct {
	Release Release         `json:"release"`
	Payload json.RawMessage `json:"payload"`
}

// Health is a validation callback invoked before activating a new generation.
// Returning a non-nil error triggers an automatic rollback.
type Health func(Bundle) error

// Work is an execution function run against an activated generation bundle.
type Work func(Bundle) (string, error)

// Store provides atomic, linearizable persistence for releases, installations,
// generations, and retention pins on a local filesystem root.
type Store struct {
	root string
	mu   *sync.Mutex
}

// storeLocks makes lifecycle writes linearizable across Store handles opened
// for the same local root.
var storeLocks sync.Map

// RetainedGenerations returns the deterministic union of every generation root.
func RetainedGenerations(i Installation) []GenerationID {
	retained := make([]GenerationID, 0, 2+len(i.Pins))
	retained = append(retained, i.Effective, i.LastKnownGood)
	for _, pin := range i.Pins {
		retained = append(retained, pin.Generation)
	}
	return sortedGenerationIDs(retained)
}

// ReclaimableGenerations returns generations that have no current, rollback,
// staged, session, or checkpoint root.
func ReclaimableGenerations(i Installation) []GenerationID {
	retained := make(map[GenerationID]struct{}, 2+len(i.Pins))
	for _, id := range RetainedGenerations(i) {
		retained[id] = struct{}{}
	}
	var reclaimable []GenerationID
	for _, id := range i.Generations {
		if _, ok := retained[id]; !ok {
			reclaimable = append(reclaimable, id)
		}
	}
	return sortedGenerationIDs(reclaimable)
}

func sortedGenerationIDs(in []GenerationID) []GenerationID {
	out := append([]GenerationID(nil), in...)
	sort.Slice(out, func(a, b int) bool { return out[a] < out[b] })
	n := 0
	for _, id := range out {
		if id != "" && (n == 0 || out[n-1] != id) {
			out[n] = id
			n++
		}
	}
	return out[:n]
}

// Open opens or initializes a managed harness Store at root, creating the
// releases and installations directories if they do not exist.
func Open(root string) (*Store, error) {
	if root == "" {
		return nil, errors.New("managed harness: root required")
	}
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	for _, d := range []string{"releases", "installations"} {
		if err := os.MkdirAll(filepath.Join(root, d), 0700); err != nil {
			return nil, err
		}
	}
	lock, _ := storeLocks.LoadOrStore(root, &sync.Mutex{})
	return &Store{root: root, mu: lock.(*sync.Mutex)}, nil
}

// BuildRelease validates product metadata and provenance, normalizes capabilities
// and layers, and constructs an immutable Bundle with its canonical SHA-256 digest.
func BuildRelease(product Product, payload any, prov Provenance) (Bundle, error) {
	if product.ID == "" || product.Variant == "" || product.Compatibility == "" || prov.Source == "" || prov.Revision == "" {
		return Bundle{}, errors.New("managed harness: incomplete release identity")
	}
	product.Capabilities = sortedUnique(product.Capabilities)
	product.Layers = sortedUnique(product.Layers)
	blob, err := json.Marshal(payload)
	if err != nil {
		return Bundle{}, err
	}
	canonical, _ := json.Marshal(struct {
		Product    Product         `json:"product"`
		Payload    json.RawMessage `json:"payload"`
		Provenance Provenance      `json:"provenance"`
	}{product, blob, prov})
	sum := sha256.Sum256(canonical)
	digest := hex.EncodeToString(sum[:])
	return Bundle{Release: Release{ID: ReleaseID(digest[:16]), Product: product, Digest: digest, Provenance: prov}, Payload: blob}, nil
}

// Publish persists an immutable release bundle into the store. It returns an error
// if a release with the same ID exists with different content, or if the bundle
// payload contains forbidden secret keywords.
func (s *Store) Publish(b Bundle) (Receipt, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := verifyBundle(b); err != nil {
		return Receipt{}, err
	}
	path := filepath.Join(s.root, "releases", string(b.Release.ID)+".json")
	blob, _ := json.MarshalIndent(b, "", "  ")
	if old, err := os.ReadFile(path); err == nil {
		if string(old) != string(blob) {
			return Receipt{}, errors.New("managed harness: immutable release collision")
		}
	} else if !os.IsNotExist(err) {
		return Receipt{}, err
	} else if err := atomicWrite(path, blob); err != nil {
		return Receipt{}, err
	}
	return Receipt{Schema: Schema, Action: "release", Desired: b.Release.ID, Status: "published", Capabilities: b.Release.Product.Capabilities}, nil
}

// Install creates a new named installation pointing at the specified release,
// executing an optional health check before activating the initial generation.
func (s *Store) Install(id InstallationID, release ReleaseID, health Health) (Receipt, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if id == "" {
		return Receipt{}, errors.New("managed harness: installation id required")
	}
	if _, err := s.readInstallation(id); err == nil {
		return Receipt{}, errors.New("managed harness: installation exists")
	}
	return s.activate(id, release, health, true)
}

// Update advances an existing installation to a new release. It verifies product
// and compatibility contracts, executes the optional health check, rolls back
// to the previous generation if health fails, and records the previous generation
// as the last known good.
func (s *Store) Update(id InstallationID, release ReleaseID, health Health) (Receipt, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.activate(id, release, health, false)
}
func (s *Store) activate(id InstallationID, release ReleaseID, health Health, create bool) (Receipt, error) {
	b, err := s.readRelease(release)
	if err != nil {
		return Receipt{}, err
	}
	inst, readErr := s.readInstallation(id)
	if create {
		inst = Installation{ID: id}
	} else if readErr != nil {
		return Receipt{}, readErr
	}
	before := inst.Effective
	if !create {
		current, err := s.currentBundle(inst)
		if err != nil {
			return Receipt{}, err
		}
		if current.Release.Product.ID != b.Release.Product.ID || current.Release.Product.Compatibility != b.Release.Product.Compatibility {
			return Receipt{Schema: Schema, Installation: id, Action: "update", Desired: release, Before: before, After: before, LastKnownGood: inst.LastKnownGood, Status: "refused", Reason: "incompatible product or compatibility version"}, nil
		}
	}
	inst.Desired = release
	gen := Generation{ID: GenerationID(string(release) + "-" + fmt.Sprint(len(inst.Generations)+1)), Release: b.Release, ActivatedAt: time.Unix(int64(len(inst.Generations)+1), 0).UTC()}
	if health != nil {
		if err := health(b); err != nil {
			return Receipt{Schema: Schema, Installation: id, Action: "update", Desired: release, Before: before, After: before, LastKnownGood: inst.LastKnownGood, Status: "rolled_back", Reason: err.Error()}, nil
		}
	}
	if before != "" {
		inst.LastKnownGood = before
	}
	inst.Effective = gen.ID
	inst.Generations = append(inst.Generations, gen.ID)
	if err := s.writeGeneration(id, gen); err != nil {
		return Receipt{}, err
	}
	if err := s.writeInstallation(inst); err != nil {
		return Receipt{}, err
	}
	action := "update"
	if create {
		action = "install"
	}
	return Receipt{Schema: Schema, Installation: id, Action: action, Desired: release, Before: before, After: gen.ID, LastKnownGood: inst.LastKnownGood, Status: "activated", Capabilities: b.Release.Product.Capabilities}, nil
}

// Run executes a work function against the currently effective release bundle
// of the specified installation, returning a receipt with the output or error.
func (s *Store) Run(id InstallationID, work Work) (Receipt, error) {
	s.mu.Lock()
	inst, err := s.readInstallation(id)
	if err != nil {
		s.mu.Unlock()
		return Receipt{}, err
	}
	b, err := s.currentBundle(inst)
	s.mu.Unlock()
	if err != nil {
		return Receipt{}, err
	}
	out, err := work(b)
	r := Receipt{Schema: Schema, Installation: id, Action: "work", Before: inst.Effective, After: inst.Effective, LastKnownGood: inst.LastKnownGood, Capabilities: b.Release.Product.Capabilities, Output: out, Status: "completed"}
	if err != nil {
		r.Status = "failed"
		r.Reason = err.Error()
	}
	return r, nil
}

// Inspect reads and returns the current persistent state of an installation.
func (s *Store) Inspect(id InstallationID) (Installation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.readInstallation(id)
}

// Pin retains a staged candidate, open-session generation, or checkpoint
// generation until the corresponding reference is released.
func (s *Store) Pin(id InstallationID, pin GenerationPin) (Receipt, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := validatePin(pin); err != nil {
		return Receipt{}, err
	}
	inst, err := s.readInstallation(id)
	if err != nil {
		return Receipt{}, err
	}
	if !containsGeneration(inst.Generations, pin.Generation) {
		return Receipt{}, fmt.Errorf("managed harness: generation %q is not owned by installation %q", pin.Generation, id)
	}
	if _, err := s.generationFileInfo(id, pin.Generation); err != nil {
		return Receipt{}, err
	}
	var before GenerationID
	replaced := false
	for n := range inst.Pins {
		if inst.Pins[n].Kind == pin.Kind && inst.Pins[n].Reference == pin.Reference {
			before = inst.Pins[n].Generation
			inst.Pins[n] = pin
			replaced = true
			break
		}
	}
	if !replaced {
		inst.Pins = append(inst.Pins, pin)
	}
	sort.Slice(inst.Pins, func(a, b int) bool {
		if inst.Pins[a].Kind != inst.Pins[b].Kind {
			return inst.Pins[a].Kind < inst.Pins[b].Kind
		}
		return inst.Pins[a].Reference < inst.Pins[b].Reference
	})
	if err := s.writeInstallation(inst); err != nil {
		return Receipt{}, err
	}
	return Receipt{Schema: Schema, Installation: id, Action: "pin", Before: before, After: pin.Generation, PinKind: pin.Kind, Reference: pin.Reference, Status: "pinned"}, nil
}

// Unpin releases one staged, session, or checkpoint reference.
func (s *Store) Unpin(id InstallationID, kind PinKind, reference string) (Receipt, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := validatePinKey(kind, reference); err != nil {
		return Receipt{}, err
	}
	inst, err := s.readInstallation(id)
	if err != nil {
		return Receipt{}, err
	}
	var before GenerationID
	kept := inst.Pins[:0]
	for _, pin := range inst.Pins {
		if pin.Kind == kind && pin.Reference == reference {
			before = pin.Generation
			continue
		}
		kept = append(kept, pin)
	}
	if before == "" {
		return Receipt{}, fmt.Errorf("managed harness: %s pin %q not found", kind, reference)
	}
	inst.Pins = kept
	if err := s.writeInstallation(inst); err != nil {
		return Receipt{}, err
	}
	return Receipt{Schema: Schema, Installation: id, Action: "unpin", Before: before, PinKind: kind, Reference: reference, Status: "released"}, nil
}

// GarbageCollect deletes only unpinned generation records owned by one
// installation. Release bundles are intentionally untouched because they may
// be shared by other installations.
func (s *Store) GarbageCollect(id InstallationID) (GCReceipt, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	receipt := GCReceipt{Schema: Schema, Installation: id, Action: "gc", Status: "completed"}
	inst, err := s.readInstallation(id)
	if err != nil {
		return receipt, err
	}
	receipt.Retained = RetainedGenerations(inst)
	for _, generation := range receipt.Retained {
		if !containsGeneration(inst.Generations, generation) {
			receipt.Status = "refused"
			receipt.Reason = fmt.Sprintf("retained generation %q is absent from installation state", generation)
			return receipt, errors.New("managed harness: " + receipt.Reason)
		}
		info, err := s.generationFileInfo(id, generation)
		if err != nil {
			receipt.Status = "refused"
			receipt.Reason = err.Error()
			return receipt, err
		}
		receipt.RetainedBytes += info.Size()
	}

	var failures []string
	for _, generation := range ReclaimableGenerations(inst) {
		path, err := s.generationPath(id, generation)
		if err != nil {
			receipt.Failed = append(receipt.Failed, generation)
			failures = append(failures, err.Error())
			continue
		}
		info, err := os.Lstat(path)
		if err != nil {
			receipt.Failed = append(receipt.Failed, generation)
			failures = append(failures, fmt.Sprintf("generation %q: %v", generation, err))
			continue
		}
		if !info.Mode().IsRegular() {
			receipt.Failed = append(receipt.Failed, generation)
			failures = append(failures, fmt.Sprintf("generation %q: owned artifact is not a regular file", generation))
			continue
		}
		if err := os.Remove(path); err != nil {
			receipt.Failed = append(receipt.Failed, generation)
			failures = append(failures, fmt.Sprintf("generation %q: %v", generation, err))
			continue
		}
		receipt.Reclaimed = append(receipt.Reclaimed, generation)
		receipt.ReclaimedBytes += info.Size()
	}
	if len(receipt.Reclaimed) > 0 {
		reclaimed := make(map[GenerationID]struct{}, len(receipt.Reclaimed))
		for _, generation := range receipt.Reclaimed {
			reclaimed[generation] = struct{}{}
		}
		remaining := inst.Generations[:0]
		for _, generation := range inst.Generations {
			if _, ok := reclaimed[generation]; !ok {
				remaining = append(remaining, generation)
			}
		}
		inst.Generations = remaining
		if err := s.writeInstallation(inst); err != nil {
			receipt.Status = "failed"
			receipt.Reason = err.Error()
			return receipt, err
		}
	}
	if len(failures) > 0 {
		receipt.Status = "partial"
		receipt.Reason = strings.Join(failures, "; ")
		return receipt, fmt.Errorf("managed harness: GC incomplete: %s", receipt.Reason)
	}
	return receipt, nil
}

func validatePin(pin GenerationPin) error {
	if pin.Generation == "" {
		return errors.New("managed harness: pin generation required")
	}
	return validatePinKey(pin.Kind, pin.Reference)
}

func validatePinKey(kind PinKind, reference string) error {
	switch kind {
	case PinStagedCandidate:
		if reference != "" {
			return errors.New("managed harness: staged candidate pin has no reference")
		}
	case PinOpenSession, PinCheckpoint:
		if reference == "" {
			return fmt.Errorf("managed harness: %s pin reference required", kind)
		}
	default:
		return fmt.Errorf("managed harness: unknown pin kind %q", kind)
	}
	return nil
}

func containsGeneration(generations []GenerationID, target GenerationID) bool {
	for _, generation := range generations {
		if generation == target {
			return true
		}
	}
	return false
}

func (s *Store) generationPath(id InstallationID, generation GenerationID) (string, error) {
	if err := safePathComponent("installation", string(id)); err != nil {
		return "", err
	}
	if err := safePathComponent("generation", string(generation)); err != nil {
		return "", err
	}
	return filepath.Join(s.root, "installations", string(id), "generations", string(generation)+".json"), nil
}

func (s *Store) generationFileInfo(id InstallationID, generation GenerationID) (os.FileInfo, error) {
	path, err := s.generationPath(id, generation)
	if err != nil {
		return nil, err
	}
	info, err := os.Lstat(path)
	if err != nil {
		return nil, fmt.Errorf("managed harness: generation %q artifact: %w", generation, err)
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("managed harness: generation %q artifact is not a regular file", generation)
	}
	return info, nil
}

func safePathComponent(kind, value string) error {
	if value == "" || value == "." || value == ".." || strings.ContainsAny(value, "/\\:\x00") {
		return fmt.Errorf("managed harness: invalid %s identity %q", kind, value)
	}
	return nil
}

func (s *Store) readRelease(id ReleaseID) (Bundle, error) {
	var b Bundle
	err := readJSON(filepath.Join(s.root, "releases", string(id)+".json"), &b)
	if err == nil {
		err = verifyBundle(b)
	}
	return b, err
}
func (s *Store) readInstallation(id InstallationID) (Installation, error) {
	var i Installation
	if err := safePathComponent("installation", string(id)); err != nil {
		return i, err
	}
	err := readJSON(filepath.Join(s.root, "installations", string(id), "state.json"), &i)
	return i, err
}
func (s *Store) currentBundle(i Installation) (Bundle, error) {
	for n := len(i.Generations) - 1; n >= 0; n-- {
		var g Generation
		path, err := s.generationPath(i.ID, i.Generations[n])
		if err != nil {
			return Bundle{}, err
		}
		if err := readJSON(path, &g); err == nil && g.ID == i.Effective {
			return s.readRelease(g.Release.ID)
		}
	}
	return Bundle{}, errors.New("managed harness: effective generation missing")
}
func (s *Store) writeGeneration(id InstallationID, g Generation) error {
	blob, _ := json.MarshalIndent(g, "", "  ")
	path, err := s.generationPath(id, g.ID)
	if err != nil {
		return err
	}
	return atomicWrite(path, blob)
}
func (s *Store) writeInstallation(i Installation) error {
	if err := safePathComponent("installation", string(i.ID)); err != nil {
		return err
	}
	blob, _ := json.MarshalIndent(i, "", "  ")
	return atomicWrite(filepath.Join(s.root, "installations", string(i.ID), "state.json"), blob)
}
func verifyBundle(b Bundle) error {
	rebuilt, err := BuildRelease(b.Release.Product, json.RawMessage(b.Payload), b.Release.Provenance)
	if err != nil {
		return err
	}
	if rebuilt.Release.Digest != b.Release.Digest || rebuilt.Release.ID != b.Release.ID {
		return errors.New("managed harness: bundle digest mismatch")
	}
	lower := strings.ToLower(string(b.Payload))
	for _, word := range []string{"secret", "token", "password"} {
		if strings.Contains(lower, `"`+word+`"`) {
			return fmt.Errorf("managed harness: installation secret field %q in release", word)
		}
	}
	return nil
}
func sortedUnique(in []string) []string {
	out := append([]string(nil), in...)
	sort.Strings(out)
	n := 0
	for _, v := range out {
		if v != "" && (n == 0 || out[n-1] != v) {
			out[n] = v
			n++
		}
	}
	return out[:n]
}
func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}
func atomicWrite(path string, b []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
