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
	"time"
)

const Schema = "fak.managed-harness/v1"

type ProductID string
type ReleaseID string
type InstallationID string
type GenerationID string

type Product struct {
	ID            ProductID `json:"id"`
	Variant       string    `json:"variant"`
	Compatibility string    `json:"compatibility"`
	Capabilities  []string  `json:"capabilities"`
	Layers        []string  `json:"layers"`
}
type Provenance struct{ Source, Revision, Builder string }
type Release struct {
	ID         ReleaseID  `json:"id"`
	Product    Product    `json:"product"`
	Digest     string     `json:"digest"`
	Provenance Provenance `json:"provenance"`
}
type Generation struct {
	ID          GenerationID `json:"id"`
	Release     Release      `json:"release"`
	ActivatedAt time.Time    `json:"activated_at"`
}
type Installation struct {
	ID            InstallationID `json:"id"`
	Desired       ReleaseID      `json:"desired"`
	Effective     GenerationID   `json:"effective"`
	LastKnownGood GenerationID   `json:"last_known_good"`
	Generations   []GenerationID `json:"generations"`
}
type Receipt struct {
	Schema        string         `json:"schema"`
	Installation  InstallationID `json:"installation"`
	Action        string         `json:"action"`
	Desired       ReleaseID      `json:"desired,omitempty"`
	Before        GenerationID   `json:"before,omitempty"`
	After         GenerationID   `json:"after,omitempty"`
	LastKnownGood GenerationID   `json:"last_known_good,omitempty"`
	Status        string         `json:"status"`
	Capabilities  []string       `json:"capabilities,omitempty"`
	Output        string         `json:"output,omitempty"`
	Reason        string         `json:"reason,omitempty"`
}

type Bundle struct {
	Release Release         `json:"release"`
	Payload json.RawMessage `json:"payload"`
}
type Health func(Bundle) error
type Work func(Bundle) (string, error)

type Store struct{ root string }

func Open(root string) (*Store, error) {
	if root == "" {
		return nil, errors.New("managed harness: root required")
	}
	for _, d := range []string{"releases", "installations"} {
		if err := os.MkdirAll(filepath.Join(root, d), 0700); err != nil {
			return nil, err
		}
	}
	return &Store{root: root}, nil
}

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
func (s *Store) Publish(b Bundle) (Receipt, error) {
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
func (s *Store) Install(id InstallationID, release ReleaseID, health Health) (Receipt, error) {
	if id == "" {
		return Receipt{}, errors.New("managed harness: installation id required")
	}
	if _, err := s.readInstallation(id); err == nil {
		return Receipt{}, errors.New("managed harness: installation exists")
	}
	return s.activate(id, release, health, true)
}
func (s *Store) Update(id InstallationID, release ReleaseID, health Health) (Receipt, error) {
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
func (s *Store) Run(id InstallationID, work Work) (Receipt, error) {
	inst, err := s.readInstallation(id)
	if err != nil {
		return Receipt{}, err
	}
	b, err := s.currentBundle(inst)
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
func (s *Store) Inspect(id InstallationID) (Installation, error) { return s.readInstallation(id) }
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
	err := readJSON(filepath.Join(s.root, "installations", string(id), "state.json"), &i)
	return i, err
}
func (s *Store) currentBundle(i Installation) (Bundle, error) {
	for n := len(i.Generations) - 1; n >= 0; n-- {
		var g Generation
		if err := readJSON(filepath.Join(s.root, "installations", string(i.ID), "generations", string(i.Generations[n])+".json"), &g); err == nil && g.ID == i.Effective {
			return s.readRelease(g.Release.ID)
		}
	}
	return Bundle{}, errors.New("managed harness: effective generation missing")
}
func (s *Store) writeGeneration(id InstallationID, g Generation) error {
	blob, _ := json.MarshalIndent(g, "", "  ")
	return atomicWrite(filepath.Join(s.root, "installations", string(id), "generations", string(g.ID)+".json"), blob)
}
func (s *Store) writeInstallation(i Installation) error {
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
