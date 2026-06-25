// Package sessionimage makes an agent SESSION a first-class, portable, model-agnostic
// VALUE — one self-describing image you can dump, archive, offload, and restore across
// hosts, users, instances, VMs, and a model change, then RESUME where it left off.
//
// # The gap it closes
//
// fak already has the three durable session primitives, but they are DISJOINT and none
// is portable as a unit:
//
//   - internal/session holds the live DRIVE state (run-state / budget / priority / pace),
//     but it is in-memory only — the design note SESSION-CONTROL-STATE-AS-FIRST-CLASS §5
//     fences "No persistence yet … a process restart re-attaches a session at its
//     defaults." (Table.Restore + this package close that fence.)
//   - internal/recall persists a finished session's CONTENT as a durable core image
//     (manifest.json page table + cas.json swap device + index.json access path), with
//     the trust gate enforced on every page-in — but it carries no drive state, no model
//     or account identity, and is query-only, not resume.
//   - internal/trajectory exports the per-turn audit corpus as JSONL — separately again.
//
// A session that lives on a laptop and must move to a server, a lightweight VM, a
// different user's instance, or simply survive a model swap has no single object to
// pick up and carry. This package is that object: it COMPOSES the existing primitives
// (it adds nothing to the frozen ABI and re-implements none of recall's gate) into one
// versioned, integrity-checked bundle plus a single-file archive for offload.
//
// # What an image is
//
// A bundle directory (or its .faksession tar, see archive.go):
//
//	image.json        — this Meta: identity, model/engine/account/residency/host,
//	                    the portability contract, a sha256 integrity index over every
//	                    other part, and the migration log.
//	session.json      — the drive State (the persistence the design note §5 named).
//	manifest.json     — recall page table     ┐ the recall core image (optional: a
//	cas.json          — recall swap device     │ brand-new session has no pages yet),
//	index.json        — recall ctxplan index  ┘ written verbatim by recall.
//	trajectory.jsonl  — the per-turn audit corpus (optional).
//
// # Why it survives a model change (the model-agnostic contract)
//
// The image stores a session's LOGICAL state — drive axes, the content-addressed page
// table, the candidate index over roles+digests, the trajectory of decisions. It
// deliberately stores NO model-specific binary state: no KV cache, no tokenizer-specific
// token ids, no sampler RNG. The KV cache is a CACHE, not state — a resumed session
// rebuilds it on the first turn — so a restore may target a DIFFERENT model and
// re-prefill from the same logical content. That move is recorded as a Migration, so
// "this session ran under model A and resumed under model B on host X" is an audited
// fact, not a silent reinterpretation. Portability.KVIncluded is false for v1 by design.
//
// # What it inherits, unchanged
//
// The load-bearing recall property rides straight through the offload boundary: a slice
// the gate QUARANTINED in the live session stays sealed after the image is packed to a
// tar, shipped to another machine, unpacked into a fresh directory, and reloaded under a
// new model. Restore re-attaches the drive faithfully (a Stopped session reloads
// Stopped). Integrity is content-addressed: every part is sha256-checked on Load, so a
// truncated or tampered offload fails closed rather than resuming a corrupt session.
package sessionimage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// Version is the on-disk portable-image format tag, stamped into image.json. A reader
// refuses an image whose Version it does not recognize (fail closed: a wrong-version
// image is worse than none), mirroring recall.ManifestVersion.
const Version = "fak.session.v1"

// The fixed sibling filenames inside a bundle. image.json is the root (it indexes the
// rest and is not listed in its own Parts); session.json is the drive sibling the design
// note §5 named. ManifestFile / CASFile / IndexFile mirror the names recall.Recorder.
// Persist and recall.PersistIndex already write (recall.IndexFile is exported; the
// manifest/cas names are recall-internal but format-stable at recall.ManifestVersion).
const (
	ImageFile      = "image.json"
	SessionFile    = "session.json"
	ManifestFile   = "manifest.json"
	CASFile        = "cas.json"
	IndexFile      = recall.IndexFile // "index.json"
	TrajectoryFile = "trajectory.jsonl"
)

// knownParts is the deterministic order parts are dumped, integrity-listed, and packed
// in. image.json is excluded (it is the index over these). Only parts that actually
// exist are written and listed; a missing optional part is simply absent.
var knownParts = []string{SessionFile, ManifestFile, CASFile, IndexFile, TrajectoryFile}

// Portability declares what a restored image guarantees and what it deliberately drops —
// the model-change contract, made explicit so an operator never assumes a KV cache rode
// along. For v1 the content is always model-agnostic and the KV is never included.
type Portability struct {
	ContentModelAgnostic bool   `json:"content_model_agnostic"` // logical content only — re-prefillable on any model
	KVIncluded           bool   `json:"kv_included"`            // false for v1: the KV cache is a cache, rebuilt on resume
	Note                 string `json:"note,omitempty"`
}

// defaultPortability is the v1 contract.
func defaultPortability(note string) Portability {
	if note == "" {
		note = "logical content only (drive + content-addressed page table + index + trajectory); " +
			"no KV cache, no token ids — a resume may target a different model and re-prefill"
	}
	return Portability{ContentModelAgnostic: true, KVIncluded: false, Note: note}
}

// Part is one sibling file's integrity record: its name, byte length, and sha256 (the
// same content-address scheme recall's CAS uses). On Load every Part is re-hashed and a
// mismatch fails closed, so a truncated or tampered offload can never resume as if whole.
type Part struct {
	Name   string `json:"name"`
	Bytes  int64  `json:"bytes"`
	Digest string `json:"digest"` // sha256 hex over the file bytes
}

// Migration records a model or host change applied when an image was restored — the
// audit trail that makes a re-home or a model swap a fact on the record, not a guess.
type Migration struct {
	WhenUnix  int64  `json:"when_unix"`
	FromModel string `json:"from_model,omitempty"`
	ToModel   string `json:"to_model,omitempty"`
	FromHost  string `json:"from_host,omitempty"`
	ToHost    string `json:"to_host,omitempty"`
	Reason    string `json:"reason,omitempty"`
}

// Meta is the identity, provenance, and integrity index of a session image — the
// content of image.json. SessionID is the recall SessionID and the drive TraceID (one
// key joins all three primitives). Parts is the sha256 index over every other file;
// Migrations grows by one each time the image is restored onto a new model or host.
type Meta struct {
	Version     string            `json:"version"`
	SessionID   string            `json:"session_id"`
	CreatedUnix int64             `json:"created_unix"`
	UpdatedUnix int64             `json:"updated_unix"`
	AppVersion  string            `json:"app_version"`
	Model       string            `json:"model,omitempty"`
	Engine      string            `json:"engine,omitempty"`
	Account     string            `json:"account,omitempty"`
	Residency   string            `json:"residency,omitempty"`
	Host        string            `json:"host,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`
	Portability Portability       `json:"portability"`
	Parts       []Part            `json:"parts"`
	Migrations  []Migration       `json:"migrations,omitempty"`
}

// Input is everything DumpDir needs to write an image. Drive is always written; the
// Recorder (page table + CAS), Index, and Trajectory are each optional — a brand-new
// session may carry only its drive. The descriptive fields (Model/Engine/Account/
// Residency/Host/Labels) are recorded verbatim into Meta so a restored session knows
// where it came from. Now is an injected unix clock for determinism; 0 uses wall time.
type Input struct {
	SessionID  string
	Drive      session.State
	Recorder   *recall.Recorder
	Index      *ctxplan.Index
	Trajectory []trajectory.Turn

	Model     string
	Engine    string
	Account   string
	Residency string
	Host      string
	Labels    map[string]string
	Note      string // optional Portability.Note override

	Now int64 // injected unix seconds; 0 => time.Now().Unix()
}

// Image is a loaded, integrity-verified session image: a handle over a bundle directory
// whose Meta and Drive have been read and whose every part hashed clean. The content
// primitives (recall Session, ctxplan index, trajectory) are paged in on demand through
// the accessors — Load verifies the bytes, Rehydrate brings the session back to life.
type Image struct {
	Dir   string
	Meta  Meta
	Drive session.State
}

// DumpDir writes a portable session image into dir: the drive (session.json), the recall
// core image (manifest.json + cas.json) when a Recorder is given, its ctxplan index
// (index.json) when an Index is given, the trajectory corpus (trajectory.jsonl) when
// any turns are given, and finally image.json — the Meta with a sha256 integrity index
// over every part written. It returns the Meta as persisted.
func DumpDir(dir string, in Input) (Meta, error) {
	id := strings.TrimSpace(in.SessionID)
	if id == "" {
		id = strings.TrimSpace(in.Drive.TraceID)
	}
	if id == "" {
		return Meta{}, fmt.Errorf("sessionimage: DumpDir requires a SessionID (or a Drive.TraceID)")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return Meta{}, err
	}

	// (1) The drive sibling — always present, keyed to the session id so the three
	// primitives share one join key.
	drive := in.Drive
	drive.TraceID = id
	db, err := json.MarshalIndent(drive, "", "  ")
	if err != nil {
		return Meta{}, err
	}
	if err := os.WriteFile(filepath.Join(dir, SessionFile), db, 0o644); err != nil {
		return Meta{}, err
	}

	// (2) The recall core image (manifest.json + cas.json), written by recall itself so
	// the bytes are byte-identical to a recall-native image — a sessionimage IS a recall
	// core image with siblings, reloadable by either reader.
	if in.Recorder != nil {
		if err := in.Recorder.Persist(dir); err != nil {
			return Meta{}, err
		}
	}

	// (3) The ctxplan candidate index (index.json) — a regenerable cache of the page
	// table; absent is never fatal (Rehydrate rebuilds it via recall.LoadOrAttachIndex).
	if in.Index != nil {
		if err := recall.PersistIndex(dir, in.Index); err != nil {
			return Meta{}, err
		}
	}

	// (4) The trajectory corpus (trajectory.jsonl) — the per-turn audit rows, one JSON
	// object per line (the same stable schema trajectory.Recorder.ExportTo writes).
	if len(in.Trajectory) > 0 {
		if err := writeTrajectory(filepath.Join(dir, TrajectoryFile), in.Trajectory); err != nil {
			return Meta{}, err
		}
	}

	// (5) The integrity index over every part now on disk, then image.json last (it is
	// not listed in its own Parts; a corrupt image.json fails the Version/JSON check).
	parts, err := indexParts(dir)
	if err != nil {
		return Meta{}, err
	}
	now := in.Now
	if now == 0 {
		now = time.Now().Unix()
	}
	meta := Meta{
		Version:     Version,
		SessionID:   id,
		CreatedUnix: now,
		UpdatedUnix: now,
		AppVersion:  appversion.Current(),
		Model:       in.Model,
		Engine:      in.Engine,
		Account:     in.Account,
		Residency:   in.Residency,
		Host:        in.Host,
		Labels:      in.Labels,
		Portability: defaultPortability(in.Note),
		Parts:       parts,
	}
	if err := writeImageJSON(dir, meta); err != nil {
		return Meta{}, err
	}
	return meta, nil
}

// writeImageJSON marshals Meta to image.json. Factored out so Rehydrate's write-back of
// an updated migration log reuses the exact serialization.
func writeImageJSON(dir string, meta Meta) error {
	mb, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, ImageFile), mb, 0o644)
}

// indexParts hashes every known sibling that exists in dir, in deterministic order, into
// a Part list. image.json is excluded by construction (it is not in knownParts).
func indexParts(dir string) ([]Part, error) {
	var parts []Part
	for _, name := range knownParts {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			if os.IsNotExist(err) {
				continue // an optional part that this session does not carry
			}
			return nil, err
		}
		parts = append(parts, Part{Name: name, Bytes: int64(len(b)), Digest: recall.Digest(b)})
	}
	return parts, nil
}

// LoadDir reads image.json + session.json from a bundle directory and verifies the
// integrity of every part (size + sha256), failing closed on a version mismatch, a
// missing listed part, or a digest that does not match its bytes. A returned Image is a
// session proven whole — safe to Rehydrate. It does NOT page in the content primitives;
// the accessors and Rehydrate do that on demand.
func LoadDir(dir string) (*Image, error) {
	mb, err := os.ReadFile(filepath.Join(dir, ImageFile))
	if err != nil {
		return nil, err
	}
	var meta Meta
	if err := json.Unmarshal(mb, &meta); err != nil {
		return nil, fmt.Errorf("sessionimage: bad %s: %w", ImageFile, err)
	}
	if meta.Version != Version {
		return nil, fmt.Errorf("sessionimage: image version %q != %q", meta.Version, Version)
	}
	if err := verifyParts(dir, meta.Parts); err != nil {
		return nil, err
	}
	// session.json is verified above as a Part; read it back for the drive record. (It
	// is required: an image always carries a drive, so a missing session.json is a
	// malformed image, not an optional-part absence.)
	sb, err := os.ReadFile(filepath.Join(dir, SessionFile))
	if err != nil {
		return nil, fmt.Errorf("sessionimage: missing %s (a session image must carry a drive): %w", SessionFile, err)
	}
	var drive session.State
	if err := json.Unmarshal(sb, &drive); err != nil {
		return nil, fmt.Errorf("sessionimage: bad %s: %w", SessionFile, err)
	}
	return &Image{Dir: dir, Meta: meta, Drive: drive}, nil
}

// verifyParts re-hashes every listed part and fails closed on the first mismatch. A part
// listed in the manifest but absent on disk is tampering/truncation (it existed at dump),
// so it errors regardless of which part it is — the content-address integrity recall
// enforces over cas.json, extended to every sibling of the image.
func verifyParts(dir string, parts []Part) error {
	for _, p := range parts {
		name, ok := safeName(p.Name)
		if !ok {
			return fmt.Errorf("sessionimage: part %q is not a safe in-bundle name", p.Name)
		}
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return fmt.Errorf("sessionimage: part %q listed but unreadable (truncated/tampered offload?): %w", name, err)
		}
		if int64(len(b)) != p.Bytes {
			return fmt.Errorf("sessionimage: part %q size %d != listed %d", name, len(b), p.Bytes)
		}
		if d := recall.Digest(b); d != p.Digest {
			return fmt.Errorf("sessionimage: part %q digest mismatch (got %s want %s) — image integrity check failed",
				name, short(d), short(p.Digest))
		}
	}
	return nil
}

// HasCoreImage reports whether the image carries a recall core image (manifest + cas) —
// i.e. whether the session accumulated any content. A drive-only image (a freshly minted
// session) returns false, and Recall/Rehydrate skip the content load.
func (img *Image) HasCoreImage() bool {
	if _, err := os.Stat(filepath.Join(img.Dir, ManifestFile)); err != nil {
		return false
	}
	_, err := os.Stat(filepath.Join(img.Dir, CASFile))
	return err == nil
}

// Recall loads the image's recall core image as a fresh, gate-enforced Session — the
// content primitive with the quarantine trust gate re-armed, so a page-in in the
// restoring process is checked exactly as in the original. Returns (nil, nil) when the
// image carries no core image (a drive-only session).
func (img *Image) Recall() (*recall.Session, error) {
	if !img.HasCoreImage() {
		return nil, nil
	}
	return recall.Load(img.Dir)
}

// Trajectory reads the per-turn audit corpus back, or nil when the image carries none.
func (img *Image) Trajectory() ([]trajectory.Turn, error) {
	f, err := os.Open(filepath.Join(img.Dir, TrajectoryFile))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	rec, _, err := trajectory.ImportFrom(f)
	if err != nil {
		return nil, err
	}
	return rec.Turns(), nil
}

// writeTrajectory writes turns as JSONL (one object per line), matching the stable schema
// trajectory.Recorder.ExportTo emits so the two are interchangeable on read.
func writeTrajectory(path string, turns []trajectory.Turn) error {
	var sb strings.Builder
	for _, t := range turns {
		b, err := json.Marshal(t)
		if err != nil {
			return err
		}
		sb.Write(b)
		sb.WriteByte('\n')
	}
	return os.WriteFile(path, []byte(sb.String()), 0o644)
}

// safeName accepts only a clean in-bundle base filename — no path separators, no "..",
// not absolute. It is the path-traversal guard shared by Unpack (untrusted tar entries)
// and verifyParts (an untrusted Meta could otherwise name ../../etc/passwd).
func safeName(name string) (string, bool) {
	if name == "" || name != filepath.Base(name) {
		return "", false
	}
	if name == "." || name == ".." {
		return "", false
	}
	// Reject any path separator (POSIX and Windows), and a colon — a Windows drive
	// (`C:foo`) or alternate-data-stream (`file:stream`) name that filepath.Base might
	// not split the way os.OpenFile resolves it. A real part name is a plain base name.
	if strings.ContainsAny(name, `/\:`) {
		return "", false
	}
	if filepath.IsAbs(name) {
		return "", false
	}
	return name, true
}

func short(d string) string {
	if len(d) > 12 {
		return d[:12]
	}
	return d
}
