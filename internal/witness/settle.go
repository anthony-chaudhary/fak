package witness

// settle.go — the SETTLED-ARTIFACT RECEIPT rung (#5646).
//
// THE GAP IT CLOSES. `path:<p>` corroborates that an artifact EXISTS. Existence is
// not completeness. A worker log, an evaluation result, an exported trace, a
// diagnostic bundle — each is produced ASYNCHRONOUSLY, and a consumer that parses,
// indexes, or publishes it the moment it appears reads a prefix and calls it the
// result. The usual "fix" is an ad-hoc sleep, which is a guess with no receipt: it
// cannot say WHAT it observed, so a slow producer silently defeats it and a fast
// one pays for nothing.
//
// THE RUNG. Settle observes an artifact's identity vector — existence, size,
// modification time, member count, and (by default) a content digest — repeatedly
// across a caller-declared interval, and returns a typed, closed-state RECEIPT
// carrying every sample and the timing basis that produced it. The consumer keeps
// the receipt: the decision is auditable after the fact, unlike a sleep.
//
//	settled         the identity vector was byte-identical across the whole window
//	                (and, when a completion marker was declared, the producer
//	                declared done)
//	still_growing   the artifact changed during the window — a consumer must wait
//	stalled         the artifact went quiet but the DECLARED completion marker never
//	                appeared: a producer that died mid-write looks exactly like this
//	missing         absent throughout, vanished mid-window, or present but bound to a
//	                DIFFERENT run (a stale pre-existing file is never this run's result)
//	unverifiable    the observation itself could not be trusted (cancelled, an
//	                irregular file, an unreadable path, an incoherent spec)
//
// WHY TRUNCATE-AND-REWRITE CANNOT PASS. A settled verdict requires every sample's
// FULL identity vector to be identical, and that vector includes a sha256 of the
// content by default. A producer that truncates to zero and rewrites changes size,
// mtime, and digest; a producer that rewrites in place to the same length still
// changes the digest. The only rewrite that survives is one that reproduces the
// same bytes — in which case "settled" is not a misclassification, it is the truth.
// A caller may set NoDigest for a very large artifact; the receipt then records the
// weaker `size+mtime` basis explicitly, so a reader can see what was actually
// observed rather than assuming the strong one.
//
// WHY A STALE FILE CANNOT PASS. Quiescence alone cannot tell "the new producer
// finished" from "last run's file is still lying there" — a stale artifact is
// perfectly quiet. Identity must be bound to the run, and this rung takes either
// binding: NotBeforeUnixNano (the artifact must not predate the run's start) or
// RunID plus a Marker whose content carries that run id. Without one of those a
// pre-existing file can only ever be reported on its own terms, never adopted as
// this run's result.
//
// READ-ONLY AND CANCELLABLE. Every observation is os.Lstat, a read-only open, and a
// lexical WalkDir. Nothing is created, written, moved, or removed — not even a lock
// or a temp file — so pointing this rung at a live producer's output can never
// perturb it. Every inter-sample wait selects on ctx.Done(), and ctx is re-checked
// before each observation, so a cancelled context returns an unverifiable receipt
// promptly instead of sleeping out the window. The total window is bounded by
// SettleMaxWindow so a claim can never park the kernel indefinitely.
//
// CLAIM FORM: `settled:<json>` where <json> is a SettleSpec. settled => CONFIRMED,
// still_growing/stalled/missing => REFUTED, unverifiable => ABSTAIN. It reads
// mutable filesystem state, so — like path:/clean:/committed: — it is never cached.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// SettleSchema tags a rendered receipt in both output shapes.
const SettleSchema = "fak.settle-receipt.v1"

// Observation-window defaults and bounds. A window is interval*(samples-1); it is
// capped so a `settled:` claim can never block the kernel indefinitely.
const (
	SettleDefaultInterval = 250 * time.Millisecond
	SettleDefaultSamples  = 3
	SettleMinSamples      = 2
	SettleMaxSamples      = 64
	SettleMaxWindow       = 30 * time.Second
)

// SettleState is the closed result vocabulary. See the file doc for each member.
type SettleState string

const (
	SettleSettled      SettleState = "settled"
	SettleStillGrowing SettleState = "still_growing"
	SettleMissing      SettleState = "missing"
	SettleStalled      SettleState = "stalled"
	SettleUnverifiable SettleState = "unverifiable"
)

// Marker states recorded per sample when a completion marker was declared.
const (
	SettleMarkerAbsent     = "absent"
	SettleMarkerReady      = "ready"
	SettleMarkerMismatch   = "mismatch"
	SettleMarkerUnreadable = "unreadable"
)

// The digest-bearing and digest-free observation bases, recorded on the receipt so
// a reader never has to assume which one produced the verdict.
const (
	SettleBasisDigest = "size+mtime+sha256"
	SettleBasisStat   = "size+mtime"
)

// SettleSpec is the caller's declaration of what to observe and how. It is the
// JSON payload of a `settled:<json>` claim, so every field is plain data.
type SettleSpec struct {
	// Path is the artifact: a regular file, or a directory observed as a whole
	// (the bundle case — member count, total size, newest mtime, tree digest).
	Path string `json:"path"`
	// Marker is an optional producer completion marker. When set, quiescence alone
	// is not enough: absent-and-quiet is reported as stalled, not settled.
	Marker string `json:"marker,omitempty"`
	// RunID binds the artifact to THIS run: the marker's content must carry it.
	// A marker naming another run reports missing, never settled.
	RunID string `json:"run_id,omitempty"`
	// NotBeforeUnixNano is the stale-file floor: an artifact older than this
	// predates the run and is reported missing rather than adopted.
	NotBeforeUnixNano int64 `json:"not_before_unix_nano,omitempty"`
	// IntervalNanos is the gap between observations (default SettleDefaultInterval).
	IntervalNanos int64 `json:"interval_nanos,omitempty"`
	// Samples is how many observations to take (default SettleDefaultSamples).
	Samples int `json:"samples,omitempty"`
	// NoDigest drops content hashing for a very large artifact. The receipt then
	// records the weaker SettleBasisStat basis.
	NoDigest bool `json:"no_digest,omitempty"`
}

// SettleSample is one observation of the artifact's identity vector.
type SettleSample struct {
	Index       int    `json:"index"`
	AtNanos     int64  `json:"at_nanos"` // offset from the first observation
	Exists      bool   `json:"exists"`
	Size        int64  `json:"size"`
	ModUnixNano int64  `json:"mod_unix_nano"`
	Files       int    `json:"files,omitempty"` // directory members; 0 for a file
	Digest      string `json:"digest,omitempty"`
	MarkerState string `json:"marker_state,omitempty"`
	Error       string `json:"error,omitempty"`
}

// identity is the comparison vector. The marker state is deliberately EXCLUDED: a
// marker appearing mid-window is the producer finishing, not the artifact moving.
func (s SettleSample) identity() string {
	return strings.Join([]string{
		strconv.FormatBool(s.Exists),
		strconv.FormatInt(s.Size, 10),
		strconv.FormatInt(s.ModUnixNano, 10),
		strconv.Itoa(s.Files),
		s.Digest,
	}, "|")
}

// SettleReceipt is the typed, auditable result: the state, why, and the complete
// evidence — every sample plus the timing basis that produced them.
type SettleReceipt struct {
	Schema        string         `json:"schema"`
	State         SettleState    `json:"state"`
	Reason        string         `json:"reason,omitempty"`
	Path          string         `json:"path"`
	Marker        string         `json:"marker,omitempty"`
	RunID         string         `json:"run_id,omitempty"`
	Basis         string         `json:"basis"`
	IntervalNanos int64          `json:"interval_nanos"`
	WantSamples   int            `json:"want_samples"`
	WindowNanos   int64          `json:"window_nanos"`
	Samples       []SettleSample `json:"samples,omitempty"`
}

// Settled reports whether the artifact may be consumed.
func (r SettleReceipt) Settled() bool { return r.State == SettleSettled }

// Err is the consumer-facing gate: nil when settled, a *SettleRefusal otherwise.
// A consumer rejects a partial artifact with `if err := rec.Err(); err != nil`.
func (r SettleReceipt) Err() error {
	if r.Settled() {
		return nil
	}
	return &SettleRefusal{Receipt: r}
}

// SettleRefusal is the typed refusal carrying the receipt that justifies it, so a
// caller that surfaces the error still has the samples behind the decision.
type SettleRefusal struct{ Receipt SettleReceipt }

func (e *SettleRefusal) Error() string {
	return fmt.Sprintf("artifact not settled: %s (%s) path=%s samples=%d",
		e.Receipt.State, e.Receipt.Reason, e.Receipt.Path, len(e.Receipt.Samples))
}

// WitnessOutcome maps the receipt onto the kernel's three-way witness contract for
// the `settled:<json>` claim. Only unverifiable abstains; every other non-settled
// state is real evidence that the artifact must not be consumed.
func (r SettleReceipt) WitnessOutcome() abi.WitnessOutcome {
	switch r.State {
	case SettleSettled:
		return abi.WitnessConfirmed
	case SettleStillGrowing, SettleMissing, SettleStalled:
		return abi.WitnessRefuted
	default:
		return abi.WitnessAbstain
	}
}

// Render is the human shape of the SAME receipt the JSON encoder emits: the state,
// the reason, the timing basis, and every sample. Both shapes are derived from one
// struct, so they cannot drift apart.
func (r SettleReceipt) Render() string {
	var b strings.Builder
	fmt.Fprintf(&b, "settled-artifact receipt (%s)\n", r.Schema)
	fmt.Fprintf(&b, "  state:    %s\n", r.State)
	if r.Reason != "" {
		fmt.Fprintf(&b, "  reason:   %s\n", r.Reason)
	}
	fmt.Fprintf(&b, "  path:     %s\n", r.Path)
	if r.Marker != "" {
		fmt.Fprintf(&b, "  marker:   %s\n", r.Marker)
	}
	if r.RunID != "" {
		fmt.Fprintf(&b, "  run_id:   %s\n", r.RunID)
	}
	fmt.Fprintf(&b, "  basis:    %s\n", r.Basis)
	fmt.Fprintf(&b, "  timing:   %d samples @ %s, window %s\n",
		r.WantSamples, time.Duration(r.IntervalNanos), time.Duration(r.WindowNanos))
	fmt.Fprintf(&b, "  samples:  %d\n", len(r.Samples))
	for _, s := range r.Samples {
		fmt.Fprintf(&b, "    [%d] +%s exists=%t size=%d mod=%d files=%d digest=%s",
			s.Index, time.Duration(s.AtNanos), s.Exists, s.Size, s.ModUnixNano, s.Files, digestOrDash(s.Digest))
		if s.MarkerState != "" {
			fmt.Fprintf(&b, " marker=%s", s.MarkerState)
		}
		if s.Error != "" {
			fmt.Fprintf(&b, " error=%s", s.Error)
		}
		b.WriteByte('\n')
	}
	return b.String()
}

func digestOrDash(d string) string {
	if d == "" {
		return "-"
	}
	return d
}

// SettleWatcher carries the timing seam. The zero value uses the real clock; tests
// inject Now/Sleep so a synthetic producer can step the artifact deterministically
// between observations with no wall-clock dependence.
type SettleWatcher struct {
	Now   func() time.Time
	Sleep func(ctx context.Context, d time.Duration) error
}

// Settle observes spec on the real clock and returns the receipt. It is read-only
// and cancellable; it never returns an error, only a typed state.
func Settle(ctx context.Context, spec SettleSpec) SettleReceipt {
	return SettleWatcher{}.Observe(ctx, spec)
}

func (w SettleWatcher) now() time.Time {
	if w.Now != nil {
		return w.Now()
	}
	return time.Now()
}

func (w SettleWatcher) sleep(ctx context.Context, d time.Duration) error {
	if w.Sleep != nil {
		return w.Sleep(ctx, d)
	}
	if d <= 0 {
		return ctx.Err()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// Observe takes the declared samples and classifies them into a receipt.
func (w SettleWatcher) Observe(ctx context.Context, spec SettleSpec) SettleReceipt {
	interval, samples := settleWindow(spec)
	rec := SettleReceipt{
		Schema:        SettleSchema,
		Path:          strings.TrimSpace(spec.Path),
		Marker:        strings.TrimSpace(spec.Marker),
		RunID:         strings.TrimSpace(spec.RunID),
		Basis:         SettleBasisDigest,
		IntervalNanos: int64(interval),
		WantSamples:   samples,
	}
	if spec.NoDigest {
		rec.Basis = SettleBasisStat
	}
	if rec.Path == "" {
		return settleUnverifiable(rec, "empty_path")
	}
	if window := interval * time.Duration(samples-1); window > SettleMaxWindow {
		return settleUnverifiable(rec, "window_exceeds_max")
	}
	if rec.RunID != "" && rec.Marker == "" {
		// A run id with nothing to bind it to cannot prove identity; saying
		// "settled" here would silently accept a stale file.
		return settleUnverifiable(rec, "run_id_without_marker")
	}

	start := w.now()
	for i := 0; i < samples; i++ {
		if i > 0 {
			if err := w.sleep(ctx, interval); err != nil {
				return settleUnverifiable(rec, "canceled")
			}
		}
		if ctx.Err() != nil {
			return settleUnverifiable(rec, "canceled")
		}
		s := observeSettle(rec, spec.NoDigest)
		s.Index = i
		s.AtNanos = w.now().Sub(start).Nanoseconds()
		rec.Samples = append(rec.Samples, s)
	}
	if n := len(rec.Samples); n > 0 {
		rec.WindowNanos = rec.Samples[n-1].AtNanos
	}
	return classifySettle(rec, spec.NotBeforeUnixNano)
}

// settleWindow normalizes the interval and sample count into their bounded forms.
func settleWindow(spec SettleSpec) (time.Duration, int) {
	interval := time.Duration(spec.IntervalNanos)
	if interval <= 0 {
		interval = SettleDefaultInterval
	}
	samples := spec.Samples
	if samples <= 0 {
		samples = SettleDefaultSamples
	}
	if samples < SettleMinSamples {
		samples = SettleMinSamples // one look can never witness quiescence
	}
	if samples > SettleMaxSamples {
		samples = SettleMaxSamples
	}
	return interval, samples
}

func settleUnverifiable(rec SettleReceipt, reason string) SettleReceipt {
	rec.State, rec.Reason = SettleUnverifiable, reason
	return rec
}

func settleState(rec SettleReceipt, state SettleState, reason string) SettleReceipt {
	rec.State, rec.Reason = state, reason
	return rec
}

// classifySettle folds the samples into a closed state. Order matters: an
// unobservable or absent artifact is decided before quiescence is even considered,
// and the run-binding gates run before any settled verdict can be reached.
func classifySettle(rec SettleReceipt, notBefore int64) SettleReceipt {
	for _, s := range rec.Samples {
		if s.Error != "" {
			return settleUnverifiable(rec, "observe_error:"+s.Error)
		}
	}
	last := rec.Samples[len(rec.Samples)-1]
	if st, bad := settleAbsence(rec, last); bad {
		return st
	}
	if notBefore > 0 && last.ModUnixNano < notBefore {
		// Quiet, present — and older than this run. A pre-existing file is not the
		// new producer's result.
		return settleState(rec, SettleMissing, "predates_run_floor")
	}
	if last.MarkerState == SettleMarkerMismatch {
		return settleState(rec, SettleMissing, "marker_bound_to_other_run")
	}
	if last.MarkerState == SettleMarkerUnreadable {
		return settleUnverifiable(rec, "marker_unreadable")
	}

	changed := settleChanged(rec.Samples)
	if rec.Marker == "" {
		if changed {
			return settleState(rec, SettleStillGrowing, "identity_changed_during_window")
		}
		return settleState(rec, SettleSettled, "quiescent")
	}
	if last.MarkerState != SettleMarkerReady {
		if changed {
			return settleState(rec, SettleStillGrowing, "producer_active_marker_absent")
		}
		// Quiet but never declared done: a producer that died mid-write.
		return settleState(rec, SettleStalled, "quiet_but_no_completion_marker")
	}
	if changed {
		return settleState(rec, SettleStillGrowing, "marker_present_but_artifact_changed")
	}
	return settleState(rec, SettleSettled, "marker_and_quiescent")
}

// settleAbsence decides the never-appeared and vanished-mid-window cases.
func settleAbsence(rec SettleReceipt, last SettleSample) (SettleReceipt, bool) {
	anyExists := false
	for _, s := range rec.Samples {
		if s.Exists {
			anyExists = true
			break
		}
	}
	if !anyExists {
		return settleState(rec, SettleMissing, "absent_throughout"), true
	}
	if !last.Exists {
		return settleState(rec, SettleMissing, "vanished_during_window"), true
	}
	return rec, false
}

func settleChanged(samples []SettleSample) bool {
	for i := 1; i < len(samples); i++ {
		if samples[i].identity() != samples[i-1].identity() {
			return true
		}
	}
	return false
}

// observeSettle takes one read-only observation of the artifact and, when one was
// declared, of the completion marker.
func observeSettle(rec SettleReceipt, noDigest bool) SettleSample {
	s := SettleSample{}
	fi, err := os.Lstat(rec.Path)
	switch {
	case err != nil && os.IsNotExist(err):
		// Absent is evidence, not an error.
	case err != nil:
		s.Error = "stat:" + err.Error()
		return s
	case fi.IsDir():
		s.Exists = true
		if e := observeTree(rec.Path, &s, noDigest); e != "" {
			s.Error = e
			return s
		}
	case fi.Mode().IsRegular():
		s.Exists = true
		s.Size, s.ModUnixNano = fi.Size(), fi.ModTime().UnixNano()
		if !noDigest {
			d, e := digestFile(rec.Path)
			if e != "" {
				s.Error = e
				return s
			}
			s.Digest = d
		}
	default:
		// A symlink or device is not an artifact this rung can reason about, and
		// following it would leave the declared path.
		s.Error = "irregular_file"
		return s
	}
	if rec.Marker != "" {
		s.MarkerState = observeMarker(rec.Marker, rec.RunID)
	}
	return s
}

// observeTree folds a directory into one identity vector: member count, total
// bytes, newest mtime, and a digest over each member's path, size, mtime and (when
// digesting) content. WalkDir is lexical, so the digest is order-stable.
func observeTree(root string, s *SettleSample, noDigest bool) string {
	h := sha256.New()
	walkErr := ""
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, we error) error {
		if we != nil {
			return we
		}
		if d.IsDir() {
			return nil
		}
		if !d.Type().IsRegular() {
			walkErr = "irregular_member:" + filepath.Base(p)
			return io.EOF // stop the walk; reported via walkErr
		}
		fi, ie := d.Info()
		if ie != nil {
			return ie
		}
		rel, _ := filepath.Rel(root, p)
		s.Files++
		s.Size += fi.Size()
		if mt := fi.ModTime().UnixNano(); mt > s.ModUnixNano {
			s.ModUnixNano = mt
		}
		fmt.Fprintf(h, "%s\x00%d\x00%d\x00", filepath.ToSlash(rel), fi.Size(), fi.ModTime().UnixNano())
		if noDigest {
			return nil
		}
		fd, de := digestFile(p)
		if de != "" {
			walkErr = de
			return io.EOF
		}
		fmt.Fprintf(h, "%s\x00", fd)
		return nil
	})
	if walkErr != "" {
		return walkErr
	}
	if err != nil {
		return "walk:" + err.Error()
	}
	s.Digest = hex.EncodeToString(h.Sum(nil))
	return ""
}

// digestFile streams a sha256 of the file. It opens read-only and buffers nothing
// beyond io.Copy's window, so a large artifact costs I/O, never memory.
func digestFile(p string) (string, string) {
	f, err := os.Open(p)
	if err != nil {
		return "", "open:" + err.Error()
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", "read:" + err.Error()
	}
	return hex.EncodeToString(h.Sum(nil)), ""
}

// observeMarker classifies the producer's completion marker. With a run id
// declared, a marker that does not carry it belongs to a DIFFERENT run — the
// stale-artifact case — and is reported as a mismatch, never as ready.
func observeMarker(path, runID string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return SettleMarkerAbsent
		}
		return SettleMarkerUnreadable
	}
	if runID == "" {
		return SettleMarkerReady
	}
	if strings.Contains(string(b), runID) {
		return SettleMarkerReady
	}
	return SettleMarkerMismatch
}

// resolveSettled adjudicates a `settled:<json>` claim. A relative path is resolved
// against the resolver's dir, so a claim means the same thing wherever the kernel
// happens to run. A malformed spec abstains — never a false CONFIRM.
func (r *Resolver) resolveSettled(ctx context.Context, raw string) abi.WitnessOutcome {
	var spec SettleSpec
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		return abi.WitnessAbstain
	}
	spec.Path = r.settleAnchor(spec.Path)
	spec.Marker = r.settleAnchor(spec.Marker)
	return Settle(ctx, spec).WitnessOutcome()
}

func (r *Resolver) settleAnchor(p string) string {
	p = strings.TrimSpace(p)
	if p == "" || r.dir == "" || isAbs(p) {
		return p
	}
	return filepath.Join(r.dir, p)
}
