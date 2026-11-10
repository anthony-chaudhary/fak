// Package phasetime records named wall-clock phases of one CLI invocation so a
// slow verb can say WHERE its time went instead of only how long it took.
//
// It exists because `fak-flow wip` and `fak-flow start` were measured at
// minutes per call with no way to attribute the time to census, per-worktree
// git probes, provisioning, or the go-run launch tax (TS-05). A recorder is
// carried on the context, so deep platform code marks a phase without a new
// parameter, and every entry point is nil-safe: a caller that did not ask for
// timings pays one context lookup per phase and records nothing.
//
// Two activation paths share one grammar:
//   - a caller-owned Recorder (e.g. `fak-flow wip --timings`) installed with
//     WithRecorder and rendered once as a table at the end; and
//   - the FAK_FLOW_TIMINGS environment knob, which streams each finished phase
//     to stderr as it completes. Streaming needs no end-of-command hook, so a
//     verb can be profiled without editing its command wiring.
package phasetime

import (
	"context"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// EnvVar enables streaming phase timings for any process that marks phases
// through Track or Mark without a context-carried Recorder. Any value other
// than "", "0", "false", or "off" (case-insensitive) turns it on.
const EnvVar = "FAK_FLOW_TIMINGS"

// Phase is one finished, named span of wall time.
type Phase struct {
	// Name is a dotted phase path, e.g. "census.refs.classify". The prefix
	// names the owning stage so nested phases group in the rendered table.
	Name string `json:"name"`
	// Offset is the phase start relative to the recorder origin.
	Offset time.Duration `json:"offset_ns"`
	// Duration is the phase wall time.
	Duration time.Duration `json:"duration_ns"`
	// Note is an optional one-line annotation (counts, the bound that fired).
	Note string `json:"note,omitempty"`
}

// Recorder collects phases. It is safe for concurrent use; a nil *Recorder is
// a valid no-op recorder.
type Recorder struct {
	mu     sync.Mutex
	origin time.Time
	now    func() time.Time
	phases []Phase
	stream io.Writer
}

// New returns a recorder whose origin is now.
func New() *Recorder {
	return &Recorder{origin: time.Now(), now: time.Now}
}

// NewStreaming returns a recorder that also writes each finished phase to w
// the moment it completes.
func NewStreaming(w io.Writer) *Recorder {
	r := New()
	r.stream = w
	return r
}

// newWithClock is the deterministic constructor for tests.
func newWithClock(now func() time.Time, stream io.Writer) *Recorder {
	return &Recorder{origin: now(), now: now, stream: stream}
}

type ctxKey struct{}

// WithRecorder returns a context carrying r. A nil r leaves ctx unchanged.
func WithRecorder(ctx context.Context, r *Recorder) context.Context {
	if r == nil {
		return ctx
	}
	return context.WithValue(ctx, ctxKey{}, r)
}

// FromContext returns the recorder carried by ctx, else the process-wide
// environment recorder (non-nil only when EnvVar is set), else nil.
func FromContext(ctx context.Context) *Recorder {
	if ctx != nil {
		if r, ok := ctx.Value(ctxKey{}).(*Recorder); ok && r != nil {
			return r
		}
	}
	return envRecorder()
}

var (
	envOnce sync.Once
	envRec  *Recorder
)

// Enabled reports whether the EnvVar knob requests streaming timings.
func Enabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(EnvVar))) {
	case "", "0", "false", "off", "no":
		return false
	}
	return true
}

func envRecorder() *Recorder {
	envOnce.Do(func() {
		if Enabled() {
			envRec = NewStreaming(os.Stderr)
		}
	})
	return envRec
}

// Start begins a phase and returns the function that ends it. The optional
// note is recorded on the phase. Calling the returned function more than once
// records only the first call. On a nil recorder both are no-ops.
func (r *Recorder) Start(name string) func(note ...string) {
	if r == nil {
		return func(...string) {}
	}
	begin := r.now()
	var once sync.Once
	return func(note ...string) {
		once.Do(func() {
			end := r.now()
			p := Phase{
				Name:     name,
				Offset:   begin.Sub(r.origin),
				Duration: end.Sub(begin),
				Note:     strings.TrimSpace(strings.Join(note, " ")),
			}
			r.mu.Lock()
			r.phases = append(r.phases, p)
			stream := r.stream
			r.mu.Unlock()
			if stream != nil {
				fmt.Fprintf(stream, "timing: %-34s +%7.2fs %8.2fs%s\n", p.Name, p.Offset.Seconds(), p.Duration.Seconds(), noteSuffix(p.Note))
			}
		})
	}
}

// Track starts a phase on the recorder carried by ctx (or the environment
// recorder). The idiom is `defer phasetime.Track(ctx, "census")()`, or
// `end := phasetime.Track(ctx, "census"); ...; end("12 refs")`.
func Track(ctx context.Context, name string) func(note ...string) {
	return FromContext(ctx).Start(name)
}

// Mark starts a phase on the environment recorder only, for code paths that
// carry no context.
func Mark(name string) func(note ...string) {
	return envRecorder().Start(name)
}

// Phases returns a copy of the finished phases ordered by start offset, then
// name, so concurrent phases render deterministically.
func (r *Recorder) Phases() []Phase {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	out := make([]Phase, len(r.phases))
	copy(out, r.phases)
	r.mu.Unlock()
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Offset != out[j].Offset {
			return out[i].Offset < out[j].Offset
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// Elapsed is the wall time from the recorder origin to now.
func (r *Recorder) Elapsed() time.Duration {
	if r == nil {
		return 0
	}
	return r.now().Sub(r.origin)
}

// WriteTable renders the finished phases as an aligned table with each
// phase's share of the total wall time. Nested phases (dotted names) are
// indented under their stage; shares of overlapping phases can sum past 100%
// because concurrent phases share wall time.
func (r *Recorder) WriteTable(w io.Writer) {
	if r == nil {
		return
	}
	total := r.Elapsed()
	fmt.Fprintf(w, "PHASE TIMINGS (wall %.2fs in-process; excludes process launch)\n", total.Seconds())
	fmt.Fprintf(w, "  %-38s %8s %8s %6s  %s\n", "phase", "start", "dur", "wall%", "note")
	for _, p := range r.Phases() {
		depth := strings.Count(p.Name, ".")
		label := strings.Repeat("  ", depth) + p.Name
		share := 0.0
		if total > 0 {
			share = 100 * float64(p.Duration) / float64(total)
		}
		fmt.Fprintf(w, "  %-38s %7.2fs %7.2fs %5.1f%%  %s\n", label, p.Offset.Seconds(), p.Duration.Seconds(), share, p.Note)
	}
}

func noteSuffix(note string) string {
	if note == "" {
		return ""
	}
	return "  " + note
}
