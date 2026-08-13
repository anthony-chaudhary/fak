// Package gardenbudget provides the durable checkpoint primitive used by
// `fak garden tick` to keep one maintenance pass inside a wall-clock budget.
//
// A garden cycle has two kinds of resumable work: collection members and acting
// phases. Both checkpoint the NEXT unit only after the current unit returns. If
// the outer watchdog kills a hung unit, the checkpoint therefore still names
// that unit and the next scheduled pass retries it instead of falsely skipping
// work. Units that completed before the kill are not replayed.
package gardenbudget

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

// Phase names one resumable unit of garden work.
type Phase string

// CursorSchema tags the durable garden-cycle checkpoint.
const CursorSchema = "fak.garden-tick-cursor.v2"

// Cursor is the durable resume point shared by collection and acting stages.
// Payload is deliberately opaque to this package; cmd/fak owns the accumulated
// member rows and action counts while gardenbudget owns atomic persistence.
type Cursor struct {
	Schema      string          `json:"schema"`
	Stage       string          `json:"stage,omitempty"`
	Next        Phase           `json:"next,omitempty"`
	Ticks       int             `json:"ticks"`
	UpdatedUnix int64           `json:"updated_unix"`
	Payload     json.RawMessage `json:"payload,omitempty"`
}

// Stamp normalizes a cursor for one new tick invocation.
func Stamp(cur Cursor, stage string, now time.Time) Cursor {
	cur.Schema = CursorSchema
	if cur.Stage == "" {
		cur.Stage = stage
	}
	cur.Ticks++
	cur.UpdatedUnix = now.Unix()
	return cur
}

// Options tunes one Execute pass.
type Options struct {
	// Budget is the whole tick's allowance. Zero or negative means unbounded.
	Budget time.Duration
	// Start is the beginning of the WHOLE tick, not merely the acting stage.
	Start time.Time
	// Now overrides the clock for deterministic tests.
	Now func() time.Time
	// Checkpoint runs after every completed phase with the next durable cursor.
	// A checkpoint error stops the pass; starting more work would make the
	// persisted resume point lie about what already completed.
	Checkpoint func(Cursor) error
}

func (o Options) now() time.Time {
	if o.Now != nil {
		return o.Now()
	}
	return time.Now()
}

// Remaining reports the allowance left in the whole-tick budget. A negative
// budget is unbounded and returns the largest duration.
func Remaining(budget time.Duration, start time.Time, now func() time.Time) time.Duration {
	if budget <= 0 {
		return time.Duration(1<<63 - 1)
	}
	if now == nil {
		now = time.Now
	}
	left := budget - now().Sub(start)
	if left < 0 {
		return 0
	}
	return left
}

// PhaseRun records one phase that actually returned this tick.
type PhaseRun struct {
	Phase  Phase  `json:"phase"`
	Millis int64  `json:"millis"`
	Err    string `json:"error,omitempty"`
}

// Result is one bounded pass's accounting.
type Result struct {
	Ran             []PhaseRun `json:"ran"`
	Deferred        []Phase    `json:"deferred"`
	Exhausted       bool       `json:"exhausted"`
	Complete        bool       `json:"complete"`
	Millis          int64      `json:"millis"`
	Next            Cursor     `json:"next"`
	CheckpointError string     `json:"checkpoint_error,omitempty"`
}

// Errors reports how many phases returned an error.
func (r Result) Errors() int {
	n := 0
	for _, p := range r.Ran {
		if p.Err != "" {
			n++
		}
	}
	return n
}

// Execute runs the suffix of phases beginning at cur.Next. It never starts a
// phase after the global budget is spent. Unlike the superseded rotation, it
// does not run one phase "for progress" after expiry: the outer watchdog's hard
// deadline is the invariant, and a fresh scheduled pass supplies fresh budget.
//
// A completed phase advances the checkpoint before another phase starts.
// Errors remain best-effort and also advance; a killed/hung phase never returns,
// so its checkpoint never advances and it is retried on the next pass.
func Execute(phases []Phase, cur Cursor, opt Options, run func(Phase) error) Result {
	start := opt.Start
	if start.IsZero() {
		start = opt.now()
	}
	res := Result{Next: cur}
	res.Next.Schema = CursorSchema
	if len(phases) == 0 {
		res.Complete = true
		res.Next.Next = ""
		res.Next.UpdatedUnix = opt.now().Unix()
		return res
	}

	begin := sequenceStart(phases, cur.Next)
	res.Next.Next = phases[begin]
	for i := begin; i < len(phases); i++ {
		p := phases[i]
		if Remaining(opt.Budget, start, opt.now) == 0 {
			res.Deferred = append(res.Deferred, phases[i:]...)
			break
		}

		phaseStart := opt.now()
		err := run(p)
		pr := PhaseRun{Phase: p, Millis: opt.now().Sub(phaseStart).Milliseconds()}
		if err != nil {
			pr.Err = err.Error()
		}
		res.Ran = append(res.Ran, pr)

		if i+1 < len(phases) {
			res.Next.Next = phases[i+1]
		} else {
			res.Next.Next = ""
			res.Complete = true
		}
		res.Next.UpdatedUnix = opt.now().Unix()
		if opt.Checkpoint != nil {
			if err := opt.Checkpoint(res.Next); err != nil {
				res.CheckpointError = err.Error()
				if i+1 < len(phases) {
					res.Deferred = append(res.Deferred, phases[i+1:]...)
				}
				break
			}
		}
	}

	res.Exhausted = len(res.Deferred) > 0
	res.Millis = opt.now().Sub(start).Milliseconds()
	res.Next.UpdatedUnix = opt.now().Unix()
	return res
}

func sequenceStart(phases []Phase, next Phase) int {
	if next == "" {
		return 0
	}
	for i, p := range phases {
		if p == next {
			return i
		}
	}
	return 0
}

// LoadCursor reads a checkpoint. Missing and wrong-schema files fail open to a
// zero cursor; malformed files report an error and also return a zero cursor.
func LoadCursor(path string) (Cursor, error) {
	if path == "" {
		return Cursor{}, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Cursor{}, nil
		}
		return Cursor{}, err
	}
	var c Cursor
	if err := json.Unmarshal(b, &c); err != nil {
		return Cursor{}, err
	}
	if c.Schema != CursorSchema {
		return Cursor{}, nil
	}
	return c, nil
}

// SaveCursor writes the checkpoint atomically.
func SaveCursor(path string, c Cursor) error {
	if path == "" {
		return nil
	}
	c.Schema = CursorSchema
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.Marshal(c)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".garden-cursor-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	if _, err := tmp.Write(append(b, '\n')); err != nil {
		tmp.Close()
		os.Remove(name)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(name)
		return err
	}
	if err := os.Rename(name, path); err != nil {
		os.Remove(name)
		return err
	}
	return nil
}
