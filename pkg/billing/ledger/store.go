package ledger

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
)

// Open opens the durable log at path (creating an empty one if absent) and
// replays it into a fresh Ledger, so balances survive a restart.
//
// The file must be a well-formed log of this format: newline-separated
// entries, each decoding strictly to the Entry schema, with sequence numbers
// starting at 1 and increasing by one. Anything else — undecodable lines,
// fields outside the schema, failed entry validation, broken sequence
// continuity, or entries the state machine refuses — is rejected with
// ErrCorruptLog and the file is left untouched. A torn last line (a crash
// mid-append) fails closed the same way: inspect and repair the file, then
// reopen.
func Open(path string) (*Ledger, error) {
	if strings.TrimSpace(path) == "" {
		return nil, ErrInvalidArgument
	}
	l := newLedger(path)
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return l, nil
	}
	if err != nil {
		return nil, fmt.Errorf("ledger: read %q: %w", path, err)
	}
	if err := l.replayBytes(data); err != nil {
		return nil, fmt.Errorf("%w: %q: %v", ErrCorruptLog, path, err)
	}
	return l, nil
}

// replayBytes applies every entry of data in order, rebuilding state and
// both indices (dedup and settled-key) from the log alone. It runs on a
// freshly constructed ledger that no other goroutine can see yet, so it
// takes no lock.
func (l *Ledger) replayBytes(data []byte) error {
	var prev uint64
	for i, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		e, err := decodeEntry([]byte(line))
		if err != nil {
			return fmt.Errorf("%w: line %d: %v", ErrInvalidEntry, i+1, err)
		}
		if e.Seq != prev+1 {
			return fmt.Errorf("%w: line %d: sequence %d, want %d",
				ErrInvalidEntry, i+1, e.Seq, prev+1)
		}
		prev = e.Seq
		if err := l.checkLocked(e); err != nil {
			return fmt.Errorf("line %d: %v", i+1, err)
		}
		l.mutateLocked(e)
	}
	if err := l.invariantsLocked(); err != nil {
		return fmt.Errorf("invariant after replay: %v", err)
	}
	return nil
}

// decodeEntry decodes exactly one serialized Entry from data and refuses
// anything the entry schema does not declare. Refusing unknown fields keeps
// the log closed: a payload that tries to carry extra state alongside an
// entry cannot be decoded into one. Decoding does not check relational rules
// (sequence continuity, state-machine fit); those are checked by the caller.
func decodeEntry(data []byte) (Entry, error) {
	var e Entry
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&e); err != nil {
		return Entry{}, err
	}
	if dec.More() {
		return Entry{}, errors.New("trailing data after entry")
	}
	if err := e.Validate(); err != nil {
		return Entry{}, err
	}
	return e, nil
}

// appendLine durably appends one entry to the log file. It opens the file
// with O_APPEND, writes exactly one line, and closes it, so an append is a
// single contiguous write and the file stays safe to append from another
// process (for example an operator CLI) while the ledger runs. A volatile
// ledger (no path) appends nowhere.
func (l *Ledger) appendLine(e Entry) error {
	if l.path == "" {
		return nil
	}
	data, err := json.Marshal(e)
	if err != nil {
		return fmt.Errorf("ledger: encode entry %s: %w", e.EventID, err)
	}
	f, err := os.OpenFile(l.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return fmt.Errorf("ledger: open %q: %w", l.path, err)
	}
	if _, err := f.Write(append(data, '\n')); err != nil {
		_ = f.Close()
		return fmt.Errorf("ledger: append %q: %w", l.path, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("ledger: close %q: %w", l.path, err)
	}
	return nil
}
