package ggufload

import (
	"bytes"
	"errors"
	"io"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
)

type retainedSourceReadsForTest interface {
	holdSourceReads() (func(), <-chan struct{}, bool)
}

type retainedReadCloseFunc func() error

func (f retainedReadCloseFunc) Close() error { return f() }

type retainedReadEvents struct {
	mu     sync.Mutex
	events []string
}

func (e *retainedReadEvents) add(s string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.events = append(e.events, s)
}

func (e *retainedReadEvents) check(t *testing.T, want ...string) {
	t.Helper()
	e.mu.Lock()
	defer e.mu.Unlock()
	if !reflect.DeepEqual(e.events, want) {
		t.Errorf("ownership order = %v, want %v", e.events, want)
	}
}

type retainedEngramReadBarrier struct {
	mu      sync.Mutex
	closed  bool
	entered chan struct{}
	resume  chan struct{}
	events  *retainedReadEvents
	payload []byte
}

func (r *retainedEngramReadBarrier) ReadAt(dst []byte, off int64) (int, error) {
	r.events.add("read-enter")
	close(r.entered)
	<-r.resume
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return 0, errors.New("reader retired during retained read")
	}
	n, err := bytes.NewReader(r.payload).ReadAt(dst, off)
	r.events.add("read-complete")
	return n, err
}

func (r *retainedEngramReadBarrier) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return errors.New("reader retired twice")
	}
	r.closed = true
	r.events.add("reader-close")
	return nil
}

// fak-test:runtime fast est=100ms lane=default
func TestWeightSourceRetainedReadClose(t *testing.T) {
	t.Run("no-lease", func(t *testing.T) {
		first := errors.New("first close failure")
		var events retainedReadEvents
		ws := &WeightSource{closers: []io.Closer{
			retainedReadCloseFunc(func() error { events.add("first"); return first }),
			retainedReadCloseFunc(func() error { events.add("second"); return errors.New("second close failure") }),
		}}
		if err := ws.Close(); err != first {
			t.Errorf("first error = %v, want original %v", err, first)
		}
		if err := ws.Close(); err != nil {
			t.Errorf("completed repeat close = %v", err)
		}
		events.check(t, "first", "second")
	})
	t.Run("retained-contract", func(t *testing.T) {
		if _, ok := any(&WeightSource{}).(retainedSourceReadsForTest); !ok {
			t.Fatal("WeightSource is missing the retained-read capability")
		}
		hold := func(t *testing.T, ws *WeightSource) (func(), <-chan struct{}) {
			t.Helper()
			release, closing, ok := any(ws).(retainedSourceReadsForTest).holdSourceReads()
			if !ok || release == nil || closing == nil {
				t.Fatal("live source refused retained read")
			}
			return release, closing
		}
		refuses := func(t *testing.T, ws *WeightSource) {
			t.Helper()
			release, closing, ok := any(ws).(retainedSourceReadsForTest).holdSourceReads()
			if ok || release != nil || closing != nil {
				if release != nil {
					release()
				}
				t.Errorf("closing source admitted a new borrower: ok=%v", ok)
			}
		}
		t.Run("blocked-engram-read", func(t *testing.T) {
			var events retainedReadEvents
			payload := bytes.Repeat([]byte{0x37}, v41EngramRowBytes)
			r := &retainedEngramReadBarrier{entered: make(chan struct{}), resume: make(chan struct{}), events: &events, payload: payload}
			ws := &WeightSource{r: r, closers: []io.Closer{r}}
			source := &V41EngramGGUFSource{r: ws.r, tableRows: 1}
			release, closing := hold(t, ws)
			readDone := make(chan error, 1)
			dst := make([]byte, v41EngramRowBytes)
			go func() {
				n, err := source.ReadRows(0, 1, dst)
				if err == nil && n != len(dst) {
					err = io.ErrUnexpectedEOF
				}
				readDone <- err
			}()
			<-r.entered
			closeDone := make(chan error, 1)
			go func() { err := ws.Close(); events.add("close-return"); closeDone <- err }()
			<-closing
			refuses(t, ws)
			close(r.resume)
			readErr := <-readDone
			events.add("read-joined")
			events.add("release")
			release()
			closeErr := <-closeDone
			if readErr != nil || closeErr != nil || !bytes.Equal(dst, payload) {
				t.Errorf("read=%v close=%v equal=%v", readErr, closeErr, bytes.Equal(dst, payload))
			}
			events.check(t, "read-enter", "read-complete", "read-joined", "release", "reader-close", "close-return")
		})
		t.Run("two-borrowers", func(t *testing.T) {
			var events retainedReadEvents
			ws := &WeightSource{closers: []io.Closer{retainedReadCloseFunc(func() error { events.add("reader-close"); return nil })}}
			first, closing := hold(t, ws)
			second, sameClosing := hold(t, ws)
			if closing != sameClosing {
				first()
				second()
				t.Fatal("borrowers did not share the actual source closing signal")
			}
			done := make(chan error, 1)
			go func() { err := ws.Close(); events.add("close-return"); done <- err }()
			<-closing
			events.add("release-first")
			first()
			first()
			refuses(t, ws)
			events.add("release-second")
			second()
			second()
			if err := <-done; err != nil {
				t.Errorf("close: %v", err)
			}
			if err := ws.Close(); err != nil {
				t.Errorf("repeat close: %v", err)
			}
			events.check(t, "release-first", "release-second", "reader-close", "close-return")
		})
		t.Run("closed-before-acquire", func(t *testing.T) {
			ws := &WeightSource{}
			if err := ws.Close(); err != nil {
				t.Fatal(err)
			}
			refuses(t, ws)
		})
		t.Run("new-borrow-close-race", func(t *testing.T) {
			for range 32 {
				var events retainedReadEvents
				ws := &WeightSource{closers: []io.Closer{retainedReadCloseFunc(func() error { events.add("reader-close"); return nil })}}
				start := make(chan struct{})
				type result struct {
					release func()
					closing <-chan struct{}
					ok      bool
				}
				borrowDone := make(chan result, 1)
				closeDone := make(chan error, 1)
				go func() {
					<-start
					release, closing, ok := any(ws).(retainedSourceReadsForTest).holdSourceReads()
					if ok {
						events.add("acquire")
					}
					borrowDone <- result{release, closing, ok}
				}()
				go func() { <-start; err := ws.Close(); events.add("close-return"); closeDone <- err }()
				close(start)
				got := <-borrowDone
				if got.ok {
					<-got.closing
					events.add("release")
					got.release()
				} else if got.release != nil || got.closing != nil {
					t.Error("refused borrow retained ownership")
					if got.release != nil {
						got.release()
					}
				}
				if err := <-closeDone; err != nil {
					t.Errorf("close: %v", err)
				}
				refuses(t, ws)
				if got.ok {
					events.check(t, "acquire", "release", "reader-close", "close-return")
				} else {
					events.check(t, "reader-close", "close-return")
				}
			}
		})
		for _, mode := range []string{"panic", "goexit"} {
			t.Run(mode, func(t *testing.T) {
				var events retainedReadEvents
				marker := errors.New("original closer panic")
				ws := &WeightSource{closers: []io.Closer{
					retainedReadCloseFunc(func() error {
						events.add("abnormal-close")
						if mode == "panic" {
							panic(marker)
						}
						runtime.Goexit()
						return nil
					}),
					retainedReadCloseFunc(func() error { events.add("must-not-run"); return nil }),
				}}
				release, _ := hold(t, ws)
				release()
				type outcome struct {
					panicValue any
					returned   bool
				}
				done := make(chan outcome, 1)
				go func() {
					returned := false
					defer func() { done <- outcome{recover(), returned} }()
					_ = ws.Close()
					returned = true
				}()
				got := <-done
				if got.returned || (mode == "panic" && got.panicValue != marker) || (mode == "goexit" && got.panicValue != nil) {
					t.Errorf("original termination lost: returned=%v panic=%v", got.returned, got.panicValue)
				}
				if err := ws.Close(); err == nil || !strings.Contains(err.Error(), "source retirement incomplete") {
					t.Errorf("abnormal repeat close = %v, want incomplete retirement", err)
				}
				refuses(t, ws)
				events.check(t, "abnormal-close")
			})
		}
	})
}
