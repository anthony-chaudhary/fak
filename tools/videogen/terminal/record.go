// record.go — the producing half of the typescript+timing format main.go reads.
//
// The 2026-07-27 capture was made with util-linux `script --timing`, which the
// NixOS rigs have and macOS does not: BSD script(1) takes no --timing and
// writes no companion file. Rather than hand-synthesise a timing file for the
// 2026-07-28 addendum — which would be fabricating the one thing the recording
// claims is real, its pacing — this mode records the deltas itself.
//
// It lives here, in the renderer, on purpose. readTiming (main.go) is the only
// other definition of this format in the tree; a producer that drifts from its
// consumer yields a replay that is silently mistimed rather than broken, which
// is the failure mode nobody notices. One file, both directions.
//
// It reads a stream on stdin rather than spawning the child itself:
//
//	command 2>&1 | go run ./tools/videogen -record-typescript X -record-timing Y
//
// No pty is involved, and none is wanted. The colour in this capture is written
// by the script's own printf escapes, not inferred from isatty, so it survives a
// pipe; what a pty would add is progress-bar redraws from nix, which are noise
// in a replay. Routing BSD script's transcript here instead was tried and is
// worse on both counts: `-F /dev/stdout` duplicates every byte, and a transcript
// pointed at a FIFO arrives as one undivided chunk, which erases the pacing this
// file exists to record.
package main

import (
	"fmt"
	"io"
	"os"
	"time"
)

// recordMode tees stdin to typescript while appending one "<delay> <nbytes>"
// line per read to timing, then copies the stream on to stdout so the operator
// can still watch the run they are recording.
//
// The delay PRECEDES its chunk, matching script(1) and the walk in main.go: the
// first line's delay is the wait before any output, not after it.
func recordMode(tsPath, timPath string) error {
	ts, err := os.Create(tsPath)
	if err != nil {
		return err
	}
	defer ts.Close()
	tim, err := os.Create(timPath)
	if err != nil {
		return err
	}
	defer tim.Close()

	// main.go drops everything up to the first newline and does not charge those
	// bytes against the timing file, so the header must be exactly one line and
	// must not be counted below.
	if _, err := fmt.Fprintf(ts, "Script started on %s\n", time.Now().Format(time.UnixDate)); err != nil {
		return err
	}

	buf := make([]byte, 64*1024)
	last := time.Now()
	// BSD script emits two EOT bytes before the child's first output. They are
	// real bytes in the stream, so dropping them silently would desynchronise
	// every subsequent chunk offset; they are dropped here, before the clock
	// starts, and never counted.
	leading := true
	for {
		n, rerr := os.Stdin.Read(buf)
		if n > 0 {
			chunk := buf[:n]
			if leading {
				for len(chunk) > 0 && chunk[0] == 0x04 {
					chunk = chunk[1:]
				}
				if len(chunk) > 0 {
					leading = false
				}
			}
			if len(chunk) > 0 {
				now := time.Now()
				if _, err := fmt.Fprintf(tim, "%.6f %d\n", now.Sub(last).Seconds(), len(chunk)); err != nil {
					return err
				}
				last = now
				if _, err := ts.Write(chunk); err != nil {
					return err
				}
				os.Stdout.Write(chunk)
			}
		}
		if rerr == io.EOF {
			return nil
		}
		if rerr != nil {
			return rerr
		}
	}
}
