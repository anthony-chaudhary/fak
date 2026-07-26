package main

import "io"

// crlfWriterTUI re-adds the LF → CRLF mapping that raw mode removes.
//
// The interactive overlay puts STDIN in raw mode to read the focus/mouse/key
// bytes (runGuardInfoOverlay). On POSIX, termios is a property of the tty
// DEVICE, not of one fd's direction: term.MakeRaw's OPOST clear therefore also
// drops ONLCR for everything this process WRITES to the same terminal. A bare
// "\n" then stops implying a carriage return, so every frame row starts at the
// column where the previous row ended — the staircase mosaic witnessed in an
// Apple Terminal companion window (#5370). Windows consoles keep LF → CRLF
// translation regardless of the input mode, which is why Windows Terminal
// panes never showed it.
//
// The writer restores the mapping at the write site: every "\n" not already
// preceded by "\r" gains one, statefully across Write calls (a chunk ending in
// "\r" followed by a chunk starting with "\n" stays one CRLF). The rewrite is
// harmless in cooked mode too ("\r\r\n" draws identically to "\r\n"), so the
// wrap does not need to track whether raw mode is currently active.
type crlfWriterTUI struct {
	w      io.Writer
	lastCR bool
}

func newCRLFWriterTUI(w io.Writer) *crlfWriterTUI { return &crlfWriterTUI{w: w} }

// Write translates and forwards p. On success it reports len(p) — the caller's
// bytes are all consumed even though more bytes reached the underlying writer.
func (c *crlfWriterTUI) Write(p []byte) (int, error) {
	out := make([]byte, 0, len(p)+8)
	for _, b := range p {
		if b == '\n' && !c.lastCR {
			out = append(out, '\r')
		}
		c.lastCR = b == '\r'
		out = append(out, b)
	}
	if _, err := c.w.Write(out); err != nil {
		return 0, err
	}
	return len(p), nil
}
