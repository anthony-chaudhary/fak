package headroom

import "strings"

// stripTerminalControl removes the terminal control noise that a model reader can
// never use, then collapses carriage-return "redraw" frames to their final state.
// It is the native compressor's largest single saving on the kind of tool output a
// coding agent actually drowns in — colorized test/build/install logs and
// progress bars — where the escape codes and the intermediate progress frames are
// pure bytes the model renders nothing from.
//
// Two transforms, both LOSSLESS TO THE MODEL:
//
//   - ansi-strip: drop ESC-introduced control sequences (SGR color, cursor
//     movement, OSC window titles, DCS/PM/APC strings) and the bare C0 control
//     bytes / DEL that carry no meaning in a tool result. Tabs and newlines are
//     preserved; UTF-8 is untouched (a C0 control byte never appears inside a
//     multi-byte rune, so dropping it cannot corrupt text).
//   - cr-collapse: a line redrawn in place with carriage returns (a progress bar:
//     "12%\r 47%\r 100%") renders, on a real terminal, as only its final frame.
//     Keep that frame; drop the overwritten ones. Only fires on an INTERIOR
//     carriage return (a genuine redraw) — a lone trailing CR (a CRLF line ending)
//     is left for the line-normalizer's trim, so this codec never mislabels a
//     plain Windows-EOL file as a redraw.
//
// The caller preserves the pre-compression bytes in the shared CAS, so both
// transforms stay reversible (the CCR promise). stripTerminalControl returns the
// cleaned bytes and the names of the transforms that ACTUALLY fired — empty when
// the input carried no terminal control, so a no-op never claims a codec. It only
// ever removes bytes, so a returned codec always means a real, smaller rendering.
func stripTerminalControl(b []byte) ([]byte, []string) {
	stripped, didEsc := stripEscapeSequences(b)
	collapsed, didCR := collapseCarriageReturns(stripped)
	var codecs []string
	if didEsc {
		codecs = append(codecs, "ansi-strip")
	}
	if didCR {
		codecs = append(codecs, "cr-collapse")
	}
	return collapsed, codecs
}

// stripEscapeSequences removes ESC-introduced control sequences and the bare C0
// control bytes (and DEL) that are noise in a tool result, preserving tab, newline
// and carriage return (the CR is handled by collapseCarriageReturns). It returns
// the input unchanged with false when there is nothing to strip, so the common
// clean-text case allocates nothing.
func stripEscapeSequences(b []byte) ([]byte, bool) {
	if !hasStrippableControl(b) {
		return b, false
	}
	out := make([]byte, 0, len(b))
	for i, n := 0, len(b); i < n; {
		c := b[i]
		switch {
		case c == 0x1b: // ESC: consume the whole escape sequence
			i = skipEscapeSequence(b, i)
		case isStrippableControl(c): // bare control byte: drop it
			i++
		default:
			out = append(out, c)
			i++
		}
	}
	return out, true
}

// hasStrippableControl reports whether b contains any byte stripEscapeSequences
// would remove (an ESC or a bare strippable control). The fast-path guard.
func hasStrippableControl(b []byte) bool {
	for _, c := range b {
		if c == 0x1b || isStrippableControl(c) {
			return true
		}
	}
	return false
}

// isStrippableControl reports whether c is a bare control byte with no meaning in
// a tool result: a C0 control other than tab (\t), newline (\n) or carriage
// return (\r), or DEL (0x7f). ESC (0x1b) is handled by skipEscapeSequence, not
// here, so it is excluded.
func isStrippableControl(c byte) bool {
	if c == 0x1b {
		return false
	}
	if c < 0x20 {
		return c != '\t' && c != '\n' && c != '\r'
	}
	return c == 0x7f
}

// skipEscapeSequence returns the index just past the escape sequence that begins
// at b[i] (b[i] must be ESC, 0x1b). It recognizes the standard families: CSI
// (ESC [ … final), OSC (ESC ] … BEL|ST), the string sequences DCS/SOS/PM/APC
// (ESC P|X|^|_ … ST), charset designation (ESC (|)|*|+ + one byte), and the
// generic two-byte escape (ESC + one byte). An unterminated sequence consumes to
// end-of-input, which is the safe direction (drop the dangling control).
func skipEscapeSequence(b []byte, i int) int {
	n := len(b)
	j := i + 1
	if j >= n {
		return n // lone trailing ESC
	}
	switch b[j] {
	case '[': // CSI: parameter/intermediate bytes then a final 0x40..0x7e
		for j++; j < n && !(b[j] >= 0x40 && b[j] <= 0x7e); j++ {
		}
		if j < n {
			j++ // include the final byte
		}
		return j
	case ']': // OSC: terminated by BEL (0x07) or ST (ESC \)
		for j++; j < n; j++ {
			if b[j] == 0x07 {
				return j + 1
			}
			if b[j] == 0x1b && j+1 < n && b[j+1] == '\\' {
				return j + 2
			}
		}
		return n
	case 'P', 'X', '^', '_': // DCS/SOS/PM/APC: string terminated by ST (ESC \)
		for j++; j < n; j++ {
			if b[j] == 0x1b && j+1 < n && b[j+1] == '\\' {
				return j + 2
			}
		}
		return n
	case '(', ')', '*', '+': // charset designation: one designation byte follows
		j++
		if j < n {
			j++
		}
		return j
	default: // generic two-byte escape (ESC + one byte)
		return j + 1
	}
}

// collapseCarriageReturns rewrites each physical line (split on \n) to the final
// frame of any in-place carriage-return redraw it contains, and reports whether
// any line changed. A line with no interior CR is left untouched (a lone trailing
// CR — a CRLF ending — is not a redraw and is left for the line-normalizer).
func collapseCarriageReturns(b []byte) ([]byte, bool) {
	if !hasInteriorCR(b) {
		return b, false
	}
	lines := strings.Split(string(b), "\n")
	changed := false
	for idx, ln := range lines {
		if c, did := collapseLine(ln); did {
			lines[idx] = c
			changed = true
		}
	}
	if !changed {
		return b, false
	}
	return []byte(strings.Join(lines, "\n")), true
}

// collapseLine reduces one line to the text after its last interior carriage
// return — the frame a terminal would leave on screen. A single trailing CR is
// treated as cursor positioning (dropped from consideration, left on the line for
// the trim pass) so a CRLF line ending is not mistaken for a redraw.
func collapseLine(line string) (string, bool) {
	core := strings.TrimSuffix(line, "\r")
	i := strings.LastIndexByte(core, '\r')
	if i < 0 {
		return line, false // no interior CR
	}
	return core[i+1:], true
}

// hasInteriorCR reports whether b contains a carriage return that is NOT a lone
// CRLF line ending (i.e. a real in-place redraw). The fast-path guard for
// collapseCarriageReturns.
func hasInteriorCR(b []byte) bool {
	for i := 0; i < len(b); i++ {
		if b[i] != '\r' {
			continue
		}
		// A CR followed by LF (or end-of-input) is a line ending, not a redraw.
		if i+1 < len(b) && b[i+1] != '\n' {
			return true
		}
	}
	return false
}
