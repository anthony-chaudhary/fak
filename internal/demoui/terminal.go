package demoui

import "os"

// Palette is the ANSI color set shared by the browserless terminal demos.
// Empty fields deliberately mean plain output.
type Palette struct {
	Red    string
	Amber  string
	Green  string
	Blue   string
	Cyan   string
	Yellow string
	Dim    string
	Bold   string
	Reset  string
}

// TerminalPalette returns ANSI colors only when out is a character device and
// NO_COLOR is empty. Stat failures and redirected output stay plain.
func TerminalPalette(out *os.File) Palette {
	tty := false
	if fi, err := out.Stat(); err == nil {
		tty = fi.Mode()&os.ModeCharDevice != 0
	}
	if os.Getenv("NO_COLOR") != "" || !tty {
		return Palette{}
	}
	return Palette{
		Red:    "\033[31m",
		Amber:  "\033[33m",
		Green:  "\033[32m",
		Blue:   "\033[34m",
		Cyan:   "\033[36m",
		Yellow: "\033[33m",
		Dim:    "\033[2m",
		Bold:   "\033[1m",
		Reset:  "\033[0m",
	}
}

// Paint wraps s in code and the palette reset. An empty code leaves s unchanged.
func (p Palette) Paint(code, s string) string {
	if code == "" {
		return s
	}
	return code + s + p.Reset
}
