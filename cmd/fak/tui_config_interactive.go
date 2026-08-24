package main

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"
	"unicode"

	"golang.org/x/term"
)

// tuiConfigInteraction is deliberately small: it offers keyboard control for
// finite registry settings while arbitrary input settings retain copyable commands.
type tuiConfigInteraction struct {
	path     string
	width    int
	at       time.Time
	report   tuiConfigReport
	settings []tuiConfigSetting
	selected int
	pending  string
	message  string
}

func runTUIConfigInteraction(stdin io.Reader, stdout io.Writer, path, initial string, width int, at time.Time) error {
	state, err := newTUIConfigInteraction(path, initial, width, at)
	if err != nil {
		return err
	}
	restore, raw, err := makeTUIConfigInputRaw(stdin)
	if err != nil {
		return fmt.Errorf("enter raw input mode: %w", err)
	}
	defer restore()
	if raw {
		if f, ok := stdout.(*os.File); ok && term.IsTerminal(int(f.Fd())) {
			stdout = newCRLFWriterTUI(stdout)
		}
	}

	state.render(stdout)
	buf := make([]byte, 1)
	for {
		n, readErr := stdin.Read(buf)
		if n > 0 {
			quit, err := state.press(rune(buf[0]))
			if err != nil {
				return err
			}
			state.render(stdout)
			if quit {
				return nil
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				return nil
			}
			return fmt.Errorf("read key: %w", readErr)
		}
	}
}

func makeTUIConfigInputRaw(stdin io.Reader) (func(), bool, error) {
	f, ok := stdin.(*os.File)
	if !ok || !term.IsTerminal(int(f.Fd())) {
		return func() {}, false, nil
	}
	oldState, err := term.MakeRaw(int(f.Fd()))
	if err != nil {
		return nil, false, err
	}
	return func() { _ = term.Restore(int(f.Fd()), oldState) }, true, nil
}

func newTUIConfigInteraction(path, initial string, width int, at time.Time) (*tuiConfigInteraction, error) {
	if width < 72 {
		width = 72
	}
	report := buildTUIConfigReport(path, at)
	if report.Status == "error" {
		return nil, fmt.Errorf("load %s: %s", report.Path, report.Error)
	}
	state := &tuiConfigInteraction{path: path, width: width, at: at, report: report}
	state.refreshSettings()
	if len(state.settings) == 0 {
		return nil, fmt.Errorf("no finite settings are available for keyboard control")
	}
	initial = strings.TrimSpace(strings.ToLower(initial))
	if initial != "" {
		found := false
		for i, setting := range state.settings {
			if tuiConfigSettingRef(setting) == initial {
				state.selected = i
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("--setting %q is not a finite, persistable setting", initial)
		}
	}
	state.pending = state.current().Effective
	return state, nil
}

func (s *tuiConfigInteraction) refreshSettings() {
	selectedRef := ""
	if len(s.settings) > 0 && s.selected < len(s.settings) {
		selectedRef = tuiConfigSettingRef(s.settings[s.selected])
	}
	s.settings = s.settings[:0]
	for _, setting := range s.report.Settings {
		if len(tuiConfigInteractiveValues(setting)) > 0 {
			s.settings = append(s.settings, setting)
		}
	}
	if selectedRef != "" {
		for i, setting := range s.settings {
			if tuiConfigSettingRef(setting) == selectedRef {
				s.selected = i
				return
			}
		}
	}
	if s.selected >= len(s.settings) {
		s.selected = maxTUI(0, len(s.settings)-1)
	}
}

func (s *tuiConfigInteraction) current() tuiConfigSetting {
	return s.settings[s.selected]
}

func (s *tuiConfigInteraction) press(key rune) (bool, error) {
	key = unicode.ToLower(key)
	switch key {
	case 'q', 3, 27:
		s.message = "exited without applying pending changes"
		return true, nil
	case 'j':
		s.selected = (s.selected + 1) % len(s.settings)
		s.pending = s.current().Effective
		s.message = "selected " + tuiConfigSettingRef(s.current())
	case 'k':
		s.selected = (s.selected - 1 + len(s.settings)) % len(s.settings)
		s.pending = s.current().Effective
		s.message = "selected " + tuiConfigSettingRef(s.current())
	case 'c', ' ':
		values := tuiConfigInteractiveValues(s.current())
		next := 0
		for i, value := range values {
			if value == s.pending {
				next = (i + 1) % len(values)
				break
			}
		}
		s.pending = values[next]
		s.message = "pending " + tuiConfigSettingRef(s.current()) + "=" + s.pending
	case 's':
		if err := s.applyCurrent("save"); err != nil {
			return false, err
		}
	case 'r':
		if err := s.applyCurrent("reset"); err != nil {
			return false, err
		}
	case '\r', '\n':
		return false, nil
	default:
		s.message = fmt.Sprintf("ignored key %q", string(key))
	}
	return false, nil
}

func (s *tuiConfigInteraction) applyCurrent(action string) error {
	ref := tuiConfigSettingRef(s.current())
	mutation := tuiConfigMutation{SetDefaults: []string{ref + "=" + s.pending}}
	if action == "reset" {
		mutation = tuiConfigMutation{UnsetDefaults: []string{ref}}
	}
	report, err := mutateTUIConfig(s.path, mutation, s.at)
	if err != nil {
		return fmt.Errorf("%s %s: %w", action, ref, err)
	}
	s.report = report
	s.refreshSettings()
	s.pending = s.current().Effective
	if action == "reset" {
		s.message = "reset " + ref + " to built-in"
	} else {
		s.message = "saved " + ref + "=" + s.pending
	}
	return nil
}

func tuiConfigInteractiveValues(setting tuiConfigSetting) []string {
	if strings.EqualFold(setting.Kind, "toggle") {
		return []string{"false", "true"}
	}
	return append([]string(nil), setting.Options...)
}

func tuiConfigSettingRef(setting tuiConfigSetting) string {
	return strings.ToLower(setting.Pane + "." + setting.Control)
}

func (s *tuiConfigInteraction) render(w io.Writer) {
	setting := s.current()
	line := func(value string) { fmt.Fprintln(w, trimTUI(value, s.width)) }
	line(fmt.Sprintf("fak console settings  interactive  %d/%d", s.selected+1, len(s.settings)))
	line("keys: j next | k previous | c/space change | s save | r reset | q exit")
	line("path=" + s.report.Path)
	line(fmt.Sprintf("> %s  effective=%s  source=%s", tuiConfigSettingRef(setting), setting.Effective, setting.Source))
	line("  choices=" + strings.Join(tuiConfigInteractiveValues(setting), "|") + "  pending=" + s.pending)
	if s.message != "" {
		line("status=" + s.message)
	}
}
