// Package tuiplugin is the in-process extension seam for fak console panes.
//
// It deliberately mirrors fak's other Register-from-init seams: pane packages
// register metadata plus a runner, the dispatcher looks panes up by id, and help
// or control surfaces read immutable descriptors without importing pane code.
// This is not dynamic loading; a pane is compiled into the fak binary like every
// other built-in driver.
package tuiplugin

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
)

// Runner executes one console pane with pane-local argv.
type Runner func(stdout, stderr io.Writer, argv []string) int

// OverviewBuilder projects a pane-specific snapshot into an overview card. The
// context is owned by the caller so the registry stays independent of cmd/fak
// report types.
type OverviewBuilder func(ctx any) (OverviewCard, error)

// OverviewCard is the registry-level summary shape a pane contributes to an
// operator overview. cmd/fak converts it to the wire type it already exposes.
type OverviewCard struct {
	Pane      string
	Status    string
	Source    string
	Summary   string
	Command   string
	Attention int
	Counts    map[string]int
	Tags      []string
}

// Control describes one operator-facing control exposed by a pane. The command
// line remains the execution boundary; these rows are the stable discovery model
// an interactive shell or external control pane can render.
type Control struct {
	ID      string   `json:"id"`
	Label   string   `json:"label"`
	Kind    string   `json:"kind"` // flag, toggle, action, input, launch
	Flag    string   `json:"flag,omitempty"`
	Default string   `json:"default,omitempty"`
	Options []string `json:"options,omitempty"`
	Command string   `json:"command,omitempty"`
	Detail  string   `json:"detail,omitempty"`
}

// Pane is the write-side registration record.
type Pane struct {
	ID       string
	Summary  string
	Usage    string
	Schema   string
	BuiltIn  bool
	Controls []Control
	Run      Runner
	Overview OverviewBuilder
}

// Descriptor is the read-side, JSON-safe pane shape. It intentionally excludes
// Run so discovery can be exposed directly to users.
type Descriptor struct {
	ID       string    `json:"id"`
	Summary  string    `json:"summary"`
	Usage    string    `json:"usage,omitempty"`
	Schema   string    `json:"schema,omitempty"`
	BuiltIn  bool      `json:"built_in"`
	Overview bool      `json:"overview"`
	Controls []Control `json:"controls,omitempty"`
}

type snapshot struct {
	ordered []Pane
	byID    map[string]Pane
	descs   []Descriptor
}

var (
	mu        sync.Mutex
	registry  = map[string]Pane{}
	published atomic.Pointer[snapshot]
)

// Register adds a console pane to the registry. It panics on malformed or
// duplicate ids so registration conflicts fail during process startup.
func Register(p Pane) {
	id := strings.TrimSpace(p.ID)
	if !validID(id) {
		panic(fmt.Sprintf("tuiplugin: invalid pane id %q", p.ID))
	}
	if p.Run == nil {
		panic(fmt.Sprintf("tuiplugin: pane %q has nil runner", id))
	}
	p.ID = id
	p.Summary = strings.TrimSpace(p.Summary)
	p.Usage = strings.TrimSpace(p.Usage)
	p.Schema = strings.TrimSpace(p.Schema)
	p.Controls = normalizeControls(p.Controls)

	mu.Lock()
	defer mu.Unlock()
	if _, exists := registry[id]; exists {
		panic(fmt.Sprintf("tuiplugin: duplicate pane id %q", id))
	}
	registry[id] = p
	publishLocked()
}

// Lookup returns a registered pane by id.
func Lookup(id string) (Pane, bool) {
	s := current()
	p, ok := s.byID[strings.TrimSpace(id)]
	if ok {
		p = clonePane(p)
	}
	return p, ok
}

// All returns the registered panes in stable id order.
func All() []Pane {
	s := current()
	out := make([]Pane, len(s.ordered))
	for i, p := range s.ordered {
		out[i] = clonePane(p)
	}
	return out
}

// Descriptors returns JSON-safe pane descriptors in stable id order.
func Descriptors() []Descriptor {
	s := current()
	out := make([]Descriptor, len(s.descs))
	for i, d := range s.descs {
		out[i] = cloneDescriptor(d)
	}
	return out
}

// ResetForTest clears the registry. It is only for package tests.
func ResetForTest() {
	mu.Lock()
	defer mu.Unlock()
	registry = map[string]Pane{}
	publishLocked()
}

func publishLocked() {
	ordered := make([]Pane, 0, len(registry))
	for _, p := range registry {
		ordered = append(ordered, p)
	}
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].ID < ordered[j].ID })
	byID := make(map[string]Pane, len(ordered))
	descs := make([]Descriptor, 0, len(ordered))
	for i, p := range ordered {
		p = clonePane(p)
		ordered[i] = p
		byID[p.ID] = p
		descs = append(descs, Descriptor{
			ID:       p.ID,
			Summary:  p.Summary,
			Usage:    p.Usage,
			Schema:   p.Schema,
			BuiltIn:  p.BuiltIn,
			Overview: p.Overview != nil,
			Controls: cloneControls(p.Controls),
		})
	}
	published.Store(&snapshot{ordered: ordered, byID: byID, descs: descs})
}

func current() *snapshot {
	if s := published.Load(); s != nil {
		return s
	}
	return &snapshot{byID: map[string]Pane{}}
}

func validID(id string) bool {
	if id == "" {
		return false
	}
	for i, r := range id {
		switch {
		case r >= 'a' && r <= 'z':
		case r >= '0' && r <= '9' && i > 0:
		case r == '-' && i > 0:
		default:
			return false
		}
	}
	return true
}

func clonePane(p Pane) Pane {
	p.Controls = cloneControls(p.Controls)
	return p
}

func cloneDescriptor(d Descriptor) Descriptor {
	d.Controls = cloneControls(d.Controls)
	return d
}

func cloneControls(in []Control) []Control {
	out := make([]Control, len(in))
	for i, c := range in {
		c.Options = append([]string(nil), c.Options...)
		out[i] = c
	}
	return out
}

func normalizeControls(in []Control) []Control {
	out := make([]Control, 0, len(in))
	for _, c := range in {
		c.ID = strings.TrimSpace(c.ID)
		c.Label = strings.TrimSpace(c.Label)
		c.Kind = strings.TrimSpace(c.Kind)
		c.Flag = strings.TrimSpace(c.Flag)
		c.Default = strings.TrimSpace(c.Default)
		c.Command = strings.TrimSpace(c.Command)
		c.Detail = strings.TrimSpace(c.Detail)
		c.Options = normalizeOptions(c.Options)
		if c.ID == "" {
			continue
		}
		out = append(out, c)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func normalizeOptions(in []string) []string {
	out := make([]string, 0, len(in))
	seen := map[string]bool{}
	for _, option := range in {
		option = strings.TrimSpace(option)
		if option == "" || seen[option] {
			continue
		}
		seen[option] = true
		out = append(out, option)
	}
	return out
}
