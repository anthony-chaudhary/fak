package main

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"regexp"
	"strconv"
	"strings"
	"syscall"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// The inkling2 family is not a fakmodel.Model: its looped/attn-res graph runs on
// ggufload's own host forward, with packed weights moved onto CUDA when present.
const serveInkling2ArchBase = "inkling2"

// serveInkling2Resource names what a refused inkling2 load is short of.
type serveInkling2Resource string

const (
	serveInkling2Disk serveInkling2Resource = "disk"
	serveInkling2RAM  serveInkling2Resource = "ram"
	serveInkling2VRAM serveInkling2Resource = "vram"
	serveInkling2CUDA serveInkling2Resource = "cuda"
)

// serveInkling2ResourceError refuses an inkling2 GGUF that cannot load at all on
// this host, naming the missing resource.
type serveInkling2ResourceError struct {
	Resource serveInkling2Resource
	Path     string
	Detail   string
	Err      error
}

func (e *serveInkling2ResourceError) Error() string {
	msg := fmt.Sprintf("inkling2: cannot load %s: insufficient %s: %s", e.Path, e.Resource, e.Detail)
	if e.Err != nil {
		msg += ": " + e.Err.Error()
	}
	return msg
}

func (e *serveInkling2ResourceError) Unwrap() error { return e.Err }

var serveInkling2ShardRe = regexp.MustCompile(`^(.*-)(\d+)-of-(\d+)\.gguf$`)

// serveInkling2ShardSet lists every shard of a -N-of-M.gguf path in order, or
// just path for a single-file GGUF.
func serveInkling2ShardSet(path string) []string {
	m := serveInkling2ShardRe.FindStringSubmatch(path)
	if m == nil {
		return []string{path}
	}
	width := len(m[2])
	count, err := strconv.Atoi(m[3])
	if err != nil || count < 1 {
		return []string{path}
	}
	out := make([]string, count)
	for i := range out {
		out[i] = fmt.Sprintf("%s%0*d-of-%s.gguf", m[1], width, i+1, m[3])
	}
	return out
}

// serveInkling2Arch returns the GGUF's general.architecture when it is an
// inkling2-family arch, "" otherwise. A non-first split shard carries no
// architecture, so the set's first shard is read instead.
func serveInkling2Arch(path string) (string, error) {
	f, err := ggufload.Open(path)
	if err != nil {
		return "", err
	}
	arch, ok := f.String("general.architecture")
	if !ok {
		if set := serveInkling2ShardSet(path); len(set) > 1 && set[0] != path {
			if f, err = ggufload.Open(set[0]); err != nil {
				return "", err
			}
			arch, _ = f.String("general.architecture")
		}
	}
	if arch == serveInkling2ArchBase || arch == serveInkling2LoopArch {
		return arch, nil
	}
	return "", nil
}

// serveInkling2Loaded is a bound inkling2 model. Backend is nil on the host
// forward.
type serveInkling2Loaded struct {
	Model     *ggufload.Inkling2LoopModel
	Backend   compute.Backend
	Arch      string
	Placement string
}

func (l *serveInkling2Loaded) Close() error { return l.Model.Close() }

// loadServeInkling2 is the non-Metal inkling2 GGUF load arm of fak run and fak
// serve. On a CUDA backend the weights are bound under the free-VRAM budget,
// spilling cold expert stacks to host with no flag; any other backend runs the
// host forward over the mapped GGUF. The placement line is written to logw once.
func loadServeInkling2(path string, backend compute.Backend, logw io.Writer) (*serveInkling2Loaded, error) {
	if err := serveInkling2DiskPreflight(path); err != nil {
		return nil, err
	}
	if backend != nil && backend.Name() == "cuda" {
		return loadServeInkling2Device(path, backend, logw)
	}
	host, err := ggufload.OpenInkling2Loop(path)
	if err != nil {
		return nil, serveInkling2ClassifyOpenError(path, err)
	}
	l := &serveInkling2Loaded{Model: host, Arch: host.Arch}
	l.Placement = fmt.Sprintf("inkling2 placement: host forward (no CUDA backend), %d layers, %d experts top-%d mapped from %d shard(s)",
		host.HP.Layers, host.HP.Experts, host.HP.ExpertsUsed, len(serveInkling2ShardSet(path)))
	if logw != nil {
		fmt.Fprintln(logw, l.Placement)
	}
	return l, nil
}

func loadServeInkling2Device(path string, backend compute.Backend, logw io.Writer) (*serveInkling2Loaded, error) {
	dev, err := ggufload.OpenInkling2LoopDevice(path, backend)
	if err != nil {
		return nil, serveInkling2ClassifyDeviceOpen(path, err)
	}
	l := &serveInkling2Loaded{Model: dev, Backend: backend, Arch: dev.Arch}
	p, ok := ggufload.Inkling2DevicePlacementOf(dev)
	if ok {
		l.Placement = p.String()
	} else {
		l.Placement = "inkling2 placement: device bind (no placement plan recorded)"
	}
	// The device open already prints a spilled placement to stderr.
	if logw != nil && !(ok && !p.FullyResident() && logw == io.Writer(os.Stderr)) {
		fmt.Fprintln(logw, l.Placement)
	}
	return l, nil
}

// serveInkling2ClassifyDeviceOpen separates a device shortfall from a
// checkpoint that would not bind on the host either. The host re-open runs only
// on this failure path, so a successful load maps the set once.
func serveInkling2ClassifyDeviceOpen(path string, err error) error {
	var over *ggufload.Inkling2DenseOverBudgetError
	if errors.As(err, &over) {
		return serveInkling2ClassifyDeviceError(path, err)
	}
	host, hostErr := ggufload.OpenInkling2Loop(path)
	if hostErr != nil {
		return serveInkling2ClassifyOpenError(path, hostErr)
	}
	host.Close()
	return serveInkling2ClassifyDeviceError(path, err)
}

// serveInkling2DiskPreflight refuses a split set with a missing, empty or
// truncated shard before any mapping, so a partial download names the shard it
// lacks. Only headers are read: the check is cheap on a ~191 GiB set.
func serveInkling2DiskPreflight(path string) error {
	for _, p := range serveInkling2ShardSet(path) {
		st, err := os.Stat(p)
		if err != nil {
			return &serveInkling2ResourceError{Resource: serveInkling2Disk, Path: path, Detail: "shard " + p + " is not on disk", Err: err}
		}
		if st.Size() == 0 {
			return &serveInkling2ResourceError{Resource: serveInkling2Disk, Path: path, Detail: "shard " + p + " is empty"}
		}
		if err := serveInkling2ShardComplete(p, st.Size()); err != nil {
			return &serveInkling2ResourceError{Resource: serveInkling2Disk, Path: path, Detail: "shard " + p + " is truncated (incomplete download)", Err: err}
		}
	}
	return nil
}

// serveInkling2ShardComplete reports a shard whose header or tensor table
// extends past its on-disk size.
func serveInkling2ShardComplete(p string, size int64) error {
	f, err := ggufload.Open(p)
	if err != nil {
		if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
			return err
		}
		return nil // not a size problem; the bind reports format errors itself
	}
	for _, t := range f.Tensors {
		if end := f.TensorDataOffset + int64(t.Offset) + t.OnDiskBytes(); end > size {
			return fmt.Errorf("tensor %s ends at byte %d of %d", t.Name, end, size)
		}
	}
	return nil
}

func serveInkling2ClassifyOpenError(path string, err error) error {
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return &serveInkling2ResourceError{Resource: serveInkling2Disk, Path: path, Detail: "a shard is not on disk", Err: err}
	case serveInkling2OutOfMemory(err):
		return &serveInkling2ResourceError{Resource: serveInkling2RAM, Path: path, Detail: "the host cannot map the weights", Err: err}
	}
	return err
}

func serveInkling2ClassifyDeviceError(path string, err error) error {
	var over *ggufload.Inkling2DenseOverBudgetError
	switch {
	case errors.As(err, &over):
		return &serveInkling2ResourceError{Resource: serveInkling2VRAM, Path: path,
			Detail: fmt.Sprintf("dense weights need %.2f GiB, device budget %.2f GiB", float64(over.DenseBytes)/(1<<30), float64(over.BudgetBytes)/(1<<30)), Err: err}
	case serveInkling2OutOfMemory(err):
		return &serveInkling2ResourceError{Resource: serveInkling2VRAM, Path: path, Detail: "device allocation failed", Err: err}
	}
	return &serveInkling2ResourceError{Resource: serveInkling2CUDA, Path: path, Detail: "the CUDA bind failed", Err: err}
}

func serveInkling2OutOfMemory(err error) bool {
	if errors.Is(err, syscall.ENOMEM) {
		return true
	}
	s := strings.ToLower(err.Error())
	for _, needle := range []string{"out of memory", "cannot allocate memory", "not enough memory", "paging file"} {
		if strings.Contains(s, needle) {
			return true
		}
	}
	return false
}
