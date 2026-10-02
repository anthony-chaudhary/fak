package model

import (
	"fmt"
	"io"
	"sort"
)

// RetainMappedQ4KResidency prepares an ordinary resident model from the
// lifetime-transferring dense loader, before publication or session creation.
// Valid mapped payloads remain shared by CPU and Metal; other payloads are read
// into owned, page-aligned storage. Models with a positive dense bound or an
// expert streaming tier are refused. Callers must keep explicit streaming loads
// on their original path, including the zero-budget stream-through policy.
// A failed read leaves all tensor state and checkpoint ownership unchanged.
func (m *Model) RetainMappedQ4KResidency() error {
	if m == nil {
		return fmt.Errorf("model: cannot retain mapped Q4_K residency on a nil model")
	}
	owner := m.ensureWeightCloser()
	owner.mu.Lock()
	defer owner.mu.Unlock()
	if owner.closing || owner.closed {
		return fmt.Errorf("model: cannot retain mapped Q4_K residency after weight close has started")
	}
	if owner.sessions != 0 {
		return fmt.Errorf("model: cannot retain mapped Q4_K residency with %d active session(s)", owner.sessions)
	}
	if m.denseResidentBoundBytes > 0 || m.expertCheckpoint != nil {
		return fmt.Errorf("model: mapped Q4_K residency requires an ordinary unbounded resident load")
	}

	var prepared []mappedResidentWeight
	for name, qt := range m.q4kw {
		if qt != nil && qt.lazy != nil && len(qt.raw) == 0 {
			prepared = append(prepared, mappedResidentWeight{name: name, q4k: qt})
		}
	}
	for name, qt := range m.kqw {
		if qt != nil && qt.lazy != nil && len(qt.raw) == 0 {
			prepared = append(prepared, mappedResidentWeight{name: name, kquant: qt})
		}
	}
	if len(prepared) == 0 {
		return nil
	}
	if owner.c == nil {
		return fmt.Errorf("model: mapped Q4_K residency requires a retained checkpoint owner")
	}
	sort.Slice(prepared, func(i, j int) bool { return prepared[i].name < prepared[j].name })
	for i := range prepared {
		weight := &prepared[i]
		var lazy *LazyQ4KRange
		if qt := weight.q4k; qt != nil {
			lazy = qt.lazy
		} else {
			lazy = weight.kquant.lazy
		}
		var err error
		weight.raw, err = mappedResidentPayload(lazy)
		if err != nil {
			return fmt.Errorf("model: retain mapped Q4_K residency for %s: %w", weight.name, err)
		}
	}
	// The dense descriptor loader also defers eligible non-Q4 K-quants. Prepare
	// them here so this resident selection preserves their eager CPU residency.
	for i := range prepared {
		weight := &prepared[i]
		if weight.q4k != nil {
			weight.q4k.raw = weight.raw
		} else {
			weight.kquant.raw = weight.raw
		}
		weight.raw = nil
		weight.name = ""
	}
	owner.c = &mappedResidentWeightCloser{owner: owner.c, weights: prepared}
	return nil
}

func mappedResidentPayload(lazy *LazyQ4KRange) ([]byte, error) {
	if span, offset, ok := mappedLazyRaw(lazy); ok {
		end := offset + lazy.Bytes
		return span[offset:end:end], nil
	}
	// The shared materializer is format-agnostic: it reads only the range's
	// byte count and offset. Use a fresh tensor and copied descriptor so reader
	// validation cannot mutate the unpublished Q4 or K-quant tensor state.
	tensor := q4kTensor{}
	if lazy != nil {
		source := *lazy
		if source.Reader != nil {
			source.Reader = mappedResidencyReaderAt{source.Reader}
		}
		tensor.lazy = &source
	}
	return tensor.materializeRaw()
}

type mappedResidencyReaderAt struct{ io.ReaderAt }

func (r mappedResidencyReaderAt) ReadAt(p []byte, offset int64) (int, error) {
	n, err := r.ReaderAt.ReadAt(p, offset)
	if n != len(p) && err == nil {
		return n, io.ErrUnexpectedEOF
	}
	return n, err
}

type mappedResidentWeight struct {
	name   string
	q4k    *q4kTensor
	kquant *kQuantTensor
	raw    []byte
}

type mappedResidentWeightCloser struct {
	owner   io.Closer
	weights []mappedResidentWeight
}

func (c *mappedResidentWeightCloser) Close() error {
	// CloseWeights has already detached sessions and released native handles.
	// Drop CPU views before the checkpoint owner unmaps their shared backing.
	for _, weight := range c.weights {
		if weight.q4k != nil {
			weight.q4k.raw, weight.q4k.lazy = nil, nil
		} else {
			weight.kquant.raw, weight.kquant.lazy = nil, nil
		}
	}
	c.weights = nil
	return c.owner.Close()
}
