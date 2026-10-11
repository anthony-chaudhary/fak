package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak#13668: the device-ring route reads its dense base from the checkpoint map, so every
// streamed-dense tensor is a lazy range backed by file pages a device upload copies directly.
// fak-test:runtime fast est=300ms lane=default
func TestStreamedExpertsMappedLoadBacksDenseBaseWithCheckpointMap(t *testing.T) {
	path := writeDeepSeek41Q2KDenseFile(t)
	opts := []Q4KLoadOption{WithDenseQ2KResident(true), WithStreamedDenseQ4KWorkingSet(1 << 20)}
	m, err := loadModelQ4KStreamedExpertsContext(t.Context(), path, nil, 0, OpenWeightsMapped, opts...)
	if err != nil {
		t.Fatalf("mapped streamed-experts load: %v", err)
	}
	defer m.CloseWeights()
	ws, err := OpenWeightsMapped(path)
	if err != nil {
		t.Fatalf("OpenWeightsMapped: %v", err)
	}
	defer ws.Close()
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	o, err := resolveQ4KLoadOptions(cfg, append(opts, WithStreamedExperts(0)))
	if err != nil {
		t.Fatalf("resolve options: %v", err)
	}
	// Native Windows has no mmap impl and degrades to ReadAt; the span half is witnessed on Linux.
	mappable := len(ws.data) > 0
	dense := 0
	for _, info := range ws.File.Tensors {
		canon, ok := CanonicalTensorNameArch(info.Name, cfg.ModelType)
		if !ok {
			continue
		}
		if canon, ok = model.QuantSourceTensorName(cfg, canon); !ok || !streamedDenseLazy(cfg, o, info.Type, canon) {
			continue
		}
		dense++
		if !m.Q4KLazy(canon) && !m.KQuantLazy(canon) {
			t.Errorf("%s was copied onto the host heap, want a lazy checkpoint range", info.Name)
		}
		if !mappable {
			continue
		}
		r, size, err := ws.tensorReader(info)
		if err != nil {
			t.Fatalf("tensorReader(%s): %v", info.Name, err)
		}
		n, err := tensorPayloadBytes(info)
		if err != nil {
			t.Fatalf("payload(%s): %v", info.Name, err)
		}
		if _, mapped := ws.mappedQ4KReader(info, r, size, int(n)).(*mappedQ4KReaderAt); !mapped {
			t.Errorf("%s has no checkpoint-map span, so its upload would stage an anonymous copy", info.Name)
		}
	}
	if dense == 0 {
		t.Fatal("fixture carries no streamed-dense tensor; the witness is vacuous")
	}
}
