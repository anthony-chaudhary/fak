package model

import (
	"errors"
	"math"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41EngramStepDeclineReusesGather(t *testing.T) {
	for _, mode := range []string{"decline-cold", "decline-warm", "handled", "gate-error", "host-invalid", "retained-owner"} {
		t.Run(mode, func(t *testing.T) {
			m, _ := v41FullEngramModel(t)
			H := m.Cfg.HiddenSize
			clear(m.tensor(layerName(0, "engram_q_norm.weight")))
			for i := range m.tensor(layerName(0, "engram_k_norm.weight")) {
				m.tensor(layerName(0, "engram_k_norm.weight"))[i] = 1
			}
			g, err := m.v41EngramInjectionGeometry(0, 1)
			if err != nil {
				t.Fatal(err)
			}
			ids := make([]uint32, g.cols)
			for i := range ids {
				ids[i] = uint32(i % 2)
			}
			cache := g.stage.caches[g.cacheIdx]
			refRows, err := GatherV41EngramRows([]*V41EngramRowCache{cache}, ids, g.cols)
			if err != nil {
				t.Fatal(err)
			}
			wantInput := make([]float32, g.cols*g.dim)
			if err := v41EngramDecodePosition(0, g.stage, g.cacheIdx, refRows, 0, g.cols, g.dim, wantInput); err != nil {
				t.Fatal(err)
			}
			if mode != "decline-warm" {
				cache.invalidateRows()
			}
			projection := make([]float32, (g.hc+1)*H)
			for i := range projection {
				projection[i] = 1
			}
			projectCalls := 0
			st := &v41ForwardState{engramProjection: func(layer int, row []float32, out, in int) ([]float32, v41DenseProjectionOutcome, error) {
				projectCalls++
				if layer != 0 || out != len(projection) || in != len(wantInput) || len(row) != len(wantInput) {
					t.Fatal("fallback projection geometry changed")
				}
				for i, value := range row {
					if math.Float32bits(value) != math.Float32bits(wantInput[i]) {
						t.Fatalf("fallback projection input[%d]=%g want=%g", i, value, wantInput[i])
					}
				}
				return projection, v41ProjectionHandled, nil
			}}
			x := make([]float32, H)
			for i := range x {
				x[i] = 1
			}
			be := newV41MHCTestBackend(true)
			s := v41DenseTestSession(t, m, be)
			resident, err := s.v41MHCDeviceSeam().newResident([][]float32{x})
			if err != nil {
				t.Fatal(err)
			}
			defer resident.free()
			gateCalls, rollbacks := 0, 0
			cause := errors.New("selected decline error must not replay")
			ops := &v41DeviceGatherOps{owner: s, engramGate: func(layer int, streams *v41DeviceStreams, gathered *v41EngramGatheredRows, positions int, geometry v41EngramGeometry, eps float32) (bool, error) {
				gateCalls++
				if layer != 0 || streams != resident.streams || positions != 1 || gathered == nil ||
					gathered.layer != layer || gathered.stage != g.stage || gathered.cacheIdx != g.cacheIdx ||
					gathered.dim != g.dim || len(gathered.rows) != len(ids) || geometry.stage != g.stage {
					t.Fatal("selected gate lost the current-call row owner")
				}
				if mode == "gate-error" {
					return false, cause
				}
				if mode == "retained-owner" {
					gathered.layer++
				}
				return mode == "handled", nil
			}}
			scratch := &v41ProjScratch{gatherOps: ops}
			active := resident
			if mode == "host-invalid" {
				active = nil
				x = x[:H-1]
			}
			before, uploads := cache.Stats(), be.streamUploads
			err = m.v41StepInjectEngram(0, v41StepEngram{stage: g.stage, rows: ids, full: true}, st, x, nil, active, scratch, func() { rollbacks++ })
			wantProjects, wantGates, wantRollbacks, wantUploads := 1, 1, 0, 1
			wantRows := int64(len(ids))
			switch mode {
			case "handled":
				wantProjects, wantUploads = 0, 0
			case "gate-error":
				wantProjects, wantUploads, wantRollbacks = 0, 0, 1
				if !errors.Is(err, cause) {
					t.Fatalf("gate error=%v, want original cause", err)
				}
			case "host-invalid":
				wantProjects, wantUploads, wantGates, wantRollbacks, wantRows = 0, 0, 0, 1, 0
				if !errors.Is(err, ErrV41NativeUnsupported) {
					t.Fatalf("malformed host input error=%v", err)
				}
			case "retained-owner":
				wantProjects, wantUploads, wantRollbacks = 0, 0, 1
				if !errors.Is(err, ErrV41NativeUnsupported) {
					t.Fatalf("mismatched retained owner error=%v, want typed refusal", err)
				}
			}
			if mode != "gate-error" && mode != "host-invalid" && mode != "retained-owner" && err != nil {
				t.Fatal(err)
			}
			if projectCalls != wantProjects || gateCalls != wantGates || rollbacks != wantRollbacks || be.streamUploads-uploads != wantUploads {
				t.Fatalf("calls projection/gate/rollback/upload=%d/%d/%d/%d want=%d/%d/%d/%d", projectCalls, gateCalls, rollbacks, be.streamUploads-uploads, wantProjects, wantGates, wantRollbacks, wantUploads)
			}
			host, err := resident.hostStreams(0)
			if err != nil {
				t.Fatal(err)
			}
			want := float32(1)
			if wantProjects == 1 {
				want = 1.5
			}
			for _, stream := range host[0] {
				for _, value := range stream {
					if math.Float32bits(value) != math.Float32bits(want) {
						t.Fatalf("resident publication=%g want=%g", value, want)
					}
				}
			}
			for _, value := range x {
				if math.Float32bits(value) != math.Float32bits(want) {
					t.Fatalf("hidden publication=%g want=%g", value, want)
				}
			}
			after := cache.Stats()
			requests := after.Hits + after.Misses - before.Hits - before.Misses
			served := after.RowsServed - before.RowsServed
			t.Logf("incremental Engram cache requests=%d rows served=%d expected=%d", requests, served, wantRows)
			if requests != wantRows || served != wantRows {
				t.Fatalf("incremental Engram repeated gather: requests=%d served=%d want one gather of %d rows", requests, served, wantRows)
			}
		})
	}
}
