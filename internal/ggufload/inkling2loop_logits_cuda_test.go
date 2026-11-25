//go:build cuda && cgo

package ggufload

import (
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/logitoracle"
)

// ink2LogitSession drives the resident decode for a logitoracle dump.
type ink2LogitSession struct {
	t *testing.T
	s *inkling2Resident
}

func (l *ink2LogitSession) feed(ids []int) {
	if l.s.batched && len(ids) > 1 {
		if _, err := l.s.prefill(ids); err != nil {
			l.t.Fatal(err)
		}
		return
	}
	for _, id := range ids {
		if _, err := l.s.step(id); err != nil {
			l.t.Fatal(err)
		}
	}
}

func (l *ink2LogitSession) read() []float32 {
	v, err := l.s.lastLogits()
	if err != nil {
		l.t.Fatal(err)
	}
	return v
}

func (l *ink2LogitSession) Prefill(ids []int) []float32 { l.feed(ids); return l.read() }
func (l *ink2LogitSession) PrefillNoLogits(ids []int)   { l.feed(ids) }
func (l *ink2LogitSession) Step(id int) []float32       { l.feed([]int{id}); return l.read() }

// FAK_INKLING2_LOGIT_DUMP=<dir> writes a logitoracle dump of the resident decode over
// the oracle prompts, one logit vector per greedy oracle position, for
// `fak compute-trace logits compare`. FAK_INKLING2_LOGIT_TEACHER=<dir> feeds another
// dump's tokens so every position compares; FAK_INKLING2_LOGIT_LABEL names the run.
func TestInkling2CUDALogitDump(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_LOGIT_DUMP")
	defer m.Close()
	if !inkling2ResidentEligible(m, be) {
		t.Skip("resident forward not eligible on this backend")
	}
	env := map[string]string{}
	for _, kv := range os.Environ() {
		if k, v, ok := strings.Cut(kv, "="); ok && strings.HasPrefix(k, "FAK_INKLING2_") && !strings.HasPrefix(k, "FAK_INKLING2_LOGIT_") {
			env[k] = v
		}
	}
	opts := logitoracle.DumpOptions{
		Dir:          os.Getenv("FAK_INKLING2_LOGIT_DUMP"),
		Label:        os.Getenv("FAK_INKLING2_LOGIT_LABEL"),
		PrefillChunk: inkling2ResidentCtx,
		TopK:         5,
		Env:          env,
		NewSession: func() (logitoracle.Session, func(), error) {
			s, err := newInkling2Resident(m, be)
			if err != nil {
				return nil, nil, err
			}
			// a CUDA stream capture is bound to the OS thread that opened it
			runtime.LockOSThread()
			return &ink2LogitSession{t: t, s: s}, func() { s.close(); runtime.UnlockOSThread() }, nil
		},
	}
	if dir := os.Getenv("FAK_INKLING2_LOGIT_TEACHER"); dir != "" {
		teacher, err := logitoracle.ReadManifest(dir)
		if err != nil {
			t.Fatal(err)
		}
		opts.Teacher = teacher
	}
	var prompts []logitoracle.PromptSpec
	for _, c := range inkling2OracleCases {
		prompts = append(prompts, logitoracle.PromptSpec{Name: c.name, IDs: c.prompt, Positions: len(c.greedy)})
	}
	mf, err := logitoracle.Dump(prompts, opts)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range mf.Prompts {
		t.Logf("logit dump %s: %d positions, prefill %.2fs, decode %.2fs, argmax head %v",
			p.Name, p.Positions, float64(p.PrefillNS)/1e9, float64(p.DecodeNS)/1e9, p.Argmax[:min(8, len(p.Argmax))])
	}
}
