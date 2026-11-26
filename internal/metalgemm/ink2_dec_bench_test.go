//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// The decode-ops microbenchmark times arms of one op as the resident forward
// runs them: many launches in one serial command buffer, so a launch pays its
// barrier drain and its latency, charged the buffer's GPU time. Arms of a case
// alternate within a repetition so a busy GPU loads them alike.

const (
	ink2OpsBenchEnv     = "FAK_INK2_MICROBENCH"
	ink2OpsBenchRepsEnv = "FAK_INK2_MICROBENCH_REPS"
	ink2OpsBenchOnlyEnv = "FAK_INK2_MICROBENCH_ONLY"
)

type ink2OpsBenchArm struct {
	name   string
	encode func(site int) error
}

type ink2OpsBenchCase struct {
	name  string
	sites int
	arms  []ink2OpsBenchArm
}

func ink2OpsBenchReps(t *testing.T) int {
	reps := 15
	if s := os.Getenv(ink2OpsBenchRepsEnv); s != "" {
		v, err := strconv.Atoi(s)
		if err != nil || v < 1 {
			t.Fatalf("%s=%q", ink2OpsBenchRepsEnv, s)
		}
		reps = v
	}
	return reps
}

func ink2OpsBenchDevice(t *testing.T) *Ink2Device {
	t.Helper()
	if os.Getenv(ink2OpsBenchEnv) != "1" {
		t.Skipf("set %s=1 to time the decode ops on the GPU", ink2OpsBenchEnv)
	}
	if !Available() {
		t.Skip("requires a Metal device")
	}
	d, err := newInk2Device(Ink2ProfileOff)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	return d
}

// ink2OpsBenchRun reports, per arm, the best and median GPU microseconds per
// launch over reps buffers of launches launches each.
func ink2OpsBenchRun(t *testing.T, d *Ink2Device, cases []ink2OpsBenchCase, launches, reps int) {
	t.Helper()
	only := os.Getenv(ink2OpsBenchOnlyEnv)
	var b strings.Builder
	for _, c := range cases {
		if only != "" && !strings.Contains(c.name, only) {
			continue
		}
		us := make([][]float64, len(c.arms))
		for rep := -1; rep < reps; rep++ {
			order := make([]int, len(c.arms))
			for i := range order {
				order[i] = i
			}
			if rep%2 != 0 {
				slices.Reverse(order)
			}
			for _, a := range order {
				before := d.Stats().GPUSeconds
				for i := range launches {
					if err := c.arms[a].encode(i % c.sites); err != nil {
						t.Fatalf("%s %s: %v", c.name, c.arms[a].name, err)
					}
				}
				if err := d.Flush(); err != nil {
					t.Fatal(err)
				}
				if rep >= 0 {
					us[a] = append(us[a], 1e6*(d.Stats().GPUSeconds-before)/float64(launches))
				}
			}
		}
		for a, arm := range c.arms {
			v := slices.Clone(us[a])
			sort.Float64s(v)
			fmt.Fprintf(&b, "DECBENCH %-34s %-22s best %7.2f  median %7.2f us\n", c.name, arm.name, v[0], v[len(v)/2])
		}
	}
	t.Log("\n" + b.String())
}

// ink2OpsBenchFill writes a cheap deterministic pattern of amplitude amp.
func ink2OpsBenchFill(t *testing.T, d *Ink2Device, n int, seed uint32, amp float32) Ink2Ref {
	t.Helper()
	r := ink2TestAlloc(t, d, n)
	v := make([]float32, n)
	x := seed*2654435761 + 1
	for i := range v {
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		v[i] = amp * (float32(x>>8)/float32(1<<23) - 1)
	}
	if err := d.Write(r, v); err != nil {
		t.Fatal(err)
	}
	return r
}

// fak-test:runtime slow est=120s
func TestInk2DecOpsMicrobench(t *testing.T) {
	d := ink2OpsBenchDevice(t)
	const n, rank, sites = 2304, 64, 16
	seed := uint32(0)
	put := func(count int, amp float32) Ink2Ref { seed++; return ink2OpsBenchFill(t, d, count, seed, amp) }
	type site struct {
		states       []Ink2Ref
		queries, out Ink2Ref
		norms        []Ink2Ref
		gate, act    Ink2Ref
		g            Ink2Ref
		ctl          Ink2Controls
	}
	ss := make([]site, sites)
	for i := range ss {
		s := &ss[i]
		for range 16 {
			s.states = append(s.states, put(n, 1))
		}
		s.queries, s.out = put(4*n, 1), ink2TestAlloc(t, d, 4*n)
		s.norms = []Ink2Ref{put(n, 1), put(n, 1), put(n, 1), put(n, 1)}
		s.gate, s.act = put(n, 1), put(n, 1)
		s.g = ink2TestAlloc(t, d, 4*rank)
		for j := range 4 {
			s.ctl.Cur = append(s.ctl.Cur, s.states[j])
			s.ctl.Msg = append(s.ctl.Msg, s.states[4+j])
			s.ctl.Down = append(s.ctl.Down, put(rank*2*n, 0.02))
			s.ctl.Gain = append(s.ctl.Gain, put(n*rank, 0.1))
			s.ctl.Forget = append(s.ctl.Forget, put(n*rank, 0.1))
		}
	}
	sub := func(c Ink2Controls, lo, hi int) Ink2Controls {
		return Ink2Controls{Cur: c.Cur[lo:hi], Msg: c.Msg[lo:hi], Down: c.Down[lo:hi], Gain: c.Gain[lo:hi], Forget: c.Forget[lo:hi]}
	}
	saved := ink2OpsLegacy
	t.Cleanup(func() { ink2OpsLegacy = saved })
	all := ink2DecLegacyOps{mix: true, ctrl: true, rms: true, route: true}
	// legacyArms is an op's arm under every first kernel and under the defaults.
	legacyArms := func(f func(i int) error) []ink2OpsBenchArm {
		return []ink2OpsBenchArm{
			{"old", func(i int) error { ink2OpsLegacy = all; return f(i) }},
			{"new", func(i int) error { ink2OpsLegacy = ink2DecLegacyOps{}; return f(i) }},
		}
	}
	var cases []ink2OpsBenchCase
	cases = append(cases, ink2OpsBenchCase{name: "floor sigmoid_mul 2304", sites: sites, arms: []ink2OpsBenchArm{
		{"old", func(i int) error { return d.SigmoidMul(ss[i].act, ss[i].gate, n) }}}})
	cases = append(cases, ink2OpsBenchCase{name: "rms 1 weighted", sites: sites, arms: legacyArms(func(i int) error { return d.RMS(ss[i].states[0], ss[i].norms[0], ss[i].out, 1, n, 1e-5) })})
	for _, biased := range []bool{false, true} {
		cases = append(cases, ink2OpsBenchCase{name: fmt.Sprintf("route 128->8 biased=%v", biased), sites: sites, arms: legacyArms(func(i int) error {
			bias := Ink2Ref{}
			if biased {
				bias = ss[i].norms[1]
			}
			return d.Route(ss[i].gate, bias, ss[i].g, ss[i].g.At(16), 128, 8, 2.5)
		})})
	}
	for _, c := range []struct {
		states, nq int
		norm       bool
	}{{2, 1, true}, {5, 1, true}, {9, 1, true}, {16, 1, true}, {9, 1, false}, {5, 3, true}, {7, 4, false}} {
		cases = append(cases, ink2OpsBenchCase{name: fmt.Sprintf("mix s=%d nq=%d norm=%v", c.states, c.nq, c.norm), sites: sites, arms: legacyArms(func(i int) error {
			var norms []Ink2Ref
			if c.norm {
				norms = ss[i].norms[:c.nq]
			}
			return d.Mix(ss[i].states[:c.states], ss[i].queries, c.nq, norms, 1e-5, ss[i].out, n)
		})})
	}
	for _, nj := range []int{3, 1} {
		cases = append(cases, ink2OpsBenchCase{name: fmt.Sprintf("control nj=%d norm", nj), sites: sites, arms: legacyArms(func(i int) error {
			return d.Control(sub(ss[i].ctl, 0, nj), ss[i].g, ss[i].out, n, rank, ss[i].norms[:nj], 1e-5)
		})})
	}
	// The controller's kernels one at a time, nj=3.
	ctrlPart := func(name string, apply bool) func(i int) error {
		return func(i int) error {
			p, err := d.PipelinePrecise(ink2LeanSources[ink2LeanBuild{9, true}], name)
			if name == "ink2_ctrl_down" || name == "ink2_ctrl_apply" {
				p, err = d.opsKernel(name, false)
			}
			if err != nil {
				return err
			}
			c := sub(ss[i].ctl, 0, 3)
			var args []Ink2Arg
			bind := func(refs []Ink2Ref, span int) {
				for _, r := range ink2OpsSlots(refs) {
					args = append(args, r.Span(span))
				}
			}
			bind(c.Cur, n)
			bind(c.Msg, n)
			shape := Ink2Const(ink2CtrlShape{n: n, rank: rank})
			if !apply {
				bind(c.Down, rank*2*n)
				args = append(args, ss[i].g.Span(3*rank), shape)
				return d.Dispatch(p, [3]int{rank, 3, 1}, [3]int{ink2OpsThreads, 1, 1}, args...)
			}
			bind(c.Gain, n*rank)
			bind(c.Forget, n*rank)
			args = append(args, ss[i].g.Span(3*rank), ss[i].out.Span(3*n), shape)
			return d.opsEach(p, n, 3, args...)
		}
	}
	cases = append(cases,
		ink2OpsBenchCase{name: "ctrl_down nj=3", sites: sites, arms: []ink2OpsBenchArm{
			{"old", ctrlPart("ink2_ctrl_down", false)}, {"lean", ctrlPart("ink2_ctrl_down_lean", false)}}},
		ink2OpsBenchCase{name: "ctrl_apply nj=3", sites: sites, arms: []ink2OpsBenchArm{
			{"old", ctrlPart("ink2_ctrl_apply", true)}, {"lean", ctrlPart("ink2_ctrl_apply_lean_64", true)}}})
	ink2OpsBenchRun(t, d, cases, 200, ink2OpsBenchReps(t))
}
