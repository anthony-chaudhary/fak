package metalgemm

import (
	"cmp"
	"fmt"
	"slices"
	"strings"
	"text/tabwriter"
	"time"
)

// Ink2ProfileEnv selects the profile mode of a device opened by NewInk2Device.
const Ink2ProfileEnv = "FAK_INK2_PROFILE"

// Ink2ProfileMode is how a device attributes GPU time to single dispatches.
type Ink2ProfileMode string

const (
	Ink2ProfileOff Ink2ProfileMode = ""
	// Ink2ProfilePass encodes each dispatch as its own compute pass of the one
	// command buffer and samples GPU timestamps at the pass boundaries.
	Ink2ProfilePass Ink2ProfileMode = "pass"
	// Ink2ProfileCommit submits and waits after every dispatch and charges it
	// the command buffer's GPU time.
	Ink2ProfileCommit Ink2ProfileMode = "commit"
)

// ink2ProfileStats samples nothing per dispatch: it only asks the decode driver
// to report the always-on counters of its decode window.
const ink2ProfileStats = "stats"

// Ink2ReportRequested is whether FAK_INK2_PROFILE asks for a decode-window
// report: any profile mode, or stats for the counters alone.
func Ink2ReportRequested(env string) bool {
	switch strings.ToLower(strings.TrimSpace(env)) {
	case "", "0", "off":
		return false
	}
	return true
}

func parseInk2ProfileMode(s string) (Ink2ProfileMode, error) {
	switch m := Ink2ProfileMode(strings.ToLower(strings.TrimSpace(s))); m {
	case "", "0", "off", ink2ProfileStats:
		return Ink2ProfileOff, nil
	case Ink2ProfilePass, Ink2ProfileCommit:
		return m, nil
	}
	return Ink2ProfileOff, fmt.Errorf("metalgemm: %s=%q, want pass, commit or stats", Ink2ProfileEnv, s)
}

// Sub returns the work done between an earlier snapshot o and s.
func (s Ink2Stats) Sub(o Ink2Stats) Ink2Stats {
	s.Submissions -= o.Submissions
	s.Dispatches -= o.Dispatches
	s.Barriers -= o.Barriers
	s.Commits -= o.Commits
	s.GPUSeconds -= o.GPUSeconds
	s.EncodeSeconds -= o.EncodeSeconds
	s.WaitSeconds -= o.WaitSeconds
	s.SchedSeconds -= o.SchedSeconds
	s.QueueSeconds -= o.QueueSeconds
	s.TailSeconds -= o.TailSeconds
	return s
}

// Ink2ProfileRow is every dispatch of one kernel at one launch size under one
// scope. Grid counts threadgroups, or threads when Threads is set.
type Ink2ProfileRow struct {
	Scope, Kernel string
	// Detail is the weight shape of a packed matvec, which the grid alone does
	// not determine.
	Detail         string
	Grid, PerGroup [3]int
	Threads        bool
	Dispatches     uint64
	GPUSeconds     float64
	EncodeSeconds  float64
}

// Ink2Profile is the per-dispatch attribution collected since the last reset.
type Ink2Profile struct {
	Mode, Requested Ink2ProfileMode
	Rows            []Ink2ProfileRow
	// BufferGPUSeconds is the command-buffer GPU time over the same window, the
	// bound the per-pass times are checked against.
	BufferGPUSeconds float64
	// Unsampled counts dispatches whose timestamps the device did not resolve.
	Unsampled uint64
}

func (p Ink2Profile) Enabled() bool { return p.Mode != Ink2ProfileOff }

func (p Ink2Profile) totals() (gpu, encode float64, dispatches uint64) {
	for _, r := range p.Rows {
		gpu, encode, dispatches = gpu+r.GPUSeconds, encode+r.EncodeSeconds, dispatches+r.Dispatches
	}
	return gpu, encode, dispatches
}

func ink2ProfileDims(v [3]int) string { return fmt.Sprintf("%dx%dx%d", v[0], v[1], v[2]) }

func (r Ink2ProfileRow) grid() string {
	if r.Threads {
		return ink2ProfileDims(r.Grid) + "t"
	}
	return ink2ProfileDims(r.Grid) + "g"
}

// ink2ProfileGroup folds rows that share a key, ranked by GPU time.
func ink2ProfileGroup(rows []Ink2ProfileRow, key func(Ink2ProfileRow) string) []Ink2ProfileRow {
	at := map[string]int{}
	var out []Ink2ProfileRow
	for _, r := range rows {
		k := key(r)
		i, ok := at[k]
		if !ok {
			i, at[k] = len(out), len(out)
			out = append(out, Ink2ProfileRow{Scope: k})
		}
		out[i].Dispatches += r.Dispatches
		out[i].GPUSeconds += r.GPUSeconds
		out[i].EncodeSeconds += r.EncodeSeconds
	}
	ink2ProfileRank(out)
	return out
}

func ink2ProfileRank(rows []Ink2ProfileRow) {
	slices.SortStableFunc(rows, func(a, b Ink2ProfileRow) int {
		return cmp.Or(cmp.Compare(b.GPUSeconds, a.GPUSeconds), cmp.Compare(a.Scope, b.Scope), cmp.Compare(a.Kernel, b.Kernel))
	})
}

// Format renders the profile of a window that decoded tokens tokens in wall,
// with st the device stats over that window: the dispatches ranked by GPU
// time, the same folded by scope and by kernel, and the per-token split of
// wall time into encode, wait and the host remainder. With profiling off only
// the split is printed.
func (p Ink2Profile) Format(tokens int, wall time.Duration, st Ink2Stats) string {
	tokens = max(tokens, 1)
	perTok := func(seconds float64) float64 { return 1e3 * seconds / float64(tokens) }
	var b strings.Builder
	gpu, encode, dispatches := p.totals()
	if p.Enabled() {
		fmt.Fprintf(&b, "ink2 profile: mode=%s", p.Mode)
		if p.Requested != p.Mode {
			fmt.Fprintf(&b, " (requested %s, stage-boundary counter sampling unsupported)", p.Requested)
		}
		fmt.Fprintf(&b, " tokens=%d dispatches=%d unsampled=%d\n", tokens, dispatches, p.Unsampled)
		share := func(seconds float64) float64 {
			if gpu <= 0 {
				return 0
			}
			return 100 * seconds / gpu
		}
		rows := slices.Clone(p.Rows)
		ink2ProfileRank(rows)
		w := tabwriter.NewWriter(&b, 0, 0, 2, ' ', tabwriter.AlignRight)
		fmt.Fprintln(w, strings.Join([]string{"#", "scope", "kernel", "detail", "grid", "group", "disp/tok", "gpu ms/tok", "us/disp", "gpu %", "cum %", "enc ms/tok", ""}, "\t"))
		cum := 0.0
		for i, r := range rows {
			cum += r.GPUSeconds
			fmt.Fprintf(w, "%d\t%s\t%s\t%s\t%s\t%s\t%.2f\t%.4f\t%.1f\t%.2f\t%.2f\t%.4f\t\n", i+1, r.Scope, r.Kernel, r.Detail, r.grid(),
				ink2ProfileDims(r.PerGroup), float64(r.Dispatches)/float64(tokens), perTok(r.GPUSeconds),
				1e6*r.GPUSeconds/float64(max(r.Dispatches, 1)), share(r.GPUSeconds), share(cum), perTok(r.EncodeSeconds))
		}
		w.Flush()
		for _, fold := range []struct {
			name string
			key  func(Ink2ProfileRow) string
		}{{"scope", func(r Ink2ProfileRow) string { return r.Scope }}, {"kernel", func(r Ink2ProfileRow) string { return r.Kernel }}} {
			fmt.Fprintf(&b, "by %s:\n", fold.name)
			w = tabwriter.NewWriter(&b, 0, 0, 2, ' ', tabwriter.AlignRight)
			fmt.Fprintf(w, "#\t%s\tdisp/tok\tgpu ms/tok\tus/disp\tgpu %%\tcum %%\tenc ms/tok\t\n", fold.name)
			cum = 0
			for i, r := range ink2ProfileGroup(p.Rows, fold.key) {
				cum += r.GPUSeconds
				fmt.Fprintf(w, "%d\t%s\t%.2f\t%.4f\t%.1f\t%.2f\t%.2f\t%.4f\t\n", i+1, r.Scope, float64(r.Dispatches)/float64(tokens),
					perTok(r.GPUSeconds), 1e6*r.GPUSeconds/float64(max(r.Dispatches, 1)), share(r.GPUSeconds), share(cum), perTok(r.EncodeSeconds))
			}
			w.Flush()
		}
		cover := 0.0
		if p.BufferGPUSeconds > 0 {
			cover = 100 * gpu / p.BufferGPUSeconds
		}
		fmt.Fprintf(&b, "profiled: gpu=%.3f ms/tok over %.3f ms/tok of command-buffer GPU time (%.1f%%), encode=%.3f ms/tok\n",
			perTok(gpu), perTok(p.BufferGPUSeconds), cover, perTok(encode))
	}
	wallS := wall.Seconds()
	fmt.Fprintf(&b, "totals: tokens=%d wall=%.3f ms/tok (%.2f tok/s) encode=%.3f wait=%.3f [queue=%.3f gpu=%.3f tail=%.3f] host_remainder=%.3f ms/tok; kernel=%.3f ms/tok submissions/tok=%.2f dispatches/tok=%.1f barriers/tok=%.1f commits/tok=%.2f\n",
		tokens, perTok(wallS), float64(tokens)/max(wallS, 1e-12), perTok(st.EncodeSeconds), perTok(st.WaitSeconds), perTok(st.QueueSeconds),
		perTok(st.GPUSeconds), perTok(st.TailSeconds), perTok(wallS-st.EncodeSeconds-st.WaitSeconds), perTok(st.SchedSeconds),
		float64(st.Submissions)/float64(tokens), float64(st.Dispatches)/float64(tokens),
		float64(st.Barriers)/float64(tokens), float64(st.Commits)/float64(tokens))
	return b.String()
}
