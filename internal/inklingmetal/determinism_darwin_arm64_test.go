//go:build darwin && arm64 && cgo

package inklingmetal

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

const (
	widthScheduleEnv = "FAK_INKLING_METAL_WIDTH_SCHEDULE"
	widthRowsEnv     = "FAK_INKLING_METAL_VERIFY_ROWS"
	widthCostsEnv    = "FAK_INKLING_METAL_WIDTH_COSTS"
	widthTraceEnv    = "FAK_INKLING_METAL_WIDTH_TRACE"
)

// The system and user framing up to "Continue this story.", then one period of
// the story passage as the model tokenizer encodes its repetition.
var (
	storyHead    = []int32{200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 107750, 149928, 152627, 13}
	storyPassage = []int32{149917, 220, 36598, 157269, 220, 34541, 169899, 150156, 149878, 167166, 158808, 150419, 220, 137375, 1500, 150109, 32426, 151371, 82, 193784, 154046, 302, 167627, 82, 149880, 220, 50348, 82, 220, 143948, 150749, 149876, 151594, 152397, 150207, 152504, 43538, 154366, 302, 154258, 150040, 52866, 149880, 149878, 158842, 86828, 176015, 149876, 151346, 170850, 158842, 149890, 150105, 189459, 150710, 196904, 149887, 150396, 156258, 150391, 150471, 154621, 149878, 150971, 153962, 149879, 149876, 150317, 177903, 156999}
	storyEnd     = int32(200010)
)

// storyPrompt has n tokens: 537 is the prompt whose greedy output varied
// between runs, 31 the short prompt of the decode baseline.
func storyPrompt(n int) []int32 {
	ids := slices.Clone(storyHead)
	for i := 0; len(ids) < n-1; i++ {
		ids = append(ids, storyPassage[i%len(storyPassage)])
	}
	return append(ids, storyEnd)
}

type widthRun struct {
	ids    []int32
	rows   []int
	wallMs []float64
	decode time.Duration
}

func (r widthRun) tokensPerSecond() float64 {
	return float64(len(r.ids)-1) / r.decode.Seconds()
}

func (r widthRun) digest() string {
	sum := sha256.Sum256([]byte(fmt.Sprint(r.ids)))
	return hex.EncodeToString(sum[:6])
}

// widthGreedy issues the fused cycles of the serving greedy loop. That loop's
// only other verify, an n-gram chain, takes its width from the history alone.
func widthGreedy(t *testing.T, m *Model, prompt []int32, gen, capacity int) widthRun {
	t.Helper()
	s, err := m.NewSession(capacity)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	next, err := s.PrefillID(prompt)
	if err != nil {
		t.Fatal(err)
	}
	var run widthRun
	start := time.Now()
	for {
		run.ids = append(run.ids, next)
		remaining := gen - len(run.ids)
		if remaining == 0 {
			break
		}
		depth := min(15, remaining-1)
		before := time.Now()
		if depth == 0 {
			ids, err := s.BeginBlockIDs([]int32{next})
			if err == nil {
				err = s.Commit(1)
			}
			if err != nil {
				t.Fatal(err)
			}
			next = ids[0]
			continue
		}
		tree, ids, kept, err := s.ProposeVerifyCommit(next, depth, 4, depth+1, -1)
		if err != nil {
			t.Fatal(err)
		}
		if len(tree) == 0 || kept&1 == 0 {
			t.Fatalf("cycle kept no anchor: %d rows, mask %b", len(tree), kept)
		}
		run.rows = append(run.rows, len(tree))
		run.wallMs = append(run.wallMs, float64(time.Since(before).Microseconds())/1000)
		last := 0
		for row := 1; row < len(tree); row++ {
			if kept&(1<<row) == 0 {
				continue
			}
			if int(tree[row][1]) != last || ids[last] != tree[row][0] {
				t.Fatalf("retained row %d is not the target's choice after row %d", row, last)
			}
			run.ids = append(run.ids, tree[row][0])
			last = row
		}
		next = ids[last]
	}
	run.decode = time.Since(start)
	return run
}

func widthModel(t *testing.T) *Model {
	t.Helper()
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	for _, name := range []string{widthScheduleEnv, widthRowsEnv, widthCostsEnv, widthTraceEnv} {
		t.Setenv(name, os.Getenv(name))
	}
	os.Unsetenv(widthRowsEnv)
	os.Unsetenv(widthTraceEnv)
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func widthEnvInt(name string, fallback int) int {
	if n, err := strconv.Atoi(os.Getenv(name)); err == nil && n > 0 {
		return n
	}
	return fallback
}

func loadAverage() string {
	out, err := exec.Command("sysctl", "-n", "vm.loadavg").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

func firstDifference[T comparable](a, b []T) int {
	for i := 0; i < min(len(a), len(b)); i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return min(len(a), len(b))
	}
	return -1
}

func median(values []float64) float64 {
	sorted := slices.Clone(values)
	slices.Sort(sorted)
	if len(sorted) == 0 {
		return 0
	}
	return (sorted[(len(sorted)-1)/2] + sorted[len(sorted)/2]) / 2
}

var widthPrompts = []struct {
	name   string
	tokens int
}{{"long", 537}, {"short", 31}}

const widthGenerated = 128

// A request's ids repeat on a reused arena that last served another prompt, on
// a fresh arena, and, given FAK_INKLING2_DETERMINISM_GOLDEN, across processes:
// the first process writes that file and later ones compare with it.
//
// fak-test:runtime slow est=90s
func TestDeterministicWidthSchedule(t *testing.T) {
	m := widthModel(t)
	t.Setenv(widthScheduleEnv, "request")
	runs := widthEnvInt("FAK_INKLING2_DETERMINISM_RUNS", 5)
	t.Logf("load average before: %s", loadAverage())
	first := map[string]widthRun{}
	for i := 0; i < runs; i++ {
		for _, p := range widthPrompts {
			capacity := p.tokens + widthGenerated + 64
			if i == runs-1 {
				// A request larger than the idle arena gets new draft state.
				capacity = widthPrompts[0].tokens + widthGenerated + 65
			}
			run := widthGreedy(t, m, storyPrompt(p.tokens), widthGenerated, capacity)
			t.Logf("%s run %d: ids %s, %d cycles, %.1f decode tok/s", p.name, i, run.digest(), len(run.rows), run.tokensPerSecond())
			want, seen := first[p.name]
			if !seen {
				first[p.name] = run
				t.Logf("%s rows per cycle: %v", p.name, run.rows)
				// Passes of one or two rows share a kernel, so only wider
				// ones show a schedule that moved.
				if slices.Max(run.rows) < 3 {
					t.Errorf("%s: no cycle verified three rows", p.name)
				}
				continue
			}
			if at := firstDifference(want.rows, run.rows); at >= 0 {
				t.Errorf("%s run %d: verify rows differ from run 0 at cycle %d", p.name, i, at)
			}
			if at := firstDifference(want.ids, run.ids); at >= 0 {
				t.Errorf("%s run %d: ids differ from run 0 at token %d", p.name, i, at)
			}
		}
	}
	t.Logf("load average after: %s", loadAverage())
	golden := os.Getenv("FAK_INKLING2_DETERMINISM_GOLDEN")
	if golden == "" {
		return
	}
	got := map[string][]int32{}
	for name, run := range first {
		got[name] = run.ids
	}
	data, err := os.ReadFile(golden)
	if os.IsNotExist(err) {
		if data, err = json.Marshal(got); err == nil {
			err = os.WriteFile(golden, data, 0o644)
		}
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", golden)
		return
	}
	var want map[string][]int32
	if err == nil {
		err = json.Unmarshal(data, &want)
	}
	if err != nil {
		t.Fatal(err)
	}
	for name, ids := range got {
		if at := firstDifference(want[name], ids); at >= 0 {
			t.Errorf("%s: ids differ from the process that wrote %s at token %d", name, golden, at)
		}
	}
	t.Logf("ids match %s", golden)
}

// FAK_INKLING2_WIDTH_COMPARE lists the schedules to time, such as
// "timed,request,session". Each runs as one block on a new arena, since a
// request-scoped block would clear the learners a session-scoped one carries.
func TestDeterministicWidthThroughput(t *testing.T) {
	plan := os.Getenv("FAK_INKLING2_WIDTH_COMPARE")
	if plan == "" {
		t.Skip("set FAK_INKLING2_WIDTH_COMPARE to a comma-separated schedule list")
	}
	m := widthModel(t)
	runs := widthEnvInt("FAK_INKLING2_DETERMINISM_RUNS", 5)
	warm := widthGreedy(t, m, storyPrompt(31), widthGenerated, 31+widthGenerated)
	t.Logf("warm-up %.1f decode tok/s", warm.tokensPerSecond())
	capacity := widthPrompts[0].tokens + widthGenerated + 64
	for block, schedule := range strings.Split(plan, ",") {
		t.Setenv(widthScheduleEnv, schedule)
		capacity++
		rates := map[string][]float64{}
		outputs := map[string]map[string]int{}
		rows, kept, cycles := map[string]int{}, map[string]int{}, map[string]int{}
		before := loadAverage()
		for i := 0; i < runs; i++ {
			for _, p := range widthPrompts {
				run := widthGreedy(t, m, storyPrompt(p.tokens), widthGenerated, capacity)
				rates[p.name] = append(rates[p.name], run.tokensPerSecond())
				if outputs[p.name] == nil {
					outputs[p.name] = map[string]int{}
				}
				outputs[p.name][run.digest()]++
				cycles[p.name] += len(run.rows)
				kept[p.name] += len(run.ids)
				for _, n := range run.rows {
					rows[p.name] += n
				}
			}
		}
		for _, p := range widthPrompts {
			r := rates[p.name]
			t.Logf("block %d %s %s: decode tok/s median %.1f range %.1f-%.1f %.1f; distinct outputs %d of %d; %.2f rows and %.2f tokens a cycle; load %s -> %s",
				block, schedule, p.name, median(r), slices.Min(r), slices.Max(r), r, len(outputs[p.name]), runs,
				float64(rows[p.name])/float64(cycles[p.name]), float64(kept[p.name])/float64(cycles[p.name]), before, loadAverage())
		}
	}
}

// With FAK_INKLING2_WIDTH_CALIBRATE set, every cycle verifies a forced row
// count and the budget logs its GPU time, the cost the timed schedule fits.
// The medians it prints are the source of the constant cost table.
func TestDeterministicWidthCalibration(t *testing.T) {
	if os.Getenv("FAK_INKLING2_WIDTH_CALIBRATE") == "" {
		t.Skip("set FAK_INKLING2_WIDTH_CALIBRATE to measure fused-cycle cost by verify rows")
	}
	m := widthModel(t)
	trace := filepath.Join(t.TempDir(), "cycles.txt")
	t.Setenv(widthTraceEnv, trace)
	t.Setenv(widthScheduleEnv, "request")
	repeats := widthEnvInt("FAK_INKLING2_DETERMINISM_RUNS", 2)
	type cell struct {
		prompt string
		rows   int
	}
	gpu, wall := map[cell][]float64{}, map[cell][]float64{}
	t.Logf("load average before: %s", loadAverage())
	for repeat := 0; repeat < repeats; repeat++ {
		for forced := 1; forced <= 16; forced++ {
			t.Setenv(widthRowsEnv, strconv.Itoa(forced))
			for _, p := range widthPrompts {
				os.Remove(trace)
				run := widthGreedy(t, m, storyPrompt(p.tokens), widthGenerated, p.tokens+widthGenerated+64)
				file, err := os.Open(trace)
				if err != nil {
					t.Fatal(err)
				}
				lines := bufio.NewScanner(file)
				for i := 0; lines.Scan(); i++ {
					var rows, tokens, history int
					var ms float64
					if _, err := fmt.Sscan(lines.Text(), &rows, &tokens, &history, &ms); err != nil {
						t.Fatal(err)
					}
					if i >= len(run.rows) {
						t.Fatalf("trace holds more cycles than the %d issued", len(run.rows))
					}
					// The first cycles after a prompt include one-time warm-up.
					if i >= 2 && ms > 0 {
						gpu[cell{p.name, rows}] = append(gpu[cell{p.name, rows}], ms)
						wall[cell{p.name, rows}] = append(wall[cell{p.name, rows}], run.wallMs[i])
					}
				}
				file.Close()
			}
		}
	}
	t.Logf("load average after: %s", loadAverage())
	var table []string
	for rows := 1; rows <= 16; rows++ {
		var both []float64
		for _, p := range widthPrompts {
			c := cell{p.name, rows}
			if len(gpu[c]) == 0 {
				t.Fatalf("no %s cycle verified %d rows", p.name, rows)
			}
			t.Logf("%s rows %2d: %3d cycles, GPU ms median %.2f min %.2f, wall ms median %.2f", p.name, rows, len(gpu[c]), median(gpu[c]), slices.Min(gpu[c]), median(wall[c]))
			both = append(both, median(gpu[c]))
		}
		table = append(table, strconv.FormatFloat((both[0]+both[1])/2, 'f', 1, 64))
	}
	t.Logf("%s=%s", widthCostsEnv, strings.Join(table, ","))
}
