package compute

import (
	"errors"
	"os"
	"regexp"
	"strconv"
	"testing"
)

// cuhDefine reads an integer `#define NAME value` from a header under cuda/.
func cuhDefine(t *testing.T, file, name string) int {
	t.Helper()
	raw, err := os.ReadFile("cuda/" + file)
	if err != nil {
		t.Fatal(err)
	}
	m := regexp.MustCompile(`(?m)^#define\s+` + regexp.QuoteMeta(name) + `\s+(\d+)\b`).FindSubmatch(raw)
	if m == nil {
		t.Fatalf("%s: #define %s not found", file, name)
	}
	n, _ := strconv.Atoi(string(m[1]))
	return n
}

// Closed host oracle for the decode task-graph ABI (cohere-megakernel, ADAPT). [SW-VERIFIED]
// structure and oracle only: no hardware, no speed claim.
const (
	mkTaskInts  = 32
	mkConsumers = 8
)

const (
	mkOpAdd = iota + 1
	mkOpMul
	mkOpScale
)

type mkTask [mkTaskInts]int32

func mkBuild(op, a, b, out, waitCtr, waitTarget, sigCtr, arg int) mkTask {
	var d mkTask
	d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7] = int32(op), int32(a), int32(b), int32(out), int32(waitCtr), int32(waitTarget), int32(sigCtr), int32(arg)
	return d
}

var errMKDeadlock = errors.New("megakernel taskgraph: no task became ready (counter never satisfied)")

// mkRun executes tasks like the persistent kernel: consumer c owns tasks i%8==c and may run its
// next task only when the task's global counter reached its target; completion bumps the signal
// counter. Returns the buffers.
func mkRun(tasks []mkTask, buf []float32, counters []int32) error {
	next := make([]int, mkConsumers) // per-consumer cursor over its own tasks
	done := 0
	for done < len(tasks) {
		progressed := false
		for c := 0; c < mkConsumers; c++ {
			for next[c] = max(next[c], 0); c+next[c]*mkConsumers < len(tasks); {
				d := tasks[c+next[c]*mkConsumers]
				if counters[d[4]] < d[5] {
					break
				}
				a, b, o := buf[d[1]], buf[d[2]], d[3]
				switch d[0] {
				case mkOpAdd:
					buf[o] = a + b
				case mkOpMul:
					buf[o] = a * b
				case mkOpScale:
					buf[o] = a * float32(d[7])
				}
				counters[d[6]]++
				next[c]++
				done++
				progressed = true
			}
		}
		if !progressed {
			return errMKDeadlock
		}
	}
	return nil
}

func TestMegakernelTaskgraph(t *testing.T) {
	t.Run("HeaderMatchesOracle", func(t *testing.T) {
		for name, want := range map[string]int{
			"FAK_MK_TASK_INTS": mkTaskInts, "FAK_MK_CONSUMER_WARPS": mkConsumers,
			"FAK_MK_CONTROLLER_WARPS": 1, "FAK_MK_PRODUCER_WARPS": 1, "FAK_MK_STORER_WARPS": 1,
			"FAK_MK_SLOT_OP": 0, "FAK_MK_SLOT_IN_A": 1, "FAK_MK_SLOT_IN_B": 2, "FAK_MK_SLOT_OUT": 3,
			"FAK_MK_SLOT_WAIT_COUNTER": 4, "FAK_MK_SLOT_WAIT_TARGET": 5, "FAK_MK_SLOT_SIGNAL_COUNTER": 6, "FAK_MK_SLOT_ARG": 7,
		} {
			if got := cuhDefine(t, "megakernel_taskgraph.cuh", name); got != want {
				t.Fatalf("%s = %d, want %d", name, got, want)
			}
		}
	})
	// Chain: b3=b0+b1 ; b4=b3*b2 ; b5=b4*3 ; b6=b5+b3. Counter i counts completions of task i's stage.
	tasks := []mkTask{
		mkBuild(mkOpAdd, 0, 1, 3, 0, 0, 1, 0),
		mkBuild(mkOpMul, 3, 2, 4, 1, 1, 2, 0),
		mkBuild(mkOpScale, 4, 4, 5, 2, 1, 3, 3),
		mkBuild(mkOpAdd, 5, 3, 6, 3, 1, 4, 0),
	}
	t.Run("MatchesPerOpReference", func(t *testing.T) {
		in := []float32{1.5, 2.5, 4, 0, 0, 0, 0}
		want := append([]float32(nil), in...)
		want[3] = want[0] + want[1]
		want[4] = want[3] * want[2]
		want[5] = want[4] * 3
		want[6] = want[5] + want[3]
		got := append([]float32(nil), in...)
		if err := mkRun(tasks, got, make([]int32, 8)); err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("buf[%d] = %v, want %v", i, got[i], want[i])
			}
		}
	})
	t.Run("UnsatisfiableCounterDeadlocksTyped", func(t *testing.T) {
		bad := append([]mkTask(nil), tasks...)
		bad[1][5] = 99
		if err := mkRun(bad, make([]float32, 7), make([]int32, 8)); !errors.Is(err, errMKDeadlock) {
			t.Fatalf("err = %v, want errMKDeadlock", err)
		}
	})
	t.Run("DescriptorLayoutIs32Int32", func(t *testing.T) {
		d := mkBuild(mkOpScale, 1, 2, 3, 4, 5, 6, 7)
		if len(d) != 32 || d[0] != mkOpScale || d[7] != 7 || d[8] != 0 || d[31] != 0 {
			t.Fatalf("descriptor = %v", d)
		}
	})
}
