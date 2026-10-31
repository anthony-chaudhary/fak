package workerworktree

import (
	"fmt"
	"strconv"
	"strings"
	"testing"
	"time"
)

// These counts measure calls through the real RecoveryEntries Git boundary,
// using deterministic read responses. They are not subprocess wall-time,
// launch throughput, or a claim about the cost of repository ancestry walks.
func recoveryProbeRows(remotes []string) string {
	var rows []string
	for i, remote := range remotes {
		sha := fmt.Sprintf("%040x", i+1)
		rows = append(rows, RecoveryMirrorPrefix+remote+fmt.Sprintf("/wt-%03d/", i)+sha+"\x00"+sha)
	}
	return strings.Join(rows, "\n")
}

func TestRecoveryEntriesBoundsRemoteFreshnessProbes(t *testing.T) {
	const n = 100
	remotes := make([]string, n)
	for i := range remotes {
		remotes[i] = "origin"
	}
	rows := recoveryProbeRows(remotes)
	calls, ancestors, probes := 0, 0, 0
	stamp := strconv.FormatInt(time.Now().Add(-time.Hour).Unix(), 10)
	g := func(root string, args []string) (int, string) {
		calls++
		if root != "/repo" {
			t.Fatalf("wrong root %q", root)
		}
		switch args[0] {
		case "for-each-ref":
			if args[len(args)-1] == RecoveryMirrorPrefix {
				return 0, rows
			}
			return 0, ""
		case "merge-base":
			ancestors++
			return 1, ""
		case "reflog":
			probes++
			if args[len(args)-1] != RecoveryMirrorStampPrefix+"origin" {
				t.Fatalf("wrong stamp %v", args)
			}
			return 0, stamp
		default:
			t.Fatalf("unexpected mutation or command %v", args)
			return 1, ""
		}
	}
	got, err := RecoveryEntries("/repo", g)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("rows=%d remote_stamps=1 actual_reads=%d ancestry_reads=%d freshness_reads=%d; old per-row formula=%d", n, calls, ancestors, probes, 2+2*n)
	if len(got) != n || ancestors != n || probes != 1 || calls != 2+n+1 {
		t.Fatalf("unbounded/replaced probes: rows=%d calls=%d ancestors=%d freshness=%d", len(got), calls, ancestors, probes)
	}
	for _, row := range got {
		if row.State != "RECOVERABLE" || row.Durability != DurabilityRemoteOnly || row.Remote != "origin" || row.MirrorAgeSec <= 0 || row.MirrorAgeSec != got[0].MirrorAgeSec || !strings.HasPrefix(row.Action, "restore local ref with git update-ref ") {
			t.Fatalf("classification changed: %+v", row)
		}
	}
}

func TestRecoveryFreshnessCacheIsPerRemoteAndInvocation(t *testing.T) {
	rows := recoveryProbeRows([]string{"origin", "backup", "origin", "backup"})
	probes := map[string]int{}
	stamp := time.Now().Add(-10 * time.Minute).Unix()
	ancestors := 0
	g := func(_ string, args []string) (int, string) {
		switch args[0] {
		case "for-each-ref":
			if args[len(args)-1] == RecoveryMirrorPrefix {
				return 0, rows
			}
			return 0, ""
		case "merge-base":
			ancestors++
			return 1, ""
		case "reflog":
			key := args[len(args)-1]
			probes[key]++
			offset := int64(0)
			if key == RecoveryMirrorStampPrefix+"backup" {
				offset = 100
			}
			return 0, strconv.FormatInt(stamp-offset, 10)
		default:
			t.Fatalf("unexpected command %v", args)
			return 1, ""
		}
	}
	first, err := RecoveryEntries("/repo", g)
	if err != nil {
		t.Fatal(err)
	}
	stamp += 200
	second, err := RecoveryEntries("/repo", g)
	if err != nil {
		t.Fatal(err)
	}
	if probes[RecoveryMirrorStampPrefix+"origin"] != 2 || probes[RecoveryMirrorStampPrefix+"backup"] != 2 || len(probes) != 2 || ancestors != 8 {
		t.Fatalf("probe isolation failed: %v ancestry=%d", probes, ancestors)
	}
	byRemote := map[string]int64{}
	for _, row := range first {
		if age, ok := byRemote[row.Remote]; ok && age != row.MirrorAgeSec {
			t.Fatal("inconsistent invocation snapshot")
		}
		byRemote[row.Remote] = row.MirrorAgeSec
	}
	if byRemote["backup"]-byRemote["origin"] < 90 {
		t.Fatalf("remote key crossed: %v", byRemote)
	}
	for _, row := range second {
		if byRemote[row.Remote]-row.MirrorAgeSec < 190 {
			t.Fatalf("cross-call stale reuse: %+v old=%d", row, byRemote[row.Remote])
		}
	}
}

func TestRecoveryFreshnessFailuresAreNotMemoizedAsKnown(t *testing.T) {
	for _, kind := range []string{"error", "malformed", "persistent-error"} {
		t.Run(kind, func(t *testing.T) {
			rows := recoveryProbeRows([]string{"origin", "origin", "origin"})
			probes := 0
			g := func(_ string, args []string) (int, string) {
				switch args[0] {
				case "for-each-ref":
					if args[len(args)-1] == RecoveryMirrorPrefix {
						return 0, rows
					}
					return 0, ""
				case "merge-base":
					return 1, ""
				case "reflog":
					probes++
					if probes == 1 || kind == "persistent-error" {
						if kind == "malformed" {
							return 0, "not-a-timestamp"
						}
						return 1, "probe failed"
					}
					return 0, strconv.FormatInt(time.Now().Add(-time.Hour).Unix(), 10)
				default:
					t.Fatalf("unexpected command %v", args)
					return 1, ""
				}
			}
			got, err := RecoveryEntries("/repo", g)
			if err != nil {
				t.Fatal(err)
			}
			if kind == "persistent-error" {
				if probes != 3 {
					t.Fatalf("failure cached: %d", probes)
				}
				for _, row := range got {
					if row.MirrorAgeSec != 0 {
						t.Fatal("failed probe invented age")
					}
				}
				return
			}
			if probes != 2 || got[0].MirrorAgeSec != 0 || got[1].MirrorAgeSec <= 0 || got[2].MirrorAgeSec != got[1].MirrorAgeSec {
				t.Fatalf("unknown became cached freshness: probes=%d rows=%+v", probes, got)
			}
		})
	}
}

// A successful age of zero must stay distinct from an unavailable observation.
func TestRecoveryMirrorAgeZeroAndFuture(t *testing.T) {
	// Keep this file compilable with the original private helper for the
	// independent old-source count regression. This control still strictly
	// requires and executes the candidate observation contract when selected.
	probe, ok := any(mirrorAge).(func(string, string, GitRunner, int64) (int64, bool))
	if !ok {
		t.Fatal("mirror age must distinguish known observations")
	}
	const now int64 = 1793490000
	for _, tc := range []struct {
		name, stamp string
		known       bool
	}{{"zero", strconv.FormatInt(now, 10), true}, {"future", strconv.FormatInt(now+1, 10), false}, {"malformed", "not-time", false}} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			g := func(_ string, args []string) (int, string) {
				calls++
				if args[0] != "reflog" {
					t.Fatalf("unexpected command %v", args)
				}
				return 0, tc.stamp
			}
			age, known := probe("/repo", "origin", g, now)
			if age != 0 || known != tc.known || calls != 1 {
				t.Fatalf("age=%d known=%v probes=%d", age, known, calls)
			}
		})
	}
}
