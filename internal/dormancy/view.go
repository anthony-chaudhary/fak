package dormancy

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"
)

// LedgerRecord is the durable loop-manager projection consumed by the view.
type LedgerRecord struct {
	LoopID          string    `json:"loop_id"`
	LastActiveAt    time.Time `json:"last_active_at"`
	HorizonSeconds  int64     `json:"horizon_seconds,omitempty"`
	ExpectedSeconds int64     `json:"expected_interval_seconds,omitempty"`
	SleepUntil      time.Time `json:"sleep_until,omitempty"`
	RestoreCount    int64     `json:"restore_count,omitempty"`
}

type BucketView struct {
	Bucket   string `json:"bucket"`
	Loops    int    `json:"loops"`
	Restores int64  `json:"restores"`
}

type LoopView struct {
	LoopID       string  `json:"loop_id"`
	GapSeconds   float64 `json:"gap_seconds"`
	Horizon      string  `json:"horizon"`
	Status       string  `json:"status"`
	RestoreCount int64   `json:"restore_count"`
}

type View struct {
	GeneratedAt   time.Time    `json:"generated_at"`
	Loops         []LoopView   `json:"loops"`
	Buckets       []BucketView `json:"buckets"`
	Dormant       int          `json:"intentionally_dormant"`
	Stuck         int          `json:"stuck"`
	OldestLoopID  string       `json:"oldest_loop_id,omitempty"`
	OldestGapSecs float64      `json:"oldest_gap_seconds,omitempty"`
}

// Fold classifies ledger records without mutating loop state.
func Fold(records []LedgerRecord, now time.Time) View {
	v := View{GeneratedAt: now.UTC()}
	buckets := map[string]*BucketView{}
	for _, r := range records {
		gap := now.Sub(r.LastActiveAt).Seconds()
		if gap < 0 {
			gap = 0
		}
		expected := time.Duration(r.ExpectedSeconds) * time.Second
		if expected <= 0 {
			expected = time.Duration(r.HorizonSeconds) * time.Second
		}
		status := "active"
		if !r.SleepUntil.IsZero() && now.Before(r.SleepUntil) {
			status = "intentionally_dormant"
			v.Dormant++
		} else if expected > 0 && gap > expected.Seconds() {
			status = "stuck"
			v.Stuck++
		}
		bucket := Bucket(time.Duration(gap * float64(time.Second))).String()
		b := buckets[bucket]
		if b == nil {
			b = &BucketView{Bucket: bucket}
			buckets[bucket] = b
		}
		b.Loops++
		b.Restores += r.RestoreCount
		v.Loops = append(v.Loops, LoopView{r.LoopID, gap, bucket, status, r.RestoreCount})
		if gap > v.OldestGapSecs {
			v.OldestGapSecs = gap
			v.OldestLoopID = r.LoopID
		}
	}
	sort.Slice(v.Loops, func(i, j int) bool { return v.Loops[i].LoopID < v.Loops[j].LoopID })
	for _, h := range []Horizon{Warm, Cool, Cold, Frozen, Ancient} {
		if b := buckets[h.String()]; b != nil {
			v.Buckets = append(v.Buckets, *b)
		}
	}
	return v
}

func ReadLedger(r io.Reader) ([]LedgerRecord, error) {
	s := bufio.NewScanner(r)
	var out []LedgerRecord
	line := 0
	for s.Scan() {
		line++
		text := strings.TrimSpace(s.Text())
		if text == "" {
			continue
		}
		var rec LedgerRecord
		if err := json.Unmarshal([]byte(text), &rec); err != nil {
			return nil, fmt.Errorf("ledger line %d: %w", line, err)
		}
		if rec.LoopID == "" || rec.LastActiveAt.IsZero() {
			return nil, fmt.Errorf("ledger line %d: loop_id and last_active_at are required", line)
		}
		out = append(out, rec)
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// Prometheus renders the acceptance-surface metric family from the same fold.
func (v View) Prometheus() string {
	var b strings.Builder
	b.WriteString("# HELP fak_dormancy_gap_seconds Current loop inactivity gap.\n# TYPE fak_dormancy_gap_seconds gauge\n")
	for _, l := range v.Loops {
		fmt.Fprintf(&b, "fak_dormancy_gap_seconds{loop_id=%q,status=%q,horizon=%q} %.0f\n", l.LoopID, l.Status, l.Horizon, l.GapSeconds)
	}
	b.WriteString("# HELP fak_dormancy_restores_total Restores grouped by inactivity horizon.\n# TYPE fak_dormancy_restores_total counter\n")
	for _, x := range v.Buckets {
		fmt.Fprintf(&b, "fak_dormancy_restores_total{horizon=%q} %d\n", x.Bucket, x.Restores)
	}
	return b.String()
}
