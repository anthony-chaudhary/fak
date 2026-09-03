// Package workspin detects sustained repository activity that is not producing
// witnessed, substantive delivery.
package workspin

import (
	"sort"
	"strings"
	"time"
)

const Schema = "fak-workspin/1"

type Kind string

const (
	Commit Kind = "commit"
	Issue  Kind = "issue"
)

type Observation struct {
	At        time.Time `json:"at"`
	Kind      Kind      `json:"kind"`
	Summary   string    `json:"summary"`
	Changed   int       `json:"changed,omitempty"`
	Witnessed bool      `json:"witnessed,omitempty"`
	Delivery  bool      `json:"delivery,omitempty"`
}

type Classified struct {
	Observation
	Class  string `json:"class"`
	Reason string `json:"reason"`
}

type Bucket struct {
	Start        time.Time    `json:"start"`
	End          time.Time    `json:"end"`
	Activity     int          `json:"activity"`
	Maintenance  int          `json:"maintenance"`
	Delivery     int          `json:"delivery"`
	Observations []Classified `json:"observations,omitempty"`
}

type Report struct {
	Schema      string   `json:"schema"`
	Verdict     string   `json:"verdict"`
	Spinning    bool     `json:"spinning"`
	Reasons     []string `json:"reasons"`
	BucketWidth string   `json:"bucket_width"`
	Buckets     []Bucket `json:"buckets"`
}

type Config struct {
	Now          time.Time
	BucketWidth  time.Duration
	BucketCount  int
	HighActivity int
	Sustained    int
	SmallChange  int
}

func DefaultConfig(now time.Time) Config {
	return Config{Now: now, BucketWidth: 7 * 24 * time.Hour, BucketCount: 4, HighActivity: 3, Sustained: 2, SmallChange: 3}
}

func Classify(o Observation, smallChange int) Classified {
	c := Classified{Observation: o}
	summary := strings.ToLower(strings.TrimSpace(o.Summary))
	prefix := summary
	if i := strings.IndexByte(prefix, ':'); i >= 0 {
		prefix = prefix[:i]
	}
	prefix = strings.TrimSuffix(strings.TrimSpace(prefix), "!")
	substantive := prefix == "feat" || prefix == "fix" || prefix == "perf" || prefix == "refactor"
	maintenance := prefix == "chore" || prefix == "docs" || prefix == "style" || prefix == "test" || prefix == "build" || prefix == "ci"

	if o.Delivery && o.Witnessed {
		c.Class, c.Reason = "delivery", "explicit witnessed delivery"
	} else if o.Kind == Commit && o.Witnessed && substantive {
		c.Class, c.Reason = "delivery", "substantive commit with fak witness"
	} else if maintenance {
		c.Class, c.Reason = "maintenance", "maintenance commit type"
	} else if o.Changed > 0 && o.Changed <= smallChange {
		c.Class, c.Reason = "maintenance", "small bounded change"
	} else {
		c.Class, c.Reason = "activity", "activity without delivery witness"
	}
	return c
}

func Audit(observations []Observation, cfg Config) Report {
	if cfg.BucketWidth <= 0 {
		cfg.BucketWidth = 7 * 24 * time.Hour
	}
	if cfg.BucketCount <= 0 {
		cfg.BucketCount = 4
	}
	if cfg.HighActivity <= 0 {
		cfg.HighActivity = 3
	}
	if cfg.Sustained <= 0 {
		cfg.Sustained = 2
	}
	if cfg.SmallChange <= 0 {
		cfg.SmallChange = 3
	}
	end := cfg.Now.UTC()
	start := end.Add(-time.Duration(cfg.BucketCount) * cfg.BucketWidth)
	r := Report{Schema: Schema, Verdict: "healthy", Reasons: []string{}, BucketWidth: cfg.BucketWidth.String(), Buckets: make([]Bucket, cfg.BucketCount)}
	for i := range r.Buckets {
		s := start.Add(time.Duration(i) * cfg.BucketWidth)
		r.Buckets[i] = Bucket{Start: s, End: s.Add(cfg.BucketWidth), Observations: []Classified{}}
	}
	sort.SliceStable(observations, func(i, j int) bool { return observations[i].At.Before(observations[j].At) })
	for _, o := range observations {
		at := o.At.UTC()
		if at.Before(start) || !at.Before(end) {
			continue
		}
		i := int(at.Sub(start) / cfg.BucketWidth)
		c := Classify(o, cfg.SmallChange)
		b := &r.Buckets[i]
		b.Activity++
		if c.Class == "delivery" {
			b.Delivery++
		}
		if c.Class == "maintenance" {
			b.Maintenance++
		}
		b.Observations = append(b.Observations, c)
	}
	run := 0
	for _, b := range r.Buckets {
		if b.Activity >= cfg.HighActivity && b.Delivery == 0 {
			run++
		} else {
			run = 0
		}
		if run >= cfg.Sustained {
			r.Spinning = true
		}
	}
	if r.Spinning {
		r.Verdict = "spinning"
		r.Reasons = append(r.Reasons, "sustained high activity without witnessed delivery")
	} else {
		r.Reasons = append(r.Reasons, "no sustained high-activity delivery gap")
	}
	return r
}
