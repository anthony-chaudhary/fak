package tokenprofile

import "fmt"

// EnvelopeClass identifies the independently budgeted scheduler resource.
type EnvelopeClass string

const (
	EnvelopeUncachedPrefill EnvelopeClass = "uncached_prefill"
	EnvelopeCachedInput     EnvelopeClass = "cached_input_kv_transfer"
	EnvelopeReservedDecode  EnvelopeClass = "reserved_decode"
	EnvelopeTotalTokens     EnvelopeClass = "total_tokens"
)

// Envelopes configures independent token-class capacities. A nil *Envelopes
// keeps legacy scalar total-token admission semantics.
type Envelopes struct {
	UncachedPrefill int64
	CachedInput     int64
	ReservedDecode  int64
}

// Capacity is one compatible routing target and its currently reserved load.
type Capacity struct {
	Name        string
	TotalTokens int64
	Envelopes   *Envelopes
	UsedTotal   int64
	Used        Envelopes
}

// Refusal is a typed admission failure naming the exhausted resource and a
// concrete recovery action.
type Refusal struct {
	Class     EnvelopeClass
	Required  int64
	Available int64
	Recovery  string
}

func (r *Refusal) Error() string {
	return fmt.Sprintf("%s envelope exhausted: required=%d available=%d; recovery: %s", r.Class, r.Required, r.Available, r.Recovery)
}

// Route chooses the first compatible target. Class envelopes are considered
// independently; targets without class policy retain legacy scalar behavior.
func Route(f Forecast, targets []Capacity) (int, *Refusal) {
	var best *Refusal
	for i := range targets {
		if refusal := targets[i].refusal(f); refusal == nil {
			return i, nil
		} else if best == nil || refusal.Available > best.Available {
			best = refusal
		}
	}
	if best == nil {
		best = &Refusal{Class: EnvelopeTotalTokens, Recovery: "configure at least one capacity target"}
	}
	return -1, best
}

// Admit reserves the request on a compatible target and returns its name.
func Admit(f Forecast, targets []Capacity) (string, *Refusal) {
	i, refusal := Route(f, targets)
	if refusal != nil {
		return "", refusal
	}
	t := &targets[i]
	t.UsedTotal += total(f)
	if t.Envelopes != nil {
		t.Used.UncachedPrefill += uncached(f)
		t.Used.CachedInput += f.CachedInputTokens
		t.Used.ReservedDecode += f.MaxOutputTokens
	}
	return t.Name, nil
}

func (c Capacity) refusal(f Forecast) *Refusal {
	if c.Envelopes == nil {
		return exhausted(EnvelopeTotalTokens, total(f), c.TotalTokens-c.UsedTotal,
			"wait for total-token capacity or route to a target with free scalar capacity")
	}
	checks := []struct {
		class               EnvelopeClass
		required, available int64
		recovery            string
	}{
		{EnvelopeUncachedPrefill, uncached(f), c.Envelopes.UncachedPrefill - c.Used.UncachedPrefill, "reuse a cached prefix, reduce uncached input, or route to prefill capacity"},
		{EnvelopeCachedInput, f.CachedInputTokens, c.Envelopes.CachedInput - c.Used.CachedInput, "wait for KV transfer capacity or route to a cache-capable target"},
		{EnvelopeReservedDecode, f.MaxOutputTokens, c.Envelopes.ReservedDecode - c.Used.ReservedDecode, "reduce reserved output, wait for decode capacity, or route to a decode-capable target"},
	}
	for _, check := range checks {
		if r := exhausted(check.class, check.required, check.available, check.recovery); r != nil {
			return r
		}
	}
	return nil
}

func exhausted(class EnvelopeClass, required, available int64, recovery string) *Refusal {
	if available < 0 {
		available = 0
	}
	if required <= available {
		return nil
	}
	return &Refusal{Class: class, Required: required, Available: available, Recovery: recovery}
}

func uncached(f Forecast) int64 {
	return f.InputTokens - f.CachedInputTokens
}

func total(f Forecast) int64 {
	return f.InputTokens + f.MaxOutputTokens
}
