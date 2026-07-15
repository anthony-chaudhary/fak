package metrics

import "github.com/anthony-chaudhary/fak/internal/negframe"

// NegationTax is the live per-turn inversion count over fak-injected prose.
// Mechanical and judgement tiers stay separate; Total is their sum.
type NegationTax struct {
	Mechanical int `json:"mechanical"`
	Judgement  int `json:"judgement"`
	Total      int `json:"total"`
}

func MeasureNegationTax(prose ...string) NegationTax {
	var tax NegationTax
	for _, text := range prose {
		for _, finding := range negframe.Classify("injected-prose", text) {
			if finding.Mechanical() {
				tax.Mechanical++
			} else {
				tax.Judgement++
			}
		}
	}
	tax.Total = tax.Mechanical + tax.Judgement
	return tax
}

// NegationTaxRecorder folds per-turn observations for Report and Prometheus.
type NegationTaxRecorder struct {
	Turns      uint64 `json:"turns"`
	Mechanical uint64 `json:"mechanical"`
	Judgement  uint64 `json:"judgement"`
}

func (r *NegationTaxRecorder) Record(prose ...string) NegationTax {
	tax := MeasureNegationTax(prose...)
	r.Turns++
	r.Mechanical += uint64(tax.Mechanical)
	r.Judgement += uint64(tax.Judgement)
	return tax
}

func (r NegationTaxRecorder) Report() NegationTaxReport {
	return NegationTaxReport{Turns: r.Turns, Mechanical: r.Mechanical, Judgement: r.Judgement, Total: r.Mechanical + r.Judgement}
}

type NegationTaxReport struct {
	Turns      uint64 `json:"turns"`
	Mechanical uint64 `json:"mechanical"`
	Judgement  uint64 `json:"judgement"`
	Total      uint64 `json:"total"`
}

// Prometheus renders the live cumulative tax in a bounded metric family.
func (r NegationTaxReport) Prometheus() string {
	return "fak_negation_tax_total{tier=\"mechanical\"} " + utoa(r.Mechanical) + "\n" +
		"fak_negation_tax_total{tier=\"judgement\"} " + utoa(r.Judgement) + "\n" +
		"fak_negation_tax_turns_total " + utoa(r.Turns) + "\n"
}

func utoa(n uint64) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
