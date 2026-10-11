package gateway

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

func byClientSeries(out, family string) map[string]string {
	rows := map[string]string{}
	prefix := family + `{client="`
	for _, line := range strings.Split(out, "\n") {
		rest, ok := strings.CutPrefix(line, prefix)
		if !ok {
			continue
		}
		client, value, ok := strings.Cut(rest, `"} `)
		if !ok {
			continue
		}
		rows[client] = value
	}
	return rows
}

// fak-test:runtime fast est=20ms lane=default
func TestInferenceTokensByClientExcludeSyntheticAndFoldEmptyToUnknown(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	turn := func(src perfSource, prompt, cached int) {
		m.observeInferenceServedTimedFrom(src, localitySelfHosted, "", prompt, 10, cached, 0, "stop", time.Second, 0)
	}
	turn(perfSource{client: perfledger.ClientPi}, 400, 9600)
	turn(perfSource{client: perfledger.ClientOpenCode}, 1000, 3000)
	turn(perfSource{client: perfledger.ClientProbe, synthetic: true}, 25, 75)
	turn(perfSource{}, 7, 3)

	out := renderInference(m)
	want := map[string]map[string]string{
		"fak_gateway_inference_prompt_tokens_by_client_total":        {perfledger.ClientPi: "400", perfledger.ClientOpenCode: "1000", perfledger.ClientUnknown: "7"},
		"fak_gateway_inference_cached_prompt_tokens_by_client_total": {perfledger.ClientPi: "9600", perfledger.ClientOpenCode: "3000", perfledger.ClientUnknown: "3"},
	}
	for family, nonzero := range want {
		rows := byClientSeries(out, family)
		if len(rows) != len(perfledger.Clients) {
			t.Fatalf("%s rendered %d rows, want %d:\n%v", family, len(rows), len(perfledger.Clients), rows)
		}
		for _, client := range perfledger.Clients {
			wantV := "0"
			if v, ok := nonzero[client]; ok {
				wantV = v
			}
			if rows[client] != wantV {
				t.Fatalf("%s{client=%q} = %q, want %q", family, client, rows[client], wantV)
			}
		}
	}
	for _, line := range []string{
		"fak_gateway_inference_prompt_tokens_total 1432\n",
		"fak_gateway_inference_cached_prompt_tokens_total 12678\n",
	} {
		if !strings.Contains(out, line) {
			t.Fatalf("unlabeled total changed; want %q in:\n%s", line, out)
		}
	}
}
