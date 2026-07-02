package fleetaccounts

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// JSONEnvelope is the top-level shape of `fleet_accounts.py json` — the machine roster
// with the policy/registry provenance and the pre-filtered available list. Field order
// matches the Python json.dumps envelope.
type JSONEnvelope struct {
	Home              string    `json:"home"`
	PolicyPath        string    `json:"policy_path"`
	PolicyExists      bool      `json:"policy_exists"`
	RegistryPath      string    `json:"registry_path"`
	RegistryExists    bool      `json:"registry_exists"`
	AvailableAccounts []Account `json:"available_accounts"`
	Accounts          []Account `json:"accounts"`
}

// BuildJSONEnvelope assembles the `json` envelope from an annotated roster.
func BuildJSONEnvelope(home, policyPath string, policyExists bool, registryPath string,
	registryExists bool, rows []Account) JSONEnvelope {
	avail := Available(rows)
	if avail == nil {
		avail = []Account{}
	}
	if rows == nil {
		rows = []Account{}
	}
	return JSONEnvelope{
		Home: home, PolicyPath: policyPath, PolicyExists: policyExists,
		RegistryPath: registryPath, RegistryExists: registryExists,
		AvailableAccounts: avail, Accounts: rows,
	}
}

// MarshalIndent renders the envelope with the same one-space indent the Python uses
// (json.dumps(..., indent=1)).
func (e JSONEnvelope) MarshalIndent() ([]byte, error) {
	return json.MarshalIndent(e, "", " ")
}

// RenderList renders the human roster table (the default `list` view) as the Python
// _cli_list does: AVAILABLE / BLOCKED / DUPLICATE / EXCLUDED / NON-ACCOUNT sections, with
// the identity summary line and the policy source footer.
func RenderList(rows []Account, home, policyPath string, policyExists bool,
	policyExampleNote string) string {
	var b strings.Builder
	buckets := map[string][]Account{
		"available": {}, "blocked": {}, "duplicate": {}, "excluded": {}, "non-account": {},
	}
	for _, r := range rows {
		switch {
		case r.Kind == KindWorker && IsDuplicateIdentity(r):
			buckets["duplicate"] = append(buckets["duplicate"], r)
		case r.Kind == KindWorker && derefBool(r.Available):
			buckets["available"] = append(buckets["available"], r)
		case r.Kind == KindWorker:
			buckets["blocked"] = append(buckets["blocked"], r)
		default:
			buckets[string(r.Kind)] = append(buckets[string(r.Kind)], r)
		}
	}
	productSet := map[string]bool{}
	for _, r := range rows {
		p := r.Product
		if p == "" {
			p = "claude"
		}
		productSet[p] = true
	}
	var products []string
	for p := range productSet {
		products = append(products, p)
	}
	sort.Strings(products)
	if len(products) == 0 {
		products = []string{"claude"}
	}
	claudeLogins := map[string]bool{}
	claudeDirs := 0
	for _, r := range rows {
		if r.Product == "claude" && r.Kind == KindWorker {
			claudeDirs++
			if u := derefStr(r.AccountUUID); u != "" {
				claudeLogins[u] = true
			}
		}
	}
	fmt.Fprintf(&b, "fleet accounts under %s  (%d dirs, products: %s)\n",
		home, len(rows), strings.Join(products, "+"))
	if claudeDirs > 0 {
		dupNote := ""
		if len(buckets["duplicate"]) > 0 {
			dupNote = fmt.Sprintf("  (%d duplicate dir(s) not offered)", len(buckets["duplicate"]))
		}
		fmt.Fprintf(&b, "identity: %d Claude worker dir(s) -> %d distinct Anthropic account(s)%s\n",
			claudeDirs, len(claudeLogins), dupNote)
	}
	b.WriteString("\n")
	fmt.Fprintf(&b, "AVAILABLE (offered to switcher now): %d\n", len(buckets["available"]))
	for _, r := range buckets["available"] {
		detail := fmt.Sprintf("%d active, %d live", derefInt(r.ActiveSessions), derefInt(r.LiveSessions))
		if r.LoginStatus != nil {
			detail += ", login=" + derefStr(r.LoginStatus)
		}
		fmt.Fprintf(&b, "  [%-8s] %-16s %-28s %-3s %-24s %s\n",
			product(r), r.Tag, r.Account, tierStr(r), derefStr(r.Model), detail)
	}
	if len(buckets["blocked"]) > 0 {
		fmt.Fprintf(&b, "\nBLOCKED (real account, do not offer now): %d\n", len(buckets["blocked"]))
		for _, r := range buckets["blocked"] {
			fmt.Fprintf(&b, "  [%-8s] %-16s %-28s %-3s %-24s %s\n",
				product(r), r.Tag, r.Account, tierStr(r), derefStr(r.Model), derefStr(r.BlockReason))
		}
	}
	if dropped := DroppedSeats(rows); len(dropped) > 0 {
		fmt.Fprintf(&b, "\nNEEDS RE-LOGIN (seat dropped from the offerable pool): %d\n", len(dropped))
		for _, d := range dropped {
			fmt.Fprintf(&b, "  %-16s %-28s %s\n", d.Tag, d.Account, d.Reason)
			fmt.Fprintf(&b, "  %-16s re-login: %s\n", "", d.NextAction)
		}
	}
	if len(buckets["duplicate"]) > 0 {
		fmt.Fprintf(&b, "\nDUPLICATE IDENTITY (same Anthropic account as a canonical dir -- not offered): %d\n",
			len(buckets["duplicate"]))
		for _, r := range buckets["duplicate"] {
			peers := strings.Join(r.IdentityPeers, ", ")
			fmt.Fprintf(&b, "  [%-8s] %-16s %-28s login=%s  shares with: %s\n",
				product(r), r.Tag, r.Account, derefStr(r.LoginEmail), peers)
		}
	}
	if len(buckets["excluded"]) > 0 {
		fmt.Fprintf(&b, "\nEXCLUDED (tombstoned): %d\n", len(buckets["excluded"]))
		for _, r := range buckets["excluded"] {
			fmt.Fprintf(&b, "  [%-8s] %-16s %-28s %s\n", product(r), r.Tag, r.Account, r.Reason)
		}
	}
	if len(buckets["non-account"]) > 0 {
		fmt.Fprintf(&b, "\nNON-ACCOUNT (ignored): %d\n", len(buckets["non-account"]))
		for _, r := range buckets["non-account"] {
			fmt.Fprintf(&b, "  [%-8s] %-16s %-28s %s\n", product(r), r.Tag, r.Account, r.Reason)
		}
	}
	polSrc := "(built-in defaults)"
	if policyExists {
		polSrc = policyPath
	} else if policyExampleNote != "" {
		polSrc = policyExampleNote
	}
	fmt.Fprintf(&b, "\npolicy: %s\n", polSrc)
	return b.String()
}

func product(r Account) string {
	if r.Product == "" {
		return "claude"
	}
	return r.Product
}

func tierStr(r Account) string {
	if r.ModelTier == nil {
		return "t?"
	}
	return "t" + itoa(*r.ModelTier)
}

// RenderSeats renders the human seat-pool view (the Python _cli_seats).
func RenderSeats(pool SeatPool) string {
	var b strings.Builder
	depleted := ""
	if pool.Depleted {
		depleted = "  DEPLETED"
	}
	fmt.Fprintf(&b, "seat pool [%s]: %d seat(s)  free=%d leased=%d blocked=%d%s\n",
		pool.Product, pool.TotalSeats, pool.FreeSeats, pool.LeasedSeats, pool.BlockedSeats, depleted)
	for _, s := range pool.Seats {
		workers := strings.Join(s.Workers, ", ")
		if workers == "" {
			workers = "-"
		}
		tier := "t?"
		if s.ModelTier != nil {
			tier = "t" + itoa(*s.ModelTier)
		}
		fmt.Fprintf(&b, "  [%-7s] %-16s %-28s %-3s -> %s\n", s.State, s.Tag, s.Account, tier, workers)
	}
	if len(pool.DoubleBooked) > 0 {
		b.WriteString("\nDOUBLE-BOOKED (one seat, >1 live worker -- INVARIANT VIOLATION):\n")
		for _, d := range pool.DoubleBooked {
			fmt.Fprintf(&b, "  %s: %s\n", d.Tag, strings.Join(d.Workers, ", "))
		}
	}
	if len(pool.UnboundLeases) > 0 {
		b.WriteString("\nUNBOUND LEASES (live worker on an account not in the pool):\n")
		for _, u := range pool.UnboundLeases {
			fmt.Fprintf(&b, "  %s: tag=%s dir=%s\n", u.Worker, u.Tag, u.Dir)
		}
	}
	return b.String()
}
