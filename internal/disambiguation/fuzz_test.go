package disambiguation_test

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/disambiguation"
)

// FuzzParseEntryMalformed drives the exported strict wire parser. Each seed is a
// different malformed-record class; accepted mutations must still satisfy the
// public record validator.
func FuzzParseEntryMalformed(f *testing.F) {
	f.Add([]byte(`{`))
	f.Add([]byte(`{"schema_version":"fak-disambiguation-entry/1"}`))
	f.Add([]byte(`{"schema_version":"fak-disambiguation-entry/999"}`))
	f.Add([]byte(`{"schema_version":"fak-disambiguation-entry/1","unknown":true}`))
	f.Add(append(mustMarshal(disambiguation.SelfTestEntry()), []byte(` {}`)...))

	f.Fuzz(func(t *testing.T, data []byte) {
		entry, err := disambiguation.ParseEntry(data)
		if err != nil {
			return
		}
		if err := entry.Validate(); err != nil {
			t.Fatalf("ParseEntry accepted a record rejected by Validate: %v", err)
		}
	})
}

// FuzzContrastGraphCycles constructs bounded directed cycles through NewIndex,
// the public graph-admission seam. Non-required contrast cycles are valid and
// must neither recurse nor hang.
func FuzzContrastGraphCycles(f *testing.F) {
	f.Add(uint8(2))
	f.Add(uint8(3))
	f.Add(uint8(17))
	f.Add(uint8(255))

	f.Fuzz(func(t *testing.T, size uint8) {
		n := 2 + int(size%31)
		entries := make([]disambiguation.Entry, n)
		for i := range entries {
			term := fmt.Sprintf("cycle-%02d", i)
			target := fmt.Sprintf("cycle-%02d", (i+1)%n)
			entries[i] = entry(term, []string{})
			entries[i].Contrasts = []disambiguation.Contrast{{
				CanonicalTerm:       target,
				Explanation:         "A bounded directed graph edge used by the fuzz witness.",
				RequiredPair:        boolPtr(false),
				ForbiddenConflation: boolPtr(false),
			}}
		}
		if _, err := disambiguation.NewIndex(entries); err != nil {
			t.Fatalf("NewIndex rejected valid %d-node contrast cycle: %v", n, err)
		}
	})
}

// FuzzDuplicateIdentities deterministically covers duplicate canonical terms,
// duplicate aliases, and canonical/alias collisions at the public index seam.
func FuzzDuplicateIdentities(f *testing.F) {
	f.Add(uint8(0), "duplicate")
	f.Add(uint8(1), "shared alias")
	f.Add(uint8(2), "canonical collision")

	f.Fuzz(func(t *testing.T, kind uint8, identity string) {
		identity = boundedIdentity(identity)
		left := entry("left", []string{"left alias"})
		right := entry("right", []string{"right alias"})
		switch kind % 3 {
		case 0:
			left.Identity.CanonicalTerm = identity
			right.Identity.CanonicalTerm = identity
		case 1:
			left.Identity.Aliases = []string{identity}
			right.Identity.Aliases = []string{identity}
		case 2:
			left.Identity.CanonicalTerm = identity
			right.Identity.Aliases = []string{identity}
		}
		if _, err := disambiguation.NewIndex([]disambiguation.Entry{left, right}); err == nil {
			t.Fatalf("NewIndex accepted duplicate identity kind %d for %q", kind%3, identity)
		}
	})
}

// FuzzUnicodeConfusables preserves exact identity semantics: visually similar
// Unicode aliases remain separate identities. It also drives the exported
// built-in resolver with the same untrusted terms and checks result isolation.
func FuzzUnicodeConfusables(f *testing.F) {
	f.Add("agent", "аgent")     // Cyrillic small a.
	f.Add("kernel", "ｋernel")   // Fullwidth small k.
	f.Add("café", "cafe\u0301") // Composed versus combining acute.
	f.Add("scope", "ſcope")     // Latin long s.

	f.Fuzz(func(t *testing.T, leftAlias, rightAlias string) {
		leftAlias = boundedIdentity(leftAlias)
		rightAlias = boundedIdentity(rightAlias)
		if leftAlias == rightAlias {
			return
		}
		left, right := entryPair("unicode-left", []string{leftAlias}, "unicode-right", []string{rightAlias})
		if _, err := disambiguation.NewIndex([]disambiguation.Entry{left, right}); err != nil {
			t.Fatalf("NewIndex conflated distinct exact Unicode identities %q and %q: %v", leftAlias, rightAlias, err)
		}

		first, err := disambiguation.Resolve(leftAlias)
		if err != nil {
			return // Most fuzzed identities are intentionally absent from the built-in index.
		}
		for i := range first.Entry.Contrasts {
			if first.Entry.Contrasts[i].RequiredPair != nil {
				*first.Entry.Contrasts[i].RequiredPair = false
			}
			if first.Entry.Contrasts[i].ForbiddenConflation != nil {
				*first.Entry.Contrasts[i].ForbiddenConflation = false
			}
		}
		second, err := disambiguation.Resolve(leftAlias)
		if err != nil {
			t.Fatalf("Resolve lost a previously resolved identity: %v", err)
		}
		for _, contrast := range second.Entry.Contrasts {
			if contrast.RequiredPair == nil || contrast.ForbiddenConflation == nil {
				t.Fatal("Resolve returned incomplete public contrast flags")
			}
		}
	})
}

// FuzzPathologicalAliasSets bounds adversarial cardinality and width while
// retaining empty, repeated, and very large deterministic seed cases.
func FuzzPathologicalAliasSets(f *testing.F) {
	f.Add(uint16(0), uint16(0), false)
	f.Add(uint16(1), uint16(4096), false)
	f.Add(uint16(256), uint16(64), false)
	f.Add(uint16(256), uint16(64), true)

	f.Fuzz(func(t *testing.T, countRaw, widthRaw uint16, duplicate bool) {
		count := int(countRaw % 257)
		width := int(widthRaw % 4097)
		aliases := make([]string, count)
		padding := strings.Repeat("x", width)
		for i := range aliases {
			aliases[i] = fmt.Sprintf("alias-%03d-%s", i, padding)
		}
		if duplicate && len(aliases) > 1 {
			aliases[len(aliases)-1] = aliases[0]
		}
		owner, peer := entryPair("alias-owner", aliases, "alias-peer", []string{})
		_, err := disambiguation.NewIndex([]disambiguation.Entry{owner, peer})
		if duplicate && len(aliases) > 1 {
			if err == nil {
				t.Fatal("NewIndex accepted a repeated alias in a pathological alias set")
			}
		} else if err != nil {
			t.Fatalf("NewIndex rejected a valid bounded alias set (count=%d width=%d): %v", count, width, err)
		}
	})
}

func entry(term string, aliases []string) disambiguation.Entry {
	value := disambiguation.SelfTestEntry()
	value.Identity.CanonicalTerm = term
	value.Identity.Aliases = aliases
	value.Contrasts = []disambiguation.Contrast{{
		CanonicalTerm:       "peer",
		Explanation:         "The fuzz identity and its peer are distinct concepts.",
		RequiredPair:        boolPtr(false),
		ForbiddenConflation: boolPtr(false),
	}}
	return value
}

func entryPair(leftTerm string, leftAliases []string, rightTerm string, rightAliases []string) (disambiguation.Entry, disambiguation.Entry) {
	left := entry(leftTerm, leftAliases)
	right := entry(rightTerm, rightAliases)
	left.Contrasts[0].CanonicalTerm = rightTerm
	right.Contrasts[0].CanonicalTerm = leftTerm
	return left, right
}

func boundedIdentity(value string) string {
	if !utf8.ValidString(value) {
		value = strings.ToValidUTF8(value, "�")
	}
	value = strings.TrimSpace(value)
	if value == "" {
		return "identity"
	}
	runes := []rune(value)
	if len(runes) > 128 {
		runes = runes[:128]
	}
	value = strings.TrimSpace(string(runes))
	if value == "" {
		return "identity"
	}
	return value
}

func boolPtr(value bool) *bool { return &value }

func mustMarshal(entry disambiguation.Entry) []byte {
	data, err := json.Marshal(entry)
	if err != nil {
		panic(err)
	}
	return data
}
