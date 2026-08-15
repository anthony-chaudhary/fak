package workloadfit

import (
	_ "embed"
	"fmt"
)

//go:embed testdata/coding-legal.json
var codingLegalFixture []byte

func Selfcheck() (Selection, Selection, error) {
	fixture, err := Parse(codingLegalFixture)
	if err != nil {
		return Selection{}, Selection{}, err
	}
	coding := Select(fixture.Contracts[0], fixture.Catalog, fixture.AsOf)
	legal := Select(fixture.Contracts[1], fixture.Catalog, fixture.AsOf)
	if coding.Status != "fit" || coding.Chosen != "ponytail@r8" {
		return Selection{}, Selection{}, fmt.Errorf("coding choice = %s/%s", coding.Status, coding.Chosen)
	}
	if legal.Status != "fit" || legal.Chosen != "legal-review-harness@r4" {
		return Selection{}, Selection{}, fmt.Errorf("legal choice = %s/%s", legal.Status, legal.Chosen)
	}
	return coding, legal, nil
}
