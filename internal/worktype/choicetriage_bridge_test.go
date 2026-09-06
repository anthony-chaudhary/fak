package worktype

import (
	"errors"
	"testing"
)

func TestChoiceTriage_MapChoiceDisposition(t *testing.T) {
	tests := []struct {
		name        string
		disposition string
		wantClass   Class
		wantOK      bool
	}{
		{
			name:        "TAKE_OBVIOUS projects to Discrete",
			disposition: "TAKE_OBVIOUS",
			wantClass:   Discrete,
			wantOK:      true,
		},
		{
			name:        "FRESH_CONTEXT projects to Discrete",
			disposition: "FRESH_CONTEXT",
			wantClass:   Discrete,
			wantOK:      true,
		},
		{
			name:        "FILE_TICKET projects to Discrete",
			disposition: "FILE_TICKET",
			wantClass:   Discrete,
			wantOK:      true,
		},
		{
			name:        "HUMAN_RESIDUAL projects to HumanOperatorEffectiveness",
			disposition: "HUMAN_RESIDUAL",
			wantClass:   HumanOperatorEffectiveness,
			wantOK:      true,
		},
		{
			name:        "case-insensitive lowercase",
			disposition: "take_obvious",
			wantClass:   Discrete,
			wantOK:      true,
		},
		{
			name:        "whitespace trimmed",
			disposition: "  HUMAN_RESIDUAL  ",
			wantClass:   HumanOperatorEffectiveness,
			wantOK:      true,
		},
		{
			name:        "unknown disposition rejected",
			disposition: "UNKNOWN_DISPOSITION",
			wantClass:   "",
			wantOK:      false,
		},
		{
			name:        "empty disposition rejected",
			disposition: "",
			wantClass:   "",
			wantOK:      false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotClass, gotOK := MapChoiceDisposition(tc.disposition)
			if gotOK != tc.wantOK {
				t.Fatalf("MapChoiceDisposition(%q) ok = %v, want %v", tc.disposition, gotOK, tc.wantOK)
			}
			if gotClass != tc.wantClass {
				t.Errorf("MapChoiceDisposition(%q) class = %q, want %q", tc.disposition, gotClass, tc.wantClass)
			}
		})
	}
}

func TestChoiceTriage_DiscreteAlias(t *testing.T) {
	if Discrete != DiscreteEpic {
		t.Errorf("Discrete (%q) != DiscreteEpic (%q)", Discrete, DiscreteEpic)
	}
}

func TestChoiceTriage_Roundtrip(t *testing.T) {
	dispositions := []string{
		ChoiceTakeObvious,
		ChoiceFreshContext,
		ChoiceFileTicket,
		ChoiceHumanResidual,
	}

	for _, d := range dispositions {
		t.Run(d, func(t *testing.T) {
			class, ok := MapChoiceDisposition(d)
			if !ok {
				t.Fatalf("MapChoiceDisposition(%q) returned ok=false", d)
			}
			if err := ValidateTaxonomyAlignment(d, class); err != nil {
				t.Fatalf("ValidateTaxonomyAlignment(%q, %q) returned error on projected pair: %v", d, class, err)
			}
		})
	}
}

func TestChoiceTriage_ValidateTaxonomyAlignment_Valid(t *testing.T) {
	validCases := []struct {
		disposition string
		class       Class
	}{
		{"TAKE_OBVIOUS", Discrete},
		{"TAKE_OBVIOUS", DiscreteEpic},
		{"TAKE_OBVIOUS", KernelOptimization},
		{"TAKE_OBVIOUS", CacheOptimization},
		{"TAKE_OBVIOUS", HumanOperatorEffectiveness},
		{"FRESH_CONTEXT", Discrete},
		{"FILE_TICKET", Discrete},
		{"HUMAN_RESIDUAL", HumanOperatorEffectiveness},
		{"HUMAN_RESIDUAL", Discrete},
		{"HUMAN_RESIDUAL", DiscreteEpic},
	}

	for _, tc := range validCases {
		if err := ValidateTaxonomyAlignment(tc.disposition, tc.class); err != nil {
			t.Errorf("ValidateTaxonomyAlignment(%q, %q) unexpected error: %v", tc.disposition, tc.class, err)
		}
	}
}

func TestChoiceTriage_Contradictions(t *testing.T) {
	contradictoryCases := []struct {
		name        string
		disposition string
		class       Class
	}{
		{
			name:        "HUMAN_RESIDUAL with KernelOptimization",
			disposition: "HUMAN_RESIDUAL",
			class:       KernelOptimization,
		},
		{
			name:        "HUMAN_RESIDUAL with CacheOptimization",
			disposition: "HUMAN_RESIDUAL",
			class:       CacheOptimization,
		},
		{
			name:        "human_residual lowercase with KernelOptimization",
			disposition: "human_residual",
			class:       KernelOptimization,
		},
		{
			name:        "human_residual with whitespace with CacheOptimization",
			disposition: "  HUMAN_RESIDUAL  ",
			class:       CacheOptimization,
		},
	}

	for _, tc := range contradictoryCases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateTaxonomyAlignment(tc.disposition, tc.class)
			if err == nil {
				t.Fatalf("ValidateTaxonomyAlignment(%q, %q) expected error, got nil", tc.disposition, tc.class)
			}
			if !errors.Is(err, ErrContradictoryOwnership) {
				t.Errorf("ValidateTaxonomyAlignment(%q, %q) error = %v, want ErrContradictoryOwnership", tc.disposition, tc.class, err)
			}
			if err != ErrContradictoryOwnership {
				t.Errorf("ValidateTaxonomyAlignment(%q, %q) direct equality failed: got %v", tc.disposition, tc.class, err)
			}
		})
	}
}

func TestChoiceTriage_InvalidInputs(t *testing.T) {
	t.Run("unknown disposition", func(t *testing.T) {
		err := ValidateTaxonomyAlignment("UNKNOWN_DISPOSITION", Discrete)
		if !errors.Is(err, ErrUnknownDisposition) {
			t.Errorf("got %v, want ErrUnknownDisposition", err)
		}
	})

	t.Run("empty disposition", func(t *testing.T) {
		err := ValidateTaxonomyAlignment("", Discrete)
		if !errors.Is(err, ErrUnknownDisposition) {
			t.Errorf("got %v, want ErrUnknownDisposition", err)
		}
	})

	t.Run("unknown work class", func(t *testing.T) {
		err := ValidateTaxonomyAlignment(ChoiceTakeObvious, Class("not-a-valid-class"))
		if !errors.Is(err, ErrUnknownWorkClass) {
			t.Errorf("got %v, want ErrUnknownWorkClass", err)
		}
	})

	t.Run("empty work class", func(t *testing.T) {
		err := ValidateTaxonomyAlignment(ChoiceTakeObvious, Class(""))
		if !errors.Is(err, ErrUnknownWorkClass) {
			t.Errorf("got %v, want ErrUnknownWorkClass", err)
		}
	})
}

func TestChoiceTriage_Helpers(t *testing.T) {
	t.Run("ChoiceDispositions length and elements", func(t *testing.T) {
		if len(ChoiceDispositions) != 4 {
			t.Fatalf("expected 4 ChoiceDispositions, got %d", len(ChoiceDispositions))
		}
		for _, d := range ChoiceDispositions {
			if !ValidChoiceDisposition(d) {
				t.Errorf("expected %q in ChoiceDispositions to be valid", d)
			}
		}
	})

	t.Run("ValidChoiceDisposition", func(t *testing.T) {
		valid := []string{
			ChoiceTakeObvious,
			ChoiceFreshContext,
			ChoiceFileTicket,
			ChoiceHumanResidual,
			"take_obvious",
			"  fresh_context  ",
		}
		for _, s := range valid {
			if !ValidChoiceDisposition(s) {
				t.Errorf("ValidChoiceDisposition(%q) = false, want true", s)
			}
		}

		invalid := []string{"", "   ", "UNKNOWN", "OTHER"}
		for _, s := range invalid {
			if ValidChoiceDisposition(s) {
				t.Errorf("ValidChoiceDisposition(%q) = true, want false", s)
			}
		}
	})

	t.Run("Class.Valid", func(t *testing.T) {
		valid := []Class{
			KernelOptimization,
			CacheOptimization,
			HumanOperatorEffectiveness,
			DiscreteEpic,
			Discrete,
		}
		for _, c := range valid {
			if !c.Valid() {
				t.Errorf("Class(%q).Valid() = false, want true", c)
			}
		}

		invalid := []Class{"", "not-a-class", "random"}
		for _, c := range invalid {
			if c.Valid() {
				t.Errorf("Class(%q).Valid() = true, want false", c)
			}
		}
	})

	t.Run("IsContradictoryOwnership", func(t *testing.T) {
		if !IsContradictoryOwnership(ChoiceHumanResidual, KernelOptimization) {
			t.Errorf("expected contradiction for HUMAN_RESIDUAL + KernelOptimization")
		}
		if !IsContradictoryOwnership(ChoiceHumanResidual, CacheOptimization) {
			t.Errorf("expected contradiction for HUMAN_RESIDUAL + CacheOptimization")
		}
		if IsContradictoryOwnership(ChoiceHumanResidual, HumanOperatorEffectiveness) {
			t.Errorf("unexpected contradiction for HUMAN_RESIDUAL + HumanOperatorEffectiveness")
		}
		if IsContradictoryOwnership(ChoiceHumanResidual, Discrete) {
			t.Errorf("unexpected contradiction for HUMAN_RESIDUAL + Discrete")
		}
		if IsContradictoryOwnership(ChoiceTakeObvious, KernelOptimization) {
			t.Errorf("unexpected contradiction for TAKE_OBVIOUS + KernelOptimization")
		}
	})
}

func TestChoiceTriage_Matrix16(t *testing.T) {
	dispositions := []string{
		ChoiceTakeObvious,
		ChoiceFreshContext,
		ChoiceFileTicket,
		ChoiceHumanResidual,
	}
	classes := []Class{
		KernelOptimization,
		CacheOptimization,
		HumanOperatorEffectiveness,
		Discrete,
	}

	for _, d := range dispositions {
		for _, c := range classes {
			d := d
			c := c
			t.Run(d+"_with_"+string(c), func(t *testing.T) {
				err := ValidateTaxonomyAlignment(d, c)
				contradiction := d == ChoiceHumanResidual && (c == KernelOptimization || c == CacheOptimization)
				if contradiction {
					if !errors.Is(err, ErrContradictoryOwnership) {
						t.Errorf("ValidateTaxonomyAlignment(%q, %q) = %v, want ErrContradictoryOwnership", d, c, err)
					}
				} else {
					if err != nil {
						t.Errorf("ValidateTaxonomyAlignment(%q, %q) unexpected error: %v", d, c, err)
					}
				}
			})
		}
	}
}
