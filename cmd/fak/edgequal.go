package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/edgequal"
)

func cmdEdgequal(argv []string) {
	os.Exit(runEdgequal(os.Stdout, os.Stderr, argv))
}

const edgequalUsage = `usage: fak edgequal <pack|validate|validate-pair> [flags]

Low-resource offline evidence validator (issue #8600).

Commands:
  pack           print embedded test pack bytes or checksum
  validate       validate a single physical execution receipt JSON
  validate-pair  validate a phone and laptop receipt pair
`

func runEdgequal(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprint(stderr, edgequalUsage)
		return 2
	}
	switch argv[0] {
	case "pack":
		return runEdgequalPack(stdout, stderr, argv[1:])
	case "validate":
		return runEdgequalValidate(stdout, stderr, argv[1:])
	case "validate-pair":
		return runEdgequalValidatePair(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprint(stdout, edgequalUsage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak edgequal: unknown subcommand %q\n\n%s", argv[0], edgequalUsage)
		return 2
	}
}

func runEdgequalPack(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak edgequal pack", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sha := fs.Bool("sha256", false, "emit only the SHA-256 digest of the pack")
	raw := fs.Bool("raw", false, "emit raw unformatted pack JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *sha {
		fmt.Fprintln(stdout, edgequal.PackSHA256())
		return 0
	}
	if *raw {
		_, _ = stdout.Write(edgequal.PackBytes())
		fmt.Fprintln(stdout)
		return 0
	}
	var pretty any
	if err := json.Unmarshal(edgequal.PackBytes(), &pretty); err != nil {
		fmt.Fprintf(stderr, "fak edgequal pack: unmarshal error: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(pretty); err != nil {
		fmt.Fprintf(stderr, "fak edgequal pack: encode error: %v\n", err)
		return 1
	}
	return 0
}

func runEdgequalValidate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak edgequal validate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	receiptPath := fs.String("receipt", "", "path to receipt JSON")
	asJSON := fs.Bool("json", false, "emit structured validation outcome as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *receiptPath == "" {
		if fs.NArg() == 1 {
			*receiptPath = fs.Arg(0)
		} else {
			fmt.Fprintln(stderr, "fak edgequal validate: missing required --receipt <path>")
			return 2
		}
	}
	b, err := os.ReadFile(*receiptPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak edgequal validate: read %s: %v\n", *receiptPath, err)
		return 1
	}
	r, err := edgequal.Parse(b)
	if err != nil {
		fmt.Fprintf(stderr, "fak edgequal validate: parse error: %v\n", err)
		return 1
	}
	valErr := edgequal.Validate(r)
	if *asJSON {
		res := map[string]any{
			"schema": edgequal.Schema,
			"valid":  valErr == nil,
			"status": r.Status,
			"device": r.Device.Class,
		}
		if valErr != nil {
			res["error"] = valErr.Error()
		}
		if r.RefusalCode != "" {
			res["refusal_code"] = r.RefusalCode
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
	} else {
		if valErr != nil {
			fmt.Fprintf(stderr, "INVALID: %v\n", valErr)
			return 1
		}
		if r.Status == "refused" {
			fmt.Fprintf(stdout, "VALID (refused): %s [%s]\n", r.RefusalCode, r.Device.Class)
		} else {
			fmt.Fprintf(stdout, "VALID: pass [%s]\n", r.Device.Class)
		}
	}
	if valErr != nil {
		return 1
	}
	return 0
}

func runEdgequalValidatePair(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak edgequal validate-pair", flag.ContinueOnError)
	fs.SetOutput(stderr)
	phonePath := fs.String("phone", "", "path to android_arm64_phone receipt JSON")
	laptopPath := fs.String("laptop", "", "path to laptop_8gib receipt JSON")
	asJSON := fs.Bool("json", false, "emit structured validation outcome as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *phonePath == "" || *laptopPath == "" {
		fmt.Fprintln(stderr, "fak edgequal validate-pair: requires both --phone <path> and --laptop <path>")
		return 2
	}
	pb, err := os.ReadFile(*phonePath)
	if err != nil {
		fmt.Fprintf(stderr, "fak edgequal validate-pair: read phone: %v\n", err)
		return 1
	}
	phone, err := edgequal.Parse(pb)
	if err != nil {
		fmt.Fprintf(stderr, "fak edgequal validate-pair: parse phone: %v\n", err)
		return 1
	}
	lb, err := os.ReadFile(*laptopPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak edgequal validate-pair: read laptop: %v\n", err)
		return 1
	}
	laptop, err := edgequal.Parse(lb)
	if err != nil {
		fmt.Fprintf(stderr, "fak edgequal validate-pair: parse laptop: %v\n", err)
		return 1
	}
	pairErr := edgequal.ValidatePair(phone, laptop)
	if *asJSON {
		res := map[string]any{
			"schema": edgequal.Schema,
			"valid":  pairErr == nil,
			"phone":  phone.Device.Class,
			"laptop": laptop.Device.Class,
		}
		if pairErr != nil {
			res["error"] = pairErr.Error()
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
	} else {
		if pairErr != nil {
			fmt.Fprintf(stderr, "INVALID PAIR: %v\n", pairErr)
			return 1
		}
		fmt.Fprintln(stdout, "VALID PAIR: physical low-resource phone and laptop verified")
	}
	if pairErr != nil {
		return 1
	}
	return 0
}
