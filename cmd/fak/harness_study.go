package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnesscontrolpacket"
	"github.com/anthony-chaudhary/fak/internal/harnesscontrolstudy"
	"github.com/anthony-chaudhary/fak/internal/harnesscreationreceipt"
	"github.com/anthony-chaudhary/fak/internal/harnesscreationstudy"
	"github.com/anthony-chaudhary/fak/internal/harnesscrossover"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func runHarnessStudy(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "receipt" {
		return runHarnessCreationReceipt(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "creation" {
		return runHarnessCreationStudy(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "control" {
		if len(argv) > 1 && argv[1] == "packet" {
			return runHarnessControlPacket(stdout, stderr, argv[2:])
		}
		return runHarnessControlStudy(stdout, stderr, argv[1:])
	}
	if len(argv) == 0 || argv[0] != "crossover" {
		fmt.Fprintln(stderr, "usage: fak harness study <control|creation|crossover> --input STUDY.json")
		return 2
	}
	fs := flag.NewFlagSet("harness study crossover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "fak.harness-crossover-study/v1alpha1 JSON")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	if *input == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak harness study crossover --input STUDY.json")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study crossover: %v\n", err)
		return 1
	}
	study, err := harnesscrossover.Parse(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study crossover: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(harnesscrossover.Evaluate(study)); err != nil {
		fmt.Fprintf(stderr, "fak harness study crossover: %v\n", err)
		return 1
	}
	return 0
}

func runHarnessCreationStudy(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness study creation", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "fak.harness-creation-study/v1alpha1 JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *input == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak harness study creation --input STUDY.json")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study creation: %v\n", err)
		return 1
	}
	study, err := harnesscreationstudy.Parse(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study creation: %v\n", err)
		return 1
	}
	if err := verifyHarnessStudyReceiptSources(*input, study); err != nil {
		fmt.Fprintf(stderr, "fak harness study creation: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(harnesscreationstudy.Evaluate(study)); err != nil {
		fmt.Fprintf(stderr, "fak harness study creation: %v\n", err)
		return 1
	}
	return 0
}

func runHarnessCreationReceipt(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness study receipt", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "fak.harness-creation-receipt/v1alpha1 JSON")
	studyPath := fs.String("study", "", "optional study JSON for duplicate run/participant refusal")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *input == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak harness study receipt --input RECEIPT.json")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study receipt: %v\n", err)
		return 1
	}
	receipt, err := harnesscreationreceipt.Parse(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study receipt: %v\n", err)
		return 1
	}
	result := harnesscreationreceipt.Evaluate(receipt)
	if *studyPath != "" {
		studyAbs, absErr := filepath.Abs(*studyPath)
		if absErr != nil {
			fmt.Fprintf(stderr, "fak harness study receipt: %v\n", absErr)
			return 1
		}
		receiptAbs, absErr := filepath.Abs(*input)
		if absErr != nil {
			fmt.Fprintf(stderr, "fak harness study receipt: %v\n", absErr)
			return 1
		}
		rel, relErr := filepath.Rel(filepath.Dir(studyAbs), receiptAbs)
		if relErr != nil || filepath.IsAbs(rel) {
			fmt.Fprintln(stderr, "fak harness study receipt: source receipt must be relative to study directory")
			return 1
		}
		sum := sha256.Sum256(raw)
		result.Row.SourceReceipt = filepath.ToSlash(rel)
		result.Row.SourceDigest = "sha256:" + hex.EncodeToString(sum[:])
		studyRaw, readErr := os.ReadFile(*studyPath)
		if readErr != nil {
			fmt.Fprintf(stderr, "fak harness study receipt: %v\n", readErr)
			return 1
		}
		if uniqueErr := harnesscreationreceipt.CheckUnique(studyRaw, result.Row); uniqueErr != nil {
			fmt.Fprintf(stderr, "fak harness study receipt: %v\n", uniqueErr)
			return 1
		}
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		fmt.Fprintf(stderr, "fak harness study receipt: %v\n", err)
		return 1
	}
	return 0
}

func verifyHarnessStudyReceiptSources(studyPath string, study harnesscreationstudy.Study) error {
	studyAbs, err := filepath.Abs(studyPath)
	if err != nil {
		return err
	}
	base := filepath.Dir(studyAbs)
	for _, run := range study.Runs {
		// Inline fixture rows keep parser/aggregate tests hermetic; production receipt paths never use this prefix.
		if run.PairID == "" || (run.SourceReceipt == "" && run.SourceDigest == "" && strings.HasPrefix(run.Receipt, "fixture:")) || (run.ParticipantClass == "maintainer-calibration" && run.SourceReceipt == "" && run.SourceDigest == "") {
			continue
		}
		if run.SourceReceipt == "" || !strings.HasPrefix(run.SourceDigest, "sha256:") {
			return fmt.Errorf("run %q requires source_receipt and source_digest", run.ID)
		}
		source := filepath.Clean(filepath.Join(base, filepath.FromSlash(run.SourceReceipt)))
		rel, err := filepath.Rel(base, source)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
			return fmt.Errorf("run %q source_receipt escapes study directory", run.ID)
		}
		raw, err := os.ReadFile(source)
		if err != nil {
			return fmt.Errorf("run %q source_receipt: %w", run.ID, err)
		}
		sum := sha256.Sum256(raw)
		if "sha256:"+hex.EncodeToString(sum[:]) != run.SourceDigest {
			return fmt.Errorf("run %q source_digest does not match archived receipt", run.ID)
		}
		receipt, err := harnesscreationreceipt.Parse(raw)
		if err != nil {
			return fmt.Errorf("run %q source_receipt: %w", run.ID, err)
		}
		row := harnesscreationreceipt.Evaluate(receipt).Row
		if row.ID != run.ID || row.ParticipantID != run.ParticipantID || row.Track != run.Track || row.Arm != run.Arm || row.PairID != run.PairID || row.TaskDigest != run.TaskDigest || row.MachineID != run.MachineID || row.PairOrder != run.PairOrder || row.ArmPosition != run.ArmPosition || row.ParticipantClass != run.ParticipantClass || row.Independent != run.Independent || row.OS != run.OS || row.CPU != run.CPU || row.NetworkState != run.NetworkState || row.CacheState != run.CacheState || row.Outcome != run.Outcome || row.ElapsedSeconds != run.ElapsedSeconds || row.Receipt != run.Receipt {
			return fmt.Errorf("run %q does not match archived receipt projection", run.ID)
		}
	}
	return nil
}

func runHarnessControlStudy(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness study control", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "fak-harness-control-study/1 JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *input == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak harness study control --input STUDY.json")
		return 2
	}
	report, err := harnesscontrolstudy.Evaluate(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study control: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		fmt.Fprintf(stderr, "fak harness study control: %v\n", err)
		return 1
	}
	if report.Verdict != "measured" {
		return 3
	}
	return 0
}

func runHarnessControlPacket(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || (argv[0] != "create" && argv[0] != "verify" && argv[0] != "receipt") {
		fmt.Fprintln(stderr, "usage: fak harness study control packet <create|verify|receipt>")
		return 2
	}
	if argv[0] == "receipt" {
		return runHarnessControlPacketReceipt(stdout, stderr, argv[1:])
	}
	if argv[0] == "verify" {
		fs := flag.NewFlagSet("harness study control packet verify", flag.ContinueOnError)
		fs.SetOutput(stderr)
		dir := fs.String("dir", "", "extracted assigned-arm packet directory")
		if err := fs.Parse(argv[1:]); err != nil {
			return 2
		}
		if *dir == "" || fs.NArg() != 0 {
			fmt.Fprintln(stderr, "fak harness study control packet verify: --dir is required")
			return 2
		}
		*dir = pathutil.ExpandTilde(*dir)
		raw, err := os.ReadFile(filepath.Join(*dir, "packet.json"))
		if err != nil {
			fmt.Fprintf(stderr, "fak harness study control packet verify: %v\n", err)
			return 1
		}
		manifest, err := harnesscontrolpacket.Parse(raw)
		if err == nil {
			err = harnesscontrolpacket.Verify(*dir, manifest)
		}
		if err != nil {
			fmt.Fprintf(stderr, "fak harness study control packet verify: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "HARNESS CONTROL PACKET | VERIFIED\narm: %s\nsource: %s\nbinary: %s\nfiles: %d\n", manifest.Arm, manifest.SourceCommit, manifest.BinaryVersion, len(manifest.Files))
		return 0
	}

	fs := flag.NewFlagSet("harness study control packet create", flag.ContinueOnError)
	fs.SetOutput(stderr)
	arm := fs.String("arm", "", "assigned arm: default-control or scratch")
	materials := fs.String("materials", "", "assigned arm materials directory")
	binary := fs.String("binary", "", "pinned Linux/amd64 fak binary")
	receipt := fs.String("receipt", "", "blank receipt template")
	output := fs.String("output", "", "new packet directory")
	commit := fs.String("source-commit", "", "exact source commit embedded in the binary")
	version := fs.String("binary-version", "", "exact output of fak version, flattened to one line")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		return 2
	}
	manifest, err := harnesscontrolpacket.Create(harnesscontrolpacket.CreateOptions{Arm: *arm, MaterialsDir: *materials, BinaryPath: *binary, ReceiptPath: *receipt, OutputDir: *output, SourceCommit: *commit, BinaryVersion: *version})
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study control packet create: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "HARNESS CONTROL PACKET | CREATED\narm: %s\nsource: %s\noutput: %s\nfiles: %d\nnext: fak harness study control packet verify --dir %s\n", manifest.Arm, manifest.SourceCommit, *output, len(manifest.Files), *output)
	return 0
}

func runHarnessControlPacketReceipt(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak harness study control packet receipt <start|finalize>")
		return 2
	}
	switch argv[0] {
	case "start":
		fs := harnessReceiptFlagSet("start", stderr)
		dir := harnessReceiptStringFlag(fs, "dir", ".", "assigned packet directory")
		participant := harnessReceiptStringFlag(fs, "participant-id", "", "anonymous participant ID")
		pair := harnessReceiptStringFlag(fs, "pair-id", "", "anonymous pair ID")
		order := harnessReceiptStringFlag(fs, "pair-order", "", "default-first or scratch-first")
		if !parseHarnessReceiptFlags(fs, argv[1:], dir) {
			return 2
		}
		r, ok := executeHarnessReceipt(stderr, "start", func() (harnesscontrolstudy.Receipt, error) {
			return harnesscontrolpacket.StartReceipt(harnesscontrolpacket.ReceiptStartOptions{Dir: *dir, ParticipantID: *participant, PairID: *pair, PairOrder: *order})
		})
		if !ok {
			return 1
		}
		fmt.Fprintf(stdout, "HARNESS CONTROL RECEIPT | STARTED\narm: %s\nposition: %d\nstarted: %s\nreceipt: %s\n", r.Arm, r.ArmPosition, r.StartedAt, filepath.Join(*dir, "receipt.json"))
		return 0
	case "finalize":
		fs := harnessReceiptFlagSet("finalize", stderr)
		dir := harnessReceiptStringFlag(fs, "dir", ".", "assigned packet directory")
		artifact := harnessReceiptStringFlag(fs, "artifact", "", "final artifact to digest")
		commands := harnessReceiptStringFlag(fs, "commands", "", "newline-delimited command transcript")
		errorsPath := harnessReceiptStringFlag(fs, "errors", "", "optional newline-delimited errors")
		succeeded := fs.Bool("succeeded", false, "task outcomes succeeded")
		verified := fs.Bool("verified", false, "independent verification succeeded")
		help := fs.Int("help-requests", 0, "facilitator help requests")
		confidence := fs.Int("confidence", 0, "confidence from 1 to 5")
		inspect := fs.Bool("inspect-captured", false, "inspect evidence captured")
		preview := fs.Bool("preview-captured", false, "preview evidence captured")
		runtime := fs.Bool("runtime-verify-captured", false, "runtime verification evidence captured")
		preference := harnessReceiptStringFlag(fs, "preference", "", "second-arm preference: default-control, scratch, or none")
		reason := harnessReceiptStringFlag(fs, "preference-reason", "", "second-arm preference reason")
		if !parseHarnessReceiptFlags(fs, argv[1:], dir) {
			return 2
		}
		r, ok := executeHarnessReceipt(stderr, "finalize", func() (harnesscontrolstudy.Receipt, error) {
			return harnesscontrolpacket.FinalizeReceipt(harnesscontrolpacket.ReceiptFinalizeOptions{Dir: *dir, ArtifactPath: *artifact, CommandsPath: *commands, ErrorsPath: *errorsPath, Succeeded: *succeeded, Verified: *verified, HelpRequests: *help, Confidence: *confidence, InspectCaptured: *inspect, PreviewCaptured: *preview, RuntimeVerifyCaptured: *runtime, Preference: *preference, PreferenceReason: *reason})
		})
		if !ok {
			return 1
		}
		fmt.Fprintf(stdout, "HARNESS CONTROL RECEIPT | FINALIZED\narm: %s\nelapsed_seconds: %.3f\nartifact: %s\nreceipt: %s\n", r.Arm, r.ElapsedSeconds, r.ArtifactDigest, filepath.Join(*dir, "receipt.json"))
		return 0
	default:
		fmt.Fprintln(stderr, "usage: fak harness study control packet receipt <start|finalize>")
		return 2
	}
}

func parseHarnessReceiptFlags(fs *flag.FlagSet, argv []string, dir *string) bool {
	if err := fs.Parse(argv); err != nil || fs.NArg() != 0 {
		return false
	}
	*dir = expandHarnessReceiptDir(*dir)
	return true
}

func harnessReceiptFlagSet(action string, stderr io.Writer) *flag.FlagSet {
	fs := flag.NewFlagSet("harness study control packet receipt "+action, flag.ContinueOnError)
	fs.SetOutput(stderr)
	return fs
}

func expandHarnessReceiptDir(dir string) string { return pathutil.ExpandTilde(dir) }

func harnessReceiptStringFlag(fs *flag.FlagSet, name, value, usage string) *string {
	return fs.String(name, value, usage)
}

func executeHarnessReceipt(stderr io.Writer, action string, execute func() (harnesscontrolstudy.Receipt, error)) (harnesscontrolstudy.Receipt, bool) {
	receipt, err := execute()
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study control packet receipt %s: %v\n", action, err)
		return receipt, false
	}
	return receipt, true
}
