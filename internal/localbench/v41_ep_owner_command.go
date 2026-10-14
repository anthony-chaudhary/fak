package localbench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
)

const v41EPOwnerCommandSchema = "fak.v41-ep-owner.command-diagnostic/v1"

// This diagnostic is input agreement and availability information, never a
// V41EPOwnerReceipt or a measurement. Nil observations remain unavailable.
type v41EPOwnerCommandDiagnostic struct {
	Schema                      string   `json:"schema"`
	Operation                   string   `json:"operation"`
	Status                      string   `json:"status"`
	RequestSHA256               *string  `json:"request_sha256"`
	Rank                        *int     `json:"rank"`
	DeclaredSourceSHA256        *string  `json:"declared_source_sha256"`
	DeclaredShardSetSHA256      *string  `json:"declared_shard_set_sha256"`
	DeclaredBackend             *string  `json:"declared_backend"`
	DeclaredDevice              *string  `json:"declared_device"`
	ObservedExecutableSHA256    *string  `json:"observed_executable_sha256"`
	ArtifactAuthentication      string   `json:"artifact_authentication"`
	SourceAttestation           string   `json:"source_attestation"`
	SupervisorEnforcement       string   `json:"supervisor_enforcement"`
	ExecutionAvailable          bool     `json:"execution_available"`
	ExecutionPerformed          bool     `json:"execution_performed"`
	NativeFactoriesEntered      bool     `json:"native_factories_entered"`
	InheritedDescriptorsAdopted bool     `json:"inherited_descriptors_adopted"`
	Unavailable                 []string `json:"unavailable"`
	EntryPointScope             string   `json:"entry_point_scope"`
}

// RunV41EPOwnerCommand is the existing benchmarks adapter's bounded diagnostic
// entry. Validation opens only its regular request file, not checkpoint files,
// endpoints, backend factories, or inherited handoff descriptors. GPU-tagged
// package initialization may already have run before this command is reached.
// The reserved owner mode refuses until native artifact issuance and binding
// are separately implemented and reviewed; no argument can supply that trust.
func RunV41EPOwnerCommand(ctx context.Context, stdout, stderr io.Writer, args []string) int {
	if ctx == nil || stdout == nil || stderr == nil {
		return 2
	}
	fail := func(code int, err error) int {
		fmt.Fprintf(stderr, "v41-ep-owner: %v\n", err)
		return code
	}
	if err := ctx.Err(); err != nil {
		return fail(1, err)
	}
	if len(args) == 1 && (args[0] == "--help" || args[0] == "-h") {
		_, err := io.WriteString(stdout, "usage: fak benchmarks run v41-ep-owner -- --validate --request FILE --rank 0|1|2\n"+
			"Validation reports declared request agreement and unavailable provenance; it performs no benchmark.\n"+
			"--supervised-owner is reserved for tools/v41epguard; native execution is unavailable.\n"+
			"Command-body validation does not guarantee driver-free package initialization.\n")
		if err != nil {
			return fail(1, err)
		}
		return 0
	}
	mode, path, rank, err := v41EPOwnerCommandArgs(args)
	if err != nil {
		return fail(2, err)
	}
	diagnostic := v41EPOwnerCommandDiagnostic{
		Schema: v41EPOwnerCommandSchema, Operation: mode, Status: "refused",
		ArtifactAuthentication: "unobserved", SourceAttestation: "unavailable", SupervisorEnforcement: "unobserved",
		Unavailable:     []string{"reviewed-native-artifact-issuer", "native-owner-binding"},
		EntryPointScope: "command body only; package initialization can precede this entry",
	}
	exitCode := 1
	if mode == "validate" {
		req, err := v41EPOwnerReadCommandRequest(ctx, path)
		if err != nil {
			return fail(1, err)
		}
		digest, err := V41EPOwnerRequestDigest(req)
		if err != nil {
			return fail(1, err)
		}
		diagnostic.Status = "request_valid"
		diagnostic.RequestSHA256, diagnostic.Rank = &digest, &rank
		diagnostic.DeclaredSourceSHA256, diagnostic.DeclaredShardSetSHA256 = &req.SourceSHA256, &req.ExpectedShardSetSHA256
		diagnostic.DeclaredBackend, diagnostic.DeclaredDevice = &req.Owners[rank].Backend, &req.Owners[rank].ExpectedDevice
		exitCode = 0
	}
	// Reserved supervised refusal intentionally precedes all inherited-descriptor
	// adoption and native hooks. A request or launch declaration cannot enable it.
	if err := ctx.Err(); err != nil {
		return fail(1, err)
	}
	if err := json.NewEncoder(stdout).Encode(diagnostic); err != nil {
		return fail(1, err)
	}
	if exitCode != 0 {
		return fail(exitCode, errors.New("native execution unavailable: reviewed artifact issuer and owner binding are missing"))
	}
	return 0
}

func v41EPOwnerCommandArgs(args []string) (mode, path string, rank int, err error) {
	seen := make(map[string]bool)
	rank = -1
	for i := 0; i < len(args); i++ {
		flag := args[i]
		if seen[flag] {
			return "", "", -1, fmt.Errorf("duplicate argument %q", flag)
		}
		seen[flag] = true
		switch flag {
		case "--validate", "--supervised-owner":
			if mode != "" {
				return "", "", -1, errors.New("conflicting operation modes")
			}
			mode = flag[2:]
		case "--request", "--rank":
			if i+1 >= len(args) {
				return "", "", -1, fmt.Errorf("%s requires a value", flag)
			}
			i++
			if flag == "--request" {
				path = args[i]
			} else {
				switch args[i] {
				case "0", "1", "2":
					rank = int(args[i][0] - '0')
				default:
					return "", "", -1, errors.New("rank must be exactly 0, 1, or 2")
				}
			}
		default:
			return "", "", -1, fmt.Errorf("unknown argument %q", flag)
		}
	}
	if mode == "supervised-owner" && len(args) == 1 {
		return mode, "", -1, nil
	}
	if mode != "validate" || path == "" || len(path) > V41EPOwnerMaxPathBytes || rank < 0 {
		return "", "", -1, errors.New("require --validate --request FILE --rank 0|1|2, or exactly --supervised-owner")
	}
	return mode, path, rank, nil
}

// Filesystem operations are synchronous; context cancellation is checked at
// operation boundaries, not claimed to preempt a blocked filesystem call.
func v41EPOwnerReadCommandRequest(ctx context.Context, path string) (req V41EPOwnerRequest, err error) {
	if err := ctx.Err(); err != nil {
		return req, err
	}
	info, err := os.Lstat(path)
	if err != nil {
		return req, err
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > V41EPOwnerMaxRequestBytes {
		return req, errors.New("request must be a nonempty bounded regular file, not a symlink")
	}
	if err := ctx.Err(); err != nil {
		return req, err
	}
	f, err := os.Open(path)
	if err != nil {
		return req, err
	}
	return v41EPOwnerReadOpenedCommandRequest(ctx, info, f)
}

type v41EPOwnerCommandFile interface {
	io.Reader
	Stat() (os.FileInfo, error)
	Close() error
}

// Ownership transfers even when descriptor validation or decoding fails.
func v41EPOwnerReadOpenedCommandRequest(ctx context.Context, info os.FileInfo, f v41EPOwnerCommandFile) (req V41EPOwnerRequest, err error) {
	defer func() { err = errors.Join(err, f.Close()) }()
	opened, err := f.Stat()
	if err != nil {
		return req, err
	}
	if !opened.Mode().IsRegular() || !os.SameFile(info, opened) || opened.Size() != info.Size() || !opened.ModTime().Equal(info.ModTime()) {
		return req, errors.New("request file changed before reading")
	}
	req, err = v41EPOwnerDecodeCommandRequest(ctx, f)
	if err != nil {
		return req, err
	}
	after, err := f.Stat()
	if err != nil {
		return req, err
	}
	if after.Size() != opened.Size() || !after.ModTime().Equal(opened.ModTime()) {
		return req, errors.New("request file changed while reading")
	}
	return req, ctx.Err()
}

func v41EPOwnerDecodeCommandRequest(ctx context.Context, reader io.Reader) (V41EPOwnerRequest, error) {
	if err := ctx.Err(); err != nil {
		return V41EPOwnerRequest{}, err
	}
	req, err := DecodeV41EPOwnerRequest(reader)
	if err := errors.Join(err, ctx.Err()); err != nil {
		return V41EPOwnerRequest{}, err
	}
	return req, nil
}
