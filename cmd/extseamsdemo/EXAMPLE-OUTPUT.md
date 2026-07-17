# Captured output

```bash
go run ./cmd/extseamsdemo
```

```text
fak extension seams
choose the least-privileged seam that works; executable in-process extensions are trusted binary code, while agent-authored improvements remain proposals until independently witnessed

SEAM                   ATTACHMENT     TRUST             USE WHEN
agent-hook             out-of-process untrusted         code is user-authored, agent-authored, replaceable, or needs process isolation
capability-resolver    lazy protocol  adjudicated       a capability body should page in only after discovery
compute-backend        in-process     trusted-compiled  a reviewed backend needs zero-copy or device-local integration
console-pane           in-process     trusted-compiled  a reviewed pane must render inside the console
improvement-proposal   artifact       untrusted         an agent proposes an improvement; an independent witness decides keep or revert
kernel-abi             in-process     trusted-compiled  a low-level mechanism must participate in the frozen kernel contract
middleware             in-process     trusted-compiled  the code must surround every call and ships in the trusted binary
policy-bundle          manifest       data              the extension can be expressed as restrictions, not executable code
quality-oracle         in-process     trusted-compiled  a stable high-volume checker is reviewed and compiled with fak
trajectory-scorer      in-process     trusted-compiled  a deterministic scorer feeds routing or trajectory control
```

Exit code: `0`.
