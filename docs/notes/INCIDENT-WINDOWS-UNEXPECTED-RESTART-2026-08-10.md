# Windows unexpected restart incident — 2026-08-10

## Verdict

At 17:15:19 PDT on 2026-08-10 this workstation stopped without a clean shutdown and
booted again at 17:48:38. Windows records this as an **ungraceful hang/reset**, not as
a bugcheck. The initiating component is not recoverable from the retained evidence,
so the root-cause statement is deliberately confidence-ranked rather than naming an
unsupported driver or hardware part.

| Confidence | Finding | Evidence |
|---|---|---|
| Confirmed | Windows did not complete a normal shutdown. | System Event 6008 says the previous shutdown at 17:15:19 was unexpected; Kernel-Power 41 was emitted at the next boot. |
| Confirmed | This occurrence was not a Windows bugcheck. | Event 41 has `BugcheckCode=0`; there is no WER-SystemErrorReporting 1001, minidump, `MEMORY.DMP`, or `volmgr` dump event for this occurrence. |
| Confirmed | The user/session service layer was already unhealthy. | At 17:10:47 nine per-user services terminated together (`CDPUserSvc`, `webthreatdefusersvc`, `NPSMSvc`, `UdkUserSvc`, `PimIndexMaintenanceSvc`, `UnistoreSvc`, `UserDataSvc`, `WpnUserService`, and `cbdhsvc`); their restart attempts then timed out. Winlogon failed to connect the user session at 17:19:15. |
| Likely | Broad scheduler/session pressure caused or accompanied the hang, after which the machine was externally reset or lost power. | The service fan-out is consistent with a session-wide stall, while Event 41 records no power-button timestamp, sleep transition, checkpoint, or bugcheck. Post-boot `fak stallscan` repeatedly observed scheduler/soft-fault pressure and process handle/thread leaks. |
| Not demonstrated | A particular GPU driver, DIMM, disk, PSU, or Windows update initiated the failure. | No WHEA error, disk/stornvme error, dump failure, planned-restart 1074, or update-restart event correlates with the boundary. Absence narrows the class but cannot prove hardware healthy during an uninstrumented hard hang. |

The 33-minute interval between the last-shutdown boundary and the next boot is
consistent with a machine that remained hung until recovery; event logs cannot tell
whether the final recovery was a reset button, power interruption, or another external
action.

## Separate historical failures

Do not merge these signatures into the August incident:

- 2026-07-14: bugcheck `0x4e` (`PFN_LIST_CORRUPT`); Windows successfully wrote
  `071426-18796-01.dmp` at the time.
- 2026-07-10: bugcheck `0x9f` (`DRIVER_POWER_STATE_FAILURE`); Windows successfully
  wrote `071026-18265-01.dmp` at the time.

Those dump files are no longer present, so attributing either historical bugcheck to a
specific module now would be speculation. A future recurrence must preserve the dump
before cleanup if it has a nonzero bugcheck code.

## Remediation applied on 2026-08-11

1. Terminated a leaking Chrome GPU-process tree observed at 10,758 handles and 567
   threads. Relaunched Chrome processes were below 2,000 handles and about 50 threads
   at read-back.
2. Restarted `FakStallscanWatch`; its bounded JSONL began advancing again.
3. Changed `FakStallscanWatch` from logon-only operation to logon **plus a five-minute
   repeating self-heal trigger** with `IgnoreNew`. A periodic trigger while the watcher
   is healthy is refused/ignored rather than creating a duplicate.
4. Registered `FakStallMonitor` with the same self-healing triggers, running
   `tools/fak_stall_monitor.ps1 -AutoMitigate`. Its actuators remain fail-closed: daemon
   mitigation is allowlisted and terminal replacement requires the leaf's consecutive
   pressure, cooldown, restorable-dashboard, and stateful-descendant gates.

At read-back both tasks were `Running`; `stallscan.jsonl` was less than one minute old.
The auto monitor was also producing `stallscan-auto.jsonl`. Current memory and storage
headroom did not indicate exhaustion (about 79.5/269.6 GiB committed, 177 GiB available,
and the system disk reported Healthy/OK with about 2 TiB free), although scheduler
pressure remained elevated. Therefore the remediation is recurrence capture and bounded
pressure relief, not a claim that an unobserved initiating defect has been proven gone.

## Recurrence decision tree

After any later unexpected boot, preserve evidence before restarting applications:

```powershell
Get-WinEvent -FilterHashtable @{LogName='System'; Id=41,6008,1001,1074} -MaxEvents 20
Get-Content "$env:LOCALAPPDATA\Fleet\stallscan.jsonl" -Tail 20
Get-Content "$env:LOCALAPPDATA\Fleet\stallscan-auto.jsonl" -Tail 20
Get-ChildItem C:\Windows\Minidump,C:\Windows\LiveKernelReports -Recurse -File -ErrorAction SilentlyContinue
```

Classify mechanically:

- nonzero Event-41 bugcheck or Event 1001: preserve and analyze the dump;
- WHEA event: escalate to CPU/RAM/PCIe/firmware hardware analysis;
- disk/stornvme reset/error: preserve storage telemetry and firmware state;
- bugcheck zero plus a pressure frame immediately before the boot boundary: attribute
  the pressure holder shown by that frame and reproduce/mitigate it;
- bugcheck zero and no pre-boundary frame despite a healthy watcher: investigate power,
  firmware, or failure below the OS logging boundary.

One privileged integrity check remains operator-runnable because this session was not
elevated:

```powershell
DISM /Online /Cleanup-Image /ScanHealth
sfc /verifyonly
```

A clean result increases confidence in the current Windows image but does not by itself
identify the cause of a hard hang.
