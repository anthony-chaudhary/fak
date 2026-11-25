"""Answer, for each [pattern-name, text] pair on stdin, whether fleet_sessions' own regex finds it."""
import json
import re
import sys

sys.path.insert(0, sys.argv[1])
import fleet_session_signals  # noqa: E402
import fleet_sessions  # noqa: E402

PATTERNS = {
    "done": fleet_sessions.DONE_RE,
    "auton": fleet_sessions.AUTON_RE,
    "superv": fleet_sessions.SUPERVISED_RE,
    "wrapper": fleet_sessions._WRAPPER_RE,
    "taskcmd": fleet_sessions._TASK_CMD_RE,
    "userclose": re.compile(r"<command-name>/(quit|exit|clear|logout|login)\b"),
    "apierr": fleet_session_signals.APIERR_RE,
    "httpstatus": fleet_session_signals.HTTP_STATUS_RE,
    "operatorstop": fleet_session_signals.OPERATOR_STOP_RE,
}

cases = json.loads(sys.stdin.buffer.read().decode("utf-8"))
json.dump([bool(PATTERNS[name].search(text)) for name, text in cases], sys.stdout)
