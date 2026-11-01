"""Fail the job unless the JUnit report shows tests that ran, passed, and were not skipped.

The database-backed tests skip themselves when RUN_DATABASE_TESTS is not set. Without this check a
lost environment variable would turn the integration suite into a silent green run.
"""

import sys
import xml.etree.ElementTree as ET


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_test_report.py JUNIT_XML")
    root = ET.parse(sys.argv[1]).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {
        name: sum(int(suite.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    print(", ".join(f"{name}={count}" for name, count in totals.items()))

    problems = []
    if totals["tests"] == 0:
        problems.append("no tests ran")
    if totals["failures"] or totals["errors"]:
        problems.append(f"{totals['failures']} failed and {totals['errors']} errored")
    if totals["skipped"]:
        problems.append(
            f"{totals['skipped']} skipped (the PostgreSQL integration tests must all run in CI)"
        )
    if problems:
        print(f"::error title=Test report::{'; '.join(problems)}.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
