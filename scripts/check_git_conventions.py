"""Check this repository's commit subjects, branches, and GitHub event metadata."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "build",
    "ci",
    "chore",
    "revert",
)
TYPE_PATTERN = "|".join(TYPES)
SUBJECT = re.compile(rf"({TYPE_PATTERN})(\([a-z0-9][a-z0-9._/-]*\))?!?: \S(?:.*\S)?")
BRANCH = re.compile(rf"(?:codex/)?({TYPE_PATTERN})/[a-z0-9]+(?:-[a-z0-9]+)*")
SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    ).stdout.strip()


def check_commit(message: str) -> None:
    subject = message.splitlines()[0] if message.splitlines() else ""
    if len(subject) > 100 or not SUBJECT.fullmatch(subject):
        raise ValueError(
            "Use a conventional subject of at most 100 characters: "
            "type(scope): description (scope and ! are optional). "
            f"Allowed types: {', '.join(TYPES)}. Got: {subject!r}"
        )


def check_branch(
    branch: str,
    *,
    default_branch: str = "main",
    allow_protected: bool = False,
    allow_dependabot: bool = False,
) -> None:
    if branch in {default_branch, "main", "master"}:
        if allow_protected:
            return
        raise ValueError(
            "Create a feature branch before committing; do not work on the default branch."
        )
    if allow_dependabot and re.fullmatch(r"dependabot/[a-zA-Z0-9._/-]+", branch):
        return
    if not BRANCH.fullmatch(branch):
        raise ValueError(
            "Use type/kebab-description or codex/type/kebab-description; "
            f"for example feat/add-folds or codex/fix/submission-order. Got: {branch!r}"
        )


def check_range(base: str, head: str, *, bootstrap: bool = False) -> None:
    if not SHA.fullmatch(head) or not SHA.fullmatch(base):
        raise ValueError("GitHub event contains an invalid commit SHA")
    if set(base) == {"0"}:
        commits = [head]  # New branch: do not retroactively lint inherited history.
    else:
        commits = git("rev-list", "--no-merges", f"{base}..{head}").splitlines()
    for commit in commits:
        parents = git("rev-list", "--parents", "-n", "1", commit).split()
        if len(parents) > 2 or bootstrap and len(parents) == 1:
            continue  # Generated merge commits and the initial template bootstrap.
        try:
            check_commit(git("show", "-s", "--format=%B", commit))
        except ValueError as error:
            raise ValueError(f"{commit[:12]}: {error}") from error


def check_event(path: Path, event_name: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    default = payload.get("repository", {}).get("default_branch", "main")
    if event_name == "pull_request":
        pr = payload["pull_request"]
        check_branch(
            pr["head"]["ref"],
            default_branch=default,
            allow_dependabot=pr["user"]["login"] == "dependabot[bot]",
        )
        check_commit(pr["title"])
        check_range(pr["base"]["sha"], pr["head"]["sha"])
    elif event_name == "push":
        ref = payload["ref"]
        if not ref.startswith("refs/heads/") or payload.get("deleted"):
            return
        check_branch(
            ref.removeprefix("refs/heads/"),
            default_branch=default,
            allow_protected=True,
            allow_dependabot=payload.get("sender", {}).get("login") == "dependabot[bot]",
        )
        check_range(payload["before"], payload["after"], bootstrap=True)
    else:
        raise ValueError(f"Unsupported GitHub event: {event_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    commit = subparsers.add_parser("commit")
    commit.add_argument("--file", type=Path, required=True)
    subparsers.add_parser("branch")
    subparsers.add_parser("github-event")
    args = parser.parse_args()
    try:
        if args.command == "commit":
            # Git's generated merge message is exempt only during a real merge.
            if not Path(git("rev-parse", "--git-path", "MERGE_HEAD")).exists():
                check_commit(args.file.read_text(encoding="utf-8-sig"))
        elif args.command == "branch":
            check_branch(git("branch", "--show-current"))
        else:
            check_event(Path(os.environ["GITHUB_EVENT_PATH"]), os.environ["GITHUB_EVENT_NAME"])
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"Git convention check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
