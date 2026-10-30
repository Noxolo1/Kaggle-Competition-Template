import json
import subprocess
import sys

import pytest

from scripts import check_git_conventions as policy


@pytest.mark.parametrize(
    "subject",
    [
        "feat: add folds",
        "fix(submission): preserve order",
        "feat(data)!: require ID",
        "docs: explain setup",
        "chore(deps): bump numpy",
        "revert: undo feature selection",
    ],
)
def test_valid_subjects(subject):
    policy.check_commit(subject + "\n\nDetails and optional footers.\n")


@pytest.mark.parametrize(
    "subject",
    [
        "",
        "Update README",
        "Feat: add folds",
        "feat:add folds",
        "fix(): repair",
        "fix: ",
        "feat: trailing space ",
        "feat: " + "x" * 100,
        "fixup! feat: add folds",
    ],
)
def test_invalid_subjects(subject):
    with pytest.raises(ValueError):
        policy.check_commit(subject)


@pytest.mark.parametrize(
    "branch",
    [
        "feat/add-folds",
        "fix/123-align-ids",
        "codex/chore/update-dependencies",
    ],
)
def test_valid_branches(branch):
    policy.check_branch(branch)


@pytest.mark.parametrize(
    "branch",
    [
        "",
        "main",
        "master",
        "feature/folds",
        "feat/Bad-name",
        "feat/a--b",
        "codex/random",
    ],
)
def test_invalid_working_branches(branch):
    with pytest.raises(ValueError):
        policy.check_branch(branch)


def test_default_and_dependabot_branches_are_contextual():
    policy.check_branch("main", allow_protected=True)
    policy.check_branch("develop", default_branch="develop", allow_protected=True)
    branch = "dependabot/uv/numpy-2.5.4"
    policy.check_branch(branch, allow_dependabot=True)
    with pytest.raises(ValueError):
        policy.check_branch(branch)


def test_pr_event_checks_branch_title_and_commit_range(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(policy, "check_range", lambda *args, **kwargs: captured.append(args))
    path = tmp_path / "event.json"
    payload = {
        "repository": {"default_branch": "main"},
        "pull_request": {
            "head": {"ref": "feat/add-folds", "sha": "a" * 40},
            "base": {"sha": "b" * 40},
            "title": "feat(validation): add grouped folds",
            "user": {"login": "contributor"},
        },
    }
    path.write_text(json.dumps(payload))
    policy.check_event(path, "pull_request")
    assert captured == [("b" * 40, "a" * 40)]
    payload["pull_request"]["title"] = "arbitrary title"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        policy.check_event(path, "pull_request")


def test_push_tags_and_deleted_branches_are_ignored(tmp_path):
    path = tmp_path / "event.json"
    for payload in ({"ref": "refs/tags/v1"}, {"ref": "refs/heads/old", "deleted": True}):
        path.write_text(json.dumps(payload))
        policy.check_event(path, "push")


def test_invalid_sha_never_reaches_git(monkeypatch):
    def unexpected(*args):
        raise AssertionError("Untrusted revision reached git")

    monkeypatch.setattr(policy, "git", unexpected)
    with pytest.raises(ValueError, match="invalid commit SHA"):
        policy.check_range("--all", "a" * 40)


def test_new_branch_does_not_lint_inherited_history(monkeypatch):
    calls = []

    def fake_git(*args):
        calls.append(args)
        if args[0] == "rev-list":
            return "a" * 40 + " " + "b" * 40
        return "feat: introduce new workflow"

    monkeypatch.setattr(policy, "git", fake_git)
    policy.check_range("0" * 40, "a" * 40)
    assert not any("--no-merges" in call for call in calls)


def test_cli_message_validation(tmp_path):
    message = tmp_path / "message.txt"
    message.write_text("invalid commit\n")
    result = subprocess.run(
        [sys.executable, "scripts/check_git_conventions.py", "commit", "--file", str(message)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "conventional subject" in result.stderr


def test_real_git_range_only_checks_new_history(tmp_path, monkeypatch):
    def command(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()

    command("init", "-b", "main")
    command("config", "user.name", "Template Test")
    command("config", "user.email", "template-test@example.invalid")
    command("config", "commit.gpgsign", "false")
    command("config", "core.hooksPath", str(tmp_path / "no-hooks"))
    command("commit", "--allow-empty", "-m", "Legacy bootstrap commit")
    root = command("rev-parse", "HEAD")
    command("commit", "--allow-empty", "-m", "feat: add workflow")
    valid = command("rev-parse", "HEAD")
    command("commit", "--allow-empty", "-m", "Unconventional change")
    invalid = command("rev-parse", "HEAD")
    monkeypatch.chdir(tmp_path)
    policy.check_range(root, valid)
    policy.check_range("0" * 40, valid)
    policy.check_range("0" * 40, root, bootstrap=True)
    with pytest.raises(ValueError, match="conventional subject"):
        policy.check_range(valid, invalid)
