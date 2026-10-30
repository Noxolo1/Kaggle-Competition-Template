# Working in a competition repository

Use a short-lived branch for each change, then open a pull request back to the default
branch. Keep experiments small enough to compare against a recorded baseline.

## Branches

Use `<type>/<kebab-description>`, optionally prefixed with `codex/` for agent work:

```sh
git switch -c feat/add-group-validation
git switch -c fix/submission-id-order
git switch -c codex/chore/update-dependencies
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
`build`, `ci`, `chore`, and `revert`. Descriptions use lowercase letters, digits,
and single hyphens. Keep branch names about the change; put experiment parameters
and scores in run records. `main` and `master` are reserved integration branches.
Dependabot's generated branches are allowed in CI for Dependabot-authored changes.

## Commits and pull requests

Follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/):

```text
feat(validation): add group holdouts
fix(submission): preserve sample ID order
docs: document Kaggle source mounts
refactor(data)!: require explicit ID columns

BREAKING CHANGE: callers must provide the competition ID column.
```

The subject is `type(scope): description`; the scope is optional. Use `!` before
the colon for a breaking change and explain the migration in the body. Repository
policy limits subjects to 100 characters and scopes to lowercase letters, digits,
dots, underscores, slashes, and hyphens. Bodies and footers may contain longer detail.
Separate the body from the subject with a blank line. Use `revert: ...` for reverts;
reword fixup/squash subjects before merging.

PR titles follow the same subject format so squash merges retain useful history.
Describe the problem, resulting behavior, validation, and any material limits.
Review staged files for data, credentials, model weights, and notebook outputs.

## Enable local checks once per clone

After installing uv and creating the project environment:

```sh
uv sync --locked --extra notebook --extra dev
git config --local core.hooksPath .githooks
```

This sets only this clone's hook path. Existing custom hook paths need to be
integrated manually; do not overwrite a hook setup you rely on. Git does not carry
local configuration into clones or newly generated competition repositories;
repeat setup there. Hooks use the existing .venv environment on Linux/macOS or Windows.
They validate the working branch before a commit and the subject before accepting
its message. Genuine merge commits retain Git's generated messages.
Detached HEAD commits are rejected; switch to a named working branch first.

GitHub CI independently validates branch names, PR titles, and new non-merge commit
subjects. PR metadata is read as JSON, never interpolated into shell commands.
PR title edits retrigger validation. Existing history is not rewritten: the first
push of a new branch checks its tip; later pushes check the pushed commit range.
A generated repository's root bootstrap commit is exempt in push checks.

Local checks can also be run explicitly:

```sh
uv run --locked python scripts/check_git_conventions.py branch
uv run --locked python scripts/check_git_conventions.py commit --file message.txt
```

## Before merging

```sh
uv run --locked --extra dev ruff check .
uv run --locked --extra dev ruff format --check .
uv run --locked --extra dev pytest -W error
```

Prefer squash merges with a conventional PR title, or retain a focused conventional
commit series. Configure GitHub branch protection/rulesets to require PRs and the
`conventions` and template check jobs on the default branch if you want server-side
enforcement. Repository files alone cannot turn on hosting-level branch protection.
