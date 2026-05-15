Thanks for your interest in contributing to **tlmtc**.

Contributions are welcome, including:

- asking questions about usage or design
- reporting bugs
- suggesting features or improvements
- improving documentation, examples, or error messages
- adding or improving tests
- contributing code

## Before you start

### Issue first, then PR

Unless your change is trivial, such as a typo correction or minor documentation improvement, please open an issue before submitting a pull request.

Opening an issue first helps avoid wasted work and keeps design decisions visible.

### AI-assisted contributions

AI-assisted work is welcome when there is clear human ownership.

You may use AI tools to help draft, revise, explore, or implement contributions, but you are responsible for the final result. Before submitting, you must review every change, verify behavior against the codebase, run the relevant checks, and make sure the contribution matches the project style and intent.

Unreviewed AI-agent or bulk-generated contributions are not accepted. Please do not submit issues or pull requests that were generated autonomously, submitted at scale, or opened without careful human review and testing.

Pull requests that appear automated, low-effort, spam-like, or submitted without clear human ownership may be closed without review, regardless of how they were produced.

### Issues

When reporting a bug, please include:

- what you expected to happen
- what actually happened
- a minimal example, if possible
- the relevant package version, Python version, and operating system
- the relevant error message or traceback, if applicable

When suggesting a feature, please describe:

- the use case
- why the current behavior is insufficient
- the expected behavior or API, if you have a concrete proposal

### Pull requests

All changes must be submitted through pull requests.

Each pull request should represent one substantively coherent change. Large, mixed-scope pull requests may be closed or split before review.

A good pull request should include:

- a short summary of what changed and why the change is needed
- a scoped list of key changes
- tests, or an explanation of why tests were not added
- documentation updates when user-facing behavior changes
- notes on how the change was tested

## Branches and commits

The `main` branch is protected. Direct commits to `main` by non-admins are not allowed.

Work should happen on short-lived, task-specific branches scoped to a single pull request.

Use simple branch names such as:

- `feature/short-description`
- `fix/short-description`
- `refactor/short-description`
- `docs/short-description`
- `tests/short-description`
- `ci/short-description`
- `chore/short-description`

Commits should follow a simple, consistent style:

- `feat: add new functionality`
- `fix: correct a bug`
- `refactor: restructure code without changing behavior`
- `docs: update documentation`
- `tests: add or adjust tests`
- `ci: update CI configuration`
- `chore: update maintenance files`

## Tests and checks

Before opening a pull request, please run the relevant local checks: `ruff`, `mypy`, and `pytest`.

The project uses Python 3.12 or newer. We recommend using `uv` for local development. Install the project with the optional dependencies when working on training, prediction, or integration behavior.

When your change affects end-to-end training or prediction behavior, please also run the integration tests.

Not every pull request needs to run every expensive check locally, but pull requests should not knowingly rely on GitHub Actions to catch obvious failures.

## Review and merge policy

Pull requests require passing CI checks and maintainer approval before merging.

Community review is welcome and may inform maintainer decisions, but final merge decisions are made by maintainers.

Pull requests are merged using squash merge.