"""Every module must import, and every command must parse.

Written after `cli.py` was committed with a syntax error that 820 passing tests
did not catch. Nothing in the suite imported the CLI, so a broken entry point
looked exactly like a healthy one -- and the next scheduled pipeline run would
have failed on its first step.

The cause was a scripted edit whose `\\n` became a literal newline inside a
string, which is a recurring failure in this repository. The lesson that keeps
not sticking is "use the editor, not shell string surgery"; this file is the
cheaper backstop for when it does not stick again.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


def python_files():
    return sorted(SRC.rglob("*.py"))


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_every_source_file_parses(path):
    """Catches string corruption anywhere, including modules no test imports."""
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_the_cli_module_imports():
    """The entry point specifically. It is the one module the rest of the suite
    never touches, and the only one whose breakage stops the pipeline dead."""
    from tradezbotz import cli

    assert callable(cli.main)


def test_every_registered_command_builds_a_parser():
    """A command with a broken argument definition fails at startup, not at use,
    so the pipeline would break on whichever step invoked it first."""
    from tradezbotz.cli import build_parser

    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    subparsers = next((a.choices for a in actions if isinstance(a.choices, dict)), {})

    assert len(subparsers) >= 15, f"only {len(subparsers)} commands registered"
    for name, sub in subparsers.items():
        assert sub.format_help(), f"{name} cannot render its help"


@pytest.mark.parametrize("command", [
    "ingest-edgar", "ingest-bulk", "ingest-sentiment", "enqueue-symbols",
    "backfill", "crosscheck", "backfill-intraday", "ingest-filings", "watch",
    "ingest-holdings", "ingest-assets", "ingest-macro", "runlog",
    "repair-symbols", "ingest-fundamentals", "measure", "status",
])
def test_command_parses_its_own_help(command):
    """Every command the workflow can invoke, by name. A typo in the workflow
    is a separate problem; this catches a command that cannot start at all."""
    from tradezbotz.cli import build_parser

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command, "--help"])
    assert exc.value.code == 0


def test_the_workflow_only_invokes_commands_that_exist():
    """The workflow is not Python and nothing type-checks it, so a renamed
    command would fail in CI rather than here."""
    import re

    from tradezbotz.cli import build_parser

    workflow = (SRC.parent / ".github" / "workflows" / "pipeline.yml").read_text(
        encoding="utf-8")
    invoked = set(re.findall(r"python -m tradezbotz ([a-z-]+)", workflow))
    parser = build_parser()
    known = next(a.choices for a in parser._actions
                 if isinstance(getattr(a, "choices", None), dict))

    unknown = invoked - set(known)
    assert not unknown, f"workflow invokes commands that do not exist: {unknown}"
