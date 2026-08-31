"""Scenarios for the subagent path, against the real `claude` binary.

Its own file rather than more cases in `test_plugin_install.py`, because what
is being claimed is different. That file asks whether an install produces a
hook that fires on a prompt; this one asks whether the harness dispatches a
`PreToolUse` hook on the Agent tool, honours the `updatedInput` it writes, and
hands the rewritten brief to the subagent — three claims about a harness this
repo does not own, none of which any assertion over memkit's own files can
settle.

Two tiers, and the split is forced rather than chosen. A `PreToolUse` hook
fires on a TOOL CALL, and a tool call requires a model to decide to make one,
so there is no harness-tier version of the delivery claim: with no model there
is no spawn, and with no spawn the hook never runs. What the harness tier can
settle is everything up to the dispatch — that the installed payload registers
the event, on the right tool, with the timeout the module declares — and that
is where the cheap half lives.

  HARNESS tier — install through the real binary and read the registration
    back out of what got installed. Runs in CI and fails rather than skips.
  LIVE tier — the delivery itself, by the sentinel technique the entry
    experiments used: spawn a subagent whose whole task is to report the text
    it was given, and look for the pointer block in what comes back. Needs the
    author's local proxy; `MEMKIT_RIG_LIVE=1` opts in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memkit import memory_prompt_recall as hook
from rig import (
    REPO,
    Profile,
    fixture_config,
    harness_tier_reason,
    live_tier_reason,
    soak_records,
    stage_plugin,
)

harness_tier = pytest.mark.skipif(
    harness_tier_reason() is not None, reason=harness_tier_reason() or ""
)
live_tier = pytest.mark.skipif(
    live_tier_reason() is not None, reason=live_tier_reason() or ""
)

# The memory a brief about a gearbox rebuild has to reach, and a brief long
# enough that the prompt path would refuse it outright. Both come from the
# committed fixtures, so this scenario and the eval slice that gates the
# thresholds are about the same pair.
BRIEF = (
    REPO / "tests" / "fixtures" / "long-briefs" / "served" / "gearbox-acceptance.md"
).read_text(encoding="utf-8").strip()
EXPECTED = "sprocket_alignment.md"


@pytest.fixture(scope="module")
def staged(tmp_path_factory) -> Path:
    return stage_plugin(tmp_path_factory.mktemp("staged") / "memkit", REPO)


@pytest.fixture
def profile(tmp_path) -> Profile:
    return Profile(tmp_path / "rig")


# --- HARNESS tier -------------------------------------------------------------


@harness_tier
def test_the_installed_payload_registers_the_event_on_the_subagent_tool(
    profile: Profile, staged: Path
) -> None:
    """Read back out of what the harness INSTALLED, not out of the working
    tree.

    The distinction is the whole reason this is a rig scenario: the repo's own
    suite asserts what `hooks/hooks.json` says, which is a claim about a file,
    and an install that dropped or rewrote the entry would leave that claim
    perfectly true. What an adopter runs is the installed copy.
    """
    profile.marketplace_add(staged)
    config = fixture_config(profile)
    profile.install("memkit@memkit", config={"memkitConfig": str(config)})

    installed = next(
        (Path(profile.config_dir) / "plugins" / "cache").rglob("hooks/hooks.json")
    )
    registration = json.loads(installed.read_text())["hooks"]
    assert set(registration) == {"UserPromptSubmit", "PreToolUse"}, sorted(registration)
    groups = registration["PreToolUse"]
    assert len(groups) == 1, groups
    assert groups[0]["matcher"] == hook.TASK_TOOL
    handler = groups[0]["hooks"][0]
    assert handler["timeout"] == hook.TASK_HARNESS_TIMEOUT
    assert handler["args"] == []
    assert handler["command"].endswith("/bin/memkit-hook")


@harness_tier
def test_the_harness_reports_both_hooks_after_the_install(
    profile: Profile, staged: Path
) -> None:
    """`plugin details` is what an adopter reads to find out whether anything
    was registered, and it counts hooks rather than naming them. One is what a
    build with only the prompt path reports, and it is also what a build whose
    second entry the harness silently rejected reports — the two states this
    number has to tell apart."""
    profile.marketplace_add(staged)
    config = fixture_config(profile)
    profile.install("memkit@memkit", config={"memkitConfig": str(config)})
    out = profile.details("memkit@memkit")
    assert out.returncode == 0, out.stderr
    assert "Hooks (2)" in out.stdout, out.stdout


# --- LIVE tier ----------------------------------------------------------------


@live_tier
def test_a_spawned_subagent_receives_the_pointer_block(
    profile: Profile, staged: Path
) -> None:
    """The delivery claim, end to end, by the sentinel technique.

    The subagent's whole task is to report the text it was handed, so what
    comes back is evidence about what the harness delivered rather than about
    what the subagent decided to do with it. That distinction is why the task
    is worded this way: asking a subagent to ACT on a retrieved pointer would
    measure the model's compliance with text the frame explicitly labels as
    data, which is the opposite of what this path wants to be true.

    Asserted twice, and the two fail differently. The soak record is what the
    hook believes it wrote; the answer is what reached the subagent. A hook
    that emitted into a closed pipe satisfies neither, and one the harness
    never dispatched leaves no `task:` record at all.
    """
    profile.marketplace_add(staged)
    config = fixture_config(profile)
    profile.install("memkit@memkit", config={"memkitConfig": str(config)})
    project = profile.project("work")

    out = profile.claude(
        "-p",
        "Use the Agent tool to spawn a general-purpose subagent. Its entire "
        "task is this brief, passed through verbatim as the tool's `prompt`, "
        "with nothing added or removed:\n\n"
        f"{BRIEF}\n\n"
        "Instruct the subagent, as the last line of that prompt, to do no "
        "work and instead reply with any file paths that appear in the "
        "instructions it received, or the word NONE. Report its reply to me "
        "verbatim.",
        "--output-format",
        "json",
        "--allowedTools",
        "Agent",
        cwd=str(project),
        timeout=600,
    )
    answer = json.loads(out.stdout)
    assert answer["is_error"] is False, answer

    records = soak_records(profile)
    injected = [r for r in records if r["outcome"] == "task:injected"]
    assert injected, [r["outcome"] for r in records]
    assert EXPECTED in injected[-1]["injected"], injected[-1]
    # And it arrived: the subagent read the path back out of its own
    # instructions, which is the half no artifact of memkit's can show.
    assert EXPECTED in answer["result"], answer["result"]


@live_tier
def test_the_brief_reaches_the_subagent_unaltered(
    profile: Profile, staged: Path
) -> None:
    """The corruption half, which the delivery case above cannot see: a hook
    that replaced the brief instead of appending to it would still produce a
    `task:injected` record and still put the pointer in front of the subagent.

    What is checked is the parent's own text, which the soak log never holds —
    the log carries a digest of the brief and nothing else — so this asks the
    subagent to read back two things that can only have arrived together: a
    sentence out of the MIDDLE of the brief, and the pointer the hook appended
    after its END. A replacement loses the first, a prefix-only rewrite loses
    it too, and a dropped emission loses the second.

    Both halves are an ECHO of delivered text, never the brief's own task. The
    brief asks for an acceptance procedure and a live subagent is free to
    decline to write one — it did, on this scenario's first run — and what it
    decides to do with a brief is not what this tier exists to prove. The
    subagent is instructed to do no work for exactly that reason, and the
    assertions are over what it was handed.
    """
    profile.marketplace_add(staged)
    config = fixture_config(profile)
    profile.install("memkit@memkit", config={"memkitConfig": str(config)})
    project = profile.project("work")

    out = profile.claude(
        "-p",
        "Use the Agent tool to spawn a general-purpose subagent whose entire "
        "task is this brief, passed through verbatim as the tool's `prompt`:\n\n"
        f"{BRIEF}\n\n"
        "Instruct the subagent, as the last line of that prompt, to do no "
        "work and instead reply with two things: the sentence in its "
        "instructions that mentions a thermal state, quoted; and every file "
        "path that appears anywhere in its instructions, or the word NONE. "
        "Report its reply verbatim.",
        "--output-format",
        "json",
        "--allowedTools",
        "Agent",
        cwd=str(project),
        timeout=600,
    )
    result = json.loads(out.stdout)
    assert result["is_error"] is False, result
    answer = result.get("result", "")
    # A phrase from the middle of the brief, well past where any prefix-only
    # rewrite would have stopped.
    assert "thermal state" in answer.lower(), answer
    # And the appended block, out of the same reply: the brief survived to its
    # middle AND the pointer arrived after its end, which is what "appended,
    # not replaced" means from the subagent's side. Two independent phrases
    # from two ends of one delivery is also the anti-vacuity check — an empty
    # or refusing answer carries neither.
    assert EXPECTED in answer, answer
    # The hook's own account of the same delivery, which fails differently: no
    # record at all is a dispatch that never happened, and a record naming
    # nothing is an emission the harness rejected.
    records = soak_records(profile)
    injected = [r for r in records if r["outcome"] == "task:injected"]
    assert injected, [r["outcome"] for r in records]
    assert EXPECTED in injected[-1]["injected"], injected[-1]
