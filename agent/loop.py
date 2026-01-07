import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = Path(os.environ.get("AUTO_LOOP_LOG_DIR", str(ROOT.parent / "auto_dev_loop_logs")))

LOG_DIR.mkdir(exist_ok=True)

# Anything under these prefixes must never be committed by the agent
FORBIDDEN_PREFIXES = (
    ".venv/",
    "logs/",
)

# If you want to prevent the agent from touching certain files at all,
# you can add checks against these exact paths as well.
FORBIDDEN_EXACT = (
    ".env",
)

DEFAULT_BRANCH = "auto-loop"

# --- Helpers ---------------------------------------------------------------


@dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str
    code: int


def run(cmd: list[str], *, env: Optional[dict[str, str]] = None) -> RunResult:
    p = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env or os.environ.copy(),
        text=True,
        capture_output=True,
    )
    return RunResult(ok=(p.returncode == 0), stdout=p.stdout, stderr=p.stderr, code=p.returncode)


def write_log(name: str, content: str) -> Path:
    path = LOG_DIR / name
    path.write_text(content, encoding="utf-8")
    return path


def git_status_porcelain() -> str:
    return run(["git", "status", "--porcelain"]).stdout


def git_has_changes() -> bool:
    return bool(git_status_porcelain().strip())


def git_diff() -> str:
    return run(["git", "diff"]).stdout


def git_diff_cached() -> str:
    return run(["git", "diff", "--cached"]).stdout


def git_changed_files_cached() -> list[str]:
    rr = run(["git", "diff", "--name-only", "--cached"])
    return [x.strip() for x in rr.stdout.splitlines() if x.strip()]


def stage_all_safe() -> None:
    """
    Stage changes, but prevent forbidden paths from being staged/committed.
    """
    run(["git", "add", "-A"])

    changed = git_changed_files_cached()
    forbidden = []

    for p in changed:
        if p in FORBIDDEN_EXACT:
            forbidden.append(p)
            continue
        if p.startswith(FORBIDDEN_PREFIXES):
            forbidden.append(p)

    if forbidden:
        # unstage forbidden paths
        run(["git", "reset", "--"] + forbidden)
        raise RuntimeError(f"Forbidden paths staged: {forbidden}")


def apply_patch(patch_text: str) -> None:
    """
    Apply unified diff patch via git apply.
    """
    p = subprocess.run(
        ["git", "apply", "--whitespace=fix", "-"],
        cwd=ROOT,
        input=patch_text,
        text=True,
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"git apply failed:\n{p.stderr}")


def git_commit(message: str) -> None:
    rr = run(["git", "commit", "-m", message])
    if not rr.ok:
        raise RuntimeError(f"git commit failed:\n{rr.stdout}\n{rr.stderr}")


def ensure_branch(branch: str) -> None:
    rr = run(["git", "checkout", branch])
    if rr.ok:
        return
    rr2 = run(["git", "checkout", "-b", branch])
    if not rr2.ok:
        raise RuntimeError(f"Failed to checkout/create branch '{branch}':\n{rr2.stdout}\n{rr2.stderr}")


# --- Patch parsing ---------------------------------------------------------

PATCH_BLOCK_RE = re.compile(r"(?s)PATCH:\s*(.*)$")  # allow empty patch


def extract_patch(llm_text: str) -> str:
    text = (llm_text or "").strip()
    m = PATCH_BLOCK_RE.search(text)
    if not m:
        return ""
    return m.group(1).strip()


# --- Quality gate ----------------------------------------------------------


def quality_gate() -> tuple[bool, str]:
    """
    Run ruff, mypy, pytest. Return (ok, logs).
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    steps = [
        (["ruff", "check", "."], "ruff"),
        (["mypy", "src"], "mypy"),
        (["pytest"], "pytest"),
    ]

    logs = []
    for cmd, name in steps:
        rr = run(cmd, env=env)
        logs.append(
            f"## {name}\n"
            f"$ {' '.join(cmd)}\n"
            f"exit={rr.code}\n"
            f"STDOUT:\n{rr.stdout}\n"
            f"STDERR:\n{rr.stderr}\n"
        )
        if not rr.ok:
            return False, "\n".join(logs)

    return True, "\n".join(logs)


# --- LLM client ------------------------------------------------------------


def call_llm(prompt: str) -> str:
    """
    Choose backend via env:
      - AUTO_LOOP_LLM=mock  (default)
      - AUTO_LOOP_LLM=openai

    For OpenAI:
      - pip install openai
      - export OPENAI_API_KEY="..."
      - export AUTO_LOOP_MODEL="gpt-4.1-mini" (optional)
    """
    backend = os.environ.get("AUTO_LOOP_LLM", "mock").lower()

    if backend == "mock":
        return "PATCH:\n"

    if backend == "openai":
        from openai import OpenAI

        client = OpenAI()
        model = os.environ.get("AUTO_LOOP_MODEL", "gpt-4.1-mini")
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=0.2,
        )
        return resp.output_text

    raise RuntimeError(f"Unknown AUTO_LOOP_LLM backend: {backend}")


# --- Prompts ---------------------------------------------------------------


def build_coder_prompt(goal: str, last_logs: str, current_diff: str) -> str:
    return f"""
You are an automated coding agent for a Python repo.

GOAL:
{goal}

RULES:
- Output MUST end with a unified diff patch in a block that starts with 'PATCH:'.
- Only include the diff inside PATCH (no markdown fences, no extra text).
- Keep changes minimal. Prefer one focused change per iteration.
- Add/adjust tests when you change behavior.
- Never modify or include files under: {FORBIDDEN_PREFIXES} or exact: {FORBIDDEN_EXACT}.
- Do not claim tests pass unless tool logs show pass.

CURRENT_GIT_DIFF (working tree):
{current_diff}

LAST_TOOL_LOGS:
{last_logs}
""".strip()


def build_reviewer_prompt(diff_text: str) -> str:
    return f"""
You are a strict code reviewer for a Python repository.

Review the DIFF and improve correctness, clarity, edge cases, and tests.

RULES:
- Output MUST end with a unified diff patch in a block that starts with 'PATCH:'.
- Only include the diff inside PATCH (no markdown fences, no extra text).
- Keep changes minimal and targeted.
- Never modify or include files under: {FORBIDDEN_PREFIXES} or exact: {FORBIDDEN_EXACT}.
- Do not remove tests unless replaced by better tests.

DIFF:
{diff_text}
""".strip()


# --- Main loop -------------------------------------------------------------


def main() -> None:
    goal_path = ROOT / "goal.md"
    if not goal_path.exists():
        goal_path.write_text(
            "Add a small feature with tests. Example:\n"
            "- Add divide(a,b) raising ValueError on b==0\n"
            "- Add tests\n",
            encoding="utf-8",
        )
        print("Created goal.md. Edit it and re-run.")
        return

    goal = goal_path.read_text(encoding="utf-8").strip()

    ensure_branch(DEFAULT_BRANCH)

    last_logs = "(none yet)"
    max_iters = int(os.environ.get("AUTO_LOOP_MAX_ITERS", "10"))

    auto_push = os.environ.get("AUTO_LOOP_AUTO_PUSH", "0") in ("1", "true", "yes")

    for i in range(1, max_iters + 1):
        print(f"\n=== ITERATION {i}/{max_iters} ===")

        current_diff = git_diff()
        coder_prompt = build_coder_prompt(goal, last_logs, current_diff)
        write_log(f"iter_{i:02d}_coder_prompt.txt", coder_prompt)

        coder_text = call_llm(coder_prompt)
        write_log(f"iter_{i:02d}_coder_llm.txt", coder_text)

        coder_patch = extract_patch(coder_text)
        if coder_patch.strip():
            apply_patch(coder_patch)
            print("Applied coder patch.")
        else:
            print("No patch produced by coder.")

        # Reviewer pass (only if there is a diff)
        diff_after = git_diff()
        if diff_after.strip():
            reviewer_prompt = build_reviewer_prompt(diff_after)
            write_log(f"iter_{i:02d}_reviewer_prompt.txt", reviewer_prompt)

            reviewer_text = call_llm(reviewer_prompt)
            write_log(f"iter_{i:02d}_reviewer_llm.txt", reviewer_text)

            reviewer_patch = extract_patch(reviewer_text)
            if reviewer_patch.strip():
                apply_patch(reviewer_patch)
                print("Applied reviewer patch.")
            else:
                print("No patch produced by reviewer.")

        ok, logs = quality_gate()
        last_logs = logs
        write_log(f"iter_{i:02d}_tools.txt", logs)

        if ok:
            if git_has_changes():
                stage_all_safe()
                git_commit(f"auto: iteration {i} pass gates")
                print("✅ Gates passed. Committed (safe).")
                if auto_push:
                    pr = run(["git", "push"])
                    if pr.ok:
                        print("⬆️ Pushed to remote.")
                    else:
                        print(f"⚠️ Push failed:\n{pr.stderr}")
            else:
                print("✅ Gates passed. Nothing to commit.")
            break

        print("❌ Gates failed. Iterating with tool logs...")

    else:
        print("Reached max iterations without passing gates.")


if __name__ == "__main__":
    main()
