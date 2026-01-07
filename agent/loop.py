import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

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

def git_diff() -> str:
    r = run(["git", "diff"])
    return r.stdout

def git_status_porcelain() -> str:
    r = run(["git", "status", "--porcelain"])
    return r.stdout

def apply_patch(patch_text: str) -> None:
    # Apply via stdin
    p = subprocess.run(
        ["git", "apply", "--whitespace=fix", "-"],
        cwd=ROOT,
        input=patch_text,
        text=True,
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"git apply failed:\n{p.stderr}")

def write_log(name: str, content: str) -> Path:
    path = LOG_DIR / name
    path.write_text(content, encoding="utf-8")
    return path

PATCH_BLOCK_RE = re.compile(r"(?s)PATCH:\s*(.*)$")

def extract_patch(llm_text: str) -> str:
    text = (llm_text or "").strip()
    m = PATCH_BLOCK_RE.search(text)
    if not m:
        # Nếu LLM không trả PATCH, coi như không patch
        return ""
    return m.group(1).strip()

def quality_gate() -> tuple[bool, str]:
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
        logs.append(f"## {name}\n$ {' '.join(cmd)}\nexit={rr.code}\nSTDOUT:\n{rr.stdout}\nSTDERR:\n{rr.stderr}\n")
        if not rr.ok:
            return False, "\n".join(logs)
    return True, "\n".join(logs)

def git_commit(message: str) -> None:
    run(["git", "add", "-A"])
    rr = run(["git", "commit", "-m", message])
    if not rr.ok:
        raise RuntimeError(f"git commit failed:\n{rr.stdout}\n{rr.stderr}")

# ====== LLM Client (mock trước, sau thay bằng API thật) ======
def call_llm(prompt: str) -> str:
    """
    Mock LLM: hiện tại trả về patch trống.
    Bạn sẽ thay hàm này bằng gọi OpenAI/LLM thực.
    """
    return "PATCH:\n"  # no-op patch


def build_prompt(goal: str, last_logs: str, current_diff: str) -> str:
    return f"""
You are an automated coding agent for a Python repo.

GOAL:
{goal}

RULES:
- Output MUST end with a unified diff patch in a block that starts with 'PATCH:'.
- Only include the diff, nothing else inside PATCH block.
- Do not claim tests pass unless tool logs show pass.
- Keep changes minimal.
- Prefer adding tests for fixes/features.

CURRENT_GIT_DIFF:
{current_diff}

LAST_TOOL_LOGS:
{last_logs}
""".strip()

def main() -> None:
    goal_path = ROOT / "goal.md"
    if not goal_path.exists():
        goal_path.write_text("Implement something small and add tests.\n", encoding="utf-8")

    goal = goal_path.read_text(encoding="utf-8").strip()

    # safety: work on a branch
    run(["git", "checkout", "-b", "auto-loop"],)

    last_logs = "(none yet)"
    max_iters = 10

    for i in range(1, max_iters + 1):
        print(f"\n=== ITERATION {i}/{max_iters} ===")

        current_diff = git_diff()
        prompt = build_prompt(goal, last_logs, current_diff)
        write_log(f"iter_{i:02d}_prompt.txt", prompt)

        llm_text = call_llm(prompt)
        write_log(f"iter_{i:02d}_llm.txt", llm_text)

        patch = extract_patch(llm_text)
        if patch.strip():
            apply_patch(patch)
        else:
            print("No patch produced by LLM.")

        ok, logs = quality_gate()
        last_logs = logs
        write_log(f"iter_{i:02d}_tools.txt", logs)

        if ok:
            if git_status_porcelain().strip():
                git_commit(f"auto: iteration {i} pass gates")
                print("✅ Gates passed. Committed.")
            else:
                print("✅ Gates passed. Nothing to commit.")
            break
        else:
            print("❌ Gates failed. Will iterate with logs.")

    else:
        print("Reached max iterations without passing gates.")

if __name__ == "__main__":
    main()
