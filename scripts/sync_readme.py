from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
EXPERIMENTS_DIR = ROOT / "experiments"
SRC_DIR = ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from rl_evo_lab.utils.config import EDERConfig, ENV_PRESETS  # noqa: E402


EXPERIMENTS_BEGIN = "<!-- BEGIN AUTO:EXPERIMENTS -->"
EXPERIMENTS_END = "<!-- END AUTO:EXPERIMENTS -->"
CONFIG_BEGIN = "<!-- BEGIN AUTO:CONFIG -->"
CONFIG_END = "<!-- END AUTO:CONFIG -->"


CONFIG_NOTES: dict[str, str] = {
    "es_sigma": "ES noise std dev. Too small = no diversity; too large = divergence",
    "es_n_workers": "ES population size before env presets override it",
    "beta": "Intrinsic reward weight",
    "use_novelty": "False = ES+DQN baseline, no IDN",
    "use_es": "False = pure DQN with epsilon-greedy",
    "novelty_warmup_episodes": "Episodes before novelty activates; IDN trains silently",
    "solved_reward": "Reward at which convergence decay begins",
    "novelty_solve_decay": "Decays beta, sigma, and worker count as learner converges",
}

FILTER_NOTES: dict[str, str] = {
    "buffer_push_alpha": "None = push all workers. 0.5 = equal fitness and novelty weight",
    "buffer_push_top_k": "Push only top-K workers by combined score",
    "buffer_novelty_floor": "Top fraction by raw novelty always enters the buffer",
}


def _format_default(value: object) -> str:
    if value is None:
        return "`None`"
    if isinstance(value, bool):
        return "`True`" if value else "`False`"
    return f"`{value}`"


def _replace_block(text: str, begin: str, end: str, replacement: str) -> str:
    start = text.index(begin) + len(begin)
    stop = text.index(end)
    return text[:start] + "\n" + replacement.rstrip() + "\n" + text[stop:]


def _experiment_metadata(path: Path) -> tuple[str, str, str]:
    module = ast.parse(path.read_text())
    question = (ast.get_docstring(module) or "").strip().splitlines()[0].strip()

    env_preset = None
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "experiment":
                    call = node.value
                    if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "Experiment":
                        for keyword in call.keywords:
                            if keyword.arg == "env" and isinstance(keyword.value, ast.Constant):
                                env_preset = keyword.value.value
                                break
    if env_preset is None:
        raise ValueError(f"Could not find Experiment(env=...) in {path}")

    env_id = str(ENV_PRESETS.get(env_preset, {}).get("env_id", env_preset))
    return path.name, env_id, question


def _generate_experiments_table() -> str:
    rows = [
        _experiment_metadata(path)
        for path in sorted(EXPERIMENTS_DIR.glob("*.py"))
        if path.name != "__init__.py"
    ]
    lines = [
        "| Script | Environment | Question |",
        "|---|---|---|",
    ]
    lines.extend(f"| `{name}` | {env_id} | {question} |" for name, env_id, question in rows)
    return "\n".join(lines)


def _generate_config_tables() -> str:
    defaults = EDERConfig()
    values = {field.name: getattr(defaults, field.name) for field in fields(EDERConfig)}

    main_lines = [
        "| Parameter | Default | Notes |",
        "|---|---|---|",
    ]
    main_lines.extend(
        f"| `{key}` | {_format_default(values[key])} | {CONFIG_NOTES[key]} |"
        for key in CONFIG_NOTES
    )

    filter_lines = [
        "**Buffer push filtering**",
        "",
        "| Parameter | Default | Notes |",
        "|---|---|---|",
    ]
    filter_lines.extend(
        f"| `{key}` | {_format_default(values[key])} | {FILTER_NOTES[key]} |"
        for key in FILTER_NOTES
    )

    return "\n".join(main_lines + [""] + filter_lines)


def render(readme_text: str) -> str:
    readme_text = _replace_block(
        readme_text,
        EXPERIMENTS_BEGIN,
        EXPERIMENTS_END,
        _generate_experiments_table(),
    )
    readme_text = _replace_block(
        readme_text,
        CONFIG_BEGIN,
        CONFIG_END,
        _generate_config_tables(),
    )
    return readme_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync generated README sections from source code")
    parser.add_argument("--check", action="store_true", help="Fail if README is out of date")
    args = parser.parse_args()

    current = README.read_text()
    updated = render(current)

    if args.check:
        if current != updated:
            print("README.md is out of date. Run: python scripts/sync_readme.py")
            return 1
        print("README.md is up to date.")
        return 0

    README.write_text(updated)
    print("Updated README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
