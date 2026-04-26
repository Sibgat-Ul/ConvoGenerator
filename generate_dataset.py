import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

from environment_simulator import EnvironmentSimulator
from conversation_generator import ConversationGenerator

load_dotenv()

def save_json(conversations: List[Dict], path: str) -> None:
    """Save the dataset as a single pretty-printed JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    print(f"  Saved JSON → {path}  ({len(conversations)} records)")

def print_stats(conversations: List[Dict]) -> None:
    total = len(conversations)
    if total == 0:
        print("  No conversations generated.")
        return

    env_counts: Dict[str, int] = {}
    total_turns = 0
    total_user_chars = 0
    total_asst_chars = 0

    for conv in conversations:
        env = conv["environment"]
        env_counts[env] = env_counts.get(env, 0) + 1
        total_turns += conv["num_turns"]
        for msg in conv["conversation"]:
            if msg["role"] == "user":
                total_user_chars += len(msg["content"])
            else:
                total_asst_chars += len(msg["content"])

    print(f"\n{'='*50}")
    print(f"  Dataset Statistics")
    print(f"{'='*50}")
    print(f"  Total conversations : {total}")
    print(f"  Total turn-pairs    : {total_turns}")
    print(f"  Avg turns/convo     : {total_turns / total:.1f}")
    print(f"  Avg user msg length : {total_user_chars / (total_turns or 1):.0f} chars")
    print(f"  Avg asst msg length : {total_asst_chars / (total_turns or 1):.0f} chars")
    print(f"\n  Per environment:")
    for env, count in sorted(env_counts.items()):
        print(f"    {env:30s}: {count} conversations")
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-turn dialogue dataset"
    )
    parser.add_argument("--per-env", type=int, default=400)
    parser.add_argument("--envs", nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument(
        "--backend",
        choices=["gemini", "gemma", "auto"],
        default="gemma",
    )
    parser.add_argument(
        "--bn-ratio",
        type=float,
        default=1.0,
        help="Ratio of Bengali conversations (0.0=all English, 1.0=all Bengali). Default: 1.0",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=1,
        help="Global minimum turn-pairs for all environments (overrides per-environment values).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=3,
        help="Global maximum turn-pairs for all environments (overrides per-environment values).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose per-turn debug logs (model outputs and backend retry details).",
    )
    parser.add_argument(
        "--env-workers",
        type=int,
        default=1,
        help="Number of parallel workers across environments. Default: 1 (sequential).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.bn_ratio <= 1.0:
        print("  ERROR: --bn-ratio must be between 0.0 and 1.0")
        sys.exit(1)

    if args.min_turns is not None and args.min_turns < 1:
        print("  ERROR: --min-turns must be >= 1")
        sys.exit(1)

    if args.max_turns is not None and args.max_turns < 1:
        print("  ERROR: --max-turns must be >= 1")
        sys.exit(1)

    if args.min_turns is not None and args.max_turns is not None and args.min_turns > args.max_turns:
        print("  ERROR: --min-turns cannot be greater than --max-turns")
        sys.exit(1)

    if args.env_workers < 1:
        print("  ERROR: --env-workers must be >= 1")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"dialogue_dataset_{timestamp}.json")

    print(f"\n{'#'*60}")
    print(f"  Synthetic Dialogue Dataset Generator")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")

    env_sim = EnvironmentSimulator(
        default_min_turns=args.min_turns,
        default_max_turns=args.max_turns,
    )

    env_names = args.envs or env_sim.list_environments()
    sample_env = next(iter(env_sim.environments.values()))
    has_global_turn_override = (
        env_sim.default_min_turns is not None or env_sim.default_max_turns is not None
    )

    print(f"  Backend       : {args.backend}")
    print(f"  Debug logs    : {'ON' if args.debug else 'OFF'}")
    print(f"  Env workers   : {args.env_workers}")
    print(f"  BN ratio      : {args.bn_ratio:.0%} Bengali / {1 - args.bn_ratio:.0%} English")
    if has_global_turn_override:
        print(
            "  Turn range    : "
            f"{sample_env.min_turns}-{sample_env.max_turns} "
            "(global override from CLI/.env)"
        )
    else:
        print("  Turn range    : per-environment defaults")
    print(f"  Environments  : {', '.join(env_names)}")
    print(f"  Per env       : {args.per_env}")
    print(f"  Total planned : {len(env_names) * args.per_env} conversations")
    print(f"  Output        : {output_path}\n")

    for name in env_names:
        if name not in env_sim.list_environments():
            available = ", ".join(env_sim.list_environments())
            print(f"  ERROR: Unknown environment '{name}'. Available: {available}")
            sys.exit(1)

    def _generate_one_environment(env_name: str) -> List[Dict]:
        local_generator = ConversationGenerator(
            backend=args.backend,
            bn_ratio=args.bn_ratio,
            debug=args.debug,
        )
        return local_generator.generate_batch(
            env_sim=env_sim,
            conversations_per_env=args.per_env,
            environments=[env_name],
            save_callback=None,
            save_every=10,
        )

    start_time = time.time()
    conversations: List[Dict] = []
    max_workers = min(args.env_workers, len(env_names))

    if max_workers == 1:
        for env_name in env_names:
            conversations.extend(_generate_one_environment(env_name))
    else:
        print(f"  Running {len(env_names)} environments with {max_workers} parallel workers...\n")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_env = {
                executor.submit(_generate_one_environment, env_name): env_name
                for env_name in env_names
            }
            for future in as_completed(future_to_env):
                env_name = future_to_env[future]
                try:
                    env_conversations = future.result()
                    conversations.extend(env_conversations)
                    print(
                        f"  ✓ Environment '{env_name}' complete: "
                        f"{len(env_conversations)} conversations"
                    )
                except Exception as exc:
                    print(f"  ✗ Environment '{env_name}' failed: {exc}")

    elapsed = time.time() - start_time

    if not conversations:
        print("\n  No conversations were generated. Check your LLM server connection.")
        sys.exit(1)

    # Final save
    save_json(conversations, output_path)

    print_stats(conversations)
    print(f"  Generation time: {elapsed:.1f}s ({elapsed / len(conversations):.1f}s per conversation)")
    print(f"\n  ✓ Done! {len(conversations)} conversations → {output_path}\n")


if __name__ == "__main__":
    main()
