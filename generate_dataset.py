import argparse
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
    parser.add_argument("--per-env", type=int, default=100)
    parser.add_argument("--envs", nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument(
        "--backend",
        choices=["gemini", "gemma", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--bn-ratio",
        type=float,
        default=0.8,
        help="Ratio of Bengali conversations (0.0=all English, 1.0=all Bengali). Default: 0.8",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"dialogue_dataset_{timestamp}.json")

    print(f"\n{'#'*60}")
    print(f"  Synthetic Dialogue Dataset Generator")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")

    env_sim = EnvironmentSimulator()
    generator = ConversationGenerator(backend=args.backend, bn_ratio=args.bn_ratio)

    env_names = args.envs or env_sim.list_environments()
    print(f"  Backend       : {args.backend}")
    print(f"  BN ratio      : {args.bn_ratio:.0%} Bengali / {1 - args.bn_ratio:.0%} English")
    print(f"  Environments  : {', '.join(env_names)}")
    print(f"  Per env       : {args.per_env}")
    print(f"  Total planned : {len(env_names) * args.per_env} conversations")
    print(f"  Output        : {output_path}\n")

    for name in env_names:
        if name not in env_sim.list_environments():
            available = ", ".join(env_sim.list_environments())
            print(f"  ERROR: Unknown environment '{name}'. Available: {available}")
            sys.exit(1)

    def _incremental_save(convos: List[Dict]):
        save_json(convos, output_path)

    start_time = time.time()
    conversations = generator.generate_batch(
        env_sim=env_sim,
        conversations_per_env=args.per_env,
        environments=env_names,
        save_callback=_incremental_save,
        save_every=10,
    )
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
