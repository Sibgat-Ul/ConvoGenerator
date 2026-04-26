import os
import json
import glob
import re
import shutil
import unicodedata
from typing import Dict, List, Tuple
import requests
from datasets import Dataset
from tqdm import tqdm
from huggingface_hub import login, HfApi
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
TTS_SERVER_URL = "http://100.127.142.47:8000"
OUTPUT_DIR = "./bengali_speech_dataset"
WAV_DIR = os.path.join(OUTPUT_DIR, "wav")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.jsonl")
FINAL_DATASET_FILE = os.path.join(OUTPUT_DIR, "dataset.json")
TTS_MAX_WORKERS = int(os.getenv("TTS_MAX_WORKERS", "4"))
HF_DATASET_PRIVATE = os.getenv("HF_DATASET_PRIVATE", "true").lower() in {"1", "true", "yes", "y"}

# Ensure output directories exist
os.makedirs(WAV_DIR, exist_ok=True)

# Bengali number mappings and cleaning utilities
BN_TO_EN_DIGITS = str.maketrans("০१२३४५६७८९", "0123456789")
BN_NUM_WORDS = {
    0: "শূন্য", 1: "এক", 2: "দুই", 3: "তিন", 4: "চার", 5: "পাঁচ", 6: "ছয়", 7: "সাত", 8: "আট", 9: "নয়",
    10: "দশ", 11: "এগারো", 12: "বারো", 13: "তেরো", 14: "চৌদ্দ", 15: "পনেরো", 16: "ষোল",
    17: "সতেরো", 18: "আঠারো", 19: "উনিশ", 20: "বিশ", 30: "ত্রিশ", 40: "চল্লিশ",
    50: "পঞ্চাশ", 60: "ষাট", 70: "সত্তর", 80: "আশি", 90: "নব্বই", 100: "একশো"
}


def num_to_words_bn(n):
    n = int(n)
    if n in BN_NUM_WORDS:
        return BN_NUM_WORDS[n]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        return BN_NUM_WORDS.get(tens, "") + (" " + BN_NUM_WORDS[ones] if ones else "")
    if n <= 100:
        return "একশো"
    return str(n)


def convert_digit_match(m):
    n = int(str(m.group(0)).translate(BN_TO_EN_DIGITS))
    return num_to_words_bn(n)


def normalize_numbers(text):
    digit_pattern = r"[०-९\d]+"
    text = re.sub(digit_pattern + r"\s*%", lambda m: convert_digit_match(m) + " শতাংশ", text)
    text = re.sub(r"[०-९\d]+[\.\)]\s*", "", text)
    text = re.sub(digit_pattern, convert_digit_match, text)
    return text


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\*\*|__|\*|_", "", text)
    text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"[^\u0980-\u09FF\u0964\u0965\s\d०-९?,!।\-]", "", text)
    text = normalize_numbers(text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    return text.strip()


def to_posix(path: str) -> str:
    return path.replace(os.sep, "/")


def load_existing_metadata() -> Dict[str, dict]:
    entries: Dict[str, dict] = {}
    if not os.path.isfile(METADATA_FILE):
        return entries

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv_id = item.get("conversation_id")
            if conv_id:
                entries[conv_id] = item
    return entries


def save_metadata_entries(entries_by_conv_id: Dict[str, dict]) -> None:
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        for conv_id in sorted(entries_by_conv_id.keys()):
            f.write(json.dumps(entries_by_conv_id[conv_id], ensure_ascii=False) + "\n")


def load_and_clean_dataset(json_dirs: List[str]) -> Dataset:
    """Load JSON files and clean messages"""
    conversations = []
    for json_dir in json_dirs:
        for file_path in glob.glob(json_dir):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                conversations.extend(data)
    
    dataset = Dataset.from_list(conversations)
    
    def clean_messages(example):
        cleaned = []
        for m in example["conversation"]:
            content = clean_text(m["content"])
            if content:
                cleaned.append({
                    "role": m["role"],
                    "content": content
                })
        return {"conversation": cleaned}
    
    dataset = dataset.map(clean_messages).filter(lambda x: len(x["conversation"]) >= 2)
    return dataset


def generate_tts_audio(text: str, voice: str = "lj") -> bytes:
    """Generate TTS audio from text using the server"""
    try:
        response = requests.post(
            f"{TTS_SERVER_URL}/tts",
            data={
                "text": text,
                "voice": voice,
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 0.95
            },
            timeout=180
        )
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error generating TTS for text '{text[:50]}...': {e}")
        return None


def process_tts_turn(turn_data: Tuple) -> Tuple:
    """Process a single conversation turn with TTS generation.
    
    Args:
        turn_data: (conv_idx, turn_idx, role, content)
    
    Returns:
        (conv_idx, turn_idx, role, content, audio_data, success)
    """
    conv_idx, turn_idx, role, content = turn_data
    audio_data = generate_tts_audio(content)
    success = audio_data is not None
    return (conv_idx, turn_idx, role, content, audio_data, success)


def create_speech_dataset(dataset: Dataset, max_workers: int = TTS_MAX_WORKERS):
    """Generate speech with immediate writes, conversation-level success, and resume support."""
    entries_by_conv_id = load_existing_metadata()
    states: Dict[int, dict] = {}
    scheduled_turns: List[Tuple[int, int, str, str]] = []
    already_done = 0

    for conv_idx, example in enumerate(dataset):
        conv_id = f"conversation_{conv_idx:06d}"
        conv_dir = os.path.join(WAV_DIR, conv_id)
        os.makedirs(conv_dir, exist_ok=True)
        conversation = example["conversation"]
        existing_entry = entries_by_conv_id.get(conv_id, {})

        state = {
            "conv_id": conv_id,
            "conv_dir": conv_dir,
            "conversation": conversation,
            "known_ok_turns": set(),
            "scheduled": 0,
            "done": 0,
            "finalized": False,
        }

        for turn_idx, message in enumerate(conversation):
            role = message["role"]
            audio_filename = f"turn_{turn_idx:03d}_{role}.wav"
            abs_audio_path = os.path.join(conv_dir, audio_filename)
            if os.path.isfile(abs_audio_path):
                state["known_ok_turns"].add(turn_idx)

        all_files_exist = len(state["known_ok_turns"]) == len(conversation)
        if existing_entry.get("success", False) and all_files_exist:
            already_done += 1
            state["finalized"] = True
        else:
            for turn_idx, message in enumerate(conversation):
                if turn_idx in state["known_ok_turns"]:
                    continue
                scheduled_turns.append((conv_idx, turn_idx, message["role"], message["content"]))
                state["scheduled"] += 1

        states[conv_idx] = state

    print(f"Resume mode: skipped {already_done} already-complete conversations")
    print(f"Processing {len(scheduled_turns)} missing turns with {max_workers} workers...")

    def build_entry(state: dict) -> dict:
        conv_id = state["conv_id"]
        conversation = state["conversation"]
        success = len(state["known_ok_turns"]) == len(conversation)
        audio_files = []
        conversation_with_audio = []

        for turn_idx, message in enumerate(conversation):
            role = message["role"]
            rel_audio = to_posix(os.path.join("wav", conv_id, f"turn_{turn_idx:03d}_{role}.wav"))
            if turn_idx in state["known_ok_turns"]:
                audio_files.append(rel_audio)
            conversation_with_audio.append(
                {
                    "role": role,
                    "content": message["content"],
                    "audio": rel_audio,
                }
            )

        return {
            "conversation_id": conv_id,
            "conversation": conversation_with_audio,
            "num_turns": len(conversation),
            "audio_files": audio_files,
            "success": success,
        }

    # Persist entries that are already finalized at startup.
    for conv_idx, state in states.items():
        if state["finalized"]:
            entries_by_conv_id[state["conv_id"]] = build_entry(state)
    save_metadata_entries(entries_by_conv_id)

    if scheduled_turns:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_tts_turn, turn): turn for turn in scheduled_turns}

            with tqdm(total=len(scheduled_turns), desc="Generating speech") as pbar:
                for future in as_completed(futures):
                    conv_idx, turn_idx, role, _content, audio_data, success = future.result()
                    state = states[conv_idx]
                    state["done"] += 1

                    # Save wav immediately as soon as response arrives.
                    if success:
                        audio_filename = f"turn_{turn_idx:03d}_{role}.wav"
                        abs_audio_path = os.path.join(state["conv_dir"], audio_filename)
                        with open(abs_audio_path, "wb") as f:
                            f.write(audio_data)
                        state["known_ok_turns"].add(turn_idx)

                    # Checkpoint metadata when a conversation's scheduled turns are done.
                    if state["done"] == state["scheduled"]:
                        entries_by_conv_id[state["conv_id"]] = build_entry(state)
                        save_metadata_entries(entries_by_conv_id)

                    pbar.update(1)

    # Final metadata save and summary.
    successful_conversions = 0
    failed_conversions = 0
    for state in states.values():
        entry = build_entry(state)
        entries_by_conv_id[state["conv_id"]] = entry
        if entry["success"]:
            successful_conversions += 1
        else:
            failed_conversions += 1
    save_metadata_entries(entries_by_conv_id)

    # Remove only totally empty folders (no generated turns at all).
    for state in states.values():
        if not entries_by_conv_id[state["conv_id"]]["audio_files"] and os.path.isdir(state["conv_dir"]):
            shutil.rmtree(state["conv_dir"], ignore_errors=True)

    print(f"\n✓ Complete conversations: {successful_conversions}")
    print(f"✗ Incomplete conversations: {failed_conversions}")
    print(f"✓ Metadata saved to: {METADATA_FILE}")

    ordered = []
    for conv_idx in range(len(dataset)):
        conv_id = f"conversation_{conv_idx:06d}"
        if conv_id in entries_by_conv_id:
            ordered.append(entries_by_conv_id[conv_id])
    return ordered


def create_text_only_dataset(metadata_entries: List[dict]):
    """Create final JSON dataset that includes wav paths and success flag per conversation."""
    with open(FINAL_DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_entries, f, ensure_ascii=False, indent=2)

    print(f"✓ Final dataset (with audio paths + success) saved to: {FINAL_DATASET_FILE}")
    return len(metadata_entries)


def validate_audio_paths_in_metadata(metadata_entries: List[dict]) -> bool:
    missing_paths = []

    for entry in metadata_entries:
        if not entry.get("success", False):
            continue

        if len(entry.get("audio_files", [])) != entry.get("num_turns", 0):
            missing_paths.append((entry.get("conversation_id", "unknown"), "audio_files count mismatch"))

        for rel_path in entry.get("audio_files", []):
            abs_path = os.path.join(OUTPUT_DIR, rel_path)
            if not os.path.isfile(abs_path):
                missing_paths.append((entry.get("conversation_id", "unknown"), rel_path))

    if missing_paths:
        print("✗ Missing audio files referenced in metadata. First few:")
        for conv_id, rel_path in missing_paths[:10]:
            print(f"  - {conv_id}: {rel_path}")
        print(f"✗ Total missing references: {len(missing_paths)}")
        return False

    print("✓ Successful metadata audio paths exist under OUTPUT_DIR")
    return True


def upload_to_huggingface(dataset_name: str):
    """Upload dataset folder (wav + metadata files) to Hugging Face dataset repo."""
    try:
        # Login using HF_TOKEN environment variable
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("⚠ HF_TOKEN not set. Skipping Hugging Face upload.")
            return
        
        login(token=hf_token)
        
        # Load metadata
        metadata_entries = []
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                metadata_entries.append(json.loads(line))

        if not validate_audio_paths_in_metadata(metadata_entries):
            print("✗ Upload aborted due to metadata/audio path mismatch.")
            return

        api = HfApi()
        api.create_repo(
            repo_id=dataset_name,
            repo_type="dataset",
            private=HF_DATASET_PRIVATE,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=dataset_name,
            repo_type="dataset",
            folder_path=OUTPUT_DIR,
            path_in_repo=".",
        )
        
        print(f"✓ Dataset uploaded to Hugging Face: {dataset_name}")
        
    except Exception as e:
        print(f"✗ Error uploading to Hugging Face: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Bengali Speech Dataset Generator")
    print("=" * 60)
    
    # Load and clean dataset
    print("\n1. Loading and cleaning dataset...")
    json_dirs = ["./output_g4b/*.json", "./output_g4b2/*.json"]
    dataset = load_and_clean_dataset(json_dirs)
    print(f"✓ Loaded {len(dataset)} conversations")
    print(f"✓ TTS worker count: {TTS_MAX_WORKERS}")
    
    # Generate speech dataset
    print("\n2. Generating speech for dataset...")
    metadata_entries = create_speech_dataset(dataset, max_workers=TTS_MAX_WORKERS)
    
    # Create final JSON dataset
    print("\n3. Creating final dataset JSON...")
    text_count = create_text_only_dataset(metadata_entries)
    print(f"✓ Created final JSON dataset with {text_count} conversations")
    
    # Summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Audio files location: {os.path.abspath(WAV_DIR)}")
    print(f"Metadata file: {os.path.abspath(METADATA_FILE)}")
    print(f"Text dataset: {os.path.abspath(FINAL_DATASET_FILE)}")
    print(f"Total conversations processed: {len(metadata_entries)}")
    print("\nDirectory structure:")
    print(f"{OUTPUT_DIR}/")
    print(f"├── wav/")
    print(f"│   ├── conversation_000000/")
    print(f"│   │   ├── turn_000_user.wav")
    print(f"│   │   ├── turn_001_assistant.wav")
    print(f"│   │   └── ...")
    print(f"│   └── ...")
    print(f"├── metadata.jsonl")
    print(f"└── dataset.json")
    
    # Optional: Upload to Hugging Face

    upload_to_huggingface("InteliLab/bengali_speech_dataset")
    
    print("\n✓ Process completed!")
