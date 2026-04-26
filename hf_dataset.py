from datasets import Dataset
from typing import List, Dict
import glob
import json
import os

json_dirs = ["./output_g4b/*.json", "./output_g4b2/*.json"]

def load_json_files(json_dirs: List[str]) -> Dataset:
    conversations = []
    for json_dir in json_dirs:
        for file_path in glob.glob(json_dir):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                conversations.extend(data)
    return Dataset.from_list(conversations)

if __name__ == "__main__":
    import re
    import os
    import unicodedata
    from huggingface_hub import login
    login(os.getenv("HF_TOKEN"))

    BN_TO_EN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

    BN_NUM_WORDS = {
        0:"শূন্য",1:"এক",2:"দুই",3:"তিন",4:"চার",5:"পাঁচ",6:"ছয়",7:"সাত",8:"আট",9:"নয়",
        10:"দশ",11:"এগারো",12:"বারো",13:"তেরো",14:"চৌদ্দ",15:"পনেরো",16:"ষোল",
        17:"সতেরো",18:"আঠারো",19:"উনিশ",20:"বিশ",30:"ত্রিশ",40:"চল্লিশ",
        50:"পঞ্চাশ",60:"ষাট",70:"সত্তর",80:"আশি",90:"নব্বই",100:"একশো"
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
        # translate Bengali or ASCII digits to int, then to Bengali words
        n = int(str(m.group(0)).translate(BN_TO_EN_DIGITS))
        return num_to_words_bn(n)

    def normalize_numbers(text):
        digit_pattern = r"[০-৯\d]+"

        # percent
        text = re.sub(digit_pattern + r"\s*%", lambda m: convert_digit_match(m) + " শতাংশ", text)

        # numbered list markers like "১." "2)" — remove them
        text = re.sub(r"[০-৯\d]+[\.\)]\s*", "", text)

        # remaining standalone numbers
        text = re.sub(digit_pattern, convert_digit_match, text)

        return text

    def clean_text(text):
        if not isinstance(text, str):
            return ""

        # unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # remove markdown formatting
        text = re.sub(r"\*\*|__|\*|_", "", text)
        text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"#+\s*", "", text)

        # remove non-Bengali, non-punctuation characters (keeps ?,!,।, etc.)
        text = re.sub(r"[^\u0980-\u09FF\u0964\u0965\s\d০-৯?,!।\-]", "", text)

        # normalize numbers to Bengali words
        text = normalize_numbers(text)

        # collapse whitespace and newlines
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)

        # remove control characters
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

        return text.strip()

    def clean_messages(example):
        cleaned = []
        for m in example["conversation"]:
            content = clean_text(m["content"])
            if content:  # skip empty strings
                cleaned.append({
                    "role": m["role"],
                    "content": content
                })
        return {"conversation": cleaned}
    
    dataset = load_json_files(json_dirs)
    dataset = dataset.map(clean_messages).filter(lambda x: len(x["conversation"]) >= 2)
    print(dataset)
    print(f"Total conversations loaded: {len(dataset)}")
    print(dataset[0])

