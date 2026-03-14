from huggingface_hub import snapshot_download, login
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("HF_KEY")

login(api_key)
model_id = "unsloth/gemma-3-27b-it-GGUF"

snapshot_download(
    model_id, 
    local_dir="./models/gemma-3-27b-it-GGUF", 
    local_dir_use_symlinks=False, 
    allow_patterns=["gemma-3-27b-it-UD-Q6_K_XL.gguf", "mmproj-BF16.gguf","gemma-3-27b-it-Q6_K.gguf" ]
)
