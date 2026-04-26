"""
Conversation generator for synthetic multi-turn dialogue dataset.

Supports two backends:
  - "gemini"  : Google Gemini API (primary, via google-genai SDK)
  - "gemma"   : Local Gemma served via OpenAI-compatible API
  - "auto"    : Try Gemini first, fall back to local Gemma on failure

Set via --backend flag or BACKEND env variable.
"""
import os
import re
import random
import time
from typing import List, Dict, Optional, Literal, Callable
from dotenv import load_dotenv

from environment_simulator import Environment, EnvironmentSimulator

load_dotenv()

Backend = Literal["gemini", "gemma", "auto"]


class ConversationGenerator:
    def __init__(
        self,
        backend: Backend = "gemma",
        bn_ratio: float = 0.8,
        # Gemini settings
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        # Gemma (local) settings
        gemma_base_url: Optional[str] = None,
        gemma_api_key: Optional[str] = None,
        gemma_model: Optional[str] = None,
        debug: bool = False,
    ):
        self.backend = backend
        self.bn_ratio = bn_ratio  # 0.0 = all English, 1.0 = all Bengali
        self.debug = debug

        if not 0.0 <= self.bn_ratio <= 1.0:
            raise ValueError(f"bn_ratio must be in [0.0, 1.0], got {self.bn_ratio}")

        # --- Gemini config ---
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.gemini_client = None

        if self.backend in ("gemini", "auto") and self.gemini_api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
                print(f"  Gemini client initialized (model: {self.gemini_model})")
            except Exception as e:
                print(f"  Warning: Could not initialise Gemini client: {e}")

        # --- Gemma (local) config ---
        self.gemma_base_url = os.getenv("GEMMA_BASE_URL", "http://0.0.0.0:8000/v1")
        self.gemma_api_key = os.getenv("GEMMA_API_KEY", "not-needed")
        self.gemma_model = os.getenv(
            "GEMMA_MODEL",
            "./models/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q6_K.gguf",
        )
        self.gemma_client = None

        if self.backend in ("gemma", "auto"):
            try:
                from openai import OpenAI
                self.gemma_client = OpenAI(api_key=self.gemma_api_key, base_url=self.gemma_base_url)
                print(f"  Gemma client initialized (model: {self.gemma_model}, server: {self.gemma_base_url})")
            except Exception as e:
                print(f"  Warning: Could not initialise Gemma client: {e}")

        print(f"ConversationGenerator ready  [backend={self.backend}, bn_ratio={self.bn_ratio}]")

    def _log(self, message: str, debug_only: bool = False) -> None:
        if debug_only and not self.debug:
            return
        print(message)

    def _call_gemini(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.85,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        """Call Gemini via google-genai SDK."""
        if not self.gemini_client:
            return None
        try:
            from google.genai import types

            # Build contents: system instruction + conversation turns
            system_text = ""
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    system_text += msg["content"] + "\n"
                elif msg["role"] == "user":
                    contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
                else:  # assistant
                    contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))

            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_text.strip() if system_text.strip() else None,
                    temperature=temperature,
                    top_p=0.92,
                    top_k=20,
                    max_output_tokens=max_tokens,
                ),
            )
            text = response.text
            if text:
                return self._clean_output(text)
        except Exception as e:
            self._log(f"  Gemini call failed: {e}", debug_only=True)
        return None

    def _call_gemma(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Optional[str]:
        """Call local Gemma via OpenAI-compatible API."""
        
        if not self.gemma_client:
            return None
        try:
            response = self.gemma_client.chat.completions.create(
                model=self.gemma_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                # extra_body={"reasoning": {"enabled": False}}
            )
            text = response.choices[0].message.content
            if text:
                return self._clean_output(text)
        except Exception as e:
            self._log(f"  Gemma call failed: {e}", debug_only=True)
        return None

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retries: int = 2,
    ) -> Optional[str]:
        """
        Route to the correct backend with retries.
        On error from one backend, immediately try the other (if auto).
          - "gemini" : Gemini only
          - "gemma"  : local Gemma only
          - "auto"   : try Gemini first, fall back to Gemma on any failure
        """
        for attempt in range(retries):
            # --- Try Gemini ---
            if self.backend in ("gemini", "auto"):
                result = self._call_gemini(messages, temperature, max_tokens)
                if result:
                    return result
                # Gemini failed — if auto, immediately try Gemma
                if self.backend == "auto":
                    self._log("  → Gemini failed, falling back to local Gemma", debug_only=True)
                    result = self._call_gemma(messages, temperature, max_tokens)
                    if result:
                        return result
                    self._log(f"  Both backends failed (attempt {attempt + 1}/{retries})", debug_only=True)
                else:
                    self._log(f"  Gemini attempt {attempt + 1}/{retries} returned nothing", debug_only=True)
                time.sleep(2)
                continue

            # --- Gemma-only mode ---
            if self.backend == "gemma":
                result = self._call_gemma(messages, temperature, max_tokens)
                if result:
                    return result
                self._log(f"  Gemma attempt {attempt + 1}/{retries} returned nothing", debug_only=True)
                time.sleep(2)

        return None

    @staticmethod
    def _clean_output(text: str) -> str:
        """Strip common artifacts from model output."""
        # Remove <think>…</think> blocks (Qwen / reasoning models)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Collapse excessive blank lines
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
        return text.strip()

    @staticmethod
    def _format_history_block(conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history as a plain-text block for embedding in a prompt."""
        lines = []
        for msg in conversation_history:
            label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{label}: {msg['content']}")
        return "\n\n".join(lines)

    def _pick_language(self) -> str:
        """Randomly pick 'Bengali' or 'English' based on bn_ratio."""
        return "Bengali" if random.random() < self.bn_ratio else "English"

    @staticmethod
    def _lang_instruction(lang: str) -> str:
        """Return a short instruction snippet enforcing the chosen language."""
        if lang == "Bengali":
            return (
                "IMPORTANT: You MUST write your ENTIRE response in Bengali (বাংলা) using Bengali script. "
                "Do NOT use English at all. Every word must be in Bengali."
            )
        return "You MUST write your entire response in English."

    def _generate_user_turn(
        self,
        env: Environment,
        user_system_prompt: str,
        conversation_history: List[Dict[str, str]],
        is_first_turn: bool = False,
        language: str = "English",
    ) -> Optional[str]:
        """Generate a single user turn.

        To satisfy vLLM's strict user/assistant alternation requirement,
        the entire conversation history is embedded as text inside a
        single user message rather than mapped onto alternating roles.
        """
        lang_instr = self._lang_instruction(language)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": f"{user_system_prompt}\n\n{lang_instr}"}
        ]

        if is_first_turn:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Start the conversation now. Write your first message to the assistant in {language}. "
                        "Remember: you are the USER, not the assistant. "
                        "Output ONLY your message, nothing else."
                    ),
                }
            )
        else:
            history_text = self._format_history_block(conversation_history)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Here is the conversation so far:\n\n"
                        f"{history_text}\n\n"
                        "---\n"
                        f"Based on the assistant's last response, write your next message as the user in {language}. "
                        "Stay in character. Keep it natural (1-4 sentences). "
                        "If the conversation has reached a natural conclusion, output exactly: CONVERSATION_END\n"
                        "Otherwise, output ONLY your next message as the user."
                    ),
                }
            )

        text = self._call_llm(messages, temperature=0.9, max_tokens=512)
        if text and "CONVERSATION_END" in text:
            return None
        return text

    def _generate_assistant_turn(
        self,
        assistant_system_prompt: str,
        conversation_history: List[Dict[str, str]],
        language: str = "English",
    ) -> Optional[str]:
        lang_instr = self._lang_instruction(language)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": f"{assistant_system_prompt}\n\n{lang_instr}"}
        ]
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        return self._call_llm(messages, temperature=0.8, max_tokens=1024)

    def generate_conversation(
        self,
        env: Environment,
        env_sim: EnvironmentSimulator,
        topic: Optional[str] = None,
        persona: Optional[str] = None,
        num_turns: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Generate a full multi-turn conversation for the given environment.

        Returns a dict:
            {
                "environment": str,
                "scenario_description": str,
                "system_prompt": str,
                "topic": str,
                "user_persona": str,
                "num_turns": int,
                "conversation": [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."},
                    ...
                ]
            }
        """
        topic = topic or env.get_random_topic()
        persona = persona or env.get_random_persona()
        num_turns = num_turns or random.randint(env.min_turns, env.max_turns)

        user_system_prompt = env_sim.build_user_system_prompt(persona, topic, env)
        assistant_system_prompt = env.system_prompt

        # Pick language for this entire conversation based on ratio
        language = self._pick_language()
        conversation_history: List[Dict[str, str]] = []

        self._log(
            f"\n  Generating conversation: env={env.name}, lang={language}, topic='{topic[:50]}…', turns={num_turns}",
            debug_only=True,
        )

        for turn_idx in range(num_turns):
            # --- User turn ---
            user_msg = self._generate_user_turn(
                env=env,
                user_system_prompt=user_system_prompt,
                conversation_history=conversation_history,
                is_first_turn=(turn_idx == 0),
                language=language,
            )
            if user_msg is None:
                # Model decided to end the conversation early
                if len(conversation_history) >= 2:
                    break
                else:
                    # Too short, retry with fallback
                    user_msg = f"Hi, I have a question about: {topic}"

            conversation_history.append({"role": "user", "content": user_msg})
            self._log(f"    Turn {turn_idx + 1} [USER]: {user_msg[:80]}…", debug_only=True)

            # --- Assistant turn ---
            assistant_msg = self._generate_assistant_turn(
                assistant_system_prompt=assistant_system_prompt,
                conversation_history=conversation_history,
                language=language,
            )
            if assistant_msg is None:
                assistant_msg = "I'd be happy to help with that! Could you tell me a bit more about what you need?"
            
            conversation_history.append({"role": "assistant", "content": assistant_msg})
            self._log(f"    Turn {turn_idx + 1} [ASST]: {assistant_msg[:80]}…", debug_only=True)

        if len(conversation_history) < 2:
            self._log("  ⚠ Conversation too short, discarding.")
            return None

        return {
            "environment": env.name,
            "scenario_description": env.description,
            "system_prompt": assistant_system_prompt,
            "topic": topic,
            "user_persona": persona,
            "language": language,
            "num_turns": len(conversation_history) // 2,
            "conversation": conversation_history,
        }

    def generate_batch(
        self,
        env_sim: EnvironmentSimulator,
        conversations_per_env: int = 5,
        environments: Optional[List[str]] = None,
        save_callback: Optional[Callable[[List[Dict]], None]] = None,
        save_every: int = 10,
    ) -> List[Dict]:
        """
        Generate a batch of conversations across all (or selected) environments.

        Args:
            env_sim: The EnvironmentSimulator with registered environments.
            conversations_per_env: How many conversations to generate per environment.
            environments: Optional list of environment names to use.
            save_callback: Optional fn(conversations) called every `save_every` convos.
            save_every: How often to call save_callback (default: 10).

        Returns:
            A list of conversation dicts.
        """
        env_names = environments or env_sim.list_environments()
        all_conversations: List[Dict] = []
        since_last_save = 0

        for env_name in env_names:
            env = env_sim.get_environment(env_name)
            print(f"\n{'='*60}")
            print(f"Environment: {env.name} ({conversations_per_env} conversations)")
            print(f"{'='*60}")

            for i in range(conversations_per_env):
                print(f"\n  [{i+1}/{conversations_per_env}]", end="")
                conv = self.generate_conversation(env=env, env_sim=env_sim)
                if conv:
                    all_conversations.append(conv)
                    since_last_save += 1
                    print(f"  ✓ Generated {conv['num_turns']} turn-pairs")

                    # Incremental save
                    if save_callback and since_last_save >= save_every:
                        save_callback(all_conversations)
                        since_last_save = 0
                else:
                    print(f"  ✗ Failed to generate conversation")

        # Final save for any remaining
        if save_callback and since_last_save > 0:
            save_callback(all_conversations)

        return all_conversations


if __name__ == "__main__":
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument("--backend", choices=["gemini", "gemma", "auto"], default="auto")
    a = p.parse_args()

    env_sim = EnvironmentSimulator()
    generator = ConversationGenerator(backend=a.backend)

    conversations = generator.generate_batch(env_sim, conversations_per_env=1)
    print(f"\n\nGenerated {len(conversations)} conversations total.")
    for conv in conversations:
        print(f"\n--- {conv['environment']} ---")
        print(f"  Topic: {conv['topic']}")
        for msg in conv["conversation"]:
            role = msg["role"].upper()
            print(f"  [{role}]: {msg['content'][:120]}…")
