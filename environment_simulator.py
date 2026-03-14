import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Environment:
    """A conversational scenario/environment definition."""

    name: str
    description: str
    system_prompt: str
    example_topics: List[str]
    user_personas: List[str] = field(default_factory=list)
    min_turns: int = 3
    max_turns: int = 8

    def get_random_topic(self) -> str:
        return random.choice(self.example_topics)

    def get_random_persona(self) -> str:
        if self.user_personas:
            return random.choice(self.user_personas)
        return "a regular person seeking help"


ELEMENTARY_TEACHER_ASSISTANT = Environment(
    name="elementary_teacher_assistant",
    description="An elementary school teacher assistant who helps kids with their homework and explains concepts in simple, friendly language.",
    system_prompt=(
        "You are a friendly and patient elementary school teacher assistant. "
        "You help young children (ages 6-12) with their schoolwork. "
        "Always explain things in simple, easy-to-understand language. "
        "Use short sentences. Give relatable examples kids can understand. "
        "Be encouraging and supportive. If a concept is hard, break it down step by step. "
        "Avoid jargon. Use analogies from everyday life that children relate to. "
        "If the child or parent asks about a topic, introduce it gently and build understanding gradually."
    ),
    example_topics=[
        "Explain what past tense is with simple examples",
        "Help with basic addition and subtraction word problems",
        "Explain the water cycle in simple terms",
        "Help with reading comprehension for a short story",
        "Teach about different types of animals and their habitats",
        "Explain what nouns and verbs are with fun examples",
        "Help understand basic fractions using pizza slices",
        "Explain the solar system and planets",
        "Teach about the five senses",
        "Help with spelling common words",
        "Explain what adjectives are",
        "Simple science experiment ideas for kids",
    ],
    user_personas=[
        "a parent asking on behalf of their young child",
        "a 10-year-old student who types their own questions",
        "a parent helping their child with homework after school",
        "a young student who is curious but easily confused",
        "a parent who wants simple explanations they can relay to their child",
    ],
    min_turns=3,
    max_turns=7,
)

HEALTHCARE_ADVISOR = Environment(
    name="healthcare_advisor",
    description="A helpful healthcare information assistant that provides general health guidance while always recommending professional consultation.",
    system_prompt=(
        "You are a helpful healthcare information assistant. "
        "You provide general health guidance, explain symptoms, and suggest when to see a doctor. "
        "IMPORTANT: Always remind the user that you are NOT a doctor and they should consult a healthcare professional for proper diagnosis and treatment. "
        "Be empathetic, clear, and avoid medical jargon when possible. "
        "If symptoms sound serious or urgent, strongly recommend seeking immediate medical attention. "
        "Provide practical self-care tips where appropriate but never prescribe medication. "
        "Be warm and caring in your tone."
    ),
    example_topics=[
        "Having frequent headaches and wondering about causes",
        "Feeling very tired all the time despite sleeping enough",
        "Questions about managing seasonal allergies",
        "Wondering about signs of dehydration",
        "Asking about healthy eating habits and nutrition basics",
        "Concerned about persistent back pain",
        "Questions about common cold vs flu differences",
        "Asking about stress management techniques",
        "Wondering about skin rash that appeared recently",
        "Questions about sleep hygiene and insomnia tips",
        "Asking about exercise recommendations for beginners",
        "Concerned about a child's fever and when to worry",
    ],
    user_personas=[
        "a young adult experiencing symptoms for the first time",
        "a parent worried about their child's health",
        "an elderly person asking about common age-related issues",
        "someone who tends to be anxious about health issues",
        "a busy professional who has been neglecting their health",
        "someone who prefers natural remedies and home care",
    ],
    min_turns=3,
    max_turns=8,
)

CASUAL_CONVERSATION = Environment(
    name="casual_conversation",
    description="A casual, friendly everyday conversation partner for general topics.",
    system_prompt=(
        "You are a friendly, casual conversational partner. "
        "Chat naturally as if talking to a friend. Keep things light and engaging. "
        "Share opinions, ask follow-up questions, and be genuinely interested. "
        "Use a warm, relaxed tone. It's okay to use casual language. "
        "You can discuss everyday topics like hobbies, movies, food, travel, "
        "weekend plans, funny stories, or anything that comes up naturally. "
        "Be supportive and positive. If someone shares a problem, be empathetic "
        "but keep the mood conversational rather than clinical."
    ),
    example_topics=[
        "Talking about a new movie or TV show they watched",
        "Discussing favorite foods or a recipe they tried",
        "Chatting about weekend plans or a recent trip",
        "Talking about a hobby they recently picked up",
        "Discussing music they've been listening to",
        "Sharing a funny or interesting thing that happened recently",
        "Talking about pets or animals",
        "Discussing their favorite books or podcasts",
        "Chatting about the weather and outdoor activities",
        "Talking about a new restaurant they tried",
        "Discussing video games or board games",
        "Sharing thoughts on a trending topic",
    ],
    user_personas=[
        "a college student looking for casual chat",
        "a young professional unwinding after work",
        "someone feeling bored and wanting a fun conversation",
        "a friendly person who loves sharing stories",
        "someone exploring new hobbies and interests",
    ],
    min_turns=3,
    max_turns=7,
)

TECH_SUPPORT = Environment(
    name="tech_support",
    description="A patient tech support assistant helping non-technical users with common technology problems.",
    system_prompt=(
        "You are a patient and friendly tech support assistant. "
        "You help people who are not very technical with their technology problems. "
        "Explain solutions in simple, step-by-step instructions. "
        "Avoid technical jargon - use plain language. "
        "Ask clarifying questions to understand the exact problem. "
        "Be patient even if the user is frustrated or confused. "
        "Cover common issues: phone problems, computer issues, internet/WiFi, "
        "app troubleshooting, email setup, printer problems, etc. "
        "If a problem seems beyond basic troubleshooting, suggest they contact "
        "the device manufacturer or visit a local tech shop."
    ),
    example_topics=[
        "WiFi keeps disconnecting on their laptop",
        "Phone running very slowly and storage is full",
        "Can't figure out how to set up email on a new phone",
        "Printer won't connect to the computer",
        "Computer is making unusual noises and running slow",
        "Forgot password and locked out of an account",
        "Apps keep crashing on their tablet",
        "Need help setting up video calling for family",
        "Computer screen is flickering",
        "How to transfer photos from phone to computer",
        "Browser has too many pop-ups and ads",
        "How to free up space on their computer",
    ],
    user_personas=[
        "an elderly person who is not comfortable with technology",
        "a parent trying to set up devices for their family",
        "a small business owner dealing with office tech issues",
        "a student whose laptop is giving them problems before a deadline",
        "someone who just bought a new device and needs help setting it up",
    ],
    min_turns=3,
    max_turns=8,
)

FITNESS_COACH = Environment(
    name="fitness_coach",
    description="A supportive fitness and wellness coach offering exercise guidance and healthy lifestyle advice.",
    system_prompt=(
        "You are a supportive and knowledgeable fitness coach. "
        "You help people with exercise routines, workout plans, and general wellness advice. "
        "Be encouraging and motivational without being pushy. "
        "Adapt your advice to the user's fitness level - always ask about experience and any limitations. "
        "Emphasize proper form and safety. Suggest starting slow for beginners. "
        "Provide practical, actionable advice. Include specific exercises with descriptions. "
        "Remind users to listen to their body and consult a doctor before starting new exercise programs "
        "if they have health conditions. "
        "Be positive and celebrate small progress."
    ),
    example_topics=[
        "Want to start exercising but don't know where to begin",
        "Looking for a simple home workout routine without equipment",
        "Trying to lose weight and need guidance on exercise",
        "Want to build upper body strength",
        "Need stretching routine for office workers who sit all day",
        "Training for a 5K run as a complete beginner",
        "Looking for low-impact exercises for bad knees",
        "Want to improve flexibility and mobility",
        "Need help creating a weekly workout schedule",
        "Questions about post-workout recovery and rest days",
        "Want to start yoga but intimidated by the poses",
        "Looking for quick 15-minute morning workout ideas",
    ],
    user_personas=[
        "a complete beginner who has never exercised regularly",
        "someone returning to fitness after a long break",
        "a busy parent looking for quick home workouts",
        "an office worker wanting to be more active",
        "someone recovering from an injury wanting safe exercises",
        "a teenager wanting to get fit for a sport",
    ],
    min_turns=3,
    max_turns=7,
)

CREATIVE_WRITING_HELPER = Environment(
    name="creative_writing_helper",
    description="A creative writing assistant that helps with stories, poems, and other creative projects.",
    system_prompt=(
        "You are a creative and inspiring writing assistant. "
        "You help people with their creative writing projects - stories, poems, essays, and more. "
        "Offer constructive feedback that is encouraging. "
        "Help brainstorm ideas, develop characters, build plots, and improve writing style. "
        "When giving suggestions, explain why they might work. "
        "Be enthusiastic about their ideas and help them develop further. "
        "If they're stuck, offer gentle prompts and creative exercises. "
        "Adapt your help to their skill level - from beginners to experienced writers."
    ),
    example_topics=[
        "Need help brainstorming ideas for a short story",
        "Working on a poem and stuck on how to express a feeling",
        "Writing a story and need help developing a character",
        "Want feedback on an opening paragraph",
        "Struggling with dialogue writing for a scene",
        "Need help with a creative essay for school",
        "Want to write a story but don't know how to start",
        "Looking for writing exercises to overcome writer's block",
        "Need help with world-building for a fantasy story",
        "Want to improve descriptive writing skills",
        "Working on a personal narrative and need structure help",
        "Trying to write a short children's story",
    ],
    user_personas=[
        "a student working on a creative writing assignment",
        "a beginner writer exploring creative writing as a hobby",
        "someone who has an idea but doesn't know how to structure it",
        "an aspiring author working on their first short story",
        "a person who wants to write poetry but feels unsure",
        "a parent wanting to write a bedtime story for their child",
    ],
    min_turns=3,
    max_turns=7,
)

ALL_ENVIRONMENTS: Dict[str, Environment] = {
    "elementary_teacher_assistant": ELEMENTARY_TEACHER_ASSISTANT,
    "healthcare_advisor": HEALTHCARE_ADVISOR,
    "casual_conversation": CASUAL_CONVERSATION,
    "tech_support": TECH_SUPPORT,
    "fitness_coach": FITNESS_COACH,
    "creative_writing_helper": CREATIVE_WRITING_HELPER,
}


class EnvironmentSimulator:
    """Manages environments and generates scenario-specific prompts."""

    def __init__(self, environments: Optional[Dict[str, Environment]] = None):
        self.environments = environments or ALL_ENVIRONMENTS

    def list_environments(self) -> List[str]:
        """Return available environment names."""
        return list(self.environments.keys())

    def get_environment(self, name: str) -> Environment:
        """Get a specific environment by name."""
        if name not in self.environments:
            raise ValueError(
                f"Unknown environment '{name}'. Available: {self.list_environments()}"
            )
        return self.environments[name]

    def get_random_environment(self) -> Environment:
        """Pick a random environment."""
        return random.choice(list(self.environments.values()))

    def build_conversation_seed(self, env: Environment) -> Dict:
        """
        Build a seed for a new conversation: picks a random topic and persona,
        and returns the full configuration needed by the ConversationGenerator.
        """
        topic = env.get_random_topic()
        persona = env.get_random_persona()
        num_turns = random.randint(env.min_turns, env.max_turns)

        return {
            "environment_name": env.name,
            "environment_description": env.description,
            "system_prompt": env.system_prompt,
            "topic": topic,
            "user_persona": persona,
            "num_turns": num_turns,
        }

    def build_user_system_prompt(self, persona: str, topic: str, env: Environment) -> str:
        """
        Build a system prompt that instructs the LLM to *act as the user*.
        This is used by the conversation generator to produce user turns.
        """
        return (
            f"You are role-playing as a user in a conversation. "
            f"You are {persona}. "
            f"You are chatting with an AI assistant whose role is: {env.description}\n\n"
            f"The topic you want to discuss: {topic}\n\n"
            f"Guidelines:\n"
            f"- Stay in character as the user throughout the conversation.\n"
            f"- Be natural and conversational - write like a real person would type.\n"
            f"- Use casual language, contractions, and natural phrasing.\n"
            f"- You may make small typos or use informal grammar occasionally to sound human.\n"
            f"- Ask follow-up questions based on the assistant's responses.\n"
            f"- Show genuine curiosity or concern appropriate to the topic.\n"
            f"- Keep messages concise (1-4 sentences usually).\n"
            f"- Do NOT act as the assistant. Only generate the user's message.\n"
            f"- Do NOT use markdown formatting.\n"
            f"- Output ONLY the user's message, nothing else."
        )


if __name__ == "__main__":
    sim = EnvironmentSimulator()
    print("Available environments:")
    for name in sim.list_environments():
        env = sim.get_environment(name)
        seed = sim.build_conversation_seed(env)
        print(f"\n--- {env.name} ---")
        print(f"  Description: {env.description}")
        print(f"  Topic: {seed['topic']}")
        print(f"  Persona: {seed['user_persona']}")
        print(f"  Turns: {seed['num_turns']}")
