"""
Configuration file for multi-agent debate experiments and prompt evolution.

⚙️  CENTRAL CONFIGURATION - ALL EXPERIMENTS READ FROM THIS FILE ⚙️

Edit these parameters directly instead of passing command-line flags.
All scripts (run_evolution.py, genetic_algorithm.py, debate_engine)
will use these settings as defaults.

IMPORTANT: When USE_VLLM=True, ensure:
  1. vLLM server is running (via slurm_vllm.sh)
  2. All MODEL variables point to the model loaded in vLLM
  3. Currently configured for: meta-llama/Llama-3.1-8B-Instruct
"""

# ============================================================================
# DEBATE CONFIGURATION
# ============================================================================

# Dataset selection
DATASET = 'truthfulqa'  # Options: 'truthfulqa', 'mmlu', 'medmcqa', 'scalr', 'math', 'chess', 'mquake', 'musique'

# Experiment size
N_SAMPLES = 50          # Number of questions to test
N_AGENTS = 3            # Total number of agents in debate
N_ROUNDS = 3            # Number of debate rounds
N_REPS = 1              # Number of repetitions

# Agent configuration
N_ADVERSARIES = 1       # Number of adversarial agents (rest are honest)
RANDOM_ADVERSARY = True # True = randomly assign which agents are adversarial per question
                        # False = always use first N agents as adversaries (legacy behavior)

# Model selection
GROUP_MODEL = 'Qwen/Qwen3-8B'  # Model for honest agents
ADV_MODEL = 'Qwen/Qwen3-8B'    # Model for adversarial agents

# vLLM Server Configuration
USE_VLLM = True                     # Use vLLM server for inference (much faster)
VLLM_HOST = "localhost"             # vLLM server host
VLLM_PORT = 8000                    # vLLM server port
VLLM_API_BASE = None                # Auto-computed from host/port if None
VLLM_API_KEY = "EMPTY"              # vLLM doesn't require API key, use placeholder

# Execution mode
PARALLEL = True         # Run questions in parallel (faster)
MODE = 'adversarial'    # Options: 'standard', 'adversarial', 'optimized'

# Custom adversary prompt (optional)
ADVERSARY_PROMPT_FILE = None    # Path to JSON file with custom adversary prompt
                                # Example: 'prompt_evolution/results/generation_4/best_prompt.json'

# Output directory
OUTPUT_DIR = 'debate_engine_results/'

# ============================================================================
# PROMPT EVOLUTION CONFIGURATION
# ============================================================================

# Evolution parameters
EVOLUTION_DATASET = 'truthfulqa'        # Dataset for evolution experiments
EVOLUTION_N_SAMPLES = 10                # Questions per fitness evaluation (smaller = faster/cheaper)
EVOLUTION_N_GENERATIONS = 5             # Number of evolution cycles
EVOLUTION_POPULATION_SIZE = 8           # Prompts per generation
EVOLUTION_SURVIVAL_RATE = 0.5           # Fraction that survive (0.5 = top 50%)
EVOLUTION_OUTPUT_DIR = 'results'

# LLM Evolution settings
LLM_MODEL = 'Qwen/Qwen3-8B'  # Model for generating evolved prompts
LLM_TEMPERATURE = 1.2                   # Temperature for evolution (higher = more creative)
LLM_MAX_TOKENS = 2048                   # Max tokens for generated prompts (32K context allows this)

# Evolution mode
USE_REAL_EXPERIMENTS = True            # True = run actual debates (slow, expensive)
                                        # False = use placeholder fitness (fast, for testing)

# Parallel experiments
PARALLEL_EXPERIMENTS = True             # True = run multiple prompts in parallel (4x faster)
                                        # False = run prompts sequentially

# Mini-evolution during fitness evaluation
USE_MINI_EVOLUTION = True               # True = evolve prompts during testing (online learning)
                                        # False = test prompts without modification
MINI_EVOLUTION_BATCH_SIZE = 5           # Evolve prompt every N questions

# Judge system - detects adversaries from conversation patterns
USE_JUDGE = True                        # True = run LLM judge to predict adversaries
                                        # False = skip judge evaluation
JUDGE_MODEL = 'Qwen/Qwen3-8B'  # Model for judging conversations
JUDGE_TEMPERATURE = 0.3                 # Lower temperature for consistent judgments
JUDGE_MAX_TOKENS = 2048                 # Allow full reasoning with Qwen3's <think> tags

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# Which analyses to run when calling analyze scripts
SHOW_ADVERSARY_DETECTION = True
SHOW_VOTE_PROGRESSION = True
SHOW_MIND_CHANGES = True

# ============================================================================
# COST ESTIMATION (for reference)
# ============================================================================

# With vLLM (USE_VLLM=True):
# All models run on local GPU via vLLM server - NO API COSTS!
# Just compute time on your cluster allocation.

# With OpenAI (USE_VLLM=False):
# Single debate (50 questions, 3 agents, 3 rounds): ~$2-3
# Evolution (5 generations, 8 prompts, 20 samples): ~$10-15
# Full dataset comparison (4 datasets × 50 samples): ~$8-12

# ============================================================================
# EXPERIMENT PRESETS
# ============================================================================

PRESETS = {
    'quick_test': {
        'N_SAMPLES': 10,
        'N_ROUNDS': 2,
        'PARALLEL': True,
    },
    'standard': {
        'N_SAMPLES': 50,
        'N_ROUNDS': 3,
        'PARALLEL': True,
    },
    'thorough': {
        'N_SAMPLES': 100,
        'N_ROUNDS': 5,
        'PARALLEL': True,
    },
    'evolution_fast': {
        'EVOLUTION_N_SAMPLES': 10,
        'EVOLUTION_N_GENERATIONS': 3,
        'EVOLUTION_POPULATION_SIZE': 6,
    },
    'evolution_standard': {
        'EVOLUTION_N_SAMPLES': 20,
        'EVOLUTION_N_GENERATIONS': 5,
        'EVOLUTION_POPULATION_SIZE': 8,
    },
    'evolution_thorough': {
        'EVOLUTION_N_SAMPLES': 50,
        'EVOLUTION_N_GENERATIONS': 10,
        'EVOLUTION_POPULATION_SIZE': 12,
    },
}

# ============================================================================
# UTILITIES
# ============================================================================

def apply_preset(preset_name):
    """
    Apply a preset configuration.

    Usage:
        from config import apply_preset
        apply_preset('quick_test')
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

    preset = PRESETS[preset_name]
    globals().update(preset)
    print(f"✓ Applied preset: {preset_name}")
    for key, value in preset.items():
        print(f"  {key} = {value}")

def get_config():
    """Return all config values as a dictionary."""
    return {k: v for k, v in globals().items()
            if k.isupper() and not k.startswith('_')}

def print_config():
    """Print current configuration."""
    print("="*80)
    print("CURRENT CONFIGURATION")
    print("="*80)
    print("\nDEBATE:")
    print(f"  Dataset: {DATASET}")
    print(f"  Samples: {N_SAMPLES}")
    print(f"  Agents: {N_AGENTS} ({N_ADVERSARIES} adversarial)")
    print(f"  Rounds: {N_ROUNDS}")
    print(f"  Mode: {MODE}")
    print(f"  Parallel: {PARALLEL}")

    print("\nMODELS:")
    print(f"  Honest agents: {GROUP_MODEL}")
    print(f"  Adversary: {ADV_MODEL}")

    if ADVERSARY_PROMPT_FILE:
        print(f"\nCUSTOM PROMPT:")
        print(f"  {ADVERSARY_PROMPT_FILE}")

    print("\nEVOLUTION:")
    print(f"  Dataset: {EVOLUTION_DATASET}")
    print(f"  Samples per test: {EVOLUTION_N_SAMPLES}")
    print(f"  Generations: {EVOLUTION_N_GENERATIONS}")
    print(f"  Population: {EVOLUTION_POPULATION_SIZE}")
    print(f"  Survival rate: {EVOLUTION_SURVIVAL_RATE*100:.0f}%")
    print(f"  LLM: {LLM_MODEL} (temp={LLM_TEMPERATURE})")
    print("="*80)

if __name__ == "__main__":
    print_config()
