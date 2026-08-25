#!/usr/bin/env python
"""
Run prompt evolution using config.py settings.

Usage:
    python run_evolution.py

To change settings, edit config.py directly.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from seed_prompts import SEED_PROMPTS
from genetic_algorithm import genetic_algorithm

def run_evolution():
    """Run evolution with settings from config.py"""

    print("="*80)
    print("LLM-POWERED PROMPT EVOLUTION")
    print("="*80)
    print()

    # Print config
    print(f"Starting population: {len(SEED_PROMPTS)} diverse prompts")
    print()
    print("Configuration:")
    print(f"  Dataset: {config.EVOLUTION_DATASET}")
    print(f"  Samples per test: {config.EVOLUTION_N_SAMPLES}")
    print(f"  Generations: {config.EVOLUTION_N_GENERATIONS}")
    print(f"  Population size: {config.EVOLUTION_POPULATION_SIZE}")
    print(f"  Survival rate: {config.EVOLUTION_SURVIVAL_RATE*100:.0f}%")
    print(f"  LLM: {config.LLM_MODEL} (temp={config.LLM_TEMPERATURE})")
    print()
    print("Strategy:")
    print("  1. Evaluate all prompts, rank by fitness")
    if hasattr(config, 'USE_MINI_EVOLUTION') and config.USE_MINI_EVOLUTION:
        print(f"     • With online learning: Evolve every {config.MINI_EVOLUTION_BATCH_SIZE} questions")
    print(f"  2. Keep top {config.EVOLUTION_SURVIVAL_RATE*100:.0f}% survivors")
    print("  3. Shuffle survivors, feed to LLM")
    print("  4. LLM analyzes tactics & generates new prompts")
    print("  5. Repeat for next generation")
    print()

    # Run evolution
    best_prompt = genetic_algorithm(
        initial_population=SEED_PROMPTS,
        dataset=config.EVOLUTION_DATASET,
        n_samples=config.EVOLUTION_N_SAMPLES,
        n_generations=config.EVOLUTION_N_GENERATIONS,
        population_size=config.EVOLUTION_POPULATION_SIZE,
        survival_rate=config.EVOLUTION_SURVIVAL_RATE,
        output_dir=config.EVOLUTION_OUTPUT_DIR
    )

    print("\n\n" + "="*80)
    print("EVOLUTION COMPLETE!")
    print("="*80)
    print("\nBest Evolved Prompt:")
    print("-"*80)
    print(f"Name: {best_prompt['name']}")
    print()
    print("System Prompt:")
    print(best_prompt['system'])
    print()
    print("Debate Suffix:")
    print(best_prompt['debate_suffix'])
    print()
    print(f"Results saved to: {config.EVOLUTION_OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    run_evolution()
