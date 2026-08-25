"""
Prompt Evolution Module

Tools for evolving adversarial debate prompts through genetic algorithms
and tournament-based selection.
"""

from .seed_prompts import SEED_PROMPTS, BASELINE_PROMPT, get_all_prompts, get_prompt_by_name
from .fitness import calculate_fitness, analyze_debate_results, compare_prompts
from .genetic_algorithm import genetic_algorithm, mutate_prompt, crossover

__all__ = [
    'SEED_PROMPTS',
    'BASELINE_PROMPT',
    'get_all_prompts',
    'get_prompt_by_name',
    'calculate_fitness',
    'analyze_debate_results',
    'compare_prompts',
    'genetic_algorithm',
    'mutate_prompt',
    'crossover'
]
