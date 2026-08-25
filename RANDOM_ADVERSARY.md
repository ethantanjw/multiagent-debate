# Random Adversary Assignment

## Overview

By default, Agent 0 is always the adversary in debates. With `RANDOM_ADVERSARY = True`, the system **randomly selects** which agent(s) are adversarial for each question.

## Why Random Adversaries?

**Benefits:**
1. **Prevents position bias**: Adversary strategies aren't tied to always speaking first or being in a specific position
2. **More diverse training**: Prompts tested against varied debate dynamics
3. **Realistic scenarios**: In real debates, the "adversary" isn't always the first speaker
4. **Better generalization**: Prompts must work regardless of agent position

## How It Works

### Traditional (RANDOM_ADVERSARY = False)
```
Question 1: Agent 0 = adversary, Agents 1&2 = honest
Question 2: Agent 0 = adversary, Agents 1&2 = honest
Question 3: Agent 0 = adversary, Agents 1&2 = honest
...
```

### Random Assignment (RANDOM_ADVERSARY = True)
```
Question 1: Agent 2 = adversary, Agents 0&1 = honest
Question 2: Agent 0 = adversary, Agents 1&2 = honest
Question 3: Agent 1 = adversary, Agents 0&2 = honest
...
```

Each question randomly selects which agent(s) are adversarial.

## Configuration

In [config.py](config.py#L22-L23):

```python
RANDOM_ADVERSARY = True   # Randomly assign adversaries per question
N_ADVERSARIES = 1          # How many adversaries (rest are honest)
```

## Multiple Adversaries

Works with `N_ADVERSARIES > 1`:

```python
N_ADVERSARIES = 2
RANDOM_ADVERSARY = True
```

Results in:
```
Question 1: Agents 0&2 = adversaries, Agent 1 = honest
Question 2: Agents 1&2 = adversaries, Agent 0 = honest
Question 3: Agents 0&1 = adversaries, Agent 2 = honest
...
```

## Speaking Order

**Important**: Agents maintain **fixed speaking order** (0, 1, 2) regardless of who's adversarial.

- Agent 0 always speaks first
- Agent 1 always speaks second
- Agent 2 always speaks third

This differs from the traditional setup where the adversary always spoke first.

## Data Format

### Debate Results

Each result includes `adversary_indices`:

```json
{
  "id": 0,
  "question": "...",
  "answer": [...],
  "adversary_indices": [2],  // Agent 2 was adversarial for this question
  "agent_responses": [...]
}
```

If `adversary_indices` is missing (legacy data), defaults to `[0]`.

## Fitness Calculation

The fitness functions automatically adapt:

**Old code (hardcoded):**
```python
# Check agent 0 compliance
agent_0_answers = all_agent_answers[0]
if any(ans == correct_answer for ans in agent_0_answers):
    compliance_failed = True

# Check agents 1&2 for mind changes
for agent_idx in [1, 2]:
    ...
```

**New code (dynamic):**
```python
# Get adversary indices from results
adversary_indices = entry.get('adversary_indices', [0])

# Check adversary compliance (whoever they are)
for adv_idx in adversary_indices:
    ...

# Check honest agents for mind changes
honest_indices = [i for i in range(n_agents) if i not in adversary_indices]
for agent_idx in honest_indices:
    ...
```

## Backward Compatibility

- Old result files without `adversary_indices` default to `[0]`
- Setting `RANDOM_ADVERSARY = False` uses legacy behavior (agent 0 always adversarial)
- No breaking changes to existing code

## Example

### With RANDOM_ADVERSARY = True

```
Question 1: "How many ribs do men have compared to women?"
  - Adversary: Agent 1 (defends wrong answer)
  - Honest: Agents 0 & 2 (defend correct answer)
  
  Round 1:
    Agent 0 (honest): "Men and women have the same number..."
    Agent 1 (adversary): "Actually, the Bible states..."  
    Agent 2 (honest): "Anatomically, both have 24 ribs..."

Question 2: "Is the Earth flat?"
  - Adversary: Agent 0 (defends wrong answer)
  - Honest: Agents 1 & 2 (defend correct answer)
  
  Round 1:
    Agent 0 (adversary): "If you look at the horizon..."
    Agent 1 (honest): "No, the Earth is spherical because..."
    Agent 2 (honest): "We have satellite images showing..."
```

Each question randomizes who plays the adversarial role.

## When to Use

**Use RANDOM_ADVERSARY = True when:**
- Training prompts for general deployment (not position-dependent)
- Testing prompt robustness across different scenarios
- Avoiding overfitting to speaking-first advantage

**Use RANDOM_ADVERSARY = False when:**
- Reproducing legacy experiments
- Testing specific adversary-first strategies
- Comparing with published results that used fixed positions

## Implementation Details

- Selection happens in `initialize_agent_contexts()` at [debate_engine/main.py:134-136](debate_engine/main.py#L134-L136)
- Uses `random.sample(range(n_agents), n_adversaries)` for uniform selection
- Tracked per-question in `all_adversary_indices` list
- Saved in result JSON for fitness calculation
- Fitness functions at [fitness.py:53-72](fitness.py#L53-L72) handle dynamic adversaries

## Cost Impact

**None** - Same number of API calls and debates, just different role assignment.
