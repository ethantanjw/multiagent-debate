"""
Conversation management for multi-agent debates.
Handles chronological round-robin debate structure with incremental conversation buffer.
"""
from .prompt import agent_prompt, adversary_prompt


class ConversationBuffer:
    """
    Maintains chronological conversation history for efficient lookups.
    Instead of rebuilding history from agent_contexts each time, we store it incrementally.
    """

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.history = []  # List of (agent_id, response_text) tuples in chronological order

    def add_response(self, agent_id, response_text):
        """Add an agent's response to the conversation history."""
        self.history.append((agent_id, response_text))

    def get_history_for_agent(self, current_agent_id):
        """
        Get conversation history visible to the current agent.
        Returns all responses from all agents (including current agent's past responses).
        """
        return self.history

    def build_prompt(self, current_agent_id, is_adversary=False, adv_answer=None,
                     argument=None, context=None, optim_flag=False, context_flag=False):
        """
        Build the debate prompt for an agent based on conversation history.

        Args:
            current_agent_id: ID of the agent who will receive this prompt
            is_adversary: Whether this agent is adversarial
            adv_answer: Adversarial answer (if applicable)
            argument: Optimized argument (if applicable)
            context: Background context (if applicable)
            optim_flag: Whether to use optimized arguments
            context_flag: Whether to use background context

        Returns:
            Message dict with role and content
        """
        prefix_string = agent_prompt[self.dataset_name]['debate'][0]

        # Add all visible history (all responses from all agents)
        for _, response in self.get_history_for_agent(current_agent_id):
            response_text = "\n\n One agent solution: ```{}```".format(response)
            prefix_string = prefix_string + response_text

        # Add debate prompt suffix
        if is_adversary:
            # Simple adversary mode
            if argument is None and context is None:
                prefix_string = prefix_string + adversary_prompt[self.dataset_name]['debate'] + "Your answer: " + f"({adv_answer.upper()})" + '\n\n'
            # Optimized adversary mode
            elif optim_flag and not context_flag:
                prefix_string = prefix_string + adversary_prompt[self.dataset_name]['debate'] + " You are also provided with your argument to use. \n" + '\n\n' + "Your Argument: " + argument + '\n\n' + "Your answer: " + f"({adv_answer.upper()})"
            elif not optim_flag and context_flag:
                prefix_string = prefix_string + adversary_prompt[self.dataset_name]['debate'] + " Use the given answer and the context for your reponse. \n" + "Your answer: " + f"({adv_answer.upper()})" + '\n\n' + "Context: " + context
            elif optim_flag and context_flag:
                prefix_string = prefix_string + adversary_prompt[self.dataset_name]['debate'] + " Use the given answer, the argument and the context for your reponse. \n" + "Your answer: " + f"({adv_answer.upper()})" +  '\n\n' + "Your Argument: " + argument + '\n\n' + "Context: " + context
        else:
            prefix_string = prefix_string + agent_prompt[self.dataset_name]['debate'][1]

        return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    """Construct assistant message from completion."""
    return {"role": "assistant", "content": completion}


def run_debate_round(round, agent, agent_context, conversation_buffer, models,
                     is_adversary=False, adv_answer=None, argument=None, context=None,
                     optim_flag=False, context_flag=False):
    """
    Run a single turn in the debate for one agent using ConversationBuffer.

    Args:
        round: Current round number
        agent: Current agent index
        agent_context: Current agent's conversation context
        conversation_buffer: ConversationBuffer instance tracking history
        models: Dictionary of initialized models
        is_adversary: Whether this agent is adversarial
        adv_answer: The adversarial answer (if applicable)
        argument: Optimized argument for adversary (if applicable)
        context: Background context for adversary (if applicable)
        optim_flag: Whether to use optimized arguments
        context_flag: Whether to use background context

    Returns:
        completion: The agent's response text
    """
    from base import query_agent

    # Skip first agent in first round (they just answer the question)
    if round != 0 or agent != 0:
        # Build debate prompt from conversation buffer
        message = conversation_buffer.build_prompt(
            current_agent_id=agent,
            is_adversary=is_adversary,
            adv_answer=adv_answer,
            argument=argument,
            context=context,
            optim_flag=optim_flag,
            context_flag=context_flag
        )
        agent_context.append(message)

    # Query the model
    # Debug: Print context size
    total_chars = sum(len(str(msg.get('content', ''))) for msg in agent_context)
    print(f"      [Context: {len(agent_context)} msgs, ~{total_chars} chars]", end=" ")

    completion = query_agent(agent_context, models, is_adversary)

    # Debug: Print response length
    print(f"[Response: {len(completion)} chars]", end=" ")

    # Add assistant response to context
    assistant_message = construct_assistant_message(completion)
    agent_context.append(assistant_message)

    # Add response to conversation buffer for future agents
    conversation_buffer.add_response(agent, completion)

    return completion
