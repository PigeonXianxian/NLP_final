

# borrowed from https://github.com/huggingface/open-r1/blob/main/recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml
OPEN_R1_SYSTEM_PROMPT = """
You are a helpful AI Assistant that provides well-reasoned and detailed responses.
You first think about the reasoning process as an internal monologue and then provide the user with the answer.
Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>
"""

# borrowed from https://github.com/knoveleng/open-rs/blob/main/recipes/grpo.yaml
OPEN_RS_SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{{}} .
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, 
i.e., <think> reasoning process here </think> <answer> answer here </answer>.
Note that respond by English, NOT use other languages.
"""

# the first question from aime 2024
FIXED_PROMPT_FOR_EVALUATION = """
Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards.
When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop.
When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop.
Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour.
Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."""

# 自定义system prompt

CUSTOM_SYSTEM_PROMPT = """
You are a responsible AI Assistant that provides thoughtful and accurate responses.
Follow this reasoning process:
1. First, think through the problem step-by-step in your internal monologue
2. Then, provide your final answer only if you are confident in its correctness
3. If you are uncertain about any part of the answer, clearly state your uncertainty

Respond in the following format:
<think>
[Your detailed reasoning process here]
</think>
<answer>
[Your final answer if confident, or "I am unsure." if uncertain]
</answer>
"""

# 自定义评测prompt
CUSTOM_EVALUATION_PROMPT = """
Who produces the Chevrolet Brookwood?"""
