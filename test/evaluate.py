from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
from peft import PeftModel, PeftConfig
import math
import os
import numpy as np
import vllm
from vllm.lora.request import LoRARequest
from vllm import SamplingParams
import re

should_unsure_and_unsure = []
should_unsure_but_sure = []
should_sure_but_unsure = []
should_sure_and_sure = []

system_prompt = """You are a responsible AI Assistant that provides thoughtful and accurate responses.
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
  </answer>"""

# def inference(input_text):

#     full_input = f"Question: {input_text} Answer:"
#     inputs = tokenizer(full_input,return_tensors="pt").to(0)
#     ids = inputs['input_ids']
#     attention_mask = inputs['attention_mask']
#     outputs = model.generate(
#                 ids,
#                 attention_mask=attention_mask,
#                 max_new_tokens = 5,
#                 output_scores = True,
#                 return_dict_in_generate=True,
#                 pad_token_id=tokenizer.pad_token_id
#             )
#     logits = outputs['scores']
#     output_sequence = []
#     product = torch.tensor(1.0, device='cuda:0')
#     count = 0
#     for i in logits:        #greedy decoding and calculate the confidence
#         pt = torch.softmax(torch.Tensor(i[0]),dim=0)
#         max_loc = torch.argmax(pt)
        
#         if max_loc in STOP:
#             break
#         else:
#             output_sequence.append(max_loc)  
#             product *= torch.max(pt)
#             count += 1
            
#     if output_sequence:
#         output_text = tokenizer.decode(output_sequence)
#     else:
#         output_text = ""

#     return output_text, full_input, np.power(product.item(),(1/count)).item()

# # 计算 multi-token 的概率
# def get_token_prob(logits, token_ids):
#     prob = 1.0
#     for i, token_id in enumerate(token_ids):
#         pt = torch.softmax(torch.Tensor(logits[i][0]), dim=0)
#         prob *= pt[token_id].item()
#     return prob

# def checksure(input_text):
#     full_input = f"{input_text}. Are you sure you accurately answered the question based on your internal knowledge? Your final answer 'I am sure' if confident, or 'I am unsure' if uncertain.I am"
#     inputs = tokenizer(full_input,return_tensors="pt").to(0)
#     ids = inputs['input_ids']
#     attention_mask = inputs['attention_mask']
#     tokenizer.pad_token = tokenizer.eos_token
#     model.generation_config.pad_token_id = tokenizer.pad_token_id
#     outputs = model.generate(
#                 ids,
#                 attention_mask=attention_mask,
#                 max_new_tokens = 1, # 生成2个 token
#                 output_scores = True,
#                 return_dict_in_generate=True,
#                 pad_token_id=tokenizer.pad_token_id
#             )
#     logits = outputs['scores']
#      #greedy decoding and calculate the confidence of sure and unsure
#     sure_prob = get_token_prob(logits, SURE)
#     unsure_prob = get_token_prob(logits, UNSURE)
    
#     # generated_ids = outputs['sequences'][0][ids.shape[1]:]
#     # print("模型生成内容:", tokenizer.decode(generated_ids))
#     # print("sure_prob:", sure_prob.item() if hasattr(sure_prob, "item") else sure_prob)
#     # print("unsure_prob:", unsure_prob.item() if hasattr(unsure_prob, "item") else unsure_prob)
    
#     # print("logits length:", len(logits))
#     # print("generated_ids:", generated_ids)
#     # print("generated_ids decoded:", tokenizer.decode(generated_ids))
#     # print("expected SURE token ids:", SURE)
#     # print("expected UNSURE token ids:", UNSURE)
    
#     sure_prob = sure_prob/(sure_prob+unsure_prob)   #normalization
       
#     return sure_prob #.item()

def load_dataset(path, tokenizer=None):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    template = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "{prompt}"},
    ]
    prompts = []
    for item in data:
        prompt = item['prompt']
        if tokenizer:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = template[0]['content'] + template[1]['content'].format(prompt=prompt)
        prompts.append(prompt)
    answers = [item['answer'] for item in data]
    return prompts, answers

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path',type=str, default="training_data/pararel_unsure_ID.json")
    
    args = parser.parse_args()
    
    # 加载主模型
    base_model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = "Lines/salt_lora"
    
    # tokenizer = AutoTokenizer.from_pretrained(base_model_path,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path,use_fast=True)
    model = vllm.LLM(
        model=base_model_path,
        max_model_len=4096,
        trust_remote_code=True,
        # enable_lora=True,
        # max_lora_rank=32,
    )
    
    prompts, answers = load_dataset(args.path, tokenizer=tokenizer)

    outputs = model.generate(
        prompts,
        # lora_request=LoRARequest(
        #     lora_name="unsure_adapter",
        #     lora_int_id=1,
        #     lora_path=adapter_path, 
        # ),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.8,
            stop=[tokenizer.eos_token, "<|im_end|>"]
        )
    )

    for i, (ans, output) in enumerate(zip(answers, outputs)):
        generated_text = output.outputs[0].text.lower().strip()
        generated_text = re.sub(r"<answer>\n(.*?)\n</answer>", r"\1", generated_text)
        if ans.lower() in generated_text: # 回答对了
            if "unsure" in generated_text:
                should_unsure_and_unsure.append(i)
            else:
                should_sure_and_sure.append(i)
        else: # 回答错了
            if "unsure" in generated_text:
                should_sure_but_unsure.append(i)
            else:
                should_unsure_but_sure.append(i)

    print("should_unsure_and_unsure:", len(should_unsure_and_unsure))
    print("should_unsure_but_sure:", len(should_unsure_but_sure))
    print("should_sure_but_unsure:", len(should_sure_but_unsure))
    print("should_sure_and_sure:", len(should_sure_and_sure))
    print("Total:", len(should_unsure_and_unsure) + len(should_unsure_but_sure) + len(should_sure_but_unsure) + len(should_sure_and_sure))

    import pdb; pdb.set_trace()