import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

pd.set_option('display.max_columns', None) ## 모든 열을 출력한다.
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

input_txt = "Transformer are the"
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
iterations = []
n_steps = 8
choice_per_step = 5

with torch.no_grad():
    for _ in range(n_steps):
        iteration = dict()
        iteration["Input"] = tokenizer.decode(input_ids[0])

        output = model(input_ids=input_ids)

        next_token_logits = output.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)

        for choice_idx in range(choice_per_step):
            token_id = sorted_ids[choice_idx]
            token_prob = next_token_probs[token_id].cpu().numpy()
            token_choice = f"[{tokenizer.decode(token_id)} ({100 * token_prob:.2f}%)"
            iteration[f"Choice {choice_idx + 1}"] = token_choice
        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)
        iterations.append(iteration)
print(pd.DataFrame(iterations))

input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
output = model.generate(input_ids, max_length=n_steps, do_sample=False)
print(tokenizer.decode(output[0]))