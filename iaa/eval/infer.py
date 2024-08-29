import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import itertools

from iaa.model.builder import load_pretrained_model
from iaa.utils import disable_torch_init
from iaa.mm_utils import process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import pdb
import sys
from pprint import pprint as pp

g_input_msg = [
    {
        "role": "system", 
        "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    }
]


def get_input(tokenizer, image_processor, model_config, rounds, query, args):

    if args.task_type in ["G","MM"]:
        g_input_msg.append({
            "role": "user", 
            "content": ("<|reserved_special_token_44|>"+ '\n' if not rounds else "") + query
        })
    else:
        g_input_msg.append({
            "role": "user", 
            "content": query
        })
    
    input_ids = tokenizer.apply_chat_template(
        g_input_msg,
        add_generation_prompt=True,
        padding="longest",
        return_tensors="pt",
    )

    if args.task_type in ["G","MM"]:
        input_id_list = input_ids[0].tolist()
        input_id_list[input_id_list.index(128049)]=-200
        image = Image.open(args.image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model_config)[0]
        input_ids = torch.tensor(input_id_list, dtype=input_ids.dtype,device=input_ids.device)

        return input_ids.unsqueeze(0), image_tensor.unsqueeze(0)
    else:
        input_id_list = input_ids[0].tolist()
        input_ids = torch.tensor(input_id_list, dtype=input_ids.dtype,device=input_ids.device)
        return input_ids.unsqueeze(0), None

    
def infer_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer.pad_token = tokenizer.eos_token
    task_type = args.task_type
    
    rounds = 0
    while 1:
        try:
            query = input("user: ")
            if query == "exit":
                break
        except:
            continue
            
        input_ids, image_tensor = get_input(tokenizer, image_processor, model.config, rounds, query, args)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                task_type=task_type,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True) if image_tensor is not None else None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                eos_token_id=[tokenizer.convert_tokens_to_ids("<|eot_id|>",)],
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        print("iaa:", outputs)
        
        g_input_msg.append({
            "role": "assistant", 
            "content": outputs
        })
        rounds += 1
        

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--task_type", type=str, default=None)
    args = parser.parse_args()
    
    infer_model(args)