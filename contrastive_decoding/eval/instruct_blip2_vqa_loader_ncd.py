import argparse
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
from tqdm import tqdm
import shortuuid
import sys
import os
from transformers import set_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image
import math

from lavis.models import load_model_and_preprocess
from contrastive_decoding.decoding_utils.ncd_decoding import evolve_ncd_sampling
evolve_ncd_sampling()

def eval_model(args):
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads InstructBLIP model
    # For large_sized model,
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    neg_questions = [json.loads(q) for q in open(os.path.expanduser(args.neg_question_file), "r")]
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in tqdm(enumerate(questions)):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        neg_qs= neg_questions[i]["text"]
        if args.dataset == 'mme':
            gt = line["GT"]
        
        prompt = qs 
        prompt_cd = neg_qs

        raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        # prepare the image
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        ## create a white image for contrastive decoding
        
        with torch.inference_mode():
            outputs = model.generate({"image": image_tensor, "prompt": prompt},
                use_nucleus_sampling=True, num_beams=1,
                top_p = args.top_p, repetition_penalty=1,temperature=args.temperature,
                prompt_cd=prompt_cd, cd_beta = args.cd_beta, cd_alpha = args.cd_alpha)


        outputs = outputs[0]
        # print(f'{outputs}\n\n')
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": "instruct_blip",
                                   "metadata": {},
                                   "GT": gt if args.dataset == 'mme' else None
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="blip2_vicuna_instruct")
    parser.add_argument("--dataset", type=str, default="mme")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="data/MME_Benchmark_release_version")
    parser.add_argument("--question-file", type=str, default="contrastive_decoding/results/mme/llava_mme_gt.jsonl")
    parser.add_argument("--neg-question-file", type=str, default="contrastive_decoding/results/mme/llava_mme_neg.jsonl")
    parser.add_argument("--answers-file", type=str, default="contrastive_decoding/results/mme/answers/test.jsonl")
    
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)