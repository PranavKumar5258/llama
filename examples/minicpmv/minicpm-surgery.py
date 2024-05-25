import argparse
import glob
import os, json
import torch
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="Path to LLaVA v1.5 model")
args = ap.parse_args()

# find the model part that includes the the multimodal projector weights
model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
checkpoint = model.state_dict()

# get a list of mm tensor names
mm_tensors = [k for k, v in checkpoint.items() if k.startswith("resampler")]

# store these tensors in a new dictionary and torch.save them
projector = {name: checkpoint[name].float().cpu() for name in mm_tensors}
torch.save(projector, f"{args.model}/llava.projector")

clip_tensors = [k for k, v in checkpoint.items() if k.startswith("vpm")]
if len(clip_tensors) > 0:
    clip = {name.replace("vpm.", ""): checkpoint[name].float().cpu() for name in clip_tensors}
    torch.save(clip, f"{args.model}/llava.clip")

    # added tokens should be removed to be able to convert Mistral models
    if os.path.exists(f"{args.model}/added_tokens.json"):
        with open(f"{args.model}/added_tokens.json", "w") as f:
            f.write("{}\n")

config = model.llm.config
config._name_or_path = "openbmb/CPM-2B"
config.auto_map = {
    "AutoConfig": "configuration_minicpm.MiniCPMConfig",
    "AutoModel": "modeling_minicpm.MiniCPMModel",
    "AutoModelForCausalLM": "modeling_minicpm.MiniCPMForCausalLM",
    "AutoModelForSeq2SeqLM": "modeling_minicpm.MiniCPMForCausalLM",
    "AutoModelForSequenceClassification": "modeling_minicpm.MiniCPMForSequenceClassification"
}
model.llm.save_pretrained(f"{args.model}/MiniCPM")
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
tok.save_pretrained(f"{args.model}/MiniCPM")
os.system(f"cp {args.model}/modeling_minicpm.py {args.model}/MiniCPM/modeling_minicpm.py")
os.system(f"cp {args.model}/tokenizer.json {args.model}/MiniCPM/tokenizer.json")
with open(f"{args.model}/MiniCPM/tokenizer_config.json", "r") as f:
    d = json.load(f)
    d.pop("auto_map")
    d["tokenizer_class"] = "LlamaTokenizer"
    if "add_prefix_space" in d:
        d.pop("add_prefix_space")
with open(f"{args.model}/MiniCPM/tokenizer_config.json", "w") as f:
    json.dump(d, f, indent=2)


print("Done!")
print(f"Now you can convert {args.model} to a regular LLaMA GGUF file.")
print(f"Also, use {args.model}/llava.projector to prepare a llava-encoder.gguf file.")
