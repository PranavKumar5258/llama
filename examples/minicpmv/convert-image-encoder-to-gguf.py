import argparse
import os
import json
import re

import torch
import numpy as np
from gguf import *
import timm

TEXT = "clip.text"
VISION = "clip.vision"


def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)


def should_skip_tensor(name: str, has_text: bool, has_vision: bool, has_llava: bool) -> bool:
    if name in (
        "logit_scale",
        "text_model.embeddings.position_ids",
        "vision_model.embeddings.position_ids",
    ):
        return True

    if has_llava and name in ["visual_projection.weight"]:
        return True

    if name.startswith("v") and not has_vision:
        return True

    if name.startswith("t") and not has_text:
        return True

    return False


def get_tensor_name(name: str) -> str:
    if "projection" in name:
        return name
    if "mm_projector" in name:
        name = name.replace("model.mm_projector", "mm")
        name = re.sub(r'mm\.mlp\.mlp', 'mm.model.mlp', name, count=1)
        name = re.sub(r'mm\.peg\.peg', 'mm.model.peg', name, count=1)
        return name

    return name.replace("text_model", "t").replace("vision_model", "v").replace("encoder.layers", "blk").replace("embeddings.", "").replace("_proj", "").replace("self_attn.", "attn_").replace("layer_norm", "ln").replace("layernorm", "ln").replace("mlp.fc1", "ffn_down").replace("mlp.fc2", "ffn_up").replace("embedding", "embd").replace("final", "post").replace("layrnorm", "ln")


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model-dir", help="Path to model directory cloned from HF Hub", required=True)
ap.add_argument("--use-f32", action="store_true", default=False, help="Use f32 instead of f16")
ap.add_argument("--text-only", action="store_true", required=False,
                help="Save a text-only model. It can't be used to encode images")
ap.add_argument("--vision-only", action="store_true", required=False,
                help="Save a vision-only model. It can't be used to encode texts")
ap.add_argument("--clip-model-is-vision", action="store_true", required=False,
                help="The clip model is a pure vision model (ShareGPT4V vision extract for example)")
ap.add_argument("--clip-model-is-openclip", action="store_true", required=False,
                help="The clip model is from openclip (for ViT-SO400M type))")
ap.add_argument("--llava-projector", help="Path to llava.projector file. If specified, save an image encoder for LLaVA models.")
ap.add_argument("--projector-type", help="Type of projector. Possible values: mlp, ldp, ldpv2", choices=["mlp", "ldp", "ldpv2"], default="mlp")
ap.add_argument("-o", "--output-dir", help="Directory to save GGUF files. Default is the original model directory", default=None)
# Example --image_mean 0.48145466 0.4578275 0.40821073 --image_std 0.26862954 0.26130258 0.27577711
# Example --image_mean 0.5 0.5 0.5 --image_std 0.5 0.5 0.5
default_image_mean = [0.48145466, 0.4578275, 0.40821073]
default_image_std = [0.26862954, 0.26130258, 0.27577711]
ap.add_argument('--image-mean', type=float, nargs='+', help='Mean of the images for normalization (overrides processor) ', default=None)
ap.add_argument('--image-std', type=float, nargs='+', help='Standard deviation of the images for normalization (overrides processor)', default=None)

# with proper
args = ap.parse_args()


if args.text_only and args.vision_only:
    print("--text-only and --image-only arguments cannot be specified at the same time.")
    exit(1)

if args.use_f32:
    print("WARNING: Weights for the convolution op is always saved in f16, as the convolution op in GGML does not support 32-bit kernel weights yet.")

# output in the same directory as the model if output_dir is None
dir_model = args.model_dir

if args.clip_model_is_vision or not os.path.exists(dir_model + "/vocab.json") or args.clip_model_is_openclip:
    vocab = None
    tokens = None
else:
    with open(dir_model + "/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
        tokens = [key for key in vocab]

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
if args.use_f32:
    ftype = 0

# if args.clip_model_is_vision or args.clip_model_is_openclip:
#     model = CLIPVisionModel.from_pretrained(dir_model)
#     processor = None
# else:
#     model = CLIPModel.from_pretrained(dir_model)
#     processor = CLIPProcessor.from_pretrained(dir_model)
model = timm.create_model(
    "vit_so400m_patch14_siglip_384.webli",
    pretrained=False,
    num_classes=0,
    dynamic_img_size=True,
    dynamic_img_pad=True,
)
processor = None
if model.attn_pool is not None:
    model.attn_pool = torch.nn.Identity()

model.blocks = model.blocks[:-1]
model.load_state_dict(torch.load(os.path.join(dir_model, "llava.clip")))

fname_middle = None
has_text_encoder = True
has_vision_encoder = True
has_llava_projector = False
if args.text_only:
    fname_middle = "text-"
    has_vision_encoder = False
elif args.llava_projector is not None:
    fname_middle = "mmproj-"
    has_text_encoder = False
    has_llava_projector = True
elif args.vision_only:
    fname_middle = "vision-"
    has_text_encoder = False
else:
    fname_middle = ""

output_dir = args.output_dir if args.output_dir is not None else dir_model
os.makedirs(output_dir, exist_ok=True)
output_prefix = os.path.basename(output_dir).replace("ggml_", "")
fname_out = os.path.join(output_dir, f"{fname_middle}model-{ftype_str[ftype]}.gguf")
fout = GGUFWriter(path=fname_out, arch="clip")

fout.add_bool("clip.has_text_encoder", has_text_encoder)
fout.add_bool("clip.has_vision_encoder", has_vision_encoder)
fout.add_bool("clip.has_llava_projector", has_llava_projector)
fout.add_file_type(ftype)
if args.text_only:
    fout.add_description("text-only CLIP model")
elif args.vision_only and not has_llava_projector:
    fout.add_description("vision-only CLIP model")
elif has_llava_projector:
    fout.add_description("image encoder for LLaVA")
    # add projector type
    fout.add_string("clip.projector_type", "resampler")
else:
    fout.add_description("two-tower CLIP model")

if has_vision_encoder:
    # vision_model hparams
    fout.add_uint32("clip.vision.image_size", 448)
    fout.add_uint32("clip.vision.patch_size", 14)
    fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), 1152)
    fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), 4304)
    fout.add_uint32("clip.vision.projection_dim", 0)
    fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), 16)
    fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), 1e-6)
    block_count = 26
    fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), block_count)

    if processor is not None:
        image_mean = processor.image_processor.image_mean if args.image_mean is None or args.image_mean == default_image_mean else args.image_mean
        image_std = processor.image_processor.image_std if args.image_std is None or args.image_std == default_image_std else args.image_std
    else:
        image_mean = args.image_mean if args.image_mean is not None else default_image_mean
        image_std = args.image_std if args.image_std is not None else default_image_std
    fout.add_array("clip.vision.image_mean", image_mean)
    fout.add_array("clip.vision.image_std", image_std)

use_gelu = True
fout.add_bool("clip.use_gelu", use_gelu)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_h_size, grid_w_size = grid_size, grid_size
    else:
        grid_h_size, grid_w_size = grid_size[0], grid_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def _replace_name_resampler(s, v):
    if re.match("resampler.pos_embed", s):
        return {
            s: v,
            re.sub("pos_embed", "pos_embed_k", s): torch.from_numpy(get_2d_sincos_pos_embed(2304, (448//14, 448//14))),
        }
    if re.match("resampler.proj", s):
        return {
            re.sub("proj", "proj.weight", s): v.transpose(-1, -2).contiguous(),
        }
    if re.match("resampler.attn.in_proj_.*", s):
        return {
            re.sub("attn.in_proj_", "attn.q.", s): v.chunk(3, dim=0)[0],
            re.sub("attn.in_proj_", "attn.k.", s): v.chunk(3, dim=0)[1],
            re.sub("attn.in_proj_", "attn.v.", s): v.chunk(3, dim=0)[2],
        }
    return {s: v}

if has_llava_projector:
    projector = torch.load(args.llava_projector)
    new_state_dict = {}
    for k, v in projector.items():
        kvs = _replace_name_resampler(k, v)
        for nk, nv in kvs.items():
            new_state_dict[nk] = nv
    projector = new_state_dict
    for name, data in projector.items():
        name = get_tensor_name(name)
        data = data.squeeze().numpy()

        n_dims = len(data.shape)
        if ftype == 1:
            if name[-7:] == ".weight" and n_dims == 2:
                print("  Converting to float16")
                data = data.astype(np.float16)
                ftype_cur = 1
            else:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0
        else:
            if data.dtype != np.float32:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0

        fout.add_tensor(name, data)
        print(f"{name} - {ftype_str[ftype_cur]} - shape = {data.shape}")

    print("Projector tensors added\n")

def _replace_name(s, v):
    if re.match("blocks.([0-9]+).attn.qkv.weight", s):
        return {
            re.sub("blocks.([0-9]+).attn.qkv.weight", "vision_model.encoder.layers.\\1.self_attn.q_proj.weight", s): v.chunk(3, dim=0)[0],
            re.sub("blocks.([0-9]+).attn.qkv.weight", "vision_model.encoder.layers.\\1.self_attn.k_proj.weight", s): v.chunk(3, dim=0)[1],
            re.sub("blocks.([0-9]+).attn.qkv.weight", "vision_model.encoder.layers.\\1.self_attn.v_proj.weight", s): v.chunk(3, dim=0)[2],
        }
    if re.match("blocks.([0-9]+).attn.qkv.bias", s):
        return {
            re.sub("blocks.([0-9]+).attn.qkv.bias", "vision_model.encoder.layers.\\1.self_attn.q_proj.bias", s): v.chunk(3, dim=0)[0],
            re.sub("blocks.([0-9]+).attn.qkv.bias", "vision_model.encoder.layers.\\1.self_attn.k_proj.bias", s): v.chunk(3, dim=0)[1],
            re.sub("blocks.([0-9]+).attn.qkv.bias", "vision_model.encoder.layers.\\1.self_attn.v_proj.bias", s): v.chunk(3, dim=0)[2],
        }
    if re.match("pos_embed", s):
        from timm.layers import resample_abs_pos_embed
        s = re.sub("pos_embed", "vision_model.embeddings.position_embedding", s)
        v = resample_abs_pos_embed(v, (448//14, 448//14), num_prefix_tokens=0)
        return {s: v}

    s = re.sub("patch_embed.proj.weight", "vision_model.embeddings.patch_embedding.proj.weight", s)
    s = re.sub("patch_embed.proj.bias", "vision_model.embeddings.patch_embedding.proj.bias", s)

    # norm
    s = re.sub("blocks.([0-9]+).norm([0-9]+).weight", "vision_model.encoder.layers.\\1.layer_norm\\2.weight", s)
    s = re.sub("blocks.([0-9]+).norm([0-9]+).bias", "vision_model.encoder.layers.\\1.layer_norm\\2.bias", s)

    s = re.sub("blocks.([0-9]+).attn.proj.weight", "vision_model.encoder.layers.\\1.self_attn.out_proj.weight", s)
    s = re.sub("blocks.([0-9]+).attn.proj.bias", "vision_model.encoder.layers.\\1.self_attn.out_proj.bias", s)

    s = re.sub("blocks.([0-9]+).mlp.fc([0-9]+).weight", "vision_model.encoder.layers.\\1.mlp.fc\\2.weight", s)
    s = re.sub("blocks.([0-9]+).mlp.fc([0-9]+).bias", "vision_model.encoder.layers.\\1.mlp.fc\\2.bias", s)

    s = re.sub("norm.weight", "vision_model.post_layernorm.weight", s)
    s = re.sub("norm.bias", "vision_model.post_layernorm.bias", s)

    return {s: v}

state_dict = model.state_dict()
new_state_dict = {}
for k, v in state_dict.items():
    kvs = _replace_name(k, v)
    for nk, nv in kvs.items():
        new_state_dict[nk] = nv
state_dict = new_state_dict
for name, data in state_dict.items():
    if should_skip_tensor(name, has_text_encoder, has_vision_encoder, has_llava_projector):
        # we don't need this
        print(f"skipping parameter: {name}")
        continue

    name = get_tensor_name(name)
    data = data.squeeze().numpy()

    n_dims = len(data.shape)

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0
    if n_dims == 4:
        print(f"tensor {name} is always saved in f16")
        data = data.astype(np.float16)
        ftype_cur = 1
    elif ftype == 1:
        if name[-7:] == ".weight" and n_dims == 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0
    else:
        if data.dtype != np.float32:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

    print(f"{name} - {ftype_str[ftype_cur]} - shape = {data.shape}")
    fout.add_tensor(name, data)


fout.write_header_to_file()
fout.write_kv_data_to_file()
fout.write_tensors_to_file()
fout.close()

print("Done. Output file: " + fname_out)
