import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from functools import partial

import pdb
import tqdm

# Get the absolute path of the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning for higher priority

from ECE import _calculate_ece, make_model_diagrams

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU, set_deterministic)
def strtobool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    
    parser.add_argument("--val_dataset", default="ReasonSeg|test", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--noise", default=0.5, type=float)
    parser.add_argument("--importance", type=lambda x: bool(strtobool(x)), default=False)

    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image, noise_step)
    return image_tensor_cd

def gaussian_noise(x, bound=0.01):
    lam = 0.5
    noise_img = torch.randn_like(x) * torch.rand(1).to(x.device) * bound
    return lam * x + (1 - lam) * noise_img

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.val_dataset,
            args.image_size,
    )
    print("Val set length:", val_dataset.__len__())

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    ECE_list = []

    for i in tqdm.tqdm(range(val_dataset.__len__())):
        image_path, images, images_clip, conversation, masks, label, resize, questions, sampled_classes, inference = val_dataset.__getitem__(i)

        if args.use_mm_start_end:
            # replace <image> token
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = conversation[0].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()
        
        output_ids, pred_masks, low_res_masks = model.evaluate(
                images_clip.bfloat16().unsqueeze(0).cuda(),
                images.bfloat16().unsqueeze(0).cuda(),
                input_ids,
                resize_list=[resize],
                original_size_list=[tuple(masks.squeeze(0).shape)],
                max_new_tokens=512,
                tokenizer=tokenizer,
                noise=args.noise,
                importance=args.importance
        )
        
        cd_output_ids, cd_pred_masks, cd_low_res_masks = model.evaluate(
                gaussian_noise(images_clip.unsqueeze(0), args.noise).bfloat16().cuda(),
                gaussian_noise(images.unsqueeze(0), args.noise).bfloat16().cuda(),
                input_ids,
                resize_list=[resize],
                original_size_list=[tuple(masks.squeeze(0).shape)],
                max_new_tokens=512,
                noise=args.noise,
                tokenizer=tokenizer,
                importance=False
        )
        
        #P_logits = pred_masks[0]
        #P_probs = P_logits.sigmoid()

        #masks_list = [masks.squeeze(0).int().cuda()]
        #output = (P_probs >= 0.5).int() ### Baseline: CIoU, GIoU: 0.5556137 0.54342693, probs ver: CIoU, GIoU: 0.5635947 0.54773134 (best)

        P_logits, Q_logits = pred_masks[0], cd_pred_masks[0]
        P_probs, Q_probs = P_logits.sigmoid(), Q_logits.sigmoid()

        IW = P_probs/Q_probs
        IW_logits = IW * P_logits
        IW_probs = F.sigmoid(IW_logits)

        masks_list = [masks.squeeze(0).int().cuda()]
        output = (IW_probs >= 0.55).int() ### Baseline: CIoU, GIoU: 0.5556137 0.54342693, probs ver: CIoU, GIoU: 0.5635947 0.54773134 (best)

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks.shape[0])

        #print(F.sigmoid(pred_masks[0])[F.sigmoid(pred_masks[0]) > 0.5], F.sigmoid(IW_probs)[F.sigmoid(IW_probs) > 0.5])
        #print(len(F.sigmoid(pred_masks[0])[F.sigmoid(pred_masks[0]) >= 0.5]), len(IW_probs[IW_probs >= 0.5]), len(IW_probs[IW_probs >= 0.5]) - len(F.sigmoid(pred_masks[0])[F.sigmoid(pred_masks[0]) >= 0.5]))


        low_res_GT = F.interpolate(masks.unsqueeze(0), (256, 256) ,mode="bilinear", align_corners=False)[0].reshape(1, 256, 256)
        low_res_masks = F.interpolate(low_res_masks, (256, 256) ,mode="bilinear", align_corners=False)[0].reshape(1, 256, 256).cpu()
        '''
        if args.importance:
            cd_low_res_masks = F.interpolate(cd_low_res_masks, (256, 256) ,mode="bilinear", align_corners=False)[0].reshape(1, 256, 256).cpu()
            low_res_masks_probs, cd_low_res_masks_probs = F.sigmoid(low_res_masks), F.sigmoid(cd_low_res_masks)
            low_res_masks_probs[low_res_masks_probs < 1e-4] = 0 #ignore v small probs
            IW =  F.sigmoid(low_res_masks/cd_low_res_masks)
            low_res_masks_probs = IW * low_res_masks_probs
        else:
        '''
        low_res_masks_probs = F.sigmoid(low_res_masks)
        low_res_masks_probs[low_res_masks_probs < 1e-5] = 0 #ignore v small probs
        #pdb.set_trace()
        ece = _calculate_ece(low_res_masks_probs.flatten(), low_res_GT.flatten(), n_bins=10)
        #ece = expected_calibration_error(low_res_masks_probs.flatten(), (low_res_masks_probs>=0.5).bool(), low_res_GT.flatten(), num_bins=15)
        ECE_list.append(ece)

        #pdb.set_trace()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    print("CIoU, GIoU:", ciou, giou)
    print("ECE:", sum(ECE_list)/len(ECE_list))


if __name__ == "__main__":
    set_deterministic(deterministic=True, seed=69)
    main(sys.argv[1:])
