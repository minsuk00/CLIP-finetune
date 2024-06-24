# Finished implementing clip loss with precomputed text embeddings, but doesn't seem to work for some reason
import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPModel
from safetensors import safe_open
from datetime import datetime, timedelta
from typing import Literal

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available

from utils import evaluate_nn
import ignite.distributed as idist

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def get_clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    # caption_loss = contrastive_loss(similarity)
    # image_loss = contrastive_loss(similarity.t())
    # return (caption_loss + image_loss) / 2.0
    image_loss = contrastive_loss(similarity)
    return image_loss

# Dataset


class MyDataset(torch.utils.data.Dataset):

    def __init__(
        self, json_file, tokenizer=None, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path="", dataset_type: Literal['mscoco', 'imagenet1k', 'imagenet100'] = 'mscoco'
    ):
        if dataset_type == "mscoco":
            raise NotImplementedError("Need to implement text precompute for mscoco")
        
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.dataset_type = dataset_type

        self.data = json.load(open(json_file))  # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.text_embedding_dict_L = torch.load("IN100_text_embedding_dict_L.pt")
        self.text_embedding_dict_with_projection_H = torch.load("IN100_text_embedding_dict_with_projection_H.pt")

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        item = self.data[idx]
        # text = item["text"]
        image_file = item["image_file"]

        # read image
        if self.dataset_type == 'imagenet1k' or self.dataset_type == 'imagenet100':
            class_id = image_file.split("_")[0]
            raw_image = Image.open(os.path.join(self.image_root_path, class_id, image_file))
        elif self.dataset_type == 'mscoco':
            raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        else:
            raise Exception("data retrieval not implemented for this dataset type")

        image = self.transform(raw_image.convert("RGB"))
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        text_sd=self.text_embedding_dict_L[class_id]
        text_clip=self.text_embedding_dict_with_projection_H[class_id]

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            # text = ""
            text_sd = self.text_embedding_dict_L[""]
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            # text = ""
            text_sd = self.text_embedding_dict_L[""]
            drop_image_embed = 1
        # get text and tokenize
        # text_input_ids = self.tokenizer(
        #     text,
        #     max_length=self.tokenizer.model_max_length,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt",
        # ).input_ids

        return {
            "image": image,
            # "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "text_embeddings_sd":text_sd,
            "text_embeddings_clip":text_clip,
        }

    def __len__(self):
        return len(self.data)


def get_loader(args):
    if args.dataset_type != 'imagenet100':
        raise Exception("implement 1nn dataloader")
    loader = {}

    transform_train = transforms.Compose(
        [
            # RandomResizedCrop(224, interpolation=3),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset_val = datasets.ImageFolder(
        os.path.join(args.data_root_path), transform=transform_train
    )
    dataset_test = datasets.ImageFolder(
        os.path.join(args.data_root_path.replace("train", "val")), transform=transform_val
    )

    loader['val'] = idist.auto_dataloader(
        dataset=dataset_val,
        batch_size=256,
        num_workers=4,
        shuffle=True, drop_last=True,
        pin_memory=True,)
    loader['test'] = idist.auto_dataloader(
        dataset=dataset_test,
        batch_size=256,
        num_workers=4,
        pin_memory=True,)

    # loader['val'] = torch.utils.data.DataLoader(
    #     dataset_val,
    #     batch_size=512,
    #     num_workers=8,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # loader['test'] = torch.utils.data.DataLoader(
    #     dataset_test,
    #     batch_size=512,
    #     num_workers=8,
    #     pin_memory=True,
    #     drop_last=False,
    # )

    return loader


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    # text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    text_sd = torch.stack([example["text_embeddings_sd"] for example in data], dim=0)
    text_clip = torch.stack([example["text_embeddings_clip"] for example in data], dim=0)
    
    return {
        "images": images,
        # "text_input_ids": text_input_ids,
        "text_embeddings_sd": text_sd,
        "text_embeddings_clip": text_clip,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
    }


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # state_dict = torch.load(ckpt_path, map_location="cpu")
        if os.path.splitext(ckpt_path)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="_",
        help="Path to resume path for accelerator. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--train_type",
        type=str,
        default="full",
        help="which components to train. clip / ip-adapter / full",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="",
        required=True,
        help="dataset name e.g. mscoco, imagenet1k",
    )
    parser.add_argument(
        "--train_modality",
        type=str,
        default="",
        required=True,
        help="modality to train on e.g. image-text or image-only",
    )
    parser.add_argument(
        "--timestep",
        type=str,
        default="all",
        required=True,
        help="time step range to train from. all means random from 0 - 1000. e.g. 400-600"
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=("The resolution for input images"),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--gradient_accum_step", type=int, default=4, help="gradient accumulation step size."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=("Save a checkpoint of the training state every X updates"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--eval_epoch", type=int, default=1, help="performs 1-NN accuracy evaluation every n epochs")
    parser.add_argument("--clip_loss_ratio", type=float, default=0.999)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    if args.resume_path != "_":
        args.output_dir = "/".join(args.resume_path.split("/")[:-1])
    else:
        args.output_dir += f"/{args.train_type}_{args.train_modality}_{args.dataset_type}_timestep-{
            args.timestep}_clip-loss-ratio-{args.clip_loss_ratio}_{datetime.now().strftime("%m-%d_%H:%M")}"
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ipg_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ipg_handler],
        gradient_accumulation_steps=args.gradient_accum_step,
    )

    # hps = {"learning_rate": args.learning_rate}
    accelerator.init_trackers(f"{datetime.now().strftime("%m-%d_%H:%M")}")

    if accelerator.is_main_process:
        print(f"Logging to {args.output_dir}")
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    # text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    # text_encoder.requires_grad_(False)
    # image_encoder.requires_grad_(False)
    # image_encoder.requires_grad_(True)
    # text_encoder_with_projection = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    # text_encoder_with_projection.requires_grad_(False)
    # clip_model.requires_grad_(False)
    # clip_model.vision_model.requires_grad_(True)
    # clip_model.visual_projection.requires_grad_(True)
    # clip_model.logit_scale.requires_grad_(True)
    # image_encoder.requires_grad_(True)
    del clip_model.text_model
    del clip_model.text_projection
    clip_model.requires_grad_(True)

    # dataloader
    train_dataset = MyDataset(
        args.data_json_file, tokenizer=None, size=args.resolution, image_root_path=args.data_root_path, dataset_type=args.dataset_type,
    )
    # print("dataset length: ", train_dataset.__len__())
    # 118287 -> 4920 steps per epoch (for mscoco) (batch = 8x3=24)
    # 1281167 -> 53382 steps per epoch (for IN1K)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size * args.gradient_accum_step,
        num_workers=args.dataloader_num_workers,
    )

    eval_loader = get_loader(args)

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    # image_encoder.to(accelerator.device, dtype=weight_dtype)
    # image_encoder.to(accelerator.device)  # let amp set dtype
    # text_encoder_with_projection.to(accelerator.device)  # let amp set dtype
    # clip_model.to(accelerator.device)
    # clip_model.vision_model.to(accelerator.device)
    # clip_model.visual_projection.to(accelerator.device)
    # image_encoder.to(accelerator.device)

    # ip-adapter
    # Linear & LN Layer
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        # clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_embeddings_dim=clip_model.config.projection_dim,
        clip_extra_context_tokens=4,
    )
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)

    if args.train_type == "only-clip":
        ip_adapter.requires_grad_(False)
        # params_to_opt = itertools.chain(image_encoder.parameters())
        params_to_opt = itertools.chain(clip_model.vision_model.parameters(), clip_model.visual_projection.parameters(),clip_model.logit_scale)
        # params_to_opt = itertools.chain(image_encoder.parameters())
    elif args.train_type == "full":
        params_to_opt = itertools.chain(
            ip_adapter.image_proj_model.parameters(), ip_adapter.adapter_modules.parameters(),
            # image_encoder.parameters(),
            clip_model.vision_model.parameters(), clip_model.visual_projection.parameters(),[clip_model.logit_scale]
            # clip_model.vision_model.parameters(), clip_model.visual_projection.parameters()
        )
    # optimizer
    # params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(), ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Prepare everything with our `accelerator`.
    # ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    # image_encoder, ip_adapter, optimizer, train_dataloader, eval_loader = accelerator.prepare(
    #     image_encoder, ip_adapter, optimizer, train_dataloader, eval_loader
    # )
    # print(next(clip_model.text_model.parameters()).device)
    
    clip_model, ip_adapter, optimizer, train_dataloader, eval_loader = accelerator.prepare(
        clip_model, ip_adapter, optimizer, train_dataloader,eval_loader
    )
    
    
    # print(next(clip_model.module.text_model.parameters()).device)
    # print(next(ip_adapter.parameters()).device)
    # print("---------")
    # clip_model.module.text_model.to('cpu')
    # clip_model.module.text_projection.to('cpu')
    # del clip_model.module.text_model
    # del clip_model.module.text_projection
    # torch.cuda.empty_cache()
    # accelerator.print("0",torch.cuda.max_memory_reserved())
    # for name, param in clip_model.named_parameters():
        # if param.requires_grad:
        # accelerator.print(name, param.data.device)
    # accelerator.print(torch.cuda.max_memory_reserved())
    # while True: pass

    global_step = 0
    start_epoch = 0
    if args.resume_path != "_":
        print("Loading resume ckpt")
        accelerator.load_state(args.resume_path)
        print(f"accelerator loaded train state from: {args.resume_path}")
        # global_step = int(args.resume_path.split("/")[-1].replace("checkpoint-",""))

        tmp = args.resume_path.split("/")[-1].split("_")
        global_step = int(tmp[1].replace("-step", ""))
        start_epoch = int(tmp[2].replace("-epoch", ""))

    begin = time.perf_counter()
    start_time = time.time()
    # accelerator.print(f"Batch size per gradient update: {args.gradient_accum_step * args.train_batch_size}")
    # accelerator.print("1",torch.cuda.max_memory_reserved())
    for epoch in range(start_epoch, args.num_train_epochs):
        # begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            # load_data_time = time.perf_counter() - begin
            # with accelerator.accumulate(image_encoder,ip_adapter):
            text_embeds = batch["text_embeddings_clip"] / batch["text_embeddings_clip"].norm(p=2, dim=-1, keepdim=True)
            for i in range(args.gradient_accum_step):
                idx_start = args.train_batch_size *i
                idx_end = args.train_batch_size * (i+1)
                minibatch_images = batch["images"][idx_start:idx_end]
                minibatch_clip_images = batch["clip_images"][idx_start:idx_end]
                minibatch_drop_image_embeds = batch["drop_image_embeds"][idx_start:idx_end]
                minibatch_text_embeddings_sd = batch["text_embeddings_sd"][idx_start:idx_end]

                # accelerator.print("2",torch.cuda.max_memory_reserved())
                with accelerator.accumulate(clip_model, ip_adapter):
                    # Convert images to latent space
                    with torch.no_grad():
                        latents = vae.encode(
                            # batch["images"].to(accelerator.device, dtype=weight_dtype)
                            minibatch_images.to(accelerator.device, dtype=weight_dtype)
                        ).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
                    # accelerator.print("3",torch.cuda.max_memory_reserved())

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    if args.timestep == "all":
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                                (bsz,), device=latents.device)
                    else:
                        s = int(args.timestep.split("-")[0])
                        e = int(args.timestep.split("-")[1])
                        timesteps = torch.randint(s, e, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    mean_timestep = torch.mean(timesteps, dtype=float).item()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # with torch.no_grad():
                    #     image_embeds = image_encoder(
                    #         batch["clip_images"].to(accelerator.device, dtype=weight_dtype)
                    #     ).image_embeds
                    # image_embeds = image_encoder(
                    #     batch["clip_images"].to(accelerator.device, dtype=weight_dtype)
                    # ).image_embeds
                    # accelerator.print("4",torch.cuda.max_memory_reserved())
                    image_embeds = clip_model.module.visual_projection(clip_model.module.vision_model(
                        # batch['clip_images'].to(accelerator.device, dtype=weight_dtype)).pooler_output)
                        minibatch_clip_images.to(accelerator.device, dtype=weight_dtype)).pooler_output)
                    image_embeds_ = []
                    # for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    for image_embed, drop_image_embed in zip(image_embeds, minibatch_drop_image_embeds):
                        if drop_image_embed == 1:
                            image_embeds_.append(torch.zeros_like(image_embed))
                        else:
                            image_embeds_.append(image_embed)
                    image_embeds_dropped = torch.stack(image_embeds_)
                    # accelerator.print("5",torch.cuda.max_memory_reserved())

                    # with torch.no_grad():
                    #     if args.train_modality == "image-only":
                    #         text = [""]*len(batch['text_input_ids'])
                    #         text_inputs = tokenizer(
                    #             text,
                    #             max_length=tokenizer.model_max_length,
                    #             padding="max_length",
                    #             truncation=True,
                    #             return_tensors="pt",
                    #         ).input_ids
                    #     elif args.train_modality == "image-text":
                    #         text_inputs = batch["text_input_ids"]
                    #     encoder_hidden_states = text_encoder(text_inputs.to(accelerator.device))[0]
                    #     text_embeds_for_clip_loss = clip_model.module.text_projection(
                    #         clip_model.module.text_model(text_inputs)[1])

                    # accelerator.print(batch['text_embeddings_sd'])
                    # accelerator.print(batch['text_embeddings_sd'].shape)
                    # noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds_dropped)
                    # noise_pred = ip_adapter(noisy_latents, timesteps, batch['text_embeddings_sd'], image_embeds_dropped)
                    noise_pred = ip_adapter(noisy_latents, timesteps, minibatch_text_embeddings_sd, image_embeds_dropped)
                    # accelerator.print("6",torch.cuda.max_memory_reserved())

                    # print(f"image embeds(shape): {image_embeds.shape}") # torch.Size([8, 1024])
                    # print(f"image embeds: {image_embeds}")
                    # print(f"text embeds(shape): {encoder_hidden_states.shape}") # torch.Size([8, 77, 768])
                    # print(f"text embeds: {encoder_hidden_states}")``
                    
                    # calculate loss
                    mse_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                    
                    # CALCULATE CLIP LOSS
                    # with torch.no_grad():
                    # text_embeds = text_encoder_with_projection(text_inputs.to(accelerator.device)).text_embeds
                    # text_embeds = text_encoder_with_projection(text_inputs.cpu()).text_embeds
                    # normalized features
                    # print("image_embeds:",image_embeds.shape)
                    # print(image_embeds)
                    # image_embeds = accelerator.gather(image_embeds)
                    # image_embeds = idist.all_gather(image_embeds)
                    # output = [torch.empty((bsz, 1024), device=accelerator.device) for _ in range(3)]
                    # image_embeds = torch.distributed.all_gather(output, image_embeds)
                    # print(image_embeds)
                    # print(output)

                    # exit()
                    # text_embeds_for_clip_loss = accelerator.gather(text_embeds_for_clip_loss)
                    # text_embeds_for_clip_loss = batch['text_embeddings_clip']
                    # accelerator.print(text_embeds_for_clip_loss.shape) # torch.Size([28, 1024])
                    # accelerator.print(image_embeds.shape) # torch.Size([7, 1024])
                    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                    # text_embeds = text_embeds_for_clip_loss / text_embeds_for_clip_loss.norm(p=2, dim=-1, keepdim=True)
                    # accelerator.print(clip_model.module.logit_scale.exp())
                    logit_scale = clip_model.module.logit_scale.exp()
                    # logits_per_text = torch.matmul(text_embeds, image_embeds.t())*logit_scale
                    logits_per_image = torch.matmul(image_embeds, text_embeds.t())*logit_scale
                    # logits_per_text = torch.matmul(text_embeds.to(accelerator.device), image_embeds.t())
                    clip_loss = get_clip_loss(logits_per_image)
                    # print(f"clip loss: {clip_loss}")
                    # print(f"mse_loss: {mse_loss}")
                    # print(f"device: {accelerator.device}")

                    # loss = clip_loss+0*mse_loss
                    loss = args.clip_loss_ratio*clip_loss + (1-args.clip_loss_ratio)*mse_loss
                    # loss = (1-args.clip_loss_ratio)*mse_loss
                    # loss = args.clip_loss_ratio*clip_loss

                    # Gather the losses across all processes for logging (if we use distributed training).
                    # avg_loss = accelerator.gather(mse_loss.repeat(args.train_batch_size)).mean().item()

                    # Backpropagate
                    # accelerator.print("7",torch.cuda.max_memory_reserved())
                    accelerator.backward(loss)
                    # accelerator.print("8",torch.cuda.max_memory_reserved())
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # accelerator.print("--------")
                    
                global_step += 1

            accelerator.log({"ip-adapter/step_loss": loss, "ip-adapter/clip_loss": clip_loss, "ip-adapter/mse_loss": mse_loss,
                            "ip-adapter/epoch": epoch, "ip-adapter/mean_timestep": mean_timestep,"ip-adapter/logit_scale":logit_scale}, step=step)
            if accelerator.is_main_process and step % 20 == 0:
                print(
                    "Global_Step {}, Epoch {}, step {}, time: {}, mean_timestep: {}, step_loss: {}, clip_loss: {}, mse_loss: {}, time_passed: {}, logit_scale: {}".format(
                        global_step, epoch, step, time.perf_counter(
                        # ) - begin, mean_timestep, loss, clip_loss, avg_loss, timedelta(seconds=(time.time()-start_time))
                        ) - begin, mean_timestep, loss, clip_loss, mse_loss, timedelta(seconds=(time.time()-start_time)),logit_scale,
                    )
                )
                begin = time.perf_counter()


                # if global_step % args.save_steps == 0:
                #     save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                #     accelerator.save_state(save_path, safe_serialization=False)

                # begin = time.perf_counter()
                # break

        save_path = os.path.join(args.output_dir, f"checkpoint_{global_step}-step_{epoch+1}-epoch")
        accelerator.save_state(save_path, safe_serialization=False)
        accelerator.print(f"checkpoint saved: {global_step}-step_{epoch+1}-epoch")

        # epoch complete. 1-nn eval
        if epoch % args.eval_epoch == 0:
            # acc = evaluate_nn(image_encoder, eval_loader['val'], eval_loader['test'])
            acc = evaluate_nn(clip_model.module.vision_model, eval_loader['val'], eval_loader['test'])
            accelerator.log({"ip-adapter/1nn": acc}, step=step)
            accelerator.print(f"1-NN accuracy: {acc}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
