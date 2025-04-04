import json
from pathlib import Path
import random
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict


def create_node_id() -> int:
    return random.randint(1, 100)


def link(
    from_node: "BaseNode", from_output: int, to_node: "BaseNode", input_name: str
) -> None:
    """
    连接两个节点

    Args:
        from_node: 源节点
        from_output: 源节点的输出索引
        to_node: 目标节点
        input_name: 目标节点的输入参数名
    """
    if not hasattr(from_node, "id") or not hasattr(to_node, "id"):
        raise ValueError("节点必须有id属性")
    setattr(to_node, input_name, [from_node.id, from_output])


@dataclass
class BaseNode:
    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": {k: v for k, v in asdict(self).items() if k != "id"},
            "class_type": self.__class__.__name__,
        }


@dataclass
class KSampler(BaseNode):
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float
    model: Optional[List[int]] = None
    positive: Optional[List[int]] = None
    negative: Optional[List[int]] = None
    latent_image: Optional[List[int]] = None


@dataclass
class KSamplerAdvanced(BaseNode):
    add_noise: str
    noise_seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    start_at_step: int
    end_at_step: int
    return_with_leftover_noise: str
    model: Optional[List[int]] = None
    positive: Optional[List[int]] = None
    negative: Optional[List[int]] = None
    latent_image: Optional[List[int]] = None


@dataclass
class CheckpointLoaderSimple(BaseNode):
    ckpt_name: str


@dataclass
class CLIPTextEncode(BaseNode):
    clip: Optional[List[int]] = None
    text: Optional[str] = None


@dataclass
class VAEDecode(BaseNode):
    samples: Optional[List[int]] = None
    vae: Optional[List[int]] = None


@dataclass
class SaveImage(BaseNode):
    filename_prefix: str
    images: Optional[List[int]] = None


@dataclass
class LoadImage(BaseNode):
    image: str
    upload: str = "image"


@dataclass
class ControlNetLoader(BaseNode):
    control_net_name: str


@dataclass
class ControlNetApplySD3(BaseNode):
    strength: float
    start_percent: float
    end_percent: float
    positive: Optional[List[int]] = None
    negative: Optional[List[int]] = None
    control_net: Optional[List[int]] = None
    vae: Optional[List[int]] = None
    image: Optional[List[int]] = None


@dataclass
class ImageScale(BaseNode):
    upscale_method: str
    width: int
    height: int
    crop: str
    image: Optional[List[int]] = None


@dataclass
class VAEEncode(BaseNode):
    pixels: Optional[List[int]] = None
    vae: Optional[List[int]] = None


@dataclass
class EmptyLatentImage(BaseNode):
    width: int
    height: int
    batch_size: int


@dataclass
class LoraLoader(BaseNode):
    lora_name: str
    strength_model: float
    strength_clip: float
    model: Optional[List[int]] = None
    clip: Optional[List[int]] = None


def create_style_transfer_workflow(
    input_image: str,
    positive_prompt: str,
    negative_prompt: str,
    checkpoint_name: str,
    controlnet_name: str,
    lora_names: List[str],
    seed: Optional[int] = None,
    steps: int = 20,
    cfg: float = 6.5,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    denoise: float = 0.75,
    lora_strength: float = 1,
    controlnet_strength: float = 0.85,
    controlnet_start_percent: float = 0,
    controlnet_end_percent: float = 1,
    output_prefix: str = "StyleTransfer",
    image_width: int = 512,
    image_height: int = 512,
) -> Dict[str, Any]:
    """
    创建一个风格迁移的workflow配置

    Args:
        input_image: 输入图像的文件名
        positive_prompt: 正向提示词
        negative_prompt: 负向提示词
        seed: 随机种子，如果为None则随机生成
        output_prefix: 输出文件名前缀

    Returns:
        Dict[str, Any]: workflow配置的JSON字典
    """
    nodes = {}
    node_counter = 0

    def create_node(func, **kwargs):
        nonlocal node_counter
        node_counter += 1
        node = func(**kwargs)
        setattr(node, "id", str(node_counter))
        nodes[str(node_counter)] = node
        return node

    # 创建各个节点
    checkpoint_loader = create_node(CheckpointLoaderSimple, ckpt_name=checkpoint_name)
    load_image = create_node(LoadImage, image=input_image)
    controlnet = create_node(ControlNetLoader, control_net_name=controlnet_name)

    final_output_model = checkpoint_loader

    for lora_name in lora_names:
        # 添加LoRA
        lora = create_node(
            LoraLoader,
            lora_name=lora_name,
            strength_model=lora_strength,
            strength_clip=lora_strength,
        )
        link(final_output_model, 0, lora, "model")
        link(final_output_model, 1, lora, "clip")
        final_output_model = lora

    # CLIP文本编码
    positive_encode = create_node(CLIPTextEncode, text=positive_prompt)
    link(final_output_model, 1, positive_encode, "clip")

    # CLIP文本编码
    negative_encode = create_node(CLIPTextEncode, text=negative_prompt)
    link(final_output_model, 1, negative_encode, "clip")

    # 图像处理
    image_scale = create_node(
        ImageScale,
        upscale_method="nearest-exact",
        width=image_width,
        height=image_height,
        crop="disabled",
    )
    link(load_image, 0, image_scale, "image")

    vae_encode = create_node(VAEEncode)
    link(image_scale, 0, vae_encode, "pixels")
    link(checkpoint_loader, 2, vae_encode, "vae")

    controlnet_apply = create_node(
        ControlNetApplySD3,
        strength=controlnet_strength,
        start_percent=controlnet_start_percent,
        end_percent=controlnet_end_percent,
    )
    link(positive_encode, 0, controlnet_apply, "positive")
    link(negative_encode, 0, controlnet_apply, "negative")
    link(controlnet, 0, controlnet_apply, "control_net")
    link(checkpoint_loader, 2, controlnet_apply, "vae")
    link(load_image, 0, controlnet_apply, "image")

    # KSampler
    if seed is None:
        seed = random.randint(1, 1000000000000000)

    ksampler = create_node(
        KSampler,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
    )
    link(final_output_model, 0, ksampler, "model")
    link(controlnet_apply, 0, ksampler, "positive")
    link(controlnet_apply, 1, ksampler, "negative")
    link(vae_encode, 0, ksampler, "latent_image")

    # VAE解码
    vae_decode = create_node(VAEDecode)
    link(ksampler, 0, vae_decode, "samples")
    link(checkpoint_loader, 2, vae_decode, "vae")

    # 保存图像
    save_image = create_node(SaveImage, filename_prefix=output_prefix)
    link(vae_decode, 0, save_image, "images")

    nodes = {k: v.to_dict() for k, v in nodes.items()}
    return nodes


if __name__ == "__main__":
    nodes = create_style_transfer_workflow(
        input_image="input_image.png",
        positive_prompt="positive_prompt",
        negative_prompt="negative_prompt",
        checkpoint_name="checkpoint_name",
        controlnet_name="controlnet_name",
        lora_names=["lora_name1", "lora_name2"],
        seed=123456,
        steps=20,
        cfg=6.5,
        sampler_name="euler",
        scheduler="normal",
        denoise=0.75,
        lora_strength=1,
        controlnet_strength=0.85,
        controlnet_start_percent=0,
        controlnet_end_percent=1,
        output_prefix="StyleTransfer",
        image_width=512,
        image_height=512,
    )
    print(json.dumps(nodes, indent=4))
