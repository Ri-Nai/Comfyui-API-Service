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
            "class_type": (
                self.class_name
                if hasattr(self, "class_name")
                else self.__class__.__name__
            ),
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


@dataclass
class CLIPTextEncodeSDXL(BaseNode):
    width: int
    height: int
    target_width: int
    target_height: int
    clip: Optional[List[int]] = None
    text_g: Optional[str] = None
    text_l: Optional[str] = None
    crop_w: int = 0
    crop_h: int = 0


@dataclass
class WD14TaggerPysssss(BaseNode):
    model: str = "wd-v1-4-moat-tagger-v2"
    threshold: float = 0.35
    character_threshold: float = 0.85
    replace_underscore: bool = False
    trailing_comma: bool = False
    exclude_tags: str = ""
    tags: str = ""
    image: Optional[List[int]] = None
    class_name: str = "WD14Tagger|pysssss"


@dataclass
class TextMultiline(BaseNode):
    text: str
    class_name: str = "Text Multiline"


@dataclass
class TextConcatenate(BaseNode):
    text_a: Optional[str] = None
    text_b: Optional[str] = None
    delimiter: str = ", "
    clean_whitespace: bool = True
    class_name: str = "Text Concatenate"


def create_style_transfer_workflow(
    input_image: str,
    style_prompt: str,
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
    image_width: int = 1024,
    image_height: int = 1024,
) -> Dict[str, Any]:
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

    style_text = create_node(TextMultiline, text=style_prompt)

    negative_text = create_node(TextMultiline, text=negative_prompt)

    tagger = create_node(WD14TaggerPysssss)

    concatenate = create_node(TextConcatenate)
    link(style_text, 0, concatenate, "text_a")
    link(tagger, 0, concatenate, "text_b")

    positive_encode = create_node(
        CLIPTextEncodeSDXL,
        width=image_width,
        height=image_height,
        target_width=image_width,
        target_height=image_height,
    )
    link(concatenate, 0, positive_encode, "text_g")
    link(concatenate, 0, positive_encode, "text_l")
    link(final_output_model, 1, positive_encode, "clip")

    negative_encode = create_node(CLIPTextEncode)
    link(negative_text, 0, negative_encode, "text")
    link(final_output_model, 1, negative_encode, "clip")
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

    vae_encode = create_node(VAEEncode)
    link(load_image, 0, vae_encode, "pixels")
    link(checkpoint_loader, 2, vae_encode, "vae")

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

    vae_decode = create_node(VAEDecode)
    link(ksampler, 0, vae_decode, "samples")
    link(checkpoint_loader, 2, vae_decode, "vae")

    save_image = create_node(SaveImage, filename_prefix=output_prefix)
    link(vae_decode, 0, save_image, "images")

    nodes = {k: v.to_dict() for k, v in nodes.items()}
    return nodes


if __name__ == "__main__":
    nodes = create_style_transfer_workflow(
        input_image="input_image.png",
        style_prompt="style_prompt",
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
        image_width=1024,
        image_height=1024,
    )
    print(json.dumps(nodes, indent=4))
