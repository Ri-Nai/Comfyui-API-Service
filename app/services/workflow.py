import json
from pathlib import Path
import random
from typing import Dict, Any, List, Tuple, Optional, Union


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


class BaseNode:
    def __init__(self):
        self.id: Optional[str] = None

    def set_id(self, id: str):
        self.id: str = id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": {k: v for k, v in self.__dict__.items() if k != "id"},
            "class_type": self.__class__.__name__,
        }


class KSampler(BaseNode):
    def __init__(
        self,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        model: Optional[List[int]] = None,
        positive: Optional[List[int]] = None,
        negative: Optional[List[int]] = None,
        latent_image: Optional[List[int]] = None,
    ):
        super().__init__()
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise
        self.model = model
        self.positive = positive
        self.negative = negative
        self.latent_image = latent_image


class CheckpointLoaderSimple(BaseNode):
    def __init__(self, ckpt_name: str):
        super().__init__()
        self.ckpt_name = ckpt_name


class CLIPTextEncode(BaseNode):
    def __init__(self, clip: Optional[List[int]] = None, text: Optional[str] = None):
        super().__init__()
        self.clip = clip
        self.text = text


class VAEDecode(BaseNode):
    def __init__(
        self, samples: Optional[List[int]] = None, vae: Optional[List[int]] = None
    ):
        super().__init__()
        self.samples = samples
        self.vae = vae


class SaveImage(BaseNode):
    def __init__(self, filename_prefix: str, images: Optional[List[int]] = None):
        super().__init__()
        self.filename_prefix = filename_prefix
        self.images = images


class LoadImage(BaseNode):
    def __init__(self, image: str, upload: str = "image"):
        super().__init__()
        self.image = image
        self.upload = upload


class ControlNetLoader(BaseNode):
    def __init__(self, control_net_name: str):
        super().__init__()
        self.control_net_name = control_net_name


class ControlNetApplySD3(BaseNode):
    def __init__(
        self,
        strength: float,
        start_percent: float,
        end_percent: float,
        positive: Optional[List[int]] = None,
        negative: Optional[List[int]] = None,
        control_net: Optional[List[int]] = None,
        vae: Optional[List[int]] = None,
        image: Optional[List[int]] = None,
    ):
        super().__init__()
        self.strength = strength
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.positive = positive
        self.negative = negative
        self.control_net = control_net
        self.vae = vae
        self.image = image


class ImageScale(BaseNode):
    def __init__(
        self,
        upscale_method: str,
        width: int,
        height: int,
        crop: str,
        image: Optional[List[int]] = None,
    ):
        super().__init__()
        self.upscale_method = upscale_method
        self.width = width
        self.height = height
        self.crop = crop
        self.image = image


class VAEEncode(BaseNode):
    def __init__(
        self, pixels: Optional[List[int]] = None, vae: Optional[List[int]] = None
    ):
        super().__init__()
        self.pixels = pixels
        self.vae = vae


class EmptyLatentImage(BaseNode):
    def __init__(self, width: int, height: int, batch_size: int):
        super().__init__()
        self.width = width
        self.height = height
        self.batch_size = batch_size


class LoraLoader(BaseNode):
    def __init__(
        self,
        lora_name: str,
        strength_model: float,
        strength_clip: float,
        model: Optional[List[int]] = None,
        clip: Optional[List[int]] = None,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.strength_model = strength_model
        self.strength_clip = strength_clip
        self.model = model
        self.clip = clip


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
    node_counter = 1

    # 创建各个节点
    checkpoint_loader = CheckpointLoaderSimple(checkpoint_name)
    checkpoint_loader.set_id(str(node_counter))
    nodes[str(node_counter)] = checkpoint_loader.to_dict()
    node_counter += 1

    load_image = LoadImage(input_image)
    load_image.set_id(str(node_counter))
    nodes[str(node_counter)] = load_image.to_dict()
    node_counter += 1

    controlnet = ControlNetLoader(controlnet_name)
    controlnet.set_id(str(node_counter))
    nodes[str(node_counter)] = controlnet.to_dict()
    node_counter += 1

    final_output_model = checkpoint_loader

    for lora_name in lora_names:
        # 添加LoRA
        lora = LoraLoader(lora_name, lora_strength, lora_strength)
        lora.set_id(str(node_counter))
        link(final_output_model, 0, lora, "model")
        link(final_output_model, 1, lora, "clip")
        nodes[str(node_counter)] = lora.to_dict()
        node_counter += 1
        final_output_model = lora

    # CLIP文本编码
    positive_encode = CLIPTextEncode(text=positive_prompt)
    positive_encode.set_id(str(node_counter))
    link(final_output_model, 1, positive_encode, "clip")
    nodes[str(node_counter)] = positive_encode.to_dict()
    node_counter += 1

    # CLIP文本编码
    negative_encode = CLIPTextEncode(text=negative_prompt)
    negative_encode.set_id(str(node_counter))
    link(final_output_model, 1, negative_encode, "clip")
    nodes[str(node_counter)] = negative_encode.to_dict()
    node_counter += 1

    # 图像处理
    image_scale = ImageScale("nearest-exact", image_width, image_height, "disabled")
    image_scale.set_id(str(node_counter))
    link(load_image, 0, image_scale, "image")
    nodes[str(node_counter)] = image_scale.to_dict()
    node_counter += 1

    vae_encode = VAEEncode()
    vae_encode.set_id(str(node_counter))
    link(image_scale, 0, vae_encode, "pixels")
    link(checkpoint_loader, 2, vae_encode, "vae")
    nodes[str(node_counter)] = vae_encode.to_dict()
    node_counter += 1

    controlnet_apply = ControlNetApplySD3(
        controlnet_strength,
        controlnet_start_percent,
        controlnet_end_percent,
    )
    controlnet_apply.set_id(str(node_counter))
    link(positive_encode, 0, controlnet_apply, "positive")
    link(negative_encode, 0, controlnet_apply, "negative")
    link(controlnet, 0, controlnet_apply, "control_net")
    link(checkpoint_loader, 2, controlnet_apply, "vae")
    link(load_image, 0, controlnet_apply, "image")
    nodes[str(node_counter)] = controlnet_apply.to_dict()
    node_counter += 1

    # KSampler
    if seed is None:
        seed = random.randint(1, 1000000000000000)

    ksampler = KSampler(
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
    )
    ksampler.set_id(str(node_counter))
    link(final_output_model, 0, ksampler, "model")
    link(controlnet_apply, 0, ksampler, "positive")
    link(controlnet_apply, 1, ksampler, "negative")
    link(vae_encode, 0, ksampler, "latent_image")
    nodes[str(node_counter)] = ksampler.to_dict()
    node_counter += 1

    # VAE解码
    vae_decode = VAEDecode()
    vae_decode.set_id(str(node_counter))
    link(ksampler, 0, vae_decode, "samples")
    link(checkpoint_loader, 2, vae_decode, "vae")
    nodes[str(node_counter)] = vae_decode.to_dict()
    node_counter += 1

    # 保存图像
    save_image = SaveImage(output_prefix)
    save_image.set_id(str(node_counter))
    link(vae_decode, 0, save_image, "images")
    nodes[str(node_counter)] = save_image.to_dict()

    return nodes
