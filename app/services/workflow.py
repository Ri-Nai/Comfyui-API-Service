import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


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
    setattr(
        to_node, input_name, [str(from_node.id), from_output]
    )  # Ensure ID is string


class BaseWorkflow:
    def __init__(self):
        self.nodes: Dict[str, "BaseNode"] = {}
        self.node_counter: int = 0

    def create_node(self, node_class: type, **kwargs) -> "BaseNode":
        """
        创建一个新节点并添加到工作流中

        Args:
            node_class: 节点类 (e.g., LoadImage, KSampler)
            **kwargs: 节点参数

        Returns:
            创建的节点实例
        """
        self.node_counter += 1
        node_id = str(self.node_counter)
        # Instantiate the node class
        node_instance = node_class(**kwargs)
        # Ensure it's a BaseNode or inherits from it
        if not isinstance(node_instance, BaseNode):
            # This check might be too strict depending on usage, adjust if needed
            raise TypeError(
                f"Node class {node_class.__name__} must inherit from BaseNode"
            )
        setattr(node_instance, "id", node_id)
        self.nodes[node_id] = node_instance
        return node_instance

    def get_nodes_dict(self) -> Dict[str, Any]:
        """
        获取节点字典表示

        Returns:
            包含所有节点配置的字典
        """
        return {k: v.to_dict() for k, v in self.nodes.items()}


@dataclass
class BaseNode:
    id: Optional[str] = field(default=None, init=False)  # Add id field

    def to_dict(self) -> Dict[str, Any]:
        node_dict = {
            "inputs": {k: v for k, v in asdict(self).items() if k != "id"},
            "class_type": (
                self.class_name
                if hasattr(self, "class_name")
                else self.__class__.__name__
            ),
        }
        # Remove None values from inputs for cleaner output
        node_dict["inputs"] = {
            k: v for k, v in node_dict["inputs"].items() if v is not None
        }
        return node_dict


# --- Node Definitions ---


@dataclass
class KSampler(BaseNode):
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float
    model: Optional[List] = None  # Simplified type hint if only ID/index is needed
    positive: Optional[List] = None
    negative: Optional[List] = None
    latent_image: Optional[List] = None


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
    model: Optional[List] = None
    positive: Optional[List] = None
    negative: Optional[List] = None
    latent_image: Optional[List] = None


@dataclass
class CheckpointLoaderSimple(BaseNode):
    ckpt_name: str


@dataclass
class CLIPTextEncode(BaseNode):
    text: Optional[Union[str, List]] = None  # Allow string or link
    clip: Optional[List] = None


@dataclass
class VAEDecode(BaseNode):
    samples: Optional[List] = None
    vae: Optional[List] = None


@dataclass
class SaveImage(BaseNode):
    filename_prefix: str
    images: Optional[List] = None


@dataclass
class LoadImage(BaseNode):
    image: str
    # upload: str = "image" # Removed as it's not used in workflows


@dataclass
class ControlNetLoader(BaseNode):
    control_net_name: str


@dataclass
class ControlNetApplyAdvanced(BaseNode):
    strength: float
    start_percent: float
    end_percent: float
    positive: Optional[List] = None
    negative: Optional[List] = None
    control_net: Optional[List] = None
    vae: Optional[List] = None
    image: Optional[List] = None


@dataclass
class ImageScale(BaseNode):
    # Reordered parameters for consistency
    image: Optional[List] = None
    upscale_method: str = "nearest-exact"
    width: int = 512  # Default values might be useful
    height: int = 512
    crop: str = "disabled"


@dataclass
class VAEEncode(BaseNode):
    pixels: Optional[List] = None
    vae: Optional[List] = None


@dataclass
class EmptyLatentImage(BaseNode):
    width: int = 512
    height: int = 512
    batch_size: int = 1


@dataclass
class LoraLoader(BaseNode):
    lora_name: str
    strength_model: float
    strength_clip: float
    model: Optional[List] = None
    clip: Optional[List] = None


@dataclass
class CLIPTextEncodeSDXL(BaseNode):
    # Non-default arguments first
    width: int
    height: int
    target_width: int
    target_height: int
    # Default arguments follow
    text_g: Optional[Union[str, List]] = None
    text_l: Optional[Union[str, List]] = None
    crop_w: int = 0
    crop_h: int = 0
    clip: Optional[List] = None


@dataclass
class WD14TaggerPysssss(BaseNode):
    image: Optional[List] = None
    model: str = "wd-v1-4-moat-tagger-v2"
    threshold: float = 0.35
    character_threshold: float = 0.85
    replace_underscore: bool = False
    trailing_comma: bool = False
    exclude_tags: str = ""
    # tags: str = "" # This seems to be an output, not input
    class_name: str = "WD14Tagger|pysssss"


@dataclass
class TextMultiline(BaseNode):
    text: str
    class_name: str = "Text Multiline"


@dataclass
class TextConcatenate(BaseNode):
    text_a: Optional[Union[str, List]] = None
    text_b: Optional[Union[str, List]] = None
    delimiter: str = ", "
    clean_whitespace: str = "true"
    class_name: str = "Text Concatenate"


# Note: Duplicate ImageScale definition removed earlier was correct.
# Keep only one definition.


@dataclass
class ImageInvert(BaseNode):
    image: Optional[List] = None


@dataclass
class IPAdapterUnifiedLoader(BaseNode):
    preset: str
    model: Optional[List] = None


@dataclass
class IPAdapterStyleComposition(BaseNode):
    model: Optional[List] = None
    ipadapter: Optional[List] = None
    image_style: Optional[List] = None
    image_composition: Optional[List] = None
    weight_style: float = 1.0
    weight_composition: float = 1.0
    expand_style: bool = False
    combine_embeds: str = "average"
    start_at: float = 0.0
    end_at: float = 1.0
    embeds_scaling: str = "V only"
    class_name: str = "IPAdapterStyleComposition"


# --- Workflow Definitions ---


def create_tagger_workflow(
    input_image: str,
    # Add parameters matching TaggerRequest fields
    tagger_model: str = "wd-v1-4-moat-tagger-v2",
    tagger_general_threshold: float = 0.35,
    tagger_character_threshold: float = 0.85,
    # output_prefix is used by SaveImage, but tagger workflow doesn't save by default
    output_prefix: str = "TaggerOutput",
    **kwargs # Capture any other unused args from the request
) -> Dict[str, Any]:
    """Generates a workflow definition for image tagging using WD14Tagger."""
    workflow = BaseWorkflow()

    load_image = workflow.create_node(LoadImage, image=input_image)
    tagger = workflow.create_node(WD14TaggerPysssss)

    link(load_image, 0, tagger, "image")

    return workflow.get_nodes_dict()


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
    lora_strength: float = 1.0,
    controlnet_strength: float = 0.5,
    controlnet_start_percent: float = 0.0,
    controlnet_end_percent: float = 0.5,
    output_prefix: str = "StyleTransfer",
    image_width: int = 1024,
    image_height: int = 1024,
    use_tagger: bool = False,
    image_scale_method: str = "nearest-exact",
    image_scale_crop: str = "center",
    clip_target_width: Optional[int] = None,
    clip_target_height: Optional[int] = None,
    clip_crop_w: int = 0,
    clip_crop_h: int = 0,
    # Tagger parameters (only used if use_tagger is True)
    tagger_model: str = "wd-v1-4-moat-tagger-v2",
    tagger_general_threshold: float = 0.35,
    tagger_character_threshold: float = 0.85,
    tagger_concat_delimiter: str = ", ",
    tagger_concat_clean_whitespace: str = "true",
) -> Dict[str, Any]:
    """Generates a style transfer workflow with ControlNet, LoRA, and optional tagging."""
    workflow = BaseWorkflow()

    # Default target width/height to image dimensions if not specified
    _clip_target_width = (
        clip_target_width if clip_target_width is not None else image_width
    )
    _clip_target_height = (
        clip_target_height if clip_target_height is not None else image_height
    )

    # --- Define Nodes ---
    checkpoint_loader = workflow.create_node(
        CheckpointLoaderSimple, ckpt_name=checkpoint_name
    )
    load_image = workflow.create_node(LoadImage, image=input_image)

    image_scale = workflow.create_node(
        ImageScale,
        width=image_width,
        height=image_height,
        upscale_method=image_scale_method,
        crop=image_scale_crop,
    )
    image_invert = workflow.create_node(ImageInvert)
    controlnet_loader = workflow.create_node(
        ControlNetLoader, control_net_name=controlnet_name
    )

    positive_text_base = workflow.create_node(TextMultiline, text=positive_prompt)
    negative_text = workflow.create_node(TextMultiline, text=negative_prompt)

    # --- Optional Tagger Branch ---
    if use_tagger:
        tagger = workflow.create_node(
            WD14TaggerPysssss,
            model=tagger_model,
            threshold=tagger_general_threshold,
            character_threshold=tagger_character_threshold,
        )
        concatenate = workflow.create_node(
            TextConcatenate,
            delimiter=tagger_concat_delimiter,
            clean_whitespace=tagger_concat_clean_whitespace,
        )
        link(image_scale, 0, tagger, "image")
        link(tagger, 0, concatenate, "text_a")
        link(positive_text_base, 0, concatenate, "text_b")
        # Use concatenated text as the final positive prompt node
        positive_text_final = concatenate
    else:
        positive_text_final = positive_text_base

    # --- LoRA Chain ---
    current_model = checkpoint_loader
    current_clip = checkpoint_loader  # Assuming initial clip comes from checkpoint[1]

    for lora_name in lora_names:
        lora_loader = workflow.create_node(
            LoraLoader,
            lora_name=lora_name,
            strength_model=lora_strength,
            strength_clip=lora_strength,
        )
        link(current_model, 0, lora_loader, "model")
        link(current_clip, 1, lora_loader, "clip")  # Link CLIP from previous node
        # Update current model and clip for the next iteration or final use
        current_model = lora_loader
        current_clip = lora_loader

    # --- Encoders ---
    positive_encode = workflow.create_node(
        CLIPTextEncodeSDXL,
        width=image_width,
        height=image_height,
        target_width=_clip_target_width,
        target_height=_clip_target_height,
        crop_w=clip_crop_w,
        crop_h=clip_crop_h,
    )
    negative_encode = workflow.create_node(CLIPTextEncode)

    # --- ControlNet ---
    controlnet_apply = workflow.create_node(
        ControlNetApplyAdvanced,
        strength=controlnet_strength,
        start_percent=controlnet_start_percent,
        end_percent=controlnet_end_percent,
    )

    # --- VAE Encode/Decode ---
    vae_encode = workflow.create_node(VAEEncode)
    vae_decode = workflow.create_node(VAEDecode)

    # --- KSampler ---
    ksampler = workflow.create_node(
        KSampler,
        seed=seed if seed is not None else create_node_id(),  # Use random seed if None
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
    )

    # --- Output ---
    save_image = workflow.create_node(SaveImage, filename_prefix=output_prefix)

    # --- Link Nodes ---

    # Image processing links
    link(load_image, 0, image_scale, "image")
    link(image_scale, 0, image_invert, "image")
    link(image_scale, 0, vae_encode, "pixels")

    # VAE links
    link(checkpoint_loader, 2, vae_encode, "vae")
    link(checkpoint_loader, 2, vae_decode, "vae")

    # Connect final CLIP from LoRA chain (or checkpoint if no LoRAs)
    link(current_clip, 1, positive_encode, "clip")
    link(current_clip, 1, negative_encode, "clip")

    # Prompt links
    link(positive_text_final, 0, positive_encode, "text_g")
    link(positive_text_final, 0, positive_encode, "text_l")  # SDXL uses both
    link(negative_text, 0, negative_encode, "text")

    # ControlNet Apply links
    link(positive_encode, 0, controlnet_apply, "positive")
    link(negative_encode, 0, controlnet_apply, "negative")
    link(controlnet_loader, 0, controlnet_apply, "control_net")
    link(image_invert, 0, controlnet_apply, "image")
    link(checkpoint_loader, 2, controlnet_apply, "vae")

    # KSampler links
    link(current_model, 0, ksampler, "model")  # Final model from LoRA chain
    link(controlnet_apply, 0, ksampler, "positive")  # Use conditioning from ControlNet
    link(controlnet_apply, 1, ksampler, "negative")
    link(vae_encode, 0, ksampler, "latent_image")

    # Decode and Save links
    link(ksampler, 0, vae_decode, "samples")
    link(vae_decode, 0, save_image, "images")

    return workflow.get_nodes_dict()


def seed_value(seed: int) -> int:
    if seed == -1:
        return random.randint(0, 1000000)
    else:
        return seed

def create_style_transfer_with_image_workflow(
    input_image: str,
    style_image: str,
    style_prompt: str,
    checkpoint_name: str,
    controlnet_name: str,
    # Optional prompts with defaults
    positive_prompt: str = "masterpiece, best quality, high resolution, ultra-detailed, intricate details",
    negative_prompt: str = "(worst quality, low quality, normal quality:1.4), blurry, noisy, jpeg artifacts, deformed, disfigured",
    # Parameters with defaults matching the JSON
    ipadapter_preset: str = "PLUS (high strength)",
    controlnet_strength: float = 0.3,
    controlnet_start_percent: float = 0.0,
    controlnet_end_percent: float = 0.5,  # Exact value from JSON
    ipadapter_weight_style: float = 1.2,
    ipadapter_weight_composition: float = 1.0,
    ksampler1_seed: int = -1,
    ksampler1_steps: int = 20,
    ksampler1_cfg: float = 2.0,
    ksampler1_sampler_name: str = "dpmpp_3m_sde_gpu",
    ksampler1_scheduler: str = "karras",
    ksampler1_end_at_step: int = 15,
    ksampler2_seed: int = -1,
    ksampler2_steps: int = 20,
    ksampler2_cfg: float = 2.0,
    ksampler2_sampler_name: str = "euler",
    ksampler2_scheduler: str = "normal",
    ksampler2_start_at_step: int = 15,
    output_prefix: str = "StyleTransferWithImage",
    image_width: int = 1024,
    image_height: int = 1024,
    image_scale_method: str = "nearest-exact",  # Added for consistency
    image_scale_crop: str = "center",  # Added for consistency
) -> Dict[str, Any]:
    """Generates a workflow using IPAdapter Style & Composition and ControlNet."""
    workflow = BaseWorkflow()

    # --- Define Nodes ---
    ckpt_loader = workflow.create_node(
        CheckpointLoaderSimple, ckpt_name=checkpoint_name
    )
    input_image_loader = workflow.create_node(LoadImage, image=input_image)
    style_image_loader = workflow.create_node(LoadImage, image=style_image)
    controlnet_loader = workflow.create_node(
        ControlNetLoader, control_net_name=controlnet_name
    )
    ipadapter_loader = workflow.create_node(
        IPAdapterUnifiedLoader, preset=ipadapter_preset
    )

    # Prompts - create nodes for all three
    style_prompt_node = workflow.create_node(TextMultiline, text=style_prompt)
    positive_prompt_node = workflow.create_node(TextMultiline, text=positive_prompt)
    negative_prompt_node = workflow.create_node(TextMultiline, text=negative_prompt)
    
    # Concatenate style and positive prompts
    prompt_concatenate = workflow.create_node(TextConcatenate)

    image_scale = workflow.create_node(
        ImageScale,
        width=image_width,
        height=image_height,
        upscale_method=image_scale_method,
        crop=image_scale_crop,
    )
    image_invert = workflow.create_node(ImageInvert)

    # Assuming standard CLIPTextEncode based on JSON (Nodes 6, 7)
    positive_encode = workflow.create_node(CLIPTextEncode)
    negative_encode = workflow.create_node(CLIPTextEncode)
    vae_encode = workflow.create_node(VAEEncode)
    vae_decode = workflow.create_node(VAEDecode)

    ipadapter_apply = workflow.create_node(
        IPAdapterStyleComposition,
        weight_style=ipadapter_weight_style,
        weight_composition=ipadapter_weight_composition,
        # Other IPAdapter params like start_at, end_at could be added if needed
    )
    controlnet_apply = workflow.create_node(
        ControlNetApplyAdvanced,
        strength=controlnet_strength,
        start_percent=controlnet_start_percent,
        end_percent=controlnet_end_percent,
    )

    ksampler1 = workflow.create_node(
        KSamplerAdvanced,
        add_noise="enable",
        noise_seed=seed_value(ksampler1_seed),
        steps=ksampler1_steps,
        cfg=ksampler1_cfg,
        sampler_name=ksampler1_sampler_name,
        scheduler=ksampler1_scheduler,
        start_at_step=0,  # Starts from beginning
        end_at_step=ksampler1_end_at_step,
        return_with_leftover_noise="enable",  # Passes noise to next stage
    )
    ksampler2 = workflow.create_node(
        KSamplerAdvanced,
        add_noise="disable",  # Uses noise from previous stage
        noise_seed=seed_value(ksampler2_seed),
        steps=ksampler2_steps,
        cfg=ksampler2_cfg,
        sampler_name=ksampler2_sampler_name,
        scheduler=ksampler2_scheduler,
        start_at_step=ksampler2_start_at_step,  # Starts where previous ended
        end_at_step=10000,  # Ends at max steps (effectively end)
        return_with_leftover_noise="disable",
    )

    save_image = workflow.create_node(SaveImage, filename_prefix=output_prefix)

    # --- Link Nodes ---

    # Image processing chain
    link(input_image_loader, 0, image_scale, "image")
    link(image_scale, 0, image_invert, "image")
    link(image_scale, 0, vae_encode, "pixels")

    # Connect Checkpoint Loader outputs
    link(ckpt_loader, 0, ipadapter_loader, "model")
    link(ckpt_loader, 1, positive_encode, "clip")
    link(ckpt_loader, 1, negative_encode, "clip")
    link(ckpt_loader, 2, vae_encode, "vae")
    link(ckpt_loader, 2, vae_decode, "vae")

    # Connect Prompt Nodes
    link(style_prompt_node, 0, prompt_concatenate, "text_a")
    link(positive_prompt_node, 0, prompt_concatenate, "text_b")
    link(prompt_concatenate, 0, positive_encode, "text")
    link(negative_prompt_node, 0, negative_encode, "text")

    # Connect ControlNet
    link(positive_encode, 0, controlnet_apply, "positive")
    link(negative_encode, 0, controlnet_apply, "negative")
    link(controlnet_loader, 0, controlnet_apply, "control_net")
    link(image_invert, 0, controlnet_apply, "image")
    link(ckpt_loader, 2, controlnet_apply, "vae")

    # Connect IPAdapter
    link(ipadapter_loader, 0, ipadapter_apply, "model")
    link(ipadapter_loader, 1, ipadapter_apply, "ipadapter")
    link(style_image_loader, 0, ipadapter_apply, "image_style")
    link(image_scale, 0, ipadapter_apply, "image_composition")

    # Connect KSamplers
    # Stage 1
    link(ipadapter_apply, 0, ksampler1, "model")
    link(controlnet_apply, 0, ksampler1, "positive")
    link(controlnet_apply, 1, ksampler1, "negative")
    link(vae_encode, 0, ksampler1, "latent_image")

    # Stage 2 (Refiner)
    link(ipadapter_apply, 0, ksampler2, "model")  # Same model input
    link(controlnet_apply, 0, ksampler2, "positive")  # Same conditioning
    link(controlnet_apply, 1, ksampler2, "negative")
    link(ksampler1, 0, ksampler2, "latent_image")  # Latent from KSampler stage 1

    # Connect VAE Decode and Save Image
    link(ksampler2, 0, vae_decode, "samples")
    link(vae_decode, 0, save_image, "images")

    return workflow.get_nodes_dict()

def create_combined_style_transfer_workflow(
    input_image: str,
    style_prompt: str,
    checkpoint_name: str,
    controlnet_name: str,
    # 可选参数
    positive_prompt: str = "masterpiece, best quality, high resolution, ultra-detailed, intricate details",
    negative_prompt: str = "(worst quality, low quality, normal quality:1.4), blurry, noisy, jpeg artifacts, deformed, disfigured",
    # IPAdapter 参数
    ipadapter_preset: str = "PLUS (high strength)",
    ipadapter_weight_style: float = 1.2,
    ipadapter_weight_composition: float = 1.0,
    # ControlNet 参数
    controlnet_strength: float = 0.8,
    controlnet_start_percent: float = 0.0,
    controlnet_end_percent: float = 0.6,
    # 采样器参数
    style_sampler_seed: int = -1,
    style_sampler_steps: int = 5,
    style_sampler_cfg: float = 1.5,
    style_sampler_name: str = "euler",
    style_sampler_scheduler: str = "normal",
    main_sampler_seed: int = -1,
    main_sampler_steps: int = 10,
    main_sampler_cfg: float = 1.5,
    main_sampler_name: str = "euler",
    main_sampler_scheduler: str = "normal",
    main_sampler_denoise: float = 0.84,
    # 图像参数 
    output_prefix: str = "CombinedStyleTransfer",
    image_width: int = 1024,
    image_height: int = 1024,
    image_scale_method: str = "nearest-exact",
    image_scale_crop: str = "disabled",
    # Tagger参数
    use_tagger: bool = False,
    tagger_model: str = "wd-v1-4-moat-tagger-v2",
    tagger_general_threshold: float = 0.35,
    tagger_character_threshold: float = 0.85,
    tagger_concat_delimiter: str = ", ",
    tagger_concat_clean_whitespace: str = "true",
) -> Dict[str, Any]:
    """
    创建结合了tagger和IPAdapter风格的工作流。
    
    该工作流包含:
    1. 可选的WD14Tagger标签生成
    2. 基于文本提示的风格图像生成
    3. 使用IPAdapter应用风格到输入图像
    4. 使用ControlNet进行图像引导
    
    Args:
        input_image: 输入内容图像路径
        style_prompt: 用于生成风格参考图像的提示词
        checkpoint_name: 模型检查点名称
        controlnet_name: ControlNet模型名称
        positive_prompt: 正向提示词
        negative_prompt: 负向提示词
        ipadapter_preset: IPAdapter预设配置
        ipadapter_weight_style: 风格权重
        ipadapter_weight_composition: 构图权重
        controlnet_strength: ControlNet强度
        controlnet_start_percent: ControlNet开始百分比
        controlnet_end_percent: ControlNet结束百分比
        style_sampler_seed: 风格图像采样器种子
        style_sampler_steps: 风格图像采样器步数
        style_sampler_cfg: 风格图像采样器CFG值
        style_sampler_name: 风格图像采样器名称
        style_sampler_scheduler: 风格图像采样器调度器
        main_sampler_seed: 主采样器种子
        main_sampler_steps: 主采样器步数
        main_sampler_cfg: 主采样器CFG值
        main_sampler_name: 主采样器名称
        main_sampler_scheduler: 主采样器调度器
        main_sampler_denoise: 主采样器去噪强度
        output_prefix: 输出文件前缀
        image_width: 图像宽度
        image_height: 图像高度
        image_scale_method: 图像缩放方法
        image_scale_crop: 图像裁剪方法
        use_tagger: 是否使用标签生成
        tagger_model: 标签模型名称
        tagger_general_threshold: 通用标签阈值
        tagger_character_threshold: 角色标签阈值
        tagger_concat_delimiter: 标签连接分隔符
        tagger_concat_clean_whitespace: 是否清理标签空白
        
    Returns:
        工作流节点字典
    """
    workflow = BaseWorkflow()

    # --- 加载基础模型 ---
    checkpoint_loader = workflow.create_node(
        CheckpointLoaderSimple, ckpt_name=checkpoint_name
    )
    
    # --- 加载输入图像 ---
    input_image_loader = workflow.create_node(LoadImage, image=input_image)
    
    # --- 缩放输入图像 ---
    image_scale = workflow.create_node(
        ImageScale,
        width=image_width,
        height=image_height,
        upscale_method=image_scale_method,
        crop=image_scale_crop,
    )
    
    # --- 提示词节点 ---
    style_prompt_node = workflow.create_node(TextMultiline, text=style_prompt)
    negative_prompt_node = workflow.create_node(TextMultiline, text=negative_prompt)
    positive_prompt_node = workflow.create_node(TextMultiline, text=positive_prompt)
    
    # --- 链接输入图像到缩放 ---
    link(input_image_loader, 0, image_scale, "image")
    
    # --- Tagger分支 (可选) ---
    if use_tagger:
        tagger = workflow.create_node(
            WD14TaggerPysssss,
            model=tagger_model,
            threshold=tagger_general_threshold,
            character_threshold=tagger_character_threshold,
        )
        prompt_concatenate = workflow.create_node(
            TextConcatenate,
            delimiter=tagger_concat_delimiter,
            clean_whitespace=tagger_concat_clean_whitespace,
        )
        link(image_scale, 0, tagger, "image")
        link(tagger, 0, prompt_concatenate, "text_a")
        link(positive_prompt_node, 0, prompt_concatenate, "text_b")
        # 使用连接后的文本作为最终正向提示词
        final_positive_prompt = prompt_concatenate
    else:
        final_positive_prompt = positive_prompt_node
    
    # --- 创建用于风格图像的空Latent ---
    empty_latent = workflow.create_node(
        EmptyLatentImage,
        width=image_width,
        height=image_height,
        batch_size=1
    )
    
    # --- 风格图像的CLIP编码 ---
    style_positive_encode = workflow.create_node(CLIPTextEncode)
    style_negative_encode = workflow.create_node(CLIPTextEncode)
    
    # --- 链接CLIP ---
    link(checkpoint_loader, 1, style_positive_encode, "clip")
    link(checkpoint_loader, 1, style_negative_encode, "clip")
    link(style_prompt_node, 0, style_positive_encode, "text")
    link(negative_prompt_node, 0, style_negative_encode, "text")
    
    # --- 风格图像的KSampler ---
    style_ksampler = workflow.create_node(
        KSampler,
        seed=seed_value(style_sampler_seed),
        steps=style_sampler_steps,
        cfg=style_sampler_cfg,
        sampler_name=style_sampler_name,
        scheduler=style_sampler_scheduler,
        denoise=1.0,  # 完全去噪以生成风格图像
    )
    
    # --- 链接风格图像KSampler ---
    link(checkpoint_loader, 0, style_ksampler, "model")
    link(style_positive_encode, 0, style_ksampler, "positive")
    link(style_negative_encode, 0, style_ksampler, "negative")
    link(empty_latent, 0, style_ksampler, "latent_image")
    
    # --- 风格图像的VAE解码 ---
    style_vae_decode = workflow.create_node(VAEDecode)
    link(style_ksampler, 0, style_vae_decode, "samples")
    link(checkpoint_loader, 2, style_vae_decode, "vae")
    
    # --- IPAdapter加载和应用 ---
    ipadapter_loader = workflow.create_node(
        IPAdapterUnifiedLoader, preset=ipadapter_preset
    )
    link(checkpoint_loader, 0, ipadapter_loader, "model")
    
    ipadapter_apply = workflow.create_node(
        IPAdapterStyleComposition,
        weight_style=ipadapter_weight_style,
        weight_composition=ipadapter_weight_composition,
        expand_style=False,
        combine_embeds="average",
        start_at=0.0,
        end_at=1.0,
        embeds_scaling="V only",
    )
    
    # --- 链接IPAdapter ---
    link(ipadapter_loader, 0, ipadapter_apply, "model")
    link(ipadapter_loader, 1, ipadapter_apply, "ipadapter")
    link(style_vae_decode, 0, ipadapter_apply, "image_style")
    link(image_scale, 0, ipadapter_apply, "image_composition")
    
    # --- ControlNet加载和应用 ---
    controlnet_loader = workflow.create_node(
        ControlNetLoader, control_net_name=controlnet_name
    )
    
    # 主要图像处理的CLIP编码
    main_positive_encode = workflow.create_node(CLIPTextEncode)
    main_negative_encode = workflow.create_node(CLIPTextEncode)
    
    # 链接CLIP
    link(checkpoint_loader, 1, main_positive_encode, "clip")
    link(checkpoint_loader, 1, main_negative_encode, "clip")
    link(final_positive_prompt, 0, main_positive_encode, "text")
    link(negative_prompt_node, 0, main_negative_encode, "text")
    
    # ControlNet应用
    controlnet_apply = workflow.create_node(
        ControlNetApplyAdvanced,
        strength=controlnet_strength,
        start_percent=controlnet_start_percent,
        end_percent=controlnet_end_percent,
    )
    
    # 链接ControlNet
    link(main_positive_encode, 0, controlnet_apply, "positive")
    link(main_negative_encode, 0, controlnet_apply, "negative")
    link(controlnet_loader, 0, controlnet_apply, "control_net")
    link(image_scale, 0, controlnet_apply, "image")
    link(checkpoint_loader, 2, controlnet_apply, "vae")
    
    # --- 输入图像的VAE编码 ---
    vae_encode = workflow.create_node(VAEEncode)
    link(image_scale, 0, vae_encode, "pixels")
    link(checkpoint_loader, 2, vae_encode, "vae")
    
    # --- 主KSampler ---
    main_ksampler = workflow.create_node(
        KSampler,
        seed=seed_value(main_sampler_seed),
        steps=main_sampler_steps,
        cfg=main_sampler_cfg,
        sampler_name=main_sampler_name,
        scheduler=main_sampler_scheduler,
        denoise=main_sampler_denoise,
    )
    
    # --- 链接主KSampler ---
    link(ipadapter_apply, 0, main_ksampler, "model")
    link(controlnet_apply, 0, main_ksampler, "positive")
    link(controlnet_apply, 1, main_ksampler, "negative")
    link(vae_encode, 0, main_ksampler, "latent_image")
    
    # --- 最终VAE解码 ---
    vae_decode = workflow.create_node(VAEDecode)
    link(main_ksampler, 0, vae_decode, "samples")
    link(checkpoint_loader, 2, vae_decode, "vae")
    
    # --- 保存图像 ---
    save_image = workflow.create_node(SaveImage, filename_prefix=output_prefix)
    link(vae_decode, 0, save_image, "images")
    
    return workflow.get_nodes_dict()


if __name__ == "__main__":
    # Example usage for create_style_transfer_workflow
    with open("../../tmp_style_transfer_nodes.json", "w") as f:
        style_transfer_nodes = create_style_transfer_workflow(
            input_image="input_image.png",
            positive_prompt="beautiful scenery, masterpiece",
            negative_prompt="ugly, deformed",
            checkpoint_name="sd_xl_base_1.0.safetensors",
            controlnet_name="control-lora-canny-rank128.safetensors",
            lora_names=["add_detail.safetensors"],
            seed=-1,
            output_prefix="StyleTransferOutput",
            use_tagger=True,  # Example with tagger
        )
        f.write(json.dumps(style_transfer_nodes, indent=2))
    with open("../../tmp_style_image_nodes.json", "w") as f:
        style_image_nodes = create_style_transfer_with_image_workflow(
            input_image="content_image.png",
            style_image="style_ref.png",
            style_prompt="in the style of Van Gogh, starry night",
            checkpoint_name="realvisxlV50_v50LightningBakedvae.safetensors",
            controlnet_name="controlnet-sd-xl-1.0-softedge-dexined.safetensors",
            output_prefix="StyleImageOutput",
        )
        f.write(json.dumps(style_image_nodes, indent=2))  # Print this one

    # Example usage for combined style transfer workflow
    with open("../../tmp_combined_style_nodes.json", "w") as f:
        combined_nodes = create_combined_style_transfer_workflow(
            input_image="content_image.png",
            style_prompt="vangogh's starry night",
            checkpoint_name="realvisxlV50_v50LightningBakedvae.safetensors",
            controlnet_name="controlnet-sd-xl-1.0-softedge-dexined.safetensors",
            positive_prompt="1 girl, cute, pink hair, smiling",
            negative_prompt="blurry, noisy, messy, lowres, jpeg, artifacts, ill, distorted, malformed",
            use_tagger=True,  # Enable tagger to analyze the input image
            output_prefix="CombinedStyleTransfer",
        )
        f.write(json.dumps(combined_nodes, indent=2))
