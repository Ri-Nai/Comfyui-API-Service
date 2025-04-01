from typing import List, Optional
from pydantic import BaseModel

class WorkflowRequest(BaseModel):
    input_image: str  # base64编码的图片数据
    positive_prompt: str
    negative_prompt: str
    checkpoint_name: str
    controlnet_name: str
    lora_names: List[str]
    seed: Optional[int] = None
    steps: Optional[int] = 20
    cfg: Optional[float] = 6.5
    sampler_name: Optional[str] = "euler"
    scheduler: Optional[str] = "normal"
    denoise: Optional[float] = 0.75
    lora_strength: Optional[float] = 1.0
    controlnet_strength: Optional[float] = 0.85
    controlnet_start_percent: Optional[float] = 0
    controlnet_end_percent: Optional[float] = 1
    output_prefix: Optional[str] = "StyleTransfer" 
    workflow_name: Optional[str] = None
