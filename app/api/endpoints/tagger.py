import logging
from fastapi import APIRouter, HTTPException
from app.services.comfyui import ComfyUIService
from app.services.workflow import create_tagger_workflow
from app.models.requests import WorkflowRequest
from app.utils.image_helper import ImageHelper

logger = logging.getLogger(__name__)

router = APIRouter()
comfyui_service = ComfyUIService()

@router.post("/tagger")
async def tagger(request: WorkflowRequest):
    try:
        logger.info("开始进行标签提取...")
        # 解码base64图片数据
        try:
            img_byte_arr, filename = ImageHelper.decode_image(request.input_image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

        # 上传图片到ComfyUI
        logger.info("正在上传图片到ComfyUI...")
        uploaded_filename = await comfyui_service.upload_image(img_byte_arr, filename)
        logger.info(f"图片上传成功: {uploaded_filename}")

        workflow = create_tagger_workflow(uploaded_filename)
        prompt_id = await comfyui_service.queue_prompt(workflow)
        await comfyui_service.wait_for_prompt(prompt_id)

        outputs = await comfyui_service.get_outputs(prompt_id)
        return {
            "status": "success",
            "message": "标签提取成功",
            "tags": outputs[0]["tags"][0]
        }
    except Exception as e:
        logger.error(f"标签提取失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
