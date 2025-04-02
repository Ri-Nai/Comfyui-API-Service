import json
import logging
import uuid
import websockets
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
from app.utils.image_helper import ImageHelper
import io
import os
import base64

logger = logging.getLogger(__name__)

class ComfyUIService:
    def __init__(self, server: str = "127.0.0.1:8188"):
        self.server = server
        self.client_id = str(uuid.uuid4())
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.timeout = 300  # 5分钟超时

    async def ensure_ws_connected(self):
        """确保 WebSocket 连接已建立"""
        if self.ws is None or not self.ws.open:
            self.ws = await websockets.connect(f"ws://{self.server}/ws?clientId={self.client_id}")

    async def close_ws(self):
        """关闭 WebSocket 连接"""
        if self.ws and self.ws.open:
            await self.ws.close()
            self.ws = None


    async def upload_image(self, image_data: bytes, filename: str) -> str:
        """上传图片到ComfyUI服务器"""
        url = f"http://{self.server}/upload/image"
        
        form_data = aiohttp.FormData()
        form_data.add_field(
            "image", image_data, filename=filename, content_type="image/png"
        )
        form_data.add_field("type", "input")
        form_data.add_field("overwrite", "true")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=form_data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to upload image: {error_text}",
                        )
                    return filename
        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to connect to ComfyUI server: {str(e)}"
            )

    async def queue_prompt(self, prompt: Dict[str, Any]) -> dict:
        """向ComfyUI发送提示"""
        url = f"http://{self.server}/prompt"
        payload = {"prompt": prompt, "client_id": self.client_id}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to queue prompt: {error_text}",
                        )
                    return await response.json()
        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to connect to ComfyUI server: {str(e)}"
            )

    async def get_history(self, prompt_id: str) -> dict:
        """获取生成历史"""
        url = f"http://{self.server}/history/{prompt_id}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to get history: {error_text}",
                        )
                    return await response.json()
        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to connect to ComfyUI server: {str(e)}"
            )

    async def get_image(self, filename: str, subfolder: str, type_: str) -> bytes:
        """获取生成的图片"""
        url = f"http://{self.server}/view"
        params = {"filename": filename, "subfolder": subfolder, "type": type_}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to get image: {error_text}",
                        )
                    return await response.read()
        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to connect to ComfyUI server: {str(e)}"
            )

    async def wait_for_prompt(self, prompt_id: str) -> None:
        """等待提示执行完成
        
        Args:
            prompt_id: 提示ID
            
        Raises:
            TimeoutError: 如果等待超过超时时间
            RuntimeError: 如果WebSocket连接出错
        """
        await self.ensure_ws_connected()
        
        try:
            async def _wait_for_completion():
                while True:
                    try:
                        message = await self.ws.recv()
                        if not isinstance(message, str):
                            continue
                            
                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析的WebSocket消息: {message}")
                            continue
                            
                        if not isinstance(data, dict):
                            continue
                            
                        msg_type = data.get("type")
                        if msg_type == "executing":
                            msg_data = data.get("data", {})
                            if not isinstance(msg_data, dict):
                                continue
                                
                            if msg_data.get("prompt_id") == prompt_id:
                                if msg_data.get("node") is None:
                                    return
                        elif msg_type == "crystools.monitor":
                            continue
                        else:
                            logger.info(f"收到消息: {message}")
                            
                    except websockets.WebSocketException as e:
                        raise RuntimeError(f"WebSocket错误: {str(e)}")

            await asyncio.wait_for(_wait_for_completion(), timeout=self.timeout)
                        
        except asyncio.TimeoutError:
            raise TimeoutError(f"等待提示 {prompt_id} 执行超时")
        except Exception as e:
            logger.error(f"等待提示时发生错误: {str(e)}")
            raise

    async def process_output(self, prompt_id: str, history: dict) -> List[Dict[str, str]]:
        """处理输出图像"""
        output_images = []
        if prompt_id in history:
            for node_id, node_output in history[prompt_id]["outputs"].items():
                if "images" in node_output:
                    for image in node_output["images"]:
                        if image["type"] == "output":
                            image_data = await self.get_image(
                                image["filename"], image["subfolder"], image["type"]
                            )
                            
                            # 使用 ImageHelper 保存图片
                            output_path = ImageHelper.get_output_path(
                                filename=image["filename"],
                                output_dir="output"
                            )
                            
                            # 将 bytes 转换为 base64 字符串并保存
                            base64_str = base64.b64encode(image_data).decode("utf-8")
                            saved_path = ImageHelper.decode_and_save_image(base64_str, output_path)
                            
                            output_images.append(
                                {
                                    "filename": image["filename"],
                                    "path": saved_path,
                                    "data": base64_str,
                                }
                            )
        return output_images 
