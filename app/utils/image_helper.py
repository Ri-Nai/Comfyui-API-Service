import base64
import io
from pathlib import Path
from PIL import Image
import os

class ImageHelper:
    @staticmethod
    def encode_image(image_path: str) -> str:
        """将图片文件编码为base64字符串"""
        with Image.open(image_path) as img:
            # 转换为RGB模式（如果是RGBA，去除透明通道）
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            
            # 将图片转换为bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return base64.b64encode(img_byte_arr).decode('utf-8')

    @staticmethod
    def decode_and_save_image(base64_str: str, save_path: str, create_dirs: bool = True) -> str:
        """解码base64字符串并保存为图片文件"""
        try:
            # 确保目录存在
            if create_dirs:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 解码base64数据
            image_data = base64.b64decode(base64_str)
            
            # 使用PIL保存图片
            with Image.open(io.BytesIO(image_data)) as img:
                img.save(save_path)
            
            return save_path
        except Exception as e:  
            raise Exception(f"保存图片失败: {str(e)}")

    @staticmethod
    def get_output_path(filename: str, output_dir: str = "outputs") -> str:
        """生成输出文件路径"""
        # 获取当前文件所在目录
        base_dir = Path(__file__).parent.parent.parent
        
        # 构建输出目录
        output_path = base_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        # 生成文件名
        return str(output_path / filename) 
