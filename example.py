import requests
import random
import time
from app.utils.image_helper import ImageHelper

BASE_URL = "http://127.0.0.1:8000" # 定义基础 URL

# --- 1. Example: Style Transfer (Single Image) --- 
url_style_transfer = f"{BASE_URL}/generate/style_transfer"
print(f"调用 Style Transfer API: {url_style_transfer}")

# 读取并编码输入图片
base64_encoded_input_image = ImageHelper.encode_image("image.png")

style_transfer_request_data = {
    "input_image": base64_encoded_input_image,
    "positive_prompt": "masterpiece, best quality, highly detailed, cyberpunk style, red light eyes, mechanic body, laser in eyes, neon city background",
    "negative_prompt": "(worst quality, low quality, normal quality:1.4), blurry, noisy, jpeg artifacts, text, watermark, username, signature, deformed, disfigured, bad anatomy, extra limbs, missing limbs",
    "checkpoint_name": "realvisxlV50_v50LightningBakedvae.safetensors",
    "controlnet_name": "controlnet-sd-xl-1.0-softedge-dexined.safetensors",
    "lora_names": [], # e.g., ["add_detail.safetensors"]
    "seed": random.randint(1, 1000000),
    "steps": 20,
    "cfg": 6.5,
    "sampler_name": "euler",
    "scheduler": "normal",
    "denoise": 0.75, # 控制风格迁移强度
    "lora_strength": 1.0,
    "controlnet_strength": 0.5, # 控制 ControlNet 强度
    "controlnet_start_percent": 0.0,
    "controlnet_end_percent": 0.5,
    "output_prefix": "StyleTransferOutput",
    "use_tagger": False, # 设为 True 以启用 tagger
    "image_width": 1024,
    "image_height": 1024,
    # 如果 use_tagger=True，可以设置以下参数
    # "tagger_model": "wd-v1-4-moat-tagger-v2",
    # "tagger_general_threshold": 0.35,
    # "tagger_character_threshold": 0.85,
}

response = requests.post(url_style_transfer, json=style_transfer_request_data)

try:
    # Raise HTTPError for bad responses (4xx or 5xx)
    response.raise_for_status()
    result = response.json()
    print("Style Transfer Response:", result)

    # Check logical success within the JSON payload
    if result.get("status") == "success" and result.get("images"):
        for img_data in result["images"]:
            try:
                save_path = ImageHelper.get_output_path(img_data["filename"])
                ImageHelper.decode_and_save_image(img_data["data"], save_path)
                print(f"Style Transfer 图片已保存为: {save_path}")
            except Exception as e:
                print(f"保存 Style Transfer 图片失败: {str(e)}")
    else:
        # Handle cases where the API call was successful (200 OK) but the operation failed logically
        print(f"Style Transfer API 返回逻辑错误: {result.get('message', 'No message')} - Detail: {result.get('detail', result)}")

except requests.exceptions.HTTPError as http_err:
    # Handle HTTP errors (4xx, 5xx)
    print(f"Style Transfer HTTP 请求失败: {http_err}")
    try:
        # Try to get more details from the response body if possible
        error_details = response.json()
        print(f"Error details: {error_details.get('detail', response.text)}")
    except requests.exceptions.JSONDecodeError:
        print(f"Response body: {response.text}")
except requests.exceptions.RequestException as req_err:
    # Handle other request errors (connection, timeout, etc.)
    print(f"Style Transfer 请求错误: {req_err}")
except Exception as e:
    # Catch any other unexpected errors during processing
    print(f"处理 Style Transfer 响应时发生意外错误: {e}")

print("\n---\n")

# --- 2. Example: Tagger --- (Commented Out)
# url_tagger = f"{BASE_URL}/generate/tagger"
# print(f"调用 Tagger API: {url_tagger}")
# tagger_request_data = {
#     "input_image": base64_encoded_input_image, # 使用上面编码的同一张图片
# }
# try:
#     response_tagger = requests.post(url_tagger, json=tagger_request_data)
#     response_tagger.raise_for_status() # Check for HTTP errors
#     result_tagger = response_tagger.json()
#     print("Tagger Response:", result_tagger)
#     if result_tagger.get("status") == "success":
#         print("提取到的标签:", result_tagger.get("tags"))
#     else:
#          print(f"Tagger API 返回逻辑错误: {result_tagger.get('message', 'No message')} - Detail: {result_tagger.get('detail', result_tagger)}")
# except requests.exceptions.HTTPError as http_err:
#     print(f"Tagger HTTP 请求失败: {http_err}")
#     try:
#         error_details = response_tagger.json()
#         print(f"Error details: {error_details.get('detail', response_tagger.text)}")
#     except requests.exceptions.JSONDecodeError:
#         print(f"Response body: {response_tagger.text}")
# except requests.exceptions.RequestException as req_err:
#     print(f"Tagger 请求错误: {req_err}")
# except Exception as e:
#     print(f"处理 Tagger 响应时发生意外错误: {e}")

# print("\n---\n")

# --- 3. Example: Style Transfer with Image (IPAdapter) --- (Commented Out)
url_style_transfer_image = f"{BASE_URL}/generate/style_transfer_with_image"
print(f"调用 Style Transfer with Image API: {url_style_transfer_image}")
# 读取并编码风格图片
base64_encoded_style_image = ImageHelper.encode_image("style.jpg") # 假设你有一张 style.jpg
style_transfer_image_request_data = {
    "input_image": base64_encoded_input_image, # 内容图片
    "style_image": base64_encoded_style_image, # 风格图片
    "style_prompt": "in the style of Van Gogh, starry night",
    "checkpoint_name": "realvisxlV50_v50LightningBakedvae.safetensors",
    "controlnet_name": "controlnet-sd-xl-1.0-softedge-dexined.safetensors",
    "positive_prompt": "masterpiece, best quality, high resolution, landscape",
    "negative_prompt": "(worst quality, low quality:1.4), blurry, text, watermark",
    "ipadapter_preset": "PLUS (high strength)",
    "controlnet_strength": 0.3,
    "ipadapter_weight_style": 1.2,
    "ksampler1_seed": random.randint(1, 1000000),
    "ksampler2_seed": random.randint(1, 1000000),
    "output_prefix": "StyleTransferWithImageOutput",
    "image_width": 1024,
    "image_height": 1024,
}
try:
    response_style_image = requests.post(url_style_transfer_image, json=style_transfer_image_request_data)
    response_style_image.raise_for_status()
    result_style_image = response_style_image.json()
    print("Style Transfer with Image Response:", result_style_image)
    # 保存返回的图片
    if result_style_image.get("status") == "success" and result_style_image.get("images"):
        for img_data in result_style_image["images"]:
            try:
                save_path = ImageHelper.get_output_path(img_data["filename"])
                ImageHelper.decode_and_save_image(img_data["data"], save_path)
                print(f"Style Transfer with Image 图片已保存为: {save_path}")
            except Exception as e:
                print(f"保存 Style Transfer with Image 图片失败: {str(e)}")
    else:
         print(f"Style Transfer with Image API 返回逻辑错误: {result_style_image.get('message', 'No message')} - Detail: {result_style_image.get('detail', result_style_image)}")
except requests.exceptions.HTTPError as http_err:
    print(f"Style Transfer with Image HTTP 请求失败: {http_err}")
    try:
        error_details = response_style_image.json()
        print(f"Error details: {error_details.get('detail', response_style_image.text)}")
    except requests.exceptions.JSONDecodeError:
        print(f"Response body: {response_style_image.text}")
except requests.exceptions.RequestException as req_err:
    print(f"Style Transfer with Image 请求错误: {req_err}")
except Exception as e:
    print(f"处理 Style Transfer with Image 响应时发生意外错误: {e}")
