import requests
import random
from app.utils.image_helper import ImageHelper

url = "http://127.0.0.1:8000/api/v1/generate"

# 读取并编码图片
base64_encoded_image_data = ImageHelper.encode_image("image.png")

request_data = {
    "input_image": base64_encoded_image_data,
    "positive_prompt": "cyberpunk style, red light eyes, mechanic body, laser in eyes, neon",
    "negative_prompt": "underwear,different color eyes,short skirt,text font ui,error,heavy breasts,text,ui,error,missing fingers,missing limb,fused fingers,one hand with more than 5 fingers,one hand with less than 5 fingers,one hand with more than 5 digit,one hand with less than 5 digit,extra digit,fewer digits,fused digit,missing digit,bad digit,liquid digit,colorful tongue,black tongue,cropped,watermark,username,blurry,JPEG artifacts,signature,3D,3D game,3D game scene,3D character,big face,long face,bad eyes,fused eyes poorly drawn eyes,extra eyes,more than two legs,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,multiple breasts,(mutated,hands and fingers:1.5 ),(long body :1.3),(mutation, poorly drawn :1.2),black-white,bad anatomy,liquid body,liquid tongue,disfigured,malformed,mutated,anatomical,nonsense,text font ui,error,malformed,hands,long neck,blurred,lowers,lowres,bad anatomy,bad proportions,bad shadow,uncoordinated body,unnatural body,fused,breasts,bad breasts,huge breasts,poorly,drawn breasts,extra breasts,liquid breasts.,heavy breasts,missing breasts,huge,haunch,huge thighs,huge calf,bad hands,fused hand,missing hand,disappearing,arms,disappearing thigh,disappearing calf,disappearing legs,fused ears,bad ears,poorly drawn ears,extra ears,liquid ears,heavy ears,missing ears,fused animal ears,bad animal ears,poorly drawn animal ears,extra animal ears,liquid animal ears,heavy,animal ears,missing animal ears,text,ui,error,missing fingers,missing limb,fused,fingers,one hand with more than 5 fingers,one hand with less than 5 fingers,one hand,with more than 5 digit,one hand with less,than 5 digit,extra digit,fewer digits,fused,digit,missing digit,bad digit,liquid digit,colorful tongue,black tongue,cropped,watermark,username,blurry,JPEG artifacts,signature,3D,3D game,3D game scene,3D,character,malformed feet,extra feet,bad,feet,poorly drawn feet,fused feet,missing,feet,extra shoes,bad shoes,fused shoes,more than two shoes,poorly drawn shoes,bad gloves,poorly drawn gloves,fused,gloves,bad cum,poorly drawn cum,fused,cum,bad hairs,poorly drawn hairs,fused,hairs,big muscles,ugly,bad face,fused,face,poorly drawn face,cloned face,big,face,long face,bad eyes,fused eyes poorly,drawn eyes,extra eyes,malformed limbs,more than 2 nipples,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,multiple breasts,(mutated,hands and fingers:1.5 ),(long body :1.3),(mutation, poorly drawn :1.2),black-white,bad anatomy,liquid body,liquid tongue,disfigured,malformed,mutated,anatomical,nonsense,text font ui,error,malformed,hands,long neck,blurred,lowers,lowres,bad anatomy,bad proportions,bad shadow,uncoordinated body,unnatural body,fused,breasts,bad breasts,huge breasts,poorly,drawn breasts,extra breasts,liquid breasts.,heavy breasts,missing breasts,huge,haunch,huge thighs,huge calf,bad hands,fused hand,missing hand,disappearing,arms,disappearing thigh,disappearing calf,disappearing legs,fused ears,bad ears,poorly drawn ears,extra ears,liquid ears,heavy ears,missing ears,fused animal ears,bad animal ears,poorly drawn animal ears,extra animal ears,liquid animal ears,heavy,animal ears,missing animal ears,text,ui,error,missing fingers,missing limb,fused,fingers,one hand with more than 5 fingers,one hand with less than 5 fingers,one hand,with more than 5 digit,one hand with less,than 5 digit,extra digit,fewer digits,fused,digit,missing digit,bad digit,liquid digit,colorful tongue,black tongue,cropped,watermark,username,blurry,JPEG artifacts,signature,3D,3D game,3D game scene,3D,character,malformed feet,extra feet,bad,feet,poorly drawn feet,fused feet,missing,feet,extra shoes,bad shoes,fused shoes,more than two shoes,poorly drawn shoes,bad gloves,poorly drawn gloves,fused,gloves,bad cum,poorly drawn cum,fused,cum,bad hairs,poorly drawn hairs,fused,hairs,big muscles,ugly,bad face,fused,face,poorly drawn face,cloned face,big,face,long face,bad eyes,fused eyes poorly,drawn eyes,extra eyes,malformed limbs,more than 2 nipples,underwear,different color eyes,short skirt,text font ui,error,heavy breasts,text,ui,error,missing fingers,missing limb,fused fingers,one hand with more than 5 fingers,one hand with less than 5 fingers,one hand with more than 5 digit,one hand with less than 5 digit,extra digit,fewer digits,fused digit,missing digit,bad digit,liquid digit,colorful tongue,black tongue,cropped,watermark,username,blurry,JPEG artifacts,signature,3D,3D game,3D game scene,3D character,big face,long face,bad eyes,fused eyes poorly drawn eyes,extra eyes,more than two legs,",
    "checkpoint_name": "animij_v20.safetensors",
    "controlnet_name": "controlnet-sd-xl-1.0-softedge-dexined.safetensors",
    "lora_names": [],
    "seed": random.randint(1, 1000000),
    "steps": 20,
    "cfg": 6.5,
    "sampler_name": "euler",
    "scheduler": "normal",
    "denoise": 0.75,
    "lora_strength": 1.0,
    "controlnet_strength": 0.85,
    "output_prefix": "MyStyleTransfer"
}




response = requests.post(url, json=request_data)
result = response.json()

# 保存返回的图片
if result["status"] == "success" and result["images"]:
    for img_data in result["images"]:
        try:
            save_path = ImageHelper.get_output_path(img_data["filename"])
            ImageHelper.decode_and_save_image(img_data["data"], save_path)
            print(f"图片已保存为: {save_path}")
        except Exception as e:
            print(f"保存图片失败: {str(e)}")
