import cv2
import numpy as np
import torch
from ultralytics.models.sam import Predictor as SAMPredictor

import json
import os
import re
import base64
import textwrap
import shutil
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path

from openai import DefaultHttpxClient, OpenAI

import logging
# 禁用 Ultralytics 的日志输出
logging.getLogger("ultralytics").setLevel(logging.WARNING)


# ----------------------- 基础工具函数 -----------------------

def encode_np_array(image_np):
    """将 numpy 图像数组（BGR）编码为 base64 字符串"""
    success, buffer = cv2.imencode('.jpg', image_np)
    if not success:
        raise ValueError("无法将图像数组编码为 JPEG")
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64



# ----------------------- 多模态模型调用（Qwen） -----------------------

def generate_robot_actions(user_command, image_input=None):
    """
    使用 base64 的方式将 numpy 图像和用户文本指令传给 Qwen 多模态模型，
    要求模型返回两部分：
      - 模型返回内容中，第一部分为自然语言响应（说明为何选择该物体），
      - 紧跟其后的部分为纯 JSON 对象，格式如下：

        {
          "name": "物体名称",
          "bbox": [左上角x, 左上角y, 右下角x, 右下角y]
        }

    返回一个 dict，包含 "response" 和 "coordinates"。
    参数 image_input 为 numpy 数组（BGR 格式）。
    """
    local_config = Path(__file__).with_name(".vlm.local.json")
    if local_config.is_file():
        config = json.loads(local_config.read_text(encoding="utf-8"))
        api_key = Path(config["api_key_file"]).expanduser().read_text(encoding="utf-8-sig").strip()
        base_url, model = config["base_url"], config["model"]
        trust_env = config.get("trust_env", True)
    else:
        api_key = os.environ.get("VLM_API_KEY", "").strip()
        base_url = os.environ.get("VLM_BASE_URL", "").strip()
        model = os.environ.get("VLM_MODEL", "").strip()
        trust_env = True
    if not all((api_key, base_url, model)):
        raise RuntimeError("请检查 .vlm.local.json 或设置 VLM_API_KEY、VLM_BASE_URL、VLM_MODEL。")

    system_prompt = textwrap.dedent("""\
    你是一个精密机械臂视觉控制系统，具备先进的多模态感知能力。请严格按照以下步骤执行任务：

    【图像分析阶段】
    1. 分析输入图像，识别图像中所有可见物体，并记录每个物体的边界框（左上角点和右下角点）及其类别名称。

    【指令解析阶段】
    2. 根据用户的自然语言指令，从识别的物体中筛选出最匹配的目标物体。

    【响应生成阶段】
    3. 输出格式必须严格如下：
    - 自然语言响应（仅包含说明为何选择该物体的文字,可以俏皮可爱地回应用户的需求，但是请注意，回答中应该只包含被选中的物体），
    - 紧跟其后，从下一行开始返回 **标准 JSON 对象**,但是不要返回json本体,格式如下：

    {
      "name": "物体名称",
      "bbox": [左上角x, 左上角y, 右下角x, 右下角y]
    }

    【注意事项】
    - JSON 必须从下一行开始；
    - 自然语言响应与 JSON 之间无其他额外文本;
    - JSON 对象不能有任何注释、额外文本或解释,包括不能有辅助标识为json文本的内容,不要有json;
    - 坐标 bbox 必须为整数；
    - 只允许使用 "bbox" 作为坐标格式。
    """)

    messages = [{"role": "system", "content": system_prompt}]
    user_content = []

    if image_input is not None:
        height, width = image_input.shape[:2]
        messages[0]["content"] += (
            f"\n原图宽 {width} 像素、高 {height} 像素。"
            "请使用原图像素坐标，原点在左上角。"
        )
        base64_img = encode_np_array(image_input)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })

    user_content.append({"type": "text", "text": user_command})
    messages.append({"role": "user", "content": user_content})

    try:
        with OpenAI(api_key=api_key, base_url=base_url,
                    http_client=DefaultHttpxClient(trust_env=trust_env)) as client:
            completion = client.chat.completions.create(
                model=model, messages=messages, temperature=0.1,
            )
        
        content = completion.choices[0].message.content
        print("原始响应：", content)

        # 使用正则表达式查找 JSON 部分
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                coord = json.loads(json_str)
            except Exception as e:
                print(f"[警告] JSON 解析失败：{e}")
                coord = {}
            natural_response = content[:match.start()].strip()
        else:
            natural_response = content.strip()
            coord = {}

        return {
            "response": natural_response,
            "coordinates": coord
        }

    except Exception as e:
        print("请求失败：" + str(e).replace(api_key, "[REDACTED]"))
        return {"response": "处理失败", "coordinates": {}}

# ----------------------- SAM 分割相关 -----------------------
def choose_model():
    """Initialize SAM predictor with proper parameters"""
    model_weight = 'sam_b.pt'
    overrides = dict(
        task='segment',
        mode='predict',
        # imgsz=1024,
        model=model_weight,
        conf=0.25,
        save=False
    )
    return SAMPredictor(overrides=overrides)

def process_sam_results(results):
    """Process SAM results to get mask and center point"""
    if not results or not results[0].masks:
        return None, None

    # Get first mask (assuming single object segmentation)
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    # Find contour and center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


# ----------------------- 语音识别与 TTS -----------------------

samplerate = 16000


def recognize_speech(seconds=5):
    """Capture a short command from the default Ubuntu microphone."""
    if shutil.which("parec") is None:
        raise RuntimeError("找不到 parec，请安装 Ubuntu 的 pulseaudio-utils。")
    print(f"请在 {seconds:g} 秒内说出指令，例如「抓苹果」。", flush=True)
    try:
        subprocess.run(
            ["parec", "--raw", "--format=s16le", "--rate=16000", "--channels=1"],
            capture_output=True, check=True, timeout=seconds,
        )
    except subprocess.TimeoutExpired as exc:
        pcm = exc.stdout or b""
        audio = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype="<i2")
        if not audio.size:
            raise RuntimeError("麦克风没有返回音频，请检查 Ubuntu 声音设置。") from exc
        return audio
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("录音失败：" + exc.stderr.decode(errors="replace").strip()) from exc
    raise RuntimeError("录音提前结束。")


def speech_to_text_offline(audio_data):
    """Transcribe 16 kHz PCM without an FFmpeg subprocess."""
    import whisper

    samples = audio_data.astype(np.float32) / 32768.0
    frame_size = samplerate // 50
    frames = samples[:samples.size // frame_size * frame_size].reshape(-1, frame_size)
    if not len(frames):
        return ""
    levels = np.sqrt(np.mean(frames**2, axis=1))
    active = np.flatnonzero(levels >= max(0.001, float(levels.max()) * 0.08))
    if not active.size:
        return ""
    padding = samplerate // 4
    first = max(0, int(active[0]) * frame_size - padding)
    last = int(active[-1] + 1) * frame_size + padding
    samples = samples[first:last]
    # Keep GPU memory available for SAM and GraspNet.
    cache = Path(__file__).resolve().parent / "outputs" / "whisper"
    model = whisper.load_model("base", device="cpu", download_root=str(cache))
    result = model.transcribe(samples, language="zh", fp16=False, temperature=0,
                              condition_on_previous_text=False, sample_len=32)
    return normalize_voice_command(result["text"])


def normalize_voice_command(text):
    """Collapse repeated copies of the same known command, including stray digits."""
    text = text.strip()
    for target in ("苹果", "蘋果", "香蕉", "锤子", "錘子", "刀", "小鸭", "小鴨"):
        command = f"抓{target}"
        if re.fullmatch(rf"(?:{command}[\s\d，。！？,.!?]*)+", text):
            return command
    return text


def tts_command(output=None):
    """Build an offline Mandarin eSpeak NG command."""
    local = Path(__file__).resolve().parent / "outputs" / "tools" / "usr" / "bin" / "espeak-ng"
    executable = shutil.which("espeak-ng") or (str(local) if local.is_file() else None)
    if executable is None:
        raise RuntimeError("找不到 espeak-ng，无法播报。")
    command = [executable, "-v", "cmn", "-s", "155"]
    if output is not None:
        command.extend(["-w", str(output)])
    return [*command, "--stdin"]


def play_tts_offline(text):
    if not text:
        return
    try:
        with TemporaryDirectory(prefix="vlm-speech-") as folder:
            audio = Path(folder) / "speech.wav"
            subprocess.run(tts_command(audio), input=text.encode(), check=True)
            subprocess.run(["paplay", str(audio)], check=True)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"离线播报失败：{exc}", flush=True)


def voice_command_to_keyword():
    text = speech_to_text_offline(recognize_speech())
    if text:
        print(f"识别到：{text}", flush=True)
    return text


# ----------------------- 主流程：图像分割 -----------------------
def segment_image(image_input, output_mask='mask1.png', *, command=None, speak=True, interactive=True):
    # 1. 使用文字获取目标指令
    print("📝 请通过文字描述目标物体及抓取指令...")
    command_text = (input("请输入: ") if command is None else command).strip()
    if not command_text:
        print("⚠️ 未收到抓取指令，请重试。")
        return None
    print(f"✅ 收到抓取指令：{command_text}")

    # # 1. 使用语音获取目标指令
    # print("🎙️ 请通过语音描述目标物体及抓取指令...")
    # command_text = voice_command_to_keyword()
    # if not command_text:
    #     print("⚠️ 未识别到语音指令，请重试。")
    #     return None
    # print(f"✅ 识别的语音指令：{command_text}")

    # 2. 通过多模态模型获取检测框
    result = generate_robot_actions(command_text, image_input)
    natural_response = result["response"]
    detection_info = result["coordinates"]
    print("自然语言回应：", natural_response)
    print("检测到的物体信息：", detection_info)

    # 仅对模型返回的自然语言回应播报
    if speak:
        play_tts_offline(natural_response)
    
    bbox = detection_info.get("bbox") if isinstance(detection_info, dict) else None
    height, width = image_input.shape[:2]
    if not (isinstance(bbox, list) and len(bbox) == 4 and
            all(type(value) is int for value in bbox) and
            0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height):
        bbox = None
    if bbox is None and not interactive:
        return None
    
    # 3. 准备图像供 SAM 使用（转换为 RGB）
    # 4. 初始化 SAM，并设置图像
    predictor = choose_model()
    predictor.set_image(image_input)

    if bbox:
        results = predictor(bboxes=[bbox])
        center, mask = process_sam_results(results)
        print(f"✅ 自动检测到目标,bbox:{bbox}")
    else:
        print("⚠️ 未检测到目标，请点击图像选择对象")
        cv2.namedWindow('Select Object', cv2.WINDOW_NORMAL)
        cv2.imshow('Select Object', image_input)
        point = []

        def click_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point.extend([x, y])
                print(f"🖱️ 点击坐标：{x}, {y}")
                cv2.setMouseCallback('Select Object', lambda *args: None)

        cv2.setMouseCallback('Select Object', click_handler)
        while True:
            key = cv2.waitKey(100)
            if point:
                break
            if cv2.getWindowProperty('Select Object', cv2.WND_PROP_VISIBLE) < 1:
                print("❌ 窗口被关闭，未进行点击")
                return None
        cv2.destroyAllWindows()
        results = predictor(points=[point], labels=[1])
        center, mask = process_sam_results(results)

    # 5. 保存分割掩码
    if mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        print(f"✅ 分割掩码已保存：{output_mask}")
    else:
        print("⚠️ 分割失败，未生成掩码")

    return mask


# ----------------------- 主程序入口 -----------------------
if __name__ == '__main__':
    seg_mask = segment_image('color_img_path.jpg')
    print("Segmentation result mask shape:", seg_mask.shape if seg_mask is not None else None)
