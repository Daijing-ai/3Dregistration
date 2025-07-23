import cv2
import torch
from CLIP import clip
import os
from PIL import Image
from tqdm import tqdm

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 本地加载 CLIP 模型（ViT-B/32）
model, preprocess = clip.load("ViT-B/32", device=device, download_root="../clip/")

# 提取关键帧函数（基于 CLIP 余弦相似度）
def extract_keyframes_clip_local(
        video_path,
        output_dir,
        threshold=0.93,
        interval=1,
        max_frames=None):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    saved_idx = 0
    prev_feat = None
    frame_idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = max_frames or total_frames

    for _ in tqdm(range(min(total_frames, max_frames))):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval != 0:
            frame_idx += 1
            continue

        # BGR → RGB → PIL
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # 预处理并提取 CLIP 特征
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor)
            feat /= feat.norm(dim=-1, keepdim=True)  # L2归一化

        # 与上一关键帧比较
        if prev_feat is None:
            prev_feat = feat
            cv2.imwrite(f"{output_dir}/frame_{saved_idx:04d}.jpg", frame)
            saved_idx += 1
        else:
            cos_sim = torch.cosine_similarity(feat, prev_feat).item()
            if cos_sim < threshold:
                prev_feat = feat
                cv2.imwrite(f"{output_dir}/frame_{saved_idx:04d}.jpg", frame)
                saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"提取完成，共 {saved_idx} 张关键帧保存在 {output_dir}")

# 调用示例
extract_keyframes_clip_local(
    video_path="/home/guestdj/sdc/Data/2.mp4",
    output_dir="./clip_keyframes",
    threshold=0.97,
    interval=1,
    max_frames=1000
)