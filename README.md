# 摄像头手势识别

这个示例使用电脑自带摄像头读取实时视频，用 MediaPipe Hand Landmarker 检测手部关键点，并识别几种常见静态手势。默认只保留手部清晰，脸部和背景会自动虚化。默认使用 640×480 分辨率以降低延迟。MediaPipe 的模型文件会在第一次运行时自动下载到 `.models/hand_landmarker.task`。

当画面中检测到两只手，且两只手的拇指和食指都伸出时，程序才会连接两只手的拇指指尖和食指指尖，并将四个指尖围成的区域保持清晰，区域外继续虚化。

## 中间画面效果

中间四点框共有 6 种画面。默认按“指尖窗口”打开次数轮换：

1. 第一次满足双手指尖条件：`Anime`，动漫风格
2. 第二次满足双手指尖条件：`Pixel`，像素画
3. 第三次满足双手指尖条件：`Sketch`，素描线稿
4. 第四次满足双手指尖条件：`Neon`，霓虹色彩
5. 第五次满足双手指尖条件：`Retro`，复古棕褐色
6. 第六次满足双手指尖条件：`Original`，原始画面

当指尖窗口暂时不满足条件时，当前画面效果保持不变；下一次重新满足条件时才切换到下一个效果。第七次重新从 `Anime` 开始。

窗口打开后按数字键 `1`–`6` 切换，也可以按 `M` 循环切换。启动时指定效果：

```bash
python gesture_camera.py --visual-mode pixel
```

手动指定效果后，不会改变自动轮换计数。

- 拳头
- 张开手掌
- 食指指向
- 胜利手势
- 点赞 / 踩

## 安装

建议使用 Python 3.9–3.12 的虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 运行

```bash
python gesture_camera.py
```

如果不需要虚化效果，可以运行：

```bash
python gesture_camera.py --no-blur
```

如果运行环境不能自动联网下载模型，请手动准备一个 `hand_landmarker.task` 文件，然后运行：

```bash
python gesture_camera.py --model-path /path/to/hand_landmarker.task
```

程序默认使用摄像头编号 `0`。如果打不开摄像头，可以依次尝试：

```bash
python gesture_camera.py --camera-index 1
```

窗口中按 `q` 或 `Esc` 退出，也可以直接点击窗口关闭按钮。Linux 下如果系统提示摄像头权限不足，需要先允许当前用户访问摄像头设备。
