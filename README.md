# 摄像头手势识别

这个示例使用电脑自带摄像头读取实时视频，用 MediaPipe Hand Landmarker 检测手部关键点，并识别几种常见静态手势。默认只保留手部清晰，脸部和背景会自动虚化。默认使用 640×480 分辨率以降低延迟。MediaPipe 的模型文件会在第一次运行时自动下载到 `.models/hand_landmarker.task`。

当画面中检测到两只手，且两只手的拇指和食指都伸出时，程序才会连接两只手的拇指指尖和食指指尖，并将四个指尖围成的区域保持清晰，区域外继续虚化。

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
