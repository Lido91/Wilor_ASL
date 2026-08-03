# WiLoR / How2Sign 参数提取工作总结

更新时间：2026-07-30（UTC）

本文记录本次在 `/home/student/hwu/Workplace/WiLoR` 中完成的环境修复、代码修改、输出格式、How2Sign 批量提取、双 GPU 运行、断点续跑，以及 MANO pose / mean / PCA 语义验证。

## 1. 最终目标与当前结论

已经完成以下工作：

1. 修复 WiLoR 环境中 `chumpy`、NumPy、PyTorch Lightning 和 `setuptools` 的兼容问题。
2. 成功运行原始图片 demo，并验证 JPG 渲染和 OBJ 导出。
3. 给 `demo.py` 增加只保存 MANO 参数的 `--params_only` 模式。
4. 新增 `extract_how2sign_wilor.py`，按 How2Sign 的 clip/frame 目录批量提取参数。
5. 输出改为用户要求的“逐帧一个 PKL”结构，并支持一帧多只手或没有手。
6. 新增双 GPU 启动脚本 `run_how2sign_wilor_2gpu.sh`。
7. 已启动 GPU 0、GPU 1 两个分片任务，依次处理 `test -> val -> train`。
8. 验证保存的 45D `hand_pose` 是完整 MANO 局部关节姿态，不是扣除 MANO mean 后的 residual。
9. 验证当前输出应使用 `use_pca=False`；`num_pca_comps=12` 不适用于保存的 45D pose。
10. 验证这些参数可以重新生成 MANO 手部 mesh/OBJ。
11. 新增固定左右手槽位的时序后处理；漏检手使用相邻同侧手的旋转 SLERP 和参数线性插值补齐。

## 2. 本次修改和新增的文件

### `demo.py`

新增：

```bash
--params_only
--save_params_only
--interpolate_missing_hands
--interpolate_hands
```

这两个参数是同一个开关。启用后：

- 每张输入图片保存一个压缩 `.npz` 参数文件。
- 不创建 renderer。
- 不保存渲染 JPG。
- 不保存 OBJ。
- `--params_only` 和 `--save_mesh` 不能同时使用。
- 没检测到手时仍保存 shape 稳定的空数组。

示例：

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --img_folder demo_img \
  --out_folder demo_params \
  --params_only \
  --fast
```

说明：曾临时增加过 demo 的 `--batch_size` 参数，但用户随后说明只是询问功能，因此已经撤销。当前 `demo.py` 的 DataLoader batch size 仍是原来的固定值 `16`。

demo 的每张图对应一个 `<image_stem>.npz`，主要字段为：

```text
source_image
global_orient_rotmat       [N,1,3,3]
hand_pose_rotmat           [N,15,3,3]
global_orient_axis_angle   [N,3]
hand_pose_axis_angle       [N,45]
mano_pose_axis_angle       [N,48]
betas                      [N,10]
pred_cam                   [N,3]
cam_t_full                 [N,3]
bbox_xyxy                  [N,4]
detection_confidence       [N]
is_right                   [N]
box_center                 [N,2]
box_size                   [N]
image_size                 [N,2]
focal_length               [N]
```

这里的 `N` 是当前图片中检测到的手数。

`--interpolate_missing_hands` 把 `img_folder` 当作按文件名排序的视频帧序列，使用与批量后处理器相同的左右手插值逻辑重新生成 mesh。可视化中紫色为真实检测，橙色为插值结果，并生成：

```text
<out_folder>/<frame>.jpg
<out_folder>/interpolation_preview.mp4
<out_folder>/interpolation_summary.json
```

示例：

```bash
CUDA_VISIBLE_DEVICES=1 python demo.py \
  --img_folder /data/hwu/how2sign/how2sign_images_test/-fZc293MpJk_4-1-rgb_front \
  --out_folder demo_out_h2s_params \
  --fast \
  --interpolate_missing_hands \
  --output_video_fps 25
```

### `extract_how2sign_wilor.py`

新增的 How2Sign 专用批量提取器，功能包括：

- 递归读取 `how2sign_images_test/val/train` 下的 clip。
- 每个输入帧生成一个 PKL。
- YOLO 检测按帧 batch 处理。
- WiLoR 按检测出的手部 crop batch 处理。
- 支持多只手。
- 支持没有检测到手的帧。
- 支持两个或更多独立 shard。
- 支持逐帧原子写入。
- 支持 clip 级 `.complete` 标记。
- 支持中断后续跑。
- 单个 clip 出错时记录错误并继续下一个 clip。

主要参数：

```text
--dataset-root
--splits
--device
--num-shards
--shard-index
--detector-batch-size
--wilor-batch-size
--confidence
--iou
--rescale-factor
--fast
--max-clips
--max-frames-per-clip
--overwrite
```

当前生产配置：

```text
detector batch size = 32
WiLoR batch size    = 64
fast mode           = enabled
number of shards    = 2
```

`--fast` 会启用 FP16 和 WiLoR backbone layer dropping，但批量脚本没有使用 `torch.compile`，因此不会出现 demo 中因为手数变化导致的大量 Dynamo 重编译。

### `run_how2sign_wilor_2gpu.sh`

启动脚本现已改为任意 GPU 数量：位置参数就是 physical GPU ID，脚本根据 GPU 数量自动设置 `--num-shards`，每张卡启动一个 worker。所有进程内部都使用 `--device 0`，因为各自通过 `CUDA_VISIBLE_DEVICES` 只能看到一张卡。
- 日志和 PID 文件放在：

```text
logs/how2sign_wilor/
```

示例：

```bash
bash run_how2sign_wilor_2gpu.sh 2
bash run_how2sign_wilor_2gpu.sh 0 1
bash run_how2sign_wilor_2gpu.sh 0 2 3
```

分别表示使用 1、2、3 张卡。不传 GPU ID 时，为兼容旧用法默认使用 GPU 0 和 1。不要在旧 worker 仍运行时改变 GPU 数量或重新启动另一套任务。

### `run_youtube_wilor_2gpu.sh`

针对 `/data/hwu/youtube_dataset/clip_fps24_img_0` 的同类动态 GPU 启动器，输出到 `/data/hwu/youtube_dataset/clip_fps24_img_0_wilor_out`：

```bash
bash run_youtube_wilor_2gpu.sh 2
bash run_youtube_wilor_2gpu.sh 0 1 3
```

### `interpolate_how2sign_wilor.py`

新增的 CPU 后处理器。它不会修改原始 `wilor_params`，而是生成独立的：

```text
wilor_params_interpolated
```

处理规则：

- 每帧固定两个槽位：index 0 为左手，index 1 为右手。
- 同一侧检测到多个候选时，保留检测置信度最高的一个。
- 某侧在中间帧漏检时，使用前后最近的同侧有效帧插值。
- rotation matrix 使用 quaternion SLERP。
- betas、camera、bbox 等连续值使用线性插值。
- clip 开头或结尾缺失时，使用最近的有效同侧帧。
- 某侧在整个 clip 中从未被检测到时，使用有限的 neutral 数值占位，并设置 `valid_mask=False`；这种情况无法仅靠插值恢复。
- 只读取带 `.complete` 的原始 clip，避免和仍在写入的提取进程竞争同一个 clip。
- 输出逐帧原子写入，并使用独立 `.complete` 标记支持续跑。

## 3. 环境与依赖修复

### Chumpy 为什么安装失败

`requirements.txt` 中包含：

```text
chumpy @ git+https://github.com/mattloper/chumpy
```

Chumpy 是旧式 Python 包，其 `setup.py` 会直接 import `pip`。新版 pip 默认创建隔离构建环境，而该临时环境里没有 `pip`，所以出现：

```text
ModuleNotFoundError: No module named 'pip'
```

使用下面的方式关闭 build isolation 后安装成功：

```bash
python -m pip install --no-build-isolation \
  "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
```

Chumpy 版本为 `0.71`，import 已验证成功。WiLoR 的完整依赖安装仍然需要它。

### 最终关键版本

```text
Python              3.10
torch               2.0.1+cu117
torchvision         0.15.2+cu117
numpy               1.26.4
opencv-python       4.10.0.84
pytorch-lightning   2.0.9
setuptools          80.9.0
chumpy              0.71
```

NumPy 原来是 `2.2.6`，与 `torch 2.0.1` 不兼容，因此降到 `1.26.4`。`setuptools` 保留在 `80.9.0`，使 Lightning 2.0.9 仍可使用 `pkg_resources`。

使用过的完整安装命令：

```bash
python -m pip install --no-build-isolation \
  -r requirements.txt \
  "torch==2.0.1" \
  "torchvision==0.15.2" \
  "pytorch-lightning==2.0.9" \
  "numpy==1.26.4" \
  "opencv-python==4.10.0.84" \
  "setuptools==80.9.0"
```

验证结果：

- `pip check` 通过。
- 关键 Python import 通过。
- `python demo.py --help` 通过。
- CUDA 可用。
- 机器上可见 4 张 NVIDIA L40S。
- `MANO_RIGHT.pkl`、`detector.pt`、`wilor_final.ckpt` 均已存在。

### 非致命 warning

运行时看到过以下 warning，不影响本次结果：

- Lightning 使用的 `pkg_resources` 已弃用。
- `timm.models.layers` 导入路径未来会弃用。
- Lightning 将旧 checkpoint 从 1.8.1 结构临时升级到 2.0.9。
- MANO 模型只有 10 个 shape coefficients。
- Ultralytics 检测到缺少 `dill` 后自动安装。
- demo 的 `torch.compile` 因每个 batch 的手数不同达到 Dynamo cache limit，随后回退执行；不是推理失败。

## 4. 原始 demo 验证

运行：

```bash
python demo.py \
  --img_folder demo_img \
  --out_folder demo_out \
  --save_mesh \
  --fast
```

结果：

- 8 张输入图均处理完成。
- 生成 8 张结果 JPG。
- 生成 45 个手部 OBJ。
- 总共生成 53 个输出文件。

这也说明一张图可以检测到多组手；OBJ 数量不是输入图片数量，而是所有图片中检测到的手数之和。

## 5. How2Sign 数据规模

输入目录：

```text
/data/hwu/how2sign/how2sign_images_test
/data/hwu/how2sign/how2sign_images_val
/data/hwu/how2sign/how2sign_images_train
```

统计：

| split | clip 数 | frame 数 | 每个 clip 帧数 |
|---|---:|---:|---|
| train | 33,390 | 5,434,505 | min 1, max 2579, mean 162.758 |
| val | 1,739 | 275,233 | min 3, max 1761, mean 158.271 |
| test | 2,343 | 381,412 | min 1, max 1319, mean 162.788 |
| 合计 | 37,472 | 6,091,150 | - |

train 图片约占 `97G`。抽样图片通常为 `1280 x 720` JPEG。

输出目录按照用户指定的 `_wilor_out` 命名：

```text
/data/hwu/how2sign/how2sign_images_test_wilor_out
/data/hwu/how2sign/how2sign_images_val_wilor_out
/data/hwu/how2sign/how2sign_images_train_wilor_out
```

根据实测单帧 PKL 大小估算，全部参数输出大约需要 `20–25 GB`。

## 6. 逐帧 PKL 输出结构

目录结构：

```text
how2sign_images_<split>_wilor_out/
└── wilor_params/
    └── <clip_id>/
        ├── frame_000000.pkl
        ├── frame_000001.pkl
        ├── ...
        └── .complete
```

每个输入帧都对应一个 PKL。PKL 内容是 Python dict。

### 核心字段

假设该帧检测到 `N` 只手：

| key | shape / type | 说明 |
|---|---|---|
| `schema_version` | int | 当前为 1 |
| `split` | str | `test`、`val` 或 `train` |
| `clip_id` | str | clip 目录名 |
| `frame_name` | str | 原图文件名 |
| `num_hands` | int | 当前帧检测到的手数 `N` |
| `hand_pose` | `[N,45]` float32 | 15 个局部关节的 axis-angle，单位 radians |
| `global_orient` | `[N,3]` float32 | 手腕全局 axis-angle |
| `mano_pose` | `[N,48]` float32 | `global_orient + hand_pose` |
| `hand_pose_axis_angle` | `[N,45]` | `hand_pose` 的明确别名 |
| `global_orient_axis_angle` | `[N,3]` | `global_orient` 的明确别名 |
| `mano_pose_axis_angle` | `[N,48]` | `mano_pose` 的明确别名 |
| `hand_pose_rotmat` | `[N,15,3,3]` | WiLoR 直接输出的局部旋转矩阵 |
| `global_orient_rotmat` | `[N,1,3,3]` | WiLoR 直接输出的全局旋转矩阵 |
| `betas` | `[N,10]` | MANO shape |
| `pred_cam` | `[N,3]` | crop camera 参数 |
| `cam_t_full` | `[N,3]` | 映射到整张图片的 camera translation |
| `bbox_xyxy` | `[N,4]` | 手部检测框 |
| `detection_confidence` | `[N]` | 检测置信度 |
| `is_right` | `[N]` float32 | `1` 为右手，`0` 为左手 |
| `box_center` | `[N,2]` | crop box center |
| `box_size` | `[N]` | crop box size |
| `image_size` | `[N,2]` | 原图尺寸 |
| `focal_length` | `[N]` | 推理使用的焦距 |

另外还保存检测阈值、IoU 阈值、bbox rescale factor 和 fast mode 标记。

### 多只手如何表示

所有手部字段的第一维都是同一个 `N`，同一个索引代表同一只手：

```python
hand_id = 0

pose = data["hand_pose"][hand_id]
shape = data["betas"][hand_id]
side = data["is_right"][hand_id]
bbox = data["bbox_xyxy"][hand_id]
```

WiLoR/YOLO 不提供跨帧稳定的 hand track ID。因此：

- 当前 PKL 可以区分同一帧里的多只手。
- 不能假设第 `i` 只手在下一帧仍然是相同身份。
- 如果后续需要稳定的左右手时间序列，应结合 `is_right`、bbox 中心和相邻帧距离做关联。

### 没检测到手

仍然会写 PKL：

```python
data["num_hands"] == 0
data["hand_pose"].shape == (0, 45)
data["betas"].shape == (0, 10)
```

这样可以保持输入帧和输出文件一一对应。

### 固定双手插值输出

原始检测结果保留在：

```text
wilor_params/<clip_id>/<frame>.pkl
```

插值后的固定双手结果写入：

```text
wilor_params_interpolated/<clip_id>/<frame>.pkl
```

插值输出中所有手部参数第一维固定为 2：

```text
index 0 = left
index 1 = right

hand_pose             [2,45]
global_orient         [2,3]
mano_pose             [2,48]
hand_pose_rotmat      [2,15,3,3]
global_orient_rotmat  [2,1,3,3]
betas                 [2,10]
bbox_xyxy             [2,4]
is_right              [0,1]
```

新增状态字段：

| key | shape | 说明 |
|---|---|---|
| `source_num_hands` | scalar | 原始帧检测数量 |
| `num_hands` | scalar | 固定为 2，代表两个槽位 |
| `num_observed_hands` | scalar | 当前帧真实检测到并选中的左右手数量 |
| `num_valid_hands` | scalar | 当前 clip 中能够检测或插值恢复的手数 |
| `hand_slot_order` | tuple | `("left", "right")` |
| `observed_mask` | `[2]` bool | 当前帧该侧来自真实检测 |
| `interpolated_mask` | `[2]` bool | 当前帧该侧来自插值或边界复制 |
| `valid_mask` | `[2]` bool | 整个 clip 是否至少出现过该侧 |
| `selected_detection_index` | `[2]` int | 选中的原始 detection index；漏检为 -1 |
| `discarded_duplicate_count` | `[2]` int | 当前帧每侧丢弃的低置信度重复检测数 |
| `interpolation_prev_frame` | `[2]` int | 每侧使用的前一个观测帧序号 |
| `interpolation_next_frame` | `[2]` int | 每侧使用的后一个观测帧序号 |
| `interpolation_alpha` | `[2]` float | 前后帧插值系数 |

插值帧的 `detection_confidence` 被设为 0，防止把生成值误认为真实检测置信度。

下游读取时：

```python
left_pose = data["hand_pose"][0]
right_pose = data["hand_pose"][1]

left_was_detected = data["observed_mask"][0]
right_was_interpolated = data["interpolated_mask"][1]
```

### 读取示例

```python
import pickle

pkl_path = (
    "/data/hwu/how2sign/"
    "how2sign_images_test_wilor_out/"
    "wilor_params/<clip_id>/<frame_name>.pkl"
)

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(data["frame_name"])
print(data["num_hands"])
print(data["hand_pose"].shape)
print(data["is_right"])
```

## 7. 双 GPU 生产任务

实际运行的两个 worker 等价于下面两条命令。

GPU 0 / shard 0：

```bash
CUDA_VISIBLE_DEVICES=0 \
MPLCONFIGDIR=/tmp/wilor_matplotlib_0 \
YOLO_CONFIG_DIR=/tmp/wilor_ultralytics_0 \
PYTHONUNBUFFERED=1 \
/home/student/hwu/miniconda3/envs/wilor/bin/python \
  extract_how2sign_wilor.py \
  --dataset-root /data/hwu/how2sign \
  --splits test val train \
  --device 0 \
  --num-shards 2 \
  --shard-index 0 \
  --detector-batch-size 32 \
  --wilor-batch-size 64 \
  --fast
```

GPU 1 / shard 1：

```bash
CUDA_VISIBLE_DEVICES=1 \
MPLCONFIGDIR=/tmp/wilor_matplotlib_1 \
YOLO_CONFIG_DIR=/tmp/wilor_ultralytics_1 \
PYTHONUNBUFFERED=1 \
/home/student/hwu/miniconda3/envs/wilor/bin/python \
  extract_how2sign_wilor.py \
  --dataset-root /data/hwu/how2sign \
  --splits test val train \
  --device 0 \
  --num-shards 2 \
  --shard-index 1 \
  --detector-batch-size 32 \
  --wilor-batch-size 64 \
  --fast
```

分片依据是 `split/clip_id` 的稳定 SHA1 hash，因此：

- 两个 worker 不会处理同一个 clip。
- 文件系统枚举顺序变化不会改变 clip 属于哪个 shard。
- 续跑时仍使用相同的 `num-shards=2` 和 shard index 即可。

### 2026-07-30 04:43 UTC 状态快照

| worker | split | 完成 clip | 累计 frame | 平均速度 | failed |
|---|---|---:|---:|---:|---:|
| GPU 0 / shard 0 | test | 164 | 27,417 | 24.45 FPS | 0 |
| GPU 1 / shard 1 | test | 165 | 28,681 | 25.46 FPS | 0 |
| 合计 | test | 329 | 56,098 | 约 49.9 FPS | 0 |

同一时间附近的直接文件系统快照为：

```text
56,801 PKL
333 个 .complete
227 MB
```

文件计数与 worker 日志之间有少量瞬时差异，是因为两个 worker 仍在写当前 clip，且两次检查不是完全同一时刻。

当前仍在处理 test；脚本会在 test 完成后自动继续 val，然后继续 train。按已测速度估计，两张 GPU 完成约 609 万帧需要 `33–36 小时`。实际时间会随每帧手数、图片读取速度和 clip 长度变化。

## 8. 中断与续跑

可以安全续跑。

提取器有两层恢复逻辑：

1. clip 中存在 `.complete`：整个 clip 跳过。
2. clip 未完成：已经存在的 frame PKL 跳过，只处理缺失帧。

每个 PKL 先写临时文件，再用 `os.replace` 原子替换。因此进程在写入中途被杀掉时，不会把半个 pickle 当成完整输出。

中断后：

- 使用原来的两个命令重新启动。
- 保持 `--num-shards 2`。
- shard 0 和 shard 1 各启动一个。
- 不要加入 `--overwrite`。

`--overwrite` 会强制重新计算已经完成的 clip，仅在明确希望全部重做时使用。

检查进度：

```bash
nvidia-smi
```

```bash
find \
  /data/hwu/how2sign/how2sign_images_test_wilor_out/wilor_params \
  -type f -name '*.pkl' | wc -l
```

```bash
find \
  /data/hwu/how2sign/how2sign_images_test_wilor_out/wilor_params \
  -type f -name '.complete' | wc -l
```

val/train 只需替换路径中的 split 名。

### 运行双手插值

直接处理一个用户指定的 `_wilor_out` 目录：

```bash
/home/student/hwu/miniconda3/envs/wilor/bin/python \
  interpolate_how2sign_wilor.py \
  --split-output-root \
    /data/hwu/how2sign/how2sign_images_test_wilor_out
```

这条命令读取：

```text
/data/hwu/how2sign/how2sign_images_test_wilor_out/wilor_params
```

并写入：

```text
/data/hwu/how2sign/how2sign_images_test_wilor_out/wilor_params_interpolated
```

原始 `wilor_params` 不会被修改。

建议等原始双 GPU 提取全部完成后，再启动两个 CPU shard：

CPU shard 0：

```bash
/home/student/hwu/miniconda3/envs/wilor/bin/python \
  interpolate_how2sign_wilor.py \
  --dataset-root /data/hwu/how2sign \
  --splits test val train \
  --num-shards 2 \
  --shard-index 0
```

CPU shard 1：

```bash
/home/student/hwu/miniconda3/envs/wilor/bin/python \
  interpolate_how2sign_wilor.py \
  --dataset-root /data/hwu/how2sign \
  --splits test val train \
  --num-shards 2 \
  --shard-index 1
```

这个步骤不使用 GPU。若在原始提取尚未结束时运行，它只处理当前已经有 `.complete` 的 clip，并跳过未完成 clip；稍后需要再次运行相同命令以处理新增完成的 clip。

插值输出同样支持断点续跑：

- 有输出 `.complete` 的 clip 自动跳过。
- 部分写完但没有 `.complete` 的 clip，只补写缺少的 frame PKL。
- 正常续跑不要使用 `--overwrite`。

## 9. Batch size

How2Sign 提取器可以调整两种 batch size：

```text
--detector-batch-size 32
--wilor-batch-size 64
```

- detector batch：一次交给 YOLO 的完整帧数量。
- WiLoR batch：一次交给 WiLoR 的检测手部 crop 数量。

如果显存不足，优先降低 WiLoR batch，例如：

```text
--detector-batch-size 16 --wilor-batch-size 32
```

如果显存充足但 GPU 利用率不高，可逐步增大。当前 `32/64` 已通过完整 clip benchmark，没有 OOM。

## 10. 视频输入

当前原始 `demo.py` 和 How2Sign 提取器都直接读取图片，不直接接收 MP4 文件。视频可以先按原始帧率解码为图片：

```bash
mkdir -p video_frames
ffmpeg -i input.mp4 -vsync 0 video_frames/frame_%06d.jpg
```

如果只需要 demo 的逐图 NPZ：

```bash
python demo.py \
  --img_folder video_frames \
  --out_folder video_params \
  --params_only \
  --fast
```

如果需要与 How2Sign 相同的逐帧 PKL，可整理成：

```text
/tmp/wilor_video/
└── how2sign_images_test/
    └── <video_id>/
        ├── frame_000001.jpg
        ├── frame_000002.jpg
        └── ...
```

然后运行：

```bash
python extract_how2sign_wilor.py \
  --dataset-root /tmp/wilor_video \
  --splits test \
  --device 0 \
  --detector-batch-size 32 \
  --wilor-batch-size 64 \
  --fast
```

当前提取是逐帧独立推理，不包含跨帧跟踪或时序平滑。若下游需要稳定的 hand identity，应额外做时序关联。

## 11. MANO pose、mean 与 PCA 的准确语义

### 保存的 45D 是否已经扣除 MANO mean

结论：

> `hand_pose` / `hand_pose_axis_angle` 是最终完整的 15 个 MANO 局部关节旋转，不是相对 MANO hand mean 的偏移量。重建时不要再加 mean。

WiLoR 的实际路径是：

1. `mano_mean_params.npz` 中的 pose 是 16 个关节的 6D rotation 初始化。
2. 网络在这个初始化上预测/refine。
3. 最终 6D rotation 被转换成 rotation matrices。
4. WiLoR 把最终 rotation matrices 直接传给 `MANOLayer`。
5. `MANOLayer` 使用 `pose2rot=False` 做 LBS，不额外添加 `pose_mean`。
6. 提取器再把这些最终 rotation matrices 转成 axis-angle。

所以，mean 只影响网络内部的预测初始化，不表示保存结果仍是一个需要加 mean 的 residual。

### `flat_hand_mean` 应该设成什么

有两条正确重建路径。

#### 路径 A：直接使用保存的 rotation matrices，推荐

使用 `MANOLayer` 和 `use_pca=False`。这一条与 WiLoR 原始前向过程最接近：

```python
import pickle
import torch
from smplx import MANOLayer

with open("frame_000000.pkl", "rb") as f:
    data = pickle.load(f)

mano = MANOLayer(
    model_path="mano_data",
    is_rhand=True,
    use_pca=False,
)

global_orient = torch.from_numpy(data["global_orient_rotmat"]).float()
hand_pose = torch.from_numpy(data["hand_pose_rotmat"]).float()
betas = torch.from_numpy(data["betas"]).float()

output = mano(
    global_orient=global_orient,
    hand_pose=hand_pose,
    betas=betas,
)

vertices = output.vertices
faces = mano.faces
```

`MANOLayer` 接收 rotation matrices，并在内部固定用 `pose2rot=False`。

#### 路径 B：使用保存的 axis-angle

普通 `smplx.MANO` 会把 axis-angle pose 与内部 `hand_mean` 组合，因此这里必须使用：

```text
use_pca=False
flat_hand_mean=True
```

示例：

```python
import pickle
import torch
from smplx import MANO

with open("frame_000000.pkl", "rb") as f:
    data = pickle.load(f)

mano = MANO(
    model_path="mano_data",
    is_rhand=True,
    use_pca=False,
    flat_hand_mean=True,
)

output = mano(
    global_orient=torch.from_numpy(data["global_orient"]).float(),
    hand_pose=torch.from_numpy(data["hand_pose"]).float(),
    betas=torch.from_numpy(data["betas"]).float(),
)
```

这里不能使用 `flat_hand_mean=False`，否则 MANO 会再次叠加非零 hand mean。

数值验证：

```text
rotation-matrix MANOLayer
vs axis-angle MANO(flat_hand_mean=True)
最大 vertex 误差 = 2.714991569519043e-05

rotation-matrix MANOLayer
vs axis-angle MANO(flat_hand_mean=False)
最大 vertex 误差 = 0.03611588478088379

MANO hand_mean L2 = 2.3246068954467773
```

因此 `flat_hand_mean=True` 的重建结果才与 WiLoR 输出一致；极小误差来自 rotation matrix 和 axis-angle 的数值转换。

### `num_pca_comps=12`

当前输出是：

```text
hand_pose.shape == [N, 45]
```

因此应使用：

```text
use_pca=False
```

当 `use_pca=False` 时，`num_pca_comps=12` 被忽略。

只有在：

```text
use_pca=True
num_pca_comps=12
```

时，MANO 才期望输入 `[N,12]` 的 PCA coefficients。把当前 45D axis-angle 投影到 12D PCA 会丢失信息，不能精确还原 WiLoR 的原始姿态。

## 12. 是否可以从参数还原 OBJ

可以。

生成 canonical MANO mesh 至少需要：

```text
global_orient
hand_pose
betas
```

推荐直接使用：

```text
global_orient_rotmat
hand_pose_rotmat
betas
```

如果要把 mesh 放进 WiLoR 预测的整图相机坐标，还可加：

```text
cam_t_full
```

例如：

```python
vertices_camera = vertices + torch.from_numpy(
    data["cam_t_full"]
).float()[:, None, :]
```

左手需要注意：

- WiLoR 使用右手 MANO canonical model。
- `is_right == 1`：右手。
- `is_right == 0`：左手，需要把生成顶点的 X 坐标镜像。
- 镜像后导出 OBJ 时建议反转三角面顶点顺序，以保持 face normal 方向。

示意：

```python
verts = vertices.detach().cpu().numpy()
faces = mano.faces.copy()

for hand_idx in range(data["num_hands"]):
    hand_verts = verts[hand_idx].copy()
    hand_faces = faces.copy()

    if data["is_right"][hand_idx] < 0.5:
        hand_verts[:, 0] *= -1
        hand_faces = hand_faces[:, [0, 2, 1]]
```

## 13. 测试与验证结果

### 参数 demo smoke test

对测试图片运行 `--params_only`：

- 检测到 2 只手。
- `is_right == [1, 0]`。
- axis-angle 转回 rotation matrix 后，最大绝对误差：

```text
1.1920928955078125e-07
```

### How2Sign 三 split smoke test

临时数据：

```text
/tmp/wilor-how2sign-smoke
```

每个 split 取 3 帧：

- 共 9 个输入帧。
- 生成 9 个 PKL。
- 共检测 18 只手。
- 0 failures。
- 所有字段 shape、dtype、第一维对齐均通过检查。

### 完整 clip benchmark

clip：

```text
test/g3Cc_1-V31U_0-3-rgb_front
```

结果：

```text
206 frames
414 hands
25.47 FPS
0 failures
```

PKL 大小：

```text
平均 3240 bytes
最小 2158 bytes
最大 4302 bytes
```

### 生产输出抽检

抽检文件：

```text
/data/hwu/how2sign/
how2sign_images_test_wilor_out/
wilor_params/FZNuNG9UBnw_2-1-rgb_front/
frame_000106.pkl
```

结果：

- `num_hands == 2`
- pose、global orient、betas、is_right、bbox shape 正确
- 核心浮点数组均为 float32
- 数值均 finite

### 双手插值合成测试

构造 5 帧 clip：

- 左手仅在第 0、4 帧观测。
- 右手仅在第 2 帧观测。
- 第 0 帧包含两个左手候选，验证选择最高置信度候选。

结果：

```text
frames=5
observed slots=3
interpolated slots=7
invalid slots=0
discarded duplicates=1
```

左手从第 0 帧的 0° 到第 4 帧的 90° 做 SLERP，第 2 帧实测为：

```text
44.99999959383233°
```

### 双手插值真实 clip 测试

真实 clip：

```text
test/g3Cc_1-V31U_0-3-rgb_front
```

原始结果：

- 206 帧。
- `frame_000119.pkl` 只有一只左手。
- 另有 3 个同侧重复检测被丢弃。

插值结果：

```text
frames=206
observed slots=411
interpolated slots=1
invalid slots=0
duplicates=3
```

`frame_000119.pkl` 的右手由第 118 和 120 帧得到：

```text
is_right            = [0, 1]
observed_mask       = [True, False]
interpolated_mask   = [False, True]
valid_mask          = [True, True]
previous frame      = [119, 118]
next frame          = [119, 120]
alpha               = [0.0, 0.5]
hand_pose shape     = [2,45]
```

所有输出参数均为 finite，真实 clip 后处理速度约为 2100 FPS（临时本地目录测试；生产存储速度可能更低）。

### 静态检查

以下检查均通过：

```text
python syntax / py_compile
demo --help
git diff --check
```

## 14. Git 工作区状态

本次相关源码状态：

```text
M  demo.py
M  wilor/utils/renderer.py
?? extract_how2sign_wilor.py
?? interpolate_how2sign_wilor.py
?? run_how2sign_wilor_2gpu.sh
?? run_youtube_wilor_2gpu.sh
?? WILOR_HOW2SIGN_EXTRACTION_SUMMARY.md
```

尚未创建 git commit。

仓库中还存在模型 checkpoint、MANO 文件、demo 输出、日志、`__pycache__` 等用户或运行生成的未跟踪/已变化文件。本次没有删除或覆盖这些文件。

## 15. 后续使用建议

1. 当前双卡任务仍在运行，不要启动第二套提取任务。
2. 定期检查 GPU、PKL 数和 `.complete` 数。
3. 任务中断后，用完全相同的两个 shard 命令继续，不加 `--overwrite`。
4. 需要固定双手时，下游优先读取 `wilor_params_interpolated` 中的 `hand_pose [2,45]`、`global_orient [2,3]`、`betas [2,10]`。
5. 下游同时保留 `observed_mask`、`interpolated_mask` 和 `valid_mask`，不要把插值值当成真实检测。
6. 精确重建 mesh 时优先使用保存的 rotation matrices。
7. axis-angle 重建必须使用 `use_pca=False, flat_hand_mean=True`。
8. 不要给保存的 45D pose 再添加 MANO mean。
9. 不要把 45D pose 直接当成 `num_pca_comps=12` 的 PCA coefficients。
