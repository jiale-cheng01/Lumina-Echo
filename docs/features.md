# Lumina Echo - 功能说明文档

## 系统架构

### 1. 音频捕获模块
```python
class AudioCapture:
    """实时音频捕获系统"""
    def __init__(self, device=None, channels=2, samplerate=44100, blocksize=2048):
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.audio_queue = queue.Queue()
```

- 支持多种音频设备
- 实时音频流处理
- 自动设备检测
- 音频格式转换

### 2. 特征提取模块

#### 基础特征
- 梅尔频谱
- 色度特征
- 谱质心
- 节奏特征
- RMS能量

#### 高级特征
- 音高检测
- 和弦分析
- 节奏模式识别
- 音色特征提取

### 3. 可视化模块

#### LED显示器
```python
class LEDVisualizer:
    """LED可视化系统"""
    def __init__(self, num_leds=60, width=1200, height=400):
        self.width = width
        self.height = height
        self.num_leds = num_leds
        self.led_width = width // num_leds
```

- 实时LED状态更新
- 平滑动画效果
- 多种显示模式
- 自适应亮度调节

## 交互功能

### 1. 模式识别

#### 节奏模式
- 检测重复节奏
- 识别节奏强度
- 响应节奏变化

#### 持续音模式
- 识别持续音调
- 检测音高稳定性
- 生成渐变效果

#### 旋律模式
- 跟踪音高变化
- 识别旋律走向
- 创建波浪效果

### 2. 视觉响应

#### 低频响应
```python
# 低频区域处理
low_freq_energy = np.power(np.mean(mel_normalized[:low_freq_index]), 1.5)
rgb_values[low_mask] = np.array([low_freq_energy * 255, energy * 50, 0])
```

- 红色渐变效果
- 强调低音能量
- 非线性增强

#### 中频响应
```python
# 中频区域处理
hue = np.mean(chroma_normalized)
mid_rgb = np.array(hsv_to_rgb(hue, 1, 1)) * 255
```

- 色度映射
- 和声可视化
- 动态颜色变化

#### 高频响应
```python
# 高频区域处理
rgb_values[high_mask] = np.array([
    high_freq_energy * 100,
    centroid_norm * 255,
    centroid_norm * 255
])
```

- 青色到白色渐变
- 高频细节展示
- 亮度自适应

## 智能特性

### 1. 自适应处理

#### 噪声抑制
```python
# 噪声处理
background_noise = 0.0005
is_signal = (audio_level > background_noise) or (mel_energy > 0.1)
```

- 动态噪声门限
- 智能信号检测
- 自适应阈值

#### 能量平衡
```python
# 能量均衡
energy = np.mean([
    np.mean(rms_normalized),
    np.mean(onset_normalized),
    np.mean(mel_normalized)
])
```

- 多维度能量计算
- 自动增益控制
- 动态范围压缩

### 2. 模式学习

#### 短期记忆
```python
self.pattern_memory = {
    'rhythm': [],      # 节奏记忆
    'melody': [],      # 旋律记忆
    'energy': []       # 能量记忆
}
```

- 存储最近模式
- 模式比较分析
- 响应优化

#### 模式适应
- 自动调整阈值
- 优化识别参数
- 改进响应策略

## 配置选项

### 1. 音频设置
- 采样率：44100Hz
- 块大小：2048样本
- 通道数：支持单声道和立体声
- 设备选择：自动或手动

### 2. 显示设置
- LED数量：可配置
- 显示尺寸：可调整
- 刷新率：可设置
- 背景样式：可自定义

### 3. 处理参数
- 频率范围：20Hz-8kHz
- 能量阈值：可调整
- 噪声门限：可配置
- 响应灵敏度：可设置

## 使用场景

### 1. 音乐可视化
- 现场演出
- 家庭娱乐
- 音乐教育

### 2. 环境照明
- 氛围营造
- 互动装置
- 空间设计

### 3. 交互艺术
- 声音装置
- 互动展览
- 艺术表演 