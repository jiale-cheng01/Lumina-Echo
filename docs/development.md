# Lumina Echo - 开发文档

## 核心概念

项目的创新点在于将音频的多维特征映射到三维空间：

### 1. 特征空间设计

- **X轴（频率维度）**
  - 使用梅尔频谱表示频率分布
  - 反映了人耳的频率感知特性
  - 低频、中频、高频的能量分布

- **Y轴（音高维度）**
  - 使用色度特征（Chroma）表示音高信息
  - 12个音高类别的能量分布
  - 反映了音乐的调性特征

- **Z轴（能量维度）**
  - 结合RMS能量和onset检测
  - 表示声音的强度变化
  - 捕捉音频的动态特征

### 2. 信号处理优化

- **噪声处理**
  - 实现了动态噪声门限
  - 添加了能量阈值检测
  - 优化了信号平滑处理

- **特征提取改进**
  - 优化了梅尔频谱计算
  - 改进了色度特征提取
  - 增强了节奏检测精度

### 3. 交互模式实现

- **模式识别**
  - 节奏模式检测
  - 持续音识别
  - 旋律变化追踪

- **视觉响应**
  - 自适应亮度调节
  - 动态颜色映射
  - 平滑过渡效果

## 关键算法实现

### 1. 音频特征提取
```python
def extract_audio_features(signal, sr, frame_length=2048, hop_length=512):
    # 梅尔频谱处理
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        n_mels=128,
        fmin=20,
        fmax=8000
    )
    
    # 色度特征提取
    chroma = librosa.feature.chroma_stft(...)
    
    # 节奏特征检测
    onset_env = librosa.onset.onset_strength(...)
```

### 2. 模式识别
```python
def detect_pattern(self, features):
    # 提取关键特征
    rhythm = np.mean(features['onset_env'])
    melody = np.mean(features['chroma'])
    energy = np.mean(features['rms'])
    
    # 模式判断逻辑
    if np.std(rhythm_pattern) > 0.2:
        return "rhythm"
    elif np.mean(energy_pattern) > 0.3:
        return "sustained"
    elif np.std(melody_pattern) > 0.15:
        return "melody"
```

### 3. 视觉映射
```python
def features_to_rgb(features, num_leds):
    # 能量检测
    energy_indicators = [
        rms_energy > energy_threshold,
        mel_energy > energy_threshold,
        onset_energy > energy_threshold * 0.5
    ]
    
    # 颜色映射
    rgb_values = np.zeros((num_leds, 3))
    # 低频区域
    rgb_values[low_mask] = [low_freq_energy * 255, energy * 50, 0]
    # 中频区域
    rgb_values[mid_mask] = mid_rgb
    # 高频区域
    rgb_values[high_mask] = [high_freq_energy * 100, centroid_norm * 255, centroid_norm * 255]
```

## 优化历程

1. **初始版本**
   - 基本的音频捕获
   - 简单的LED控制
   - 固定的颜色映射

2. **第一次优化**
   - 添加了噪声处理
   - 实现了动态范围调整
   - 改进了颜色映射算法

3. **交互增强**
   - 实现了模式识别
   - 添加了视觉反馈
   - 优化了响应机制

4. **性能优化**
   - 改进了数据处理效率
   - 优化了内存使用
   - 减少了处理延迟

## 未来改进方向

1. **特征提取**
   - 添加更多音频特征
   - 优化特征提取算法
   - 实现实时频率分析

2. **交互体验**
   - 增加更多交互模式
   - 改进模式识别准确度
   - 添加用户自定义功能

3. **视觉效果**
   - 实现更多视觉模式
   - 优化动画效果
   - 添加3D显示支持 