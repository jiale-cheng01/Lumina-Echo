import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import queue
import threading
import time
import sys
import pygame


# ----------------------
# 1. 音频捕获
# ----------------------
class AudioCapture:
    def __init__(self, device=None, channels=2, samplerate=44100, blocksize=2048):
        """初始化音频捕获

        Args:
            device: 音频设备索引或名称，None表示使用默认设备
            channels: 通道数
            samplerate: 采样率
            blocksize: 每次捕获的样本数
        """
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.audio_queue = queue.Queue()
        self.running = False

    def callback(self, indata, frames, time, status):
        """音频流回调函数"""
        if status:
            print(status)
        # 如果是立体声，转换为单声道
        if self.channels == 2:
            indata = np.mean(indata, axis=1)
        self.audio_queue.put(indata.copy())

    def start(self):
        """开始音频捕获"""
        self.running = True
        self.stream = sd.InputStream(
            device=self.device,
            channels=self.channels,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            callback=self.callback
        )
        self.stream.start()

    def stop(self):
        """停止音频捕获"""
        self.running = False
        self.stream.stop()
        self.stream.close()

    def get_audio_data(self):
        """获取音频数据"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None


# ----------------------
# 2. 实时处理控制器
# ----------------------
class LEDVisualizer:
    def __init__(self, num_leds=60, width=1200, height=400):
        """初始化 LED 可视化器"""
        pygame.init()
        self.width = width
        self.height = height
        self.num_leds = num_leds
        self.led_width = width // num_leds
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("LED Audio Visualizer")
        
        # 创建渐变背景
        self.background = pygame.Surface((width, height))
        self.create_gradient_background()
        
    def create_gradient_background(self):
        """创建渐变背景"""
        for y in range(self.height):
            # 从深蓝到黑色的渐变
            color = (0, 0, int(40 * (1 - y/self.height)))
            pygame.draw.line(self.background, color, (0, y), (self.width, y))
            
    def visualize(self, rgb_values):
        """显示 LED 状态"""
        # 处理 Pygame 事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
                
        # 绘制背景
        self.screen.blit(self.background, (0, 0))
        
        # 绘制 LED 灯效果
        for i, rgb in enumerate(rgb_values):
            x = i * self.led_width
            # 创建发光效果
            self.draw_glowing_led(x, rgb)
            
        pygame.display.flip()
        return True
        
    def draw_glowing_led(self, x, rgb):
        """绘制具有发光效果的 LED"""
        # LED 中心位置
        center_y = self.height // 2
        
        # 绘制发光效果（从暗到亮的渐变圆）
        for radius in range(30, 0, -5):
            alpha = int(255 * (radius / 30))
            glow_color = (*rgb, alpha)
            glow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, glow_color, (radius, radius), radius)
            self.screen.blit(glow_surface, 
                           (x + self.led_width//2 - radius, 
                            center_y - radius))
        
        # 绘制 LED 中心
        pygame.draw.circle(self.screen, rgb, 
                         (x + self.led_width//2, center_y), 5)
    
    def cleanup(self):
        """清理 Pygame 资源"""
        pygame.quit()

class AudioVisualizer:
    def __init__(self, num_leds=60, device=None):
        self.num_leds = num_leds
        self.frame_length = 2048
        self.hop_length = 512
        self.audio_capture = AudioCapture(
            device=device,
            blocksize=self.frame_length
        )
        self.running = False
        self.led_visualizer = LEDVisualizer(num_leds=num_leds)
        
        # 添加数据监控
        self.debug_mode = True
        self.data_monitor = {
            'audio_level': [],
            'mel_energy': [],
            'chroma_energy': [],
            'onset_strength': [],
            'rms_energy': []
        }
        self.monitor_window_size = 100  # 保存最近100帧的数据
        
        # 添加交互模式
        self.interaction_mode = True
        self.pattern_memory = {
            'rhythm': [],      # 记住节奏模式
            'melody': [],      # 记住旋律模式
            'energy': []       # 记住能量模式
        }
        self.memory_length = 50  # 记忆长度
        self.response_threshold = 0.6  # 响应阈值

    def detect_pattern(self, features):
        """检测输入的音频模式"""
        # 提取关键特征
        rhythm = np.mean(features['onset_env'])
        melody = np.mean(features['chroma'])
        energy = np.mean(features['rms'])
        
        # 更新模式记忆
        self.pattern_memory['rhythm'].append(rhythm)
        self.pattern_memory['melody'].append(melody)
        self.pattern_memory['energy'].append(energy)
        
        # 保持记忆长度
        for key in self.pattern_memory:
            if len(self.pattern_memory[key]) > self.memory_length:
                self.pattern_memory[key] = self.pattern_memory[key][-self.memory_length:]
        
        # 检测模式特征
        pattern_type = "none"
        if len(self.pattern_memory['rhythm']) > 10:
            rhythm_pattern = np.array(self.pattern_memory['rhythm'][-10:])
            energy_pattern = np.array(self.pattern_memory['energy'][-10:])
            
            # 检测节奏模式
            if np.std(rhythm_pattern) > 0.2:  # 强节奏
                pattern_type = "rhythm"
            # 检测持续音
            elif np.mean(energy_pattern) > 0.3 and np.std(energy_pattern) < 0.1:
                pattern_type = "sustained"
            # 检测旋律
            elif np.std(self.pattern_memory['melody'][-10:]) > 0.15:
                pattern_type = "melody"
        
        return pattern_type

    def generate_response(self, pattern_type):
        """根据检测到的模式生成视觉响应"""
        if pattern_type == "rhythm":
            # 节奏模式：脉冲式闪烁
            response = np.sin(np.linspace(0, 2*np.pi, self.num_leds)) * 255
            return np.column_stack([response, np.zeros_like(response), response])
        elif pattern_type == "sustained":
            # 持续音：渐变效果
            gradient = np.linspace(0, 1, self.num_leds)
            return np.column_stack([gradient*255, (1-gradient)*255, np.zeros_like(gradient)])
        elif pattern_type == "melody":
            # 旋律：彩虹波浪
            hues = np.linspace(0, 1, self.num_leds)
            colors = np.array([hsv_to_rgb(h, 1, 1) for h in hues]) * 255
            return colors
        else:
            return np.zeros((self.num_leds, 3), dtype=np.uint8)

    def monitor_data(self, audio_block, features):
        """监控数据流"""
        if not self.debug_mode:
            return

        # 计算音频电平（添加更多精确的计算）
        audio_level = np.sqrt(np.mean(audio_block**2))
        
        # 获取各特征的能量并添加更多的预处理
        mel_energy = np.mean(features['mel_spec'])
        chroma_energy = np.mean(features['chroma'])
        onset_strength = np.mean(features['onset_env'])
        rms_energy = np.mean(features['rms'])
        
        # 添加背景噪声检测
        background_noise = 0.0005
        is_signal = (audio_level > background_noise) or (mel_energy > 0.1)
        
        # 在交互模式下检测模式
        if self.interaction_mode and is_signal:
            pattern_type = self.detect_pattern(features)
            if pattern_type != "none":
                print(f"\r检测到模式: {pattern_type}", end="")
        
        # 更新监控数据
        self.data_monitor['audio_level'].append(audio_level)
        self.data_monitor['mel_energy'].append(mel_energy)
        self.data_monitor['chroma_energy'].append(chroma_energy)
        self.data_monitor['onset_strength'].append(onset_strength)
        self.data_monitor['rms_energy'].append(rms_energy)
        
        # 限制数据长度
        for key in self.data_monitor:
            if len(self.data_monitor[key]) > self.monitor_window_size:
                self.data_monitor[key] = self.data_monitor[key][-self.monitor_window_size:]
        
        # 打印实时数据（添加信号指示和频段信息）
        status = "【有效信号】" if is_signal else "【背景噪声】"
        
        # 计算低中高频段的能量
        mel_bands = features['mel_spec']
        low_freq = np.mean(mel_bands[:42])    # 低频段 (20-200Hz)
        mid_freq = np.mean(mel_bands[42:84])  # 中频段 (200-2kHz)
        high_freq = np.mean(mel_bands[84:])   # 高频段 (2k-8kHz)
        
        print(f"\r{status} | 音频电平: {audio_level:.6f} | "
              f"梅尔能量: {mel_energy:.6f} "
              f"[低:{low_freq:.3f} 中:{mid_freq:.3f} 高:{high_freq:.3f}] | "
              f"色度能量: {chroma_energy:.6f} | "
              f"节奏强度: {onset_strength:.6f} | "
              f"RMS能量: {rms_energy:.6f}", end="")

    def process_audio_block(self, audio_block):
        """处理单个音频块"""
        # 提取特征
        features = extract_audio_features(
            audio_block,
            self.audio_capture.samplerate,
            self.frame_length,
            self.hop_length
        )

        # 获取当前帧的特征
        current_features = {
            'mel_spec': features['mel_spec'],
            'chroma': features['chroma'],
            'spectral_centroid': features['spectral_centroid'],
            'onset_env': features['onset_env'],
            'rms': features['rms']
        }

        # 如果没有检测到特定模式，使用普通的可视化
        rgb_values = features_to_rgb(features, self.num_leds)
        return rgb_values

    def run(self):
        """运行可视化"""
        self.running = True
        self.audio_capture.start()

        try:
            while self.running:
                audio_block = self.audio_capture.get_audio_data()
                if audio_block is not None:
                    rgb_values = self.process_audio_block(audio_block.flatten())
                    # 使用 Pygame 可视化 LED 状态
                    if not self.led_visualizer.visualize(rgb_values):
                        break
                time.sleep(0.001)  # 小的延迟以避免CPU过载

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.stop()

    def stop(self):
        """停止可视化"""
        self.running = False
        self.audio_capture.stop()
        self.led_visualizer.cleanup()


# ----------------------
# 1. 读取音频文件
# ----------------------
def load_audio(file_path):
    """加载音频文件，返回信号和采样率"""
    signal, sr = librosa.load(file_path, sr=None, mono=True)  # 强制单声道
    return signal, sr


# ----------------------
# 2. 基频检测（音高提取）
# ----------------------
def detect_pitch(signal, sr, frame_length=2048, hop_length=512):
    """使用 librosa 的 YIN 算法检测基频"""
    # 计算基频（单位：Hz）
    pitches, magnitudes = librosa.piptrack(
        y=signal,
        sr=sr,
        S=None,
        n_fft=frame_length,
        hop_length=hop_length,
        fmin=50,  # 最低检测频率（Hz）
        fmax=2000  # 最高检测频率（Hz）
    )

    # 使用 NumPy 向量化操作替代循环
    max_magnitude_indices = magnitudes.argmax(axis=0)
    pitch = pitches[max_magnitude_indices, np.arange(pitches.shape[1])]
    return pitch[pitch > 0]  # 过滤无效值


# ----------------------
# 3. 频率转音符名称
# ----------------------
def freq_to_note(freq):
    """将频率转换为最接近的音符名称（如 C4, A4）"""
    A4 = 440.0  # 标准音高 A4 = 440Hz
    if freq == 0:
        return "N/A"
    semitone = 12 * np.log2(freq / A4) + 69  # 计算半音值
    semitone = round(semitone)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = semitone // 12 - 1
    note = notes[semitone % 12]
    return f"{note}{octave}"


# ----------------------
# 3.1 和弦分析
# ----------------------
def get_note_without_octave(note):
    """从音符名称中提取音名（不含八度）"""
    return note[:-1] if note != "N/A" else "N/A"


def identify_chord(notes):
    """识别和弦类型

    基于同时出现的音符识别可能的和弦类型
    支持以下和弦类型：
    - 大三和弦 (Major)
    - 小三和弦 (Minor)
    - 增三和弦 (Augmented)
    - 减三和弦 (Diminished)
    - 大七和弦 (Major 7th)
    - 小七和弦 (Minor 7th)
    """
    if not notes or len(notes) < 2:
        return "No Chord"

    # 移除重复音符和八度信息
    unique_notes = list(set([get_note_without_octave(n) for n in notes if n != "N/A"]))
    if len(unique_notes) < 2:
        return "No Chord"

    # 定义常见和弦模式
    chord_patterns = {
        # 大三和弦
        'major': [0, 4, 7],
        # 小三和弦
        'minor': [0, 3, 7],
        # 增三和弦
        'augmented': [0, 4, 8],
        # 减三和弦
        'diminished': [0, 3, 6],
        # 大七和弦
        'major7': [0, 4, 7, 11],
        # 小七和弦
        'minor7': [0, 3, 7, 10],
    }

    # 将音符转换为相对半音数
    notes_to_semitones = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3,
        'E': 4, 'F': 5, 'F#': 6, 'G': 7,
        'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }

    # 计算音符间的相对间隔
    intervals = []
    root_note = unique_notes[0]
    root_value = notes_to_semitones[root_note]

    for note in unique_notes[1:]:
        value = notes_to_semitones[note]
        interval = (value - root_value) % 12
        intervals.append(interval)
    intervals.sort()

    # 匹配和弦模式
    for chord_name, pattern in chord_patterns.items():
        if len(intervals) >= len(pattern) - 1:
            matches = all(i in pattern[1:] for i in intervals)
            if matches:
                return f"{root_note} {chord_name}"

    return "Unknown Chord"


def analyze_chord_progression(notes_sequence, window_size=4):
    """分析和弦进行

    Args:
        notes_sequence: 音符序列
        window_size: 同时分析的音符数量

    Returns:
        和弦进行列表
    """
    chords = []
    for i in range(0, len(notes_sequence), window_size):
        window = notes_sequence[i:i + window_size]
        chord = identify_chord(window)
        if chord != "No Chord" and chord not in chords[-1:]:  # 避免重复相邻和弦
            chords.append(chord)
    return chords


# ----------------------
# 4. 可视化结果
# ----------------------
def plot_results(signal, sr, pitches):
    """绘制音频波形和音高变化"""
    plt.figure(figsize=(12, 8))

    # 波形图
    plt.subplot(2, 1, 1)
    time = np.arange(len(signal)) / sr
    plt.plot(time, signal, alpha=0.5)
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # 音高图
    plt.subplot(2, 1, 2)
    pitch_times = librosa.frames_to_time(
        np.arange(len(pitches)),
        sr=sr,
        hop_length=hop_length,
        n_fft=frame_length
    )
    plt.plot(pitch_times, pitches, label='Pitch (Hz)', color='r')
    plt.title("Pitch Detection")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()


# ----------------------
# 1. 音频特征提取
# ----------------------
def extract_audio_features(signal, sr, frame_length=2048, hop_length=512):
    """提取音频的多个特征

    Returns:
        dict: 包含多个音频特征的字典
    """
    # 1. 频谱
    D = librosa.stft(signal, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(D)

    # 2. 梅尔频谱 - 改进处理
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        n_mels=128,
        fmin=20,  # 设置最低频率
        fmax=8000  # 设置最高频率
    )
    # 转换为分贝单位并归一化
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)

    # 3. 色度特征（音高类别）
    chroma = librosa.feature.chroma_stft(
        S=magnitude,
        sr=sr,
        hop_length=hop_length
    )

    # 4. 谱质心（音色亮度）
    spectral_centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length
    )

    # 5. 节奏特征
    onset_env = librosa.onset.onset_strength(
        y=signal,
        sr=sr,
        hop_length=hop_length
    )

    # 6. RMS能量
    rms = librosa.feature.rms(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length
    )

    return {
        'mel_spec': mel_spec,
        'chroma': chroma,
        'spectral_centroid': spectral_centroid,
        'onset_env': onset_env,
        'rms': rms
    }


# ----------------------
# 2. 特征到RGB映射
# ----------------------
def features_to_rgb(features, num_leds):
    """将音频特征映射到RGB值，使用笛卡尔坐标系处理多维特征"""
    # 获取当前帧的特征值并进行预处理
    mel_spec = features['mel_spec']
    chroma = features['chroma']
    centroid = features['spectral_centroid']
    onset = features['onset_env']
    rms = features['rms']

    # 改进的能量检测和噪声门限
    energy_threshold = 0.005  # 降低能量门限
    noise_floor = 0.002      # 降低噪声底线
    
    # 计算综合能量指标
    rms_energy = np.mean(rms)
    mel_energy = np.mean(mel_spec)
    onset_energy = np.mean(onset)
    
    # 使用多个指标来判断是否有实际的音频信号
    energy_indicators = [
        rms_energy > energy_threshold,
        mel_energy > energy_threshold,
        onset_energy > energy_threshold * 0.5  # 节奏检测可以稍微宽松一些
    ]
    
    # 只有当多个指标都表明有信号时才继续处理
    if not any(energy_indicators):
        return np.zeros((num_leds, 3), dtype=np.uint8)

    # 创建三维特征空间
    mel_bands = np.linspace(0, 1, mel_spec.shape[0])
    mel_energy = np.mean(mel_spec, axis=1)
    
    # 改进的归一化处理
    def normalize_with_dynamic_range(data, min_val=0.0, max_val=1.0):
        """带动态范围的归一化"""
        if np.all(data == 0):
            return np.zeros_like(data)
        data_range = np.ptp(data)
        if data_range < 1e-6:  # 避免除以零
            return np.zeros_like(data)
        return np.clip(
            (data - np.min(data)) / data_range * (max_val - min_val) + min_val - noise_floor,
            0, 1
        )

    mel_normalized = normalize_with_dynamic_range(mel_energy)
    chroma_energy = np.mean(chroma, axis=1)
    chroma_normalized = normalize_with_dynamic_range(chroma_energy)
    rms_normalized = normalize_with_dynamic_range(rms.flatten())
    onset_normalized = normalize_with_dynamic_range(onset.flatten(), min_val=0.3)
    
    # 使用平滑的能量计算
    energy = np.mean([
        np.mean(rms_normalized),
        np.mean(onset_normalized),
        np.mean(mel_normalized)
    ])
    energy = np.clip(energy, 0, 1)

    # 创建LED索引数组
    led_indices = np.arange(num_leds)
    section_size = num_leds // 3

    # 预计算每个区域的颜色值，添加非线性映射
    low_freq_index = len(mel_bands) // 3
    low_freq_energy = np.power(np.mean(mel_normalized[:low_freq_index]), 1.5)  # 非线性增强低频
    
    high_freq_index = 2 * len(mel_bands) // 3
    high_freq_energy = np.power(np.mean(mel_normalized[high_freq_index:]), 1.2)  # 非线性增强高频
    
    centroid_norm = np.mean(np.interp(centroid.flatten(), (centroid.min(), centroid.max()), (0, 1)))
    hue = np.mean(chroma_normalized)
    mid_rgb = np.array(hsv_to_rgb(hue, 1, 1)) * 255

    # 使用布尔索引和广播创建RGB数组
    rgb_values = np.zeros((num_leds, 3))

    # 低频区域 - 红色渐变
    low_mask = led_indices < section_size
    rgb_values[low_mask] = np.array([low_freq_energy * 255, energy * 50, 0])

    # 中频区域 - 使用色度特征
    mid_mask = (led_indices >= section_size) & (led_indices < 2 * section_size)
    rgb_values[mid_mask] = mid_rgb

    # 高频区域 - 青色到白色渐变
    high_mask = led_indices >= 2 * section_size
    rgb_values[high_mask] = np.array([
        high_freq_energy * 100,  # 减少红色分量
        centroid_norm * 255,
        centroid_norm * 255
    ])

    # 使用平滑的能量特征调制整体亮度
    rgb_values = rgb_values * energy

    return rgb_values.astype(np.uint8)


def hsv_to_rgb(h, s, v):
    """HSV颜色空间转RGB"""
    if s == 0.0:
        return (v, v, v)

    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    if i == 5:
        return (v, p, q)


# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    # 列出可用的音频设备
    print("可用的音频设备:")
    devices = sd.query_devices()

    # 创建一个集合来存储唯一的设备名称
    unique_devices = {}
    for i, device in enumerate(devices):
        # 提取基本设备名称（去除API信息）
        base_name = device['name'].split(',')[0].strip()
        if base_name not in unique_devices:
            unique_devices[base_name] = {
                'index': i,
                'inputs': device['max_input_channels'],
                'outputs': device['max_output_channels']
            }

    # 显示唯一的设备列表
    print("\n可用的音频设备（已去重）:")
    for name, info in unique_devices.items():
        print(f"{info['index']}: {name} ({info['inputs']} in, {info['outputs']} out)")

    # 查找虚拟声卡设备
    virtual_cable = None
    for name, info in unique_devices.items():
        # 寻找VB-CABLE输出设备
        if 'CABLE Output' in name and info['inputs'] > 0:
            virtual_cable = info['index']
            break

    if virtual_cable is None:
        print("\n未找到虚拟声卡设备。请按照以下步骤设置：")
        print("1. 下载并安装 VB-CABLE (https://vb-audio.com/Cable/)")
        print("2. 在Windows声音设置中将CABLE Input设置为默认播放设备")
        print("3. 重新运行此程序")
        sys.exit(1)

    print(f"\n使用虚拟声卡设备: {devices[virtual_cable]['name']}")
    print("提示：如果想同时听到声音，请在Windows声音设置中：")
    print("1. 打开声音设置 -> 声音 -> 录制")
    print("2. 找到'CABLE Output'")
    print("3. 右键点击 -> 属性 -> 侦听")
    print("4. 勾选'监听此设备'并选择您的扬声器")

    # 创建可视化器实例
    visualizer = AudioVisualizer(num_leds=60, device=virtual_cable)

    # 运行可视化
    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\n程序已停止")
