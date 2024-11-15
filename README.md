
## 项目概述  
  
本项目是一个"音频相似性分析工具"，旨在通过提取音频文件的MFCC（梅尔频率倒谱系数）特征，并使用余弦相似度算法来找出指定目录中具有高度相似性的音频文件。  
  
## 主要功能  
  
- **提取MFCC特征**：使用`librosa`库加载音频文件，并提取其MFCC特征。  
- **扫描目录**：遍历指定目录及其子目录，查找支持格式的音频文件（如".mp3", ".wav"等），并提取特征。  
- **查找相似音频**：使用余弦相似度算法比较音频文件的特征向量，找出相似音频。  
- **输出结果**：将相似音频对及其相似度值写入文本文件。  
  
## 使用方法  
  
1. **准备环境**：  
   - 安装Python及其必要的库：`librosa`, `numpy`, `scikit-learn`。  
   - 可以使用以下命令进行安装：  
     ```bash  
     pip install librosa numpy scikit-learn  
     ```  
  
2. **运行程序**：  
   - 将项目代码保存为一个Python文件（例如`audio_similarity.py`）。  
   - 修改`main`函数中的`directory_path`变量，指向包含音频文件的目录路径。  
   - 在命令行中运行Python脚本：  
     ```bash  
     python audio_similarity.py  
     ```  
  
3. **查看结果**：  
   - 程序运行后，将在同一目录下生成一个名为`similar_audios.txt`的文件，其中列出了相似音频的信息。  
  
## 注意事项  
  
- **性能考虑**：对于大型音频库或高分辨率音频文件，特征提取和相似度计算可能需要较长时间和大量内存。  
- **格式支持**：当前版本仅支持上述提到的音频格式。如果需要支持更多格式，请自行修改代码。  
- **阈值调整**：相似度阈值和返回的最相似音频数量可以根据具体需求进行调整。  
- **错误处理**：程序包含了基本的错误处理逻辑。如果遇到问题，请检查音频文件是否损坏或格式是否正确。  
  
## 依赖库  
  
- `librosa`：用于音频处理和特征提取。  
- `numpy`：用于数值计算和数组操作。  
- `scikit-learn`：提供余弦相似度算法。  
- `os`：用于文件和目录操作。  
  
## 贡献与扩展  
  
- 欢迎对本项目进行贡献，包括代码优化、功能扩展和错误修复。  
- 如果有任何建议或需要支持更多功能，请随时联系我们。  
  
## 联系方式  
  
- 如果有任何问题，请通过[GitHub Issues](https://github.com/ikros1/MusicFind/issues)进行联系
- 也可以发送电子邮件至[ikros1@outlook.com](ikros1@outlook.com)、列表、代码块等）。