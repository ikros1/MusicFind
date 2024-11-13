import os
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

def extract_mfcc_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        print('解析特征：' + str(file_path))
        return mfccs_mean
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


def scan_directory_for_music_files(directory):
    supported_formats = ('.mp3', '.wav', '.ogg', '.flac')
    features_dict = {}
    file_list_queue = Queue()
    lock = threading.Lock()
    
    # 遍历目录，将符合条件的文件路径加入队列
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                file_path = os.path.join(root, file)
                file_list_queue.put(file_path)
    
    # 定义线程工作函数
    def worker():
        while True:
            try:
                file_path = file_list_queue.get_nowait()
            except Empty:
                break
            mfccs = extract_mfcc_features(file_path)
            if mfccs is not None:
                with lock:
                    features_dict[file_path] = mfccs
            file_list_queue.task_done()
    
    # 使用 ThreadPoolExecutor 来管理线程池
    with ThreadPoolExecutor(max_workers=80) as executor:
        # 提交所有工作线程
        for _ in range(80):
            executor.submit(worker)
        
        # 等待所有文件处理完毕
        file_list_queue.join()
    
    return features_dict


def find_similar_audios(features_dict, threshold=0.9999, top_n=5):
    if not features_dict:
        print("No features to compare.")
        return []

    all_features = np.vstack(list(features_dict.values()))
    similarities = cosine_similarity(all_features)

    similar_audios = {}
    for i in range(len(similarities)):
        original_file_path = list(features_dict.keys())[i]
        # Create a mask to exclude the comparison with itself
        mask = np.ones_like(similarities[i], dtype=bool)
        mask[i] = False  # Set to False for the self-comparison

        # Get sorted indices excluding the self-comparison
        sorted_indices = np.argsort(-similarities[i, mask])[:top_n]

        # Map these indices back to the original indices considering the mask
        original_sorted_indices = np.where(mask)[0][sorted_indices]

        similar_list = [(list(features_dict.keys())[j], similarities[i, j]) for j in original_sorted_indices if
                        similarities[i, j] >= threshold]
        if similar_list:
            similar_audios[original_file_path] = similar_list

    return similar_audios


def main():
    directory_path = 'path/'
    features_dict = scan_directory_for_music_files(directory_path)
    similar_audios = find_similar_audios(features_dict)

    output_file = 'similar_audios.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for orig_path, similar_list in similar_audios.items():
            f.write(f"\nOriginal File: {orig_path}\n")
            for sim_path, similarity in similar_list:
                f.write(f"Similar File: {sim_path} (Similarity: {similarity:.4f})\n")


if __name__ == "__main__":
    main()
