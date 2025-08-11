# source_code_feature_extraction.py
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch.nn as nn
import os
from unixcoder import UniXcoder

def convert_code_to_vector(code_list):
    """
    Convert a list of source code files to feature vectors using UniXcoder

    Args:
        code_list (list): List of source code strings

    Returns:
        list: List of numpy arrays containing 240-dimensional feature vectors
    """
    # Initialize model
    model_name = "microsoft/unixcoder-base-unimodal"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unixcoder_model = UniXcoder(model_name).to(device)
    unixcoder_model.eval()

    vectors = []
    for code in code_list:
        input_ids = unixcoder_model.tokenize([code], mode="<encoder-only>")
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = unixcoder_model(input_ids)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            sentence_embeddings = outputs[:, 0, :].squeeze().cpu().numpy()
        vector = np.resize(sentence_embeddings, 240)  # Resize to target dimension
        vectors.append(vector)
    return vectors


def read_cpp_files_and_labels(directory_path):
    """
    Read C++ source code files and their labels from a directory

    Args:
        directory_path (str): Path to directory containing .cpp files

    Returns:
        tuple: (list of code strings, list of labels, list of filenames)
    """
    code_files = []
    labels = []
    filenames = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".cpp"):
            label = int(filename.split('_')[0])
            labels.append(label)
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    code = file.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as file:
                    code = file.read()
            code_files.append(code)
            filenames.append(filename)
    return code_files, labels, filenames