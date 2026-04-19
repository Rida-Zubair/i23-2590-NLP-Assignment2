from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_curve(values: List[float], title: str, ylabel: str, path: Path, second_values: List[float] = None, second_label: str = 'Validation'):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(values) + 1), values, label='Train')
    if second_values is not None:
        plt.plot(range(1, len(second_values) + 1), second_values, label=second_label)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, path: Path):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha='right')
    plt.yticks(ticks, labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_attention_heatmap(weights: np.ndarray, tokens: List[str], title: str, path: Path):
    plt.figure(figsize=(10, 8))
    plt.imshow(weights, aspect='auto')
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(title)
    plt.xlabel('Key tokens')
    plt.ylabel('Query tokens')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
