from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Any, Dict, Tuple, Set, Iterable

from matplotlib import pyplot as plt
import numpy as np
import PIL.Image
import spacy.tokens
import torch
import torch.nn.functional as F

from .evaluate import compute_ioa
from .utils import compute_token_merge_indices, cached_nlp, auto_autocast

__all__ = ['GlobalHeatMap', 'RawHeatMapCollection', 'WordHeatMap', 'ParsedHeatMap', 'SyntacticHeatMapPair']


def plot_overlay_heat_map(im, heat_map, figsize: Tuple[int, int] = (10,10)):
    # type: (PIL.Image.Image | np.ndarray, torch.Tensor) -> None

    with auto_autocast(dtype=torch.float32):
        plt.figure(figsize=figsize)
        plt.axis('off')
        im = np.array(im)
        plt.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet')

        im = torch.from_numpy(im).float() / 255
        im = torch.cat((im, (1 - heat_map.unsqueeze(-1))), dim=-1)
        plt.imshow(im)


class WordHeatMap:
    def __init__(self, heatmap: torch.Tensor, word: str = None, word_idx: int = None):
        self.word = word
        self.word_idx = word_idx
        self.heatmap = heatmap

    @property
    def value(self):
        return self.heatmap

    def plot_overlay(self, image, figsize: Tuple[int, int] = (10,10)):
        # type: (PIL.Image.Image | np.ndarray, Path, bool, plt.Axes, Dict[str, Any]) -> None
        plot_overlay_heat_map(image, self.expand_as(image), figsize)

    def expand_as(self, image):
        # type: (PIL.Image.Image, bool, float, bool, Dict[str, Any]) -> torch.Tensor

        im = self.heatmap.unsqueeze(0).unsqueeze(0)
        im = F.interpolate(im.float().detach(), size=(image.size[0], image.size[1]), mode='bicubic')
        im = im[0,0]
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)
        im = im.cpu().detach().squeeze()
        
        return im

    def compute_ioa(self, other: 'WordHeatMap'):
        return compute_ioa(self.heatmap, other.heatmap)


@dataclass
class SyntacticHeatMapPair:
    head_heat_map: WordHeatMap
    dep_heat_map: WordHeatMap
    head_text: str
    dep_text: str
    relation: str


@dataclass
class ParsedHeatMap:
    word_heat_map: WordHeatMap
    token: spacy.tokens.Token


class GlobalHeatMap:
    def __init__(self, tokenizer: Any, prompt: str, heat_maps: torch.Tensor):
        self.tokenizer = tokenizer
        self.heat_maps = heat_maps
        self.prompt = prompt
        self.compute_word_heat_map = lru_cache(maxsize=50)(self.compute_word_heat_map)

    def compute_word_heat_map(self, word: str, word_idx: int = None, offset_idx: int = 0) -> WordHeatMap:
        merge_idxs, word_idx = compute_token_merge_indices(self.tokenizer, self.prompt, word, word_idx, offset_idx)
        return WordHeatMap(self.heat_maps[merge_idxs].mean(0), word, word_idx)

    def parsed_heat_maps(self) -> Iterable[ParsedHeatMap]:
        for token in cached_nlp(self.prompt):
            try:
                heat_map = self.compute_word_heat_map(token.text)
                yield ParsedHeatMap(heat_map, token)
            except ValueError:
                pass

    def dependency_relations(self) -> Iterable[SyntacticHeatMapPair]:
        for token in cached_nlp(self.prompt):
            if token.dep_ != 'ROOT':
                try:
                    dep_heat_map = self.compute_word_heat_map(token.text)
                    head_heat_map = self.compute_word_heat_map(token.head.text)

                    yield SyntacticHeatMapPair(head_heat_map, dep_heat_map, token.head.text, token.text, token.dep_)
                except ValueError:
                    pass


RawHeatMapKey = Tuple[int, int, int]  # factor, layer, head


class RawHeatMapCollection:
    def __init__(self):
        self.ids_to_heatmaps: Dict[RawHeatMapKey, torch.Tensor] = defaultdict(lambda: 0.0)
        self.ids_to_num_maps: Dict[RawHeatMapKey, int] = defaultdict(lambda: 0)

    def update(self, factor: int, layer_idx: int, head_idx: int, heatmap: torch.Tensor):
        with auto_autocast(dtype=torch.float32):
            key = (factor, layer_idx, head_idx)
            self.ids_to_heatmaps[key] = self.ids_to_heatmaps[key] + heatmap

    def factors(self) -> Set[int]:
        return set(key[0] for key in self.ids_to_heatmaps.keys())

    def layers(self) -> Set[int]:
        return set(key[1] for key in self.ids_to_heatmaps.keys())

    def heads(self) -> Set[int]:
        return set(key[2] for key in self.ids_to_heatmaps.keys())

    def __iter__(self):
        return iter(self.ids_to_heatmaps.items())

    def clear(self):
        self.ids_to_heatmaps.clear()
        self.ids_to_num_maps.clear()
