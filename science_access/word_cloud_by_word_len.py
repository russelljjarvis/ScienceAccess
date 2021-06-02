from __future__ import division

import numpy as np
from random import Random


import warnings
from random import Random
import io
import os
import re
import base64
import sys
import colorsys
import matplotlib
import numpy as np
from operator import itemgetter
from xml.sax import saxutils

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageFont

# from wordcloud import WordCloud

from wordcloud.query_integral_image import query_integral_image
from wordcloud.tokenization import unigrams_and_bigrams, process_tokens
from nltk.corpus import words as english_words

import dask
import pandas as pd

FILE = os.path.dirname(__file__)
# FONT_PATH = os.environ.get('FONT_PATH', os.path.join(FILE, 'DroidSansMono.ttf'))
# STOPWORDS = set(map(str.strip, open(os.path.join(FILE, 'stopwords')).readlines()))


class IntegralOccupancyMap(object):
    def __init__(self, height, width, mask):
        self.height = height
        self.width = width
        if mask is not None:
            # the order of the cumsum's is important for speed ?!
            self.integral = np.cumsum(np.cumsum(255 * mask, axis=1), axis=0).astype(
                np.uint32
            )
        else:
            self.integral = np.zeros((height, width), dtype=np.uint32)

    def sample_position(self, size_x, size_y, random_state):
        return query_integral_image(self.integral, size_x, size_y, random_state)

    def update(self, img_array, pos_x, pos_y):
        partial_integral = np.cumsum(
            np.cumsum(img_array[pos_x:, pos_y:], axis=1), axis=0
        )
        # paste recomputed part into old image
        # if x or y is zero it is a bit annoying
        if pos_x > 0:
            if pos_y > 0:
                partial_integral += (
                    self.integral[pos_x - 1, pos_y:]
                    - self.integral[pos_x - 1, pos_y - 1]
                )
            else:
                partial_integral += self.integral[pos_x - 1, pos_y:]
        if pos_y > 0:
            partial_integral += self.integral[pos_x:, pos_y - 1][:, np.newaxis]

        self.integral[pos_x:, pos_y:] = partial_integral


def wrapper(w):
    try:
        if w[0] in english_words.words():
            return w
        else:
            return None
    except:
        return None


import copy

from nltk.tokenize import word_tokenize
import streamlit as st


def generate_from_lengths(self, words, max_font_size=None, verbose=False):  # noqa: C901
    """Create a word_cloud from words and frequencies.
    Parameters
    ----------
    frequencies : dict from string to float
        A contains words and associated frequency.
    max_font_size : int
        Use this font-size instead of self.max_font_size
    Returns
    -------
    self
    """
    # make sure frequencies are sorted and normalized
    """
    frequencies = sorted(frequencies.items(), key=itemgetter(1), reverse=True)
    if len(frequencies) <= 0:
        raise ValueError("We need at least 1 word to plot a word cloud, "
                            "got %d." % len(frequencies))
    frequencies = frequencies[:self.max_words]
    """
    # largest entry will be 1
    self.max_words = 30
    words = word_tokenize(words)
    wordss = list(set(words))
    wordss = [word for word in wordss if len(word)]

    sizes = [len(word) for word in wordss if len(word)]

    if verbose:
        st.text(wordss)
        st.text(sizes)

    max_len = np.max(sizes)

    frequencies = [(word, word_len / max_len) for word, word_len in zip(words, sizes)]
    real_frequencies = [w for w in frequencies if w is not None]

    frequencies = sorted(frequencies, key=lambda item: item[1],reverse=True)
    if len(frequencies)>30:
        frequencies = frequencies[0:30]
    max_frequency = float(frequencies[0][1])

    #real_frequencies = [(wrapper)(w) for w in frequencies]
    #frequencies = sorted(real_frequencies, key=lambda item: item[1])#, reverse=True)
    self.word_counts_fz = None
    self.word_counts_fz = frequencies

    self.biggest_words = None
    self.biggest_words = frequencies  # [0:2]
    if self.random_state is not None:
        random_state = self.random_state
    else:
        random_state = Random()

    if self.mask is not None:
        boolean_mask = self._get_bolean_mask(self.mask)
        width = self.mask.shape[1]
        height = self.mask.shape[0]
    else:
        boolean_mask = None
        height, width = self.height, self.width
    occupancy = IntegralOccupancyMap(height, width, boolean_mask)

    # create image
    img_grey = Image.new("L", (width, height))
    draw = ImageDraw.Draw(img_grey)
    img_array = np.asarray(img_grey)
    font_sizes, positions, orientations, colors = [], [], [], []

    last_freq = 1.0

    if max_font_size is None:
        # if not provided use default font_size
        max_font_size = self.max_font_size*2.5

    if max_font_size is None:
        # figure out a good font size by trying to draw with
        # just the first two words
        if len(frequencies) == 1:
            # we only have one word. We make it big!
            font_size = self.height
        else:
            if hasattr(self,'layout_'):
                sizes = [x[1] for x in self.layout_]
            try:
                font_size = int(2 * sizes[0] * sizes[1] / (sizes[0] + sizes[1]))
            # quick fix for if self.layout_ contains less than 2 values
            # on very small images it can be empty
            except IndexError:
                try:
                    font_size = sizes[0]
                except IndexError:
                    raise ValueError(
                        "Couldn't find space to draw. Either the Canvas size"
                        " is too small or too much of the image is masked "
                        "out."
                    )
    else:
        font_size = max_font_size

    # we set self.words_ here because we called generate_from_frequencies
    # above... hurray for good design?
    self.words_ = dict(frequencies)

    if self.repeat and len(frequencies) < self.max_words:
        # pad frequencies with repeating words.
        times_extend = int(np.ceil(self.max_words / len(frequencies))) - 1
        # get smallest frequency
        frequencies_org = list(frequencies)
        downweight = frequencies[-1][1]
        for i in range(times_extend):
            frequencies.extend(
                [(word, freq * downweight ** (i + 1)) for word, freq in frequencies_org]
            )

    # start drawing grey image
    for word, freq in frequencies:
        if freq == 0:
            continue
        # select the font size
        rs = self.relative_scaling
        if rs != 0:
            font_size = int(
                round((rs * (freq / float(last_freq)) + (1 - rs)) * font_size)
            )
        if random_state.random() < self.prefer_horizontal:
            orientation = None
        else:
            orientation = Image.ROTATE_90
        tried_other_orientation = False
        while True:
            # try to find a position
            font = ImageFont.truetype(self.font_path, font_size)
            # transpose font optionally
            transposed_font = ImageFont.TransposedFont(font, orientation=orientation)
            # get size of resulting text
            box_size = draw.textsize(word, font=transposed_font)
            # find possible places using integral image:
            result = occupancy.sample_position(
                box_size[1] + self.margin, box_size[0] + self.margin, random_state
            )
            if result is not None or font_size < self.min_font_size:
                # either we found a place or font-size went too small
                break
            # if we didn't find a place, make font smaller
            # but first try to rotate!
            if not tried_other_orientation and self.prefer_horizontal < 1:
                orientation = (
                    Image.ROTATE_90 if orientation is None else Image.ROTATE_90
                )
                tried_other_orientation = True
            else:
                font_size -= self.font_step
                orientation = None

        if font_size < self.min_font_size:
            # we were unable to draw any more
            break

        x, y = np.array(result) + self.margin // 2
        # actually draw the text
        draw.text((y, x), word, fill="white", font=transposed_font)
        positions.append((x, y))
        orientations.append(orientation)
        font_sizes.append(font_size)
        colors.append(
            self.color_func(
                word,
                font_size=font_size,
                position=(x, y),
                orientation=orientation,
                random_state=random_state,
                font_path=self.font_path,
            )
        )
        # recompute integral image
        if self.mask is None:
            img_array = np.asarray(img_grey)
        else:
            img_array = np.asarray(img_grey) + boolean_mask
        # recompute bottom right
        # the order of the cumsum's is important for speed ?!
        occupancy.update(img_array, x, y)
        last_freq = freq

    self.layout_ = list(zip(frequencies, font_sizes, positions, orientations, colors))
    return self
