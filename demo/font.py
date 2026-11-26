"""A 5×7 bitmap font for the RGB scoreboard."""

import numpy as np


_GLYPHS = {
    "A": "0e11111f111111", "B": "1e11111e11111e", "C": "0e11101010110e",
    "D": "1e11111111111e", "E": "1f10101e10101f", "F": "1f10101e101010",
    "G": "0e11101711110f", "H": "1111111f111111", "I": "0e04040404040e",
    "J": "0702020212120c", "K": "11121418141211", "L": "1010101010101f",
    "M": "111b1515111111", "N": "11191915131311", "O": "0e11111111110e",
    "P": "1e11111e101010", "Q": "0e11111115120d", "R": "1e11111e141211",
    "S": "0f10100e01011e", "T": "1f040404040404", "U": "1111111111110e",
    "V": "11111111110a04", "W": "1111111515150a", "X": "11110a040a1111",
    "Y": "11110a04040404", "Z": "1f01020408101f",
    "0": "0e11131519110e", "1": "040c040404040e", "2": "0e11010204081f",
    "3": "1e01010601111e", "4": "02060a121f0202", "5": "1f10101e01011e",
    "6": "0e10101e11110e", "7": "1f010204080808", "8": "0e11110e11110e",
    "9": "0e11110f01010e",
    " ": "00000000000000", "'": "04040800000000", "!": "04040404040004",
    "-": "0000001f000000", ".": "00000000000404", "%": "19190204081313",
}
GLYPHS = {
    char: (np.frombuffer(bytes.fromhex(rows), dtype=np.uint8)[:, None] & (16, 8, 4, 2, 1)) != 0
    for char, rows in _GLYPHS.items()
}


def text_width(text, scale):
    return max(0, (6 * len(text) - 1) * scale)


def draw_text(image, x, y, text, color, scale=3):
    for char in text.upper():
        mask = GLYPHS.get(char, GLYPHS[" "]).repeat(scale, axis=0).repeat(scale, axis=1)
        w, h = 5 * scale, 7 * scale
        left, top = max(0, x), max(0, y)
        right, bottom = min(image.shape[1], x + w), min(image.shape[0], y + h)
        if left < right and top < bottom:
            image[top:bottom, left:right][mask[top-y:bottom-y, left-x:right-x]] = color
        x += 6 * scale
