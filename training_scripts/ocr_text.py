"""Lightweight OCR text normalization shared by evaluation and deployment."""

import re

import jaconv


def post_process(text):
    text="".join(text.split()).replace("\u2026", "...")
    text=re.sub("[\u30fb.]{2,}", lambda match:(match.end()-match.start())*".", text)
    return jaconv.h2z(text, ascii=True, digit=True)
