#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gradio uygulamasını başlatma scripti"""

from src.ui.gradio_app import interface

if __name__ == "__main__":
    print("Gradio arayüzü başlatılıyor...")
    interface.launch(debug=True, share=False)
