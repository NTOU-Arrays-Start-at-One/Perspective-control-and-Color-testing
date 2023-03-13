import os
import matplotlib.pyplot as plt

def save_image_file(file, idx = 0):
    while os.path.exists(f"{file}{idx}.png"): idx += 1
    if idx == 0: plt.savefig(f"{file}.png")
    else: plt.savefig(f"{file}{idx}.png")