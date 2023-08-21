# Generative Pre-trained Transformer

A small GPT trained using PyTorch, and bigram language model from Karpathy lecture. Support added for training on Intel Arc A750 GPU.

Model is saved to file `model.pt` if running with no arguments. Pretrained model can be loaded by appending filename to command line argument `python bigram.py model.pt`. Output is saved to `output.txt`.

Input was Foundation by Isaac Asimov and training took around 20 minutes on my GPU. You can use your own input or get the full text here: https://archive.org/stream/AsimovTheFoundation/Asimov_the_foundation_djvu.txt

