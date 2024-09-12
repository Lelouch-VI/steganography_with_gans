# steganography_with_gans
An alternative implementation of Steganography using Generative Adversarial Networks (GANs for short). This project based on and heavily inspired by the paper [_SteganoGAN: High Capacity Image Steganography with GANs_]([url](https://arxiv.org/abs/1901.03892)), by Zhang, et al. (2019) and some other works.

**THIS PROJECT IS STILL A WORK IN PROGRESS**

**Novelties from SteganoGAN:**
- The original conducted training and testing on the [MSCOCO]([url](https://cocodataset.org/)) and [DIV2K]([url](https://data.vision.ee.ethz.ch/cvl/DIV2K/)) datasets. This project is trained on data from the [IStego100k]([url](https://github.com/YangzlTHU/IStego100K)), which is more recent and specifically tailored to steganography and steganalysis
- Throughout the project, the LeakyReLU activation function which was implemented in the original (and several tangential works) is replaced with the newer, more novel Mish function. Further information on Mish can be found in the paper [_Mish: A Self Regularized Non-Monotonic Activation Function_]([url](https://arxiv.org/abs/1908.08681)), by Diganta Misra (2020).
- The original implemented a basic encoder and decoder, as well as ones with dense and residual connections, but only implemented a basic critic. Building on that, this project implements a critic with dense and residual connections

The research paper for this project, as of its state in 8/2024 can be found in the 'Research' folder.
Testing of individual project files and functions can be found in the 'File testing' folder.

**Current state:** Project files uploaded, but not yet implemented in such a way which would allow for use or modular testing

**Future goals 9/2024:**
- Link files into a more concise format so that this repo may be used for more modular testing
- Clean up code for efficiency and aesthetics
- Implement testing more robustly across critic/encoder-decoder pairings
- Implement testing and validation on larger datasets, taken as samples from IStego100k
- Implement testing with bigger batch sizes and more epochs
