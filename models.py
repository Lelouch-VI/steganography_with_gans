''' LIBRARIES '''
import json
import datetime
import matplotlib.pyplot as plt
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torchvision import datasets, transforms
from IPython.display import clear_output
import torchvision
from torch.optim import Adam # Adam optimizer
from tqdm import tqdm
import torch
import os
import gc
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# My own work
from loader  import MyDataset
from encoder import BasicMishEncoder, DenseMishEncoder, ResidualMishEncoder
from decoder import BasicMishDecoder, DenseMishDecoder
from critics import BasicMishCritic, DenseMishCritic, ResidualMishCritic

# SteganoGAN utils file
from utils import bits_to_bytearray, bytearray_to_text, ssim, text_to_bits

# Constants
DEFAULT_PATH = os.path.join(
    os.getcwd(),
    'train'
)
# os.makedirs(DEFAULT_PATH, exist_ok=True)

# Metrics
METRIC_FIELDS = [
  'val.encoder_mse',
  'val.decoder_loss',
  'val.decoder_acc',
  'val.cover_score',
  'val.generated_score',
  'val.ssim',
  'val.psnr',
  'val.bpp',
  'train.encoder_mse',
  'train.decoder_loss',
  'train.decoder_acc',
  'train.cover_score',
  'train.generated_score',
]





class MISHSteganoGAN():
  # Initialization and Setup
  def __init__(self, data_depth, encoder, decoder, critic,
                cuda=False, log_dir=None, verbose=False):
    """ 
    Parameters:
    - data_depth: int, depth of the data being embedded.
    - encoder: Encoder instance or class.
    - decoder: Decoder instance or class.
    - critic: Critic instance or class.
    - cuda: bool, whether to use GPU (default=False).
    - log_dir: str, path to log directory (default=None).
    - verbose: bool, verbosity flag (default=False).
    """
    self.data_depth = data_depth
    self.encoder = encoder
    self.decoder = decoder
    self.critic = critic
    self.verbose = verbose
    self.log_dir = log_dir
    self.cuda = cuda

    # Call the method to set the device (GPU or CPU)
    self._set_device()

    # Move models to the appropriate device
    self.encoder.to(self.device)
    self.decoder.to(self.device)
    self.critic.to(self.device)

    # Initialize optimizers and other training components later
    self.critic_optimizer = None
    self.encoder_decoder_optimizer = None

    if self.log_dir:
      os.makedirs(self.log_dir, exist_ok=True)

    if self.verbose:
      print(f"Model initialized with data depth {self.data_depth}")

  def _set_device(self):
    """
    Set the device to GPU if available and requested, otherwise use CPU.

    I am using CUDA anyway but this was the most secure way to get this done
    """
    if self.cuda and torch.cuda.is_available():
        self.device = torch.device('cuda')
        if self.verbose:
            print("Using CUDA device.")
    else:
        self.device = torch.device('cpu')
        if self.cuda and not torch.cuda.is_available() and self.verbose:
            print("CUDA is not available. Defaulting to CPU.")
        elif self.verbose:
            print("Using CPU device.")


  # Payload Preparation
  def _random_data(self, cover):
    """
    Parameters:
    - cover: torch.Tensor, the cover image tensor.

    Returns:
    - torch.Tensor, random binary data tensor to embed in the cover.
    """
    N, _, H, W = cover.size()
    return torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)

  def _make_payload(self, width, height, depth, text):
    """
    Parameters:
    - width: int, width of the image.
    - height: int, height of the image.
    - depth: int, depth of the data.
    - text: str, the text data to be encoded.

    Returns:
    - torch.Tensor, payload tensor formatted for encoding.
    """
    message = text_to_bits(text) + [0] * 32  # Convert text to bits and pad with zeros.
    payload = message
    while len(payload) < width * height * depth:
        payload += message
    payload = payload[:width * height * depth]  # Truncate payload to fit within the image dimensions.
    return torch.FloatTensor(payload).view(1, depth, height, width).to(self.device)


  # Encoding and Decoding
  def _encode(self, cover, payload):
    """
    Parameters:
    - cover: torch.Tensor, the cover image tensor.
    - payload: torch.Tensor, the data tensor to be encoded.

    Returns:
    - torch.Tensor, the steganographic image with embedded data.
    """
    generated = self.encoder(cover, payload)
    return generated

  def _decode(self, stego):
    """
    Parameters:
    - stego: torch.Tensor, the steganographic image tensor.

    Returns:
    - torch.Tensor, the decoded data tensor.
    """
    decoded = self.decoder(stego)
    return decoded

  def _encode_decode(self, cover, quantize=False):
    """
    Parameters:
    - cover: torch.Tensor, the cover image tensor.
    - quantize: bool, whether to quantize the stego image (default=False).

    Returns:
    - tuple(torch.Tensor, torch.Tensor, torch.Tensor):
      - The generated stego image.
      - The original payload tensor.
      - The decoded data tensor.
    """
    # Generate random data to embed
    payload = self._random_data(cover)

    # Encode the data into the cover image using the specified encoder
    generated = self._encode(cover, payload)

    # Quantization step (optional)
    if quantize:
        generated = (255.0 * (generated + 1.0) / 2.0).long()
        generated = 2.0 * generated.float() / 255.0 - 1.0

    # Decode the data from the steganographic image using the specified decoder
    decoded = self._decode(generated)

    return generated, payload, decoded

  # Training
  def fit(self, train_loader, validate_loader, epochs=5):
    """
    Parameters:
    - train_loader: DataLoader, the DataLoader for the training dataset.
    - validate_loader: DataLoader, the DataLoader for the validation dataset.
    - epochs: int, number of epochs to train (default=5).
    """
    if self.critic_optimizer is None or self.decoder_optimizer is None:
      self.critic_optimizer, self.decoder_optimizer = self._get_optimizers()

    for epoch in range(1, epochs + 1):
      print(f"Epoch {epoch}/{epochs}")

      # Initialize metrics
      '''
      metrics = {field: [] for field in [ 'train.cover_score', 'train.generated_score',
                                          'train.encoder_mse', 'train.decoder_loss',
                                          'train.decoder_acc',
                                          'val.cover_score', 'val.generated_score', 
                                          'val.encoder_mse', 'val.decoder_loss', 
                                          'val.decoder_acc', 
                                          'val.ssim', # structural similarity
                                          'val.psnr', # peak signal to noise ratio
                                          'val.bpp']} # reed solomon bits per pixel
      '''
      # Initialize metrics using METRIC_FIELDS
      metrics = {field: [] for field in METRIC_FIELDS}
  
      # Train the critic and encoder/decoder
      self._fit_critic(train_loader, metrics)
      self._fit_coders(train_loader, metrics)

      # Validate the model
      self._validate(validate_loader, metrics)

      # Logging and storing metrics could be added here

      print(f"Metrics after epoch {epoch}:")
      for key, value in metrics.items():
        print(f"{key}: {sum(value)/len(value):.4f}")

  def _fit_critic(self, train_loader, metrics):
    """
    Parameters:
    - train_loader: DataLoader, the DataLoader for the training dataset.
    - metrics: dict, dictionary to store training metrics.
    """
    for cover, _ in tqdm(train_loader, desc="Training Critic", leave=False):
      gc.collect()
      cover = cover.to(self.device)
      payload = self._random_data(cover)
      generated = self._encode(cover, payload)

      cover_score = torch.mean(self.critic(cover))
      generated_score = torch.mean(self.critic(generated))

      self.critic_optimizer.zero_grad()
      (cover_score - generated_score).backward()
      self.critic_optimizer.step()

      # Clip critic weights to enforce Lipschitz continuity
      for p in self.critic.parameters():
        p.data.clamp_(-0.1, 0.1)

      metrics['train.cover_score'].append(cover_score.item())
      metrics['train.generated_score'].append(generated_score.item())

  def _fit_coders(self, train_loader, metrics):
    """
    Parameters:
    - train_loader: DataLoader, the DataLoader for the training dataset.
    - metrics: dict, dictionary to store training metrics.
    """
    for cover, _ in tqdm(train_loader, desc="Training Encoder/Decoder", leave=False):
      gc.collect()
      cover = cover.to(self.device)
      generated, payload, decoded = self._encode_decode(cover)

      encoder_mse = torch.nn.functional.mse_loss(generated, cover)
      decoder_loss = torch.nn.functional.binary_cross_entropy_with_logits(decoded, payload)
      decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
      generated_score = torch.mean(self.critic(generated))

      self.decoder_optimizer.zero_grad()
      (100.0 * encoder_mse + decoder_loss + generated_score).backward()
      self.decoder_optimizer.step()

      metrics['train.encoder_mse'].append(encoder_mse.item())
      metrics['train.decoder_loss'].append(decoder_loss.item())
      metrics['train.decoder_acc'].append(decoder_acc.item())

  def _validate(self, validate_loader, metrics):
    """
    Parameters:
    - validate_loader: DataLoader, the DataLoader for the validation dataset.
    - metrics: dict, dictionary to store validation metrics.
    """
    for cover, _ in tqdm(validate_loader, desc="Validating", leave=False):
      gc.collect()
      cover = cover.to(self.device)
      generated, payload, decoded = self._encode_decode(cover, quantize=True)

      encoder_mse = torch.nn.functional.mse_loss(generated, cover)
      decoder_loss = torch.nn.functional.binary_cross_entropy_with_logits(decoded, payload)
      decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
      generated_score = torch.mean(self.critic(generated))
      cover_score = torch.mean(self.critic(cover))
      ssim_score = ssim(cover, generated)
      psnr_score = 10 * torch.log10(4 / encoder_mse)
      bpp_score = self.data_depth * (2 * decoder_acc.item() - 1)

      metrics['val.encoder_mse'].append(encoder_mse.item())
      metrics['val.decoder_loss'].append(decoder_loss.item())
      metrics['val.decoder_acc'].append(decoder_acc.item())
      metrics['val.cover_score'].append(cover_score.item())
      metrics['val.generated_score'].append(generated_score.item())
      metrics['val.ssim'].append(ssim_score.item())
      metrics['val.psnr'].append(psnr_score.item())
      metrics['val.bpp'].append(bpp_score)


  # Optimization and Loss Functions
  def _get_optimizers(self):
    """
    Create optimizers for the critic and encoder/decoder.
    Set the learning rate at a constant of 1e-4, the same as the original

    Returns (tuple):
    - critic_optimizer: Optimizer, the optimizer for the critic.
    - decoder_optimizer: Optimizer, the optimizer for the decoder and encoder combined.
    """
    # Optimizer for the critic
    critic_optimizer = Adam(self.critic.parameters(), lr=1e-4)

    # Optimizer for both the encoder and decoder
    decoder_optimizer = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-4)

    return critic_optimizer, decoder_optimizer
  
  # Saving and Loading
  def save(self, path):
    """
    Parameters:
    - path: str, the file path to save the model.
    """
    model_state = {
      'encoder': self.encoder.state_dict(),
      'decoder': self.decoder.state_dict(),
      'critic': self.critic.state_dict(),
      'critic_optimizer': self.critic_optimizer.state_dict(),
      'decoder_optimizer': self.decoder_optimizer.state_dict(),
      'epochs': self.epochs
    }
    torch.save(model_state, path)
    if self.verbose:
      print(f"Model saved to {path}")

  
  def load(self, path, cuda=True, verbose=False):
    """
    Parameters:
    - path: str, the file path to load the model from.
    - cuda: bool, whether to use GPU (default=True).
    """
    model_state = torch.load(path, map_location='cuda' if cuda and torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = self.__class__(data_depth=model_state['encoder']['weight'].size(1),
                encoder=self.encoder, 
                decoder=self.decoder, 
                critic=self.critic,
                cuda=cuda, 
                verbose=verbose)

    # Load model states
    model.encoder.load_state_dict(model_state['encoder'])
    model.decoder.load_state_dict(model_state['decoder'])
    model.critic.load_state_dict(model_state['critic'])
    model.critic_optimizer.load_state_dict(model_state['critic_optimizer'])
    model.decoder_optimizer.load_state_dict(model_state['decoder_optimizer'])
    model.epochs = model_state['epochs']

    if model.verbose:
      print(f"Model loaded from {path}")

    return model


  # Logging and Visualization Methods
  def _log_metrics(self, metrics, epoch):
    """
    MISSING PARAMETERS:
    - Logging directory or file path.

    Parameters:
    - metrics: dict, dictionary of metrics to log.
    - epoch: int, current training epoch.
    """
    metrics['epoch'] = epoch

    if self.log_dir:
      if not os.path.exists(self.log_dir):
        os.makedirs(self.log_dir)
      metrics_path = os.path.join(self.log_dir, 'metrics.json')

      # Append the new metrics to the log file
      with open(metrics_path, 'a') as f:
        json.dump(metrics, f)
        f.write('\n')

    if self.verbose:
      print(f"Metrics logged to {metrics_path}")

  def _plot_metrics(self, metrics, epoch):
    """
    Plot the training and validation metrics for the current epoch.

    Parameters:
    - metrics (dict): Dictionary of metrics to plot.
    - epoch (int): The current epoch number.
    """
    # Define the metrics to plot
    metric_names = ['encoder_mse', 'decoder_loss', 'decoder_acc', 'cover_score', 'generated_score', 'ssim', 'psnr', 'bpp']

    # Create a figure for plotting
    plt.figure(figsize=(12, 8))

    for i, metric in enumerate(metric_names):
      # Plot the metric
      plt.subplot(2, 4, i+1)
      plt.plot(metrics['train.' + metric], label=f'Train {metric}')
      plt.plot(metrics['val.' + metric], label=f'Val {metric}')
      plt.xlabel('Epoch')
      plt.ylabel(metric)
      plt.title(f'{metric} over epochs')
      plt.legend()
      plt.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    # Save the plot if log_dir is provided
    if self.log_dir:
      plot_path = os.path.join(self.log_dir, f'metrics_epoch_{epoch}.png')
      plt.savefig(plot_path)
      if self.verbose:
        print(f"Plot saved to {plot_path}")
