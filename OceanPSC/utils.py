from PIL import Image
import PIL
import skimage.measure
import numpy as np
import collections
import csv
from matplotlib.colors import LightSource, LinearSegmentedColormap
import matplotlib.pyplot as plt
import scipy as sp
import scipy.spatial

PIL.Image.MAX_IMAGE_PIXELS = 466560010

def load_data(path,reduction=(4,4)):
    img=Image.open(path)
    rsl=np.array(img)
    rsl = skimage.measure.block_reduce(rsl, (4,4), np.mean)
    return rsl

# Various common functions.



# Open CSV file as a dict.
def read_csv(csv_path):
  with open(csv_path, 'r') as csv_file:
    return list(csv.DictReader(csv_file))


# Renormalizes the values of `x` to `bounds`
def normalize(x, bounds=(0, 1)):
  return np.interp(x, (x.min(), x.max()), bounds)


# Fourier-based power law noise with frequency bounds.
def fbm(shape, p, lower=-np.inf, upper=np.inf):
  freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in shape)
  freq_radial = np.hypot(*np.meshgrid(*freqs))
  envelope = (np.power(freq_radial, p, where=freq_radial!=0) *
              (freq_radial > lower) * (freq_radial < upper))
  envelope[0][0] = 0.0
  phase_noise = np.exp(2j * np.pi * np.random.rand(*shape))
  return normalize(np.real(np.fft.ifft2(np.fft.fft2(phase_noise) * envelope)))


# Returns each value of `a` with coordinates offset by `offset` (via complex 
# values). The values at the new coordiantes are the linear interpolation of
# neighboring values in `a`.
def sample(a, offset):
  shape = np.array(a.shape)
  delta = np.array((offset.real, offset.imag))
  coords = np.array(np.meshgrid(*map(range, shape))) - delta

  lower_coords = np.floor(coords).astype(int)
  upper_coords = lower_coords + 1
  coord_offsets = coords - lower_coords 
  lower_coords %= shape[:, np.newaxis, np.newaxis]
  upper_coords %= shape[:, np.newaxis, np.newaxis]

  result = lerp(lerp(a[lower_coords[1], lower_coords[0]],
                     a[lower_coords[1], upper_coords[0]],
                     coord_offsets[0]),
                lerp(a[upper_coords[1], lower_coords[0]],
                     a[upper_coords[1], upper_coords[0]],
                     coord_offsets[0]),
                coord_offsets[1])
  return result


# Takes each value of `a` and offsets them by `delta`. Treats each grid point
# like a unit square.
def displace(a, delta):
  fns = {
      -1: lambda x: -x,
      0: lambda x: 1 - np.abs(x),
      1: lambda x: x,
  }
  result = np.zeros_like(a)
  for dx in range(-1, 2):
    wx = np.maximum(fns[dx](delta.real), 0.0)
    for dy in range(-1, 2):
      wy = np.maximum(fns[dy](delta.imag), 0.0)
      result += np.roll(np.roll(wx * wy * a, dy, axis=0), dx, axis=1)

  return result


# Returns the gradient of the gaussian blur of `a` encoded as a complex number. 
def gaussian_gradient(a, sigma=1.0):
  [fy, fx] = np.meshgrid(*(np.fft.fftfreq(n, 1.0 / n) for n in a.shape))
  sigma2 = sigma**2
  g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
  dg = lambda x: g(x) * (x / sigma2)

  fa = np.fft.fft2(a)
  dy = np.fft.ifft2(np.fft.fft2(dg(fy) * g(fx)) * fa).real
  dx = np.fft.ifft2(np.fft.fft2(g(fy) * dg(fx)) * fa).real
  return 1j * dx + dy


# Simple gradient by taking the diff of each cell's horizontal and vertical
# neighbors.
def simple_gradient(a):
  dx = 0.5 * (np.roll(a, 1, axis=0) - np.roll(a, -1, axis=0))
  dy = 0.5 * (np.roll(a, 1, axis=1) - np.roll(a, -1, axis=1))
  return 1j * dx + dy


# Loads the terrain height array (and optionally the land mask from the given 
# file.
def load_from_file(path):
  result = np.load(path)
  if type(result) == np.lib.npyio.NpzFile:
    return (result['height'], result['land_mask'])
  else:
    return (result, None)


# Saves the array as a PNG image. Assumes all input values are [0, 1]
def save_as_png(a, path):
  image = Image.fromarray(np.round(a * 255).astype('uint8'))
  image.save(path)



# Linear interpolation of `x` to `y` with respect to `a`
def lerp(x, y, a): return (1.0 - a) * x + a * y


# Returns a list of grid coordinates for every (x, y) position bounded by
# `shape`
def make_grid_points(shape):
  [Y, X] = np.meshgrid(np.arange(shape[0]), np.arange(shape[1])) 
  grid_points = np.column_stack([X.flatten(), Y.flatten()])
  return grid_points





# Peforms a gaussian blur of `a`.
def gaussian_blur(a, sigma=1.0):
  freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in a.shape)
  freq_radial = np.hypot(*np.meshgrid(*freqs))
  sigma2 = sigma**2
  g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
  kernel = g(freq_radial)
  kernel /= kernel.sum()
  return np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(kernel)).real

# Smooth Step function
def S(x):
  return 3*x*x-2*x*x*x

# Pseudo-random numbers


def a(i, j):
    u, v = 50*(i/np.pi-np.floor(i/np.pi)), 50*(j/np.pi-np.floor(j/np.pi))
    return 2*(u*v*(u+v)-np.floor(u*v*(u+v)))-1

# Conditions de raccordement


def b(i, j): return a(i+1, j)
def c(i, j): return a(i, j+1)
def d(i, j): return a(i+1, j+1)

# Noise local


def N(x, y):
    i, j = np.floor(x), np.floor(y)
    return a(i, j)+(b(i, j)-a(i, j))*S(x-i)+(c(i, j)-a(i, j))*S(y-j)+(a(i, j)-b(i, j)-c(i, j)+d(i, j))*S(x-i)*S(y-j)

# Génératrice du terrain


def f(x, y):
    result = 0
    p = 1
    for i in range(10):
        result += N(p*x, p*y)/p
        p *= 2
        temp = x
        x = 3/5*x-4/5*y
        y = 4/5*temp+3/5*y
    return result

# laplacien au point (x,y)


def Laplacien(x, y):
    S = 0
    if x > 0:
        S += carte[x-1][y]
    if x < resolution-1:
        S += carte[x+1][y]
    if y > 0:
        S += carte[x][y-1]
    if y < resolution-1:
        S += carte[x][y+1]
    return S-4*carte[x][y]


