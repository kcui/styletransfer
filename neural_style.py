import tensorflow as tf
import numpy as np
import scipy.io
import argparse
import struct
import errno
import time
import cv2
import os
import vgg19

# Weights + biases of VGG-19 network
vgg_path = '../imagenet-vgg-verydeep-19.mat'
# Weight for content loss
content_weight = 5e0
# Interpolation weight of the style image(s)
style_imgs_weights = [1.0]

# Weight for style loss
style_weight = 1e4
# Weight for total variational loss
total_variational_loss_weight = 1e-3
# Weight for temporal loss
temporal_weight = 2e2
# VGG19 layers to be used for content image
content_layers = ['conv4_2']
# Contribution of each content layer to loss
content_layer_weights = [1.0]
# VGG19 layers to be used for style image
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
# Contribution of each style layer to loss
style_layer_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
# Max num iterations for L-BFGS optimizer
max_iterations = 1000

# Indices of previous frames to be considered for long-term temporal consistency
prev_frame_indices = [1]

'''
  parsing and configuration
'''
def parse_args():

  desc = "TensorFlow implementation of 'A Neural Algorithm for Artistic Style'"
  parser = argparse.ArgumentParser(description=desc)

  parser.add_argument('--style_imgs', nargs='+', type=str,
    help='Filenames of the style images (example: starry-night.jpg)',
    required=True)

  parser.add_argument('--content_img', type=str,
    help='Filename of the content image (example: lion.jpg)')

  parser.add_argument('--max_size', type=int,
    default=512,
    help='Maximum width or height of the input images. (default: %(default)s)')

  parser.add_argument('--device', type=str,
    default='/gpu:0',
    choices=['/gpu:0', '/cpu:0'],
    help='GPU or CPU mode.  GPU mode requires NVIDIA CUDA. (default|recommended: %(default)s)')

  parser.add_argument('--video', action='store_true',
    help='Boolean flag indicating if the user is generating a video.')

  parser.add_argument('--end_frame', type=int,
    default=1,
    help='Last frame number.')

  parser.add_argument('--video_input_dir', type=str,
    default='./video_input',
    help='Relative or absolute directory path to input frames.')

  args = parser.parse_args()

  # normalize weights
  global style_layer_weights
  global content_layer_weights
  global style_imgs_weights
  style_layer_weights   = normalize(style_layer_weights)
  content_layer_weights = normalize(content_layer_weights)
  style_imgs_weights = normalize(style_imgs_weights)

  # create directories for output
  if args.video:
    maybe_make_directory('./video_output')
  else:
    maybe_make_directory('./image_output')

  return args

'''
  'a neural algorithm for artistic style' loss functions
'''
def content_layer_loss(p, x):
  _, h, w, d = p.get_shape()
  M = h.value * w.value
  N = d.value
  K = 1. / (2. * N**0.5 * M**0.5)
  loss = K * tf.reduce_sum(tf.pow((x - p), 2))
  return loss

def style_layer_loss(a, x):
  _, h, w, d = a.get_shape()
  M = h.value * w.value
  N = d.value
  A = gram_matrix(a, M, N)
  G = gram_matrix(x, M, N)
  loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
  return loss

def gram_matrix(x, area, depth):
  F = tf.reshape(x, (area, depth))
  G = tf.matmul(tf.transpose(F), F)
  return G

def sum_style_losses(sess, net, style_imgs):
  total_style_loss = 0.
  weights = style_imgs_weights
  for img, img_weight in zip(style_imgs, weights):
    sess.run(net['input'].assign(img))
    style_loss = 0.
    for layer, weight in zip(style_layers, style_layer_weights):
      a = sess.run(net[layer])
      x = net[layer]
      a = tf.convert_to_tensor(a)
      style_loss += style_layer_loss(a, x) * weight
    style_loss /= float(len(style_layers))
    total_style_loss += (style_loss * img_weight)
  total_style_loss /= float(len(style_imgs))
  return total_style_loss

def sum_content_losses(sess, net, content_img):
  sess.run(net['input'].assign(content_img))
  content_loss = 0.
  for layer, weight in zip(content_layers, content_layer_weights):
    p = sess.run(net[layer])
    x = net[layer]
    p = tf.convert_to_tensor(p)
    content_loss += content_layer_loss(p, x) * weight
  content_loss /= float(len(content_layers))
  return content_loss

'''
  'artistic style transfer for videos' loss functions
'''
def temporal_loss(x, w, c):
  c = c[np.newaxis,:,:,:]
  D = float(x.size)
  loss = (1. / D) * tf.reduce_sum(c * tf.nn.l2_loss(x - w))
  loss = tf.cast(loss, tf.float32)
  return loss

def get_longterm_weights(i, j):
  c_sum = 0.
  for k in range(prev_frame_indices):
    if i - k > i - j:
      c_sum += get_content_weights(i, i - k)
  c = get_content_weights(i, i - j)
  c_max = tf.maximum(c - c_sum, 0.)
  return c_max

def sum_longterm_temporal_losses(sess, net, frame, input_img):
  x = sess.run(net['input'].assign(input_img))
  loss = 0.
  for j in range(prev_frame_indices):
    prev_frame = frame - j
    w = get_prev_warped_frame(frame)
    c = get_longterm_weights(frame, prev_frame)
    loss += temporal_loss(x, w, c)
  return loss

def sum_shortterm_temporal_losses(sess, net, frame, input_img):
  x = sess.run(net['input'].assign(input_img))
  prev_frame = frame - 1
  w = get_prev_warped_frame(frame)
  c = get_content_weights(frame, prev_frame)
  loss = temporal_loss(x, w, c)
  return loss

'''
  utilities and i/o
'''
def read_image(path):
  # bgr image
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  check_image(img, path)
  img = img.astype(np.float32)
  img = preprocess(img)
  return img

def write_image(path, img):
  img = postprocess(img)
  cv2.imwrite(path, img)

def preprocess(img):
  imgpre = np.copy(img)
  # bgr to rgb
  imgpre = imgpre[...,::-1]
  # shape (h, w, d) to (1, h, w, d)
  imgpre = imgpre[np.newaxis,:,:,:]
  imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  return imgpre

def postprocess(img):
  imgpost = np.copy(img)
  imgpost += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  # shape (1, h, w, d) to (h, w, d)
  imgpost = imgpost[0]
  imgpost = np.clip(imgpost, 0, 255).astype('uint8')
  # rgb to bgr
  imgpost = imgpost[...,::-1]
  return imgpost

def read_flow_file(path):
  with open(path, 'rb') as f:
    # 4 bytes header
    header = struct.unpack('4s', f.read(4))[0]
    # 4 bytes width, height
    w = struct.unpack('i', f.read(4))[0]
    h = struct.unpack('i', f.read(4))[0]
    flow = np.ndarray((2, h, w), dtype=np.float32)
    for y in range(h):
      for x in range(w):
        flow[0,y,x] = struct.unpack('f', f.read(4))[0]
        flow[1,y,x] = struct.unpack('f', f.read(4))[0]
  return flow

def read_weights_file(path):
  lines = open(path).readlines()
  header = list(map(int, lines[0].split(' ')))
  w = header[0]
  h = header[1]
  vals = np.zeros((h, w), dtype=np.float32)
  for i in range(1, len(lines)):
    line = lines[i].rstrip().split(' ')
    vals[i-1] = np.array(list(map(np.float32, line)))
    vals[i-1] = list(map(lambda x: 0. if x < 255. else 1., vals[i-1]))
  # expand to 3 channels
  weights = np.dstack([vals.astype(np.float32)] * 3)
  return weights

def normalize(weights):
  denom = sum(weights)
  if denom > 0.:
    return [float(i) / denom for i in weights]
  else: return [0.] * len(weights)

def maybe_make_directory(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def check_image(img, path):
  if img is None:
    raise OSError(errno.ENOENT, "No such file", path)

'''
  rendering -- where the magic happens
'''
def stylize(content_img, style_imgs, init_img, frame=None):
  with tf.device(args.device), tf.Session() as sess:
    # setup network
    vgg = vgg19.VGG19(content_img, vgg_path=vgg_path)
    net = vgg.get_model()

    # style loss
    L_style = sum_style_losses(sess, net, style_imgs)

    # content loss
    L_content = sum_content_losses(sess, net, content_img)

    # denoising loss
    L_tv = tf.image.total_variation(net['input'])

    # loss weights
    alpha = content_weight
    beta  = style_weight
    theta = total_variational_loss_weight

    # total loss
    L_total  = alpha * L_content
    L_total += beta  * L_style
    L_total += theta * L_tv

    # video temporal loss
    if args.video and frame > 1:
      gamma      = temporal_weight
      L_temporal = sum_shortterm_temporal_losses(sess, net, frame, init_img)
      L_total   += gamma * L_temporal

    # optimization algorithm
    optimizer = get_optimizer(L_total)

    minimize_with_lbfgs(sess, net, optimizer, init_img)

    output_img = sess.run(net['input'])


    if args.video:
      write_video_output(frame, output_img)
    else:
      write_image_output(output_img, content_img, style_imgs, init_img)

def minimize_with_lbfgs(sess, net, optimizer, init_img):
  print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  sess.run(net['input'].assign(init_img))
  optimizer.minimize(sess)

def get_optimizer(loss):

  optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B',
    options={'maxiter': max_iterations, 'disp': 50}) #50 print iterations
  return optimizer

def write_video_output(frame, output_img):
  fn = 'frame_{}.ppm'.format(str(frame).zfill(4))
  path = os.path.join('./video_output', fn)
  write_image(path, output_img)

def write_image_output(output_img, content_img, style_imgs, init_img):
  out_dir = os.path.join('./image_output', 'result')
  maybe_make_directory(out_dir)
  img_path = os.path.join(out_dir, 'result'+'.png')
  content_path = os.path.join(out_dir, 'content.png')
  init_path = os.path.join(out_dir, 'init.png')

  write_image(img_path, output_img)
  write_image(content_path, content_img)
  write_image(init_path, init_img)
  index = 0
  for style_img in style_imgs:
    path = os.path.join(out_dir, 'style_'+str(index)+'.png')
    write_image(path, style_img)
    index += 1

  # save the configuration settings
  out_file = os.path.join(out_dir, 'meta_data.txt')
  f = open(out_file, 'w')
  f.write('image_name: {}\n'.format('result'))
  f.write('content: {}\n'.format(args.content_img))
  index = 0
  for style_img, weight in zip(args.style_imgs, style_imgs_weights):
    f.write('styles['+str(index)+']: {} * {}\n'.format(weight, style_img))
    index += 1
  index = 0

  f.write('init_type: {}\n'.format('content'))
  f.write('content_weight: {}\n'.format(content_weight))
  f.write('style_weight: {}\n'.format(style_weight))
  f.write('tv_weight: {}\n'.format(total_variational_loss_weight))
  f.write('content_layers: {}\n'.format(content_layers))
  f.write('style_layers: {}\n'.format(style_layers))
  f.write('max_iterations: {}\n'.format(max_iterations))
  f.write('max_image_size: {}\n'.format(args.max_size))
  f.close()

'''
  image loading and processing
'''
def get_init_image(init_type, content_img, style_imgs, frame=None):
  if init_type == 'content':
    return content_img
  elif init_type == 'prev_warped':
    init_img = get_prev_warped_frame(frame)
    return init_img

def get_content_frame(frame):
  fn = 'frame_{}.ppm'.format(str(frame).zfill(4))
  path = os.path.join(args.video_input_dir, fn)
  img = read_image(path)
  return img

def get_content_image(content_img):
  path = os.path.join('./image_input', content_img)
   # bgr image
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  check_image(img, path)
  img = img.astype(np.float32)
  h, w, d = img.shape
  mx = args.max_size
  # resize if > max size
  if h > w and h > mx:
    w = (float(mx) / float(h)) * w
    img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
  if w > mx:
    h = (float(mx) / float(w)) * h
    img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
  img = preprocess(img)
  return img

def get_style_images(content_img):
  _, ch, cw, cd = content_img.shape
  style_imgs = []
  for style_fn in args.style_imgs:
    path = os.path.join('./styles', style_fn)
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    img = img.astype(np.float32)
    img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
    img = preprocess(img)
    style_imgs.append(img)
  return style_imgs

def get_prev_warped_frame(frame):
  prevframe = frame - 1

  # get path to previous frame
  imagepath = os.path.join(
    './video_output', '
    ''frame_{}.ppm'.format(str(prevframe).zfill(4)))

  # read in previous image
  prev_image = cv2.imread(path, cv2.IMREAD_COLOR)
  check_image(image, imagepath)

  path = os.path.join(args.video_input_dir,
    'backward_{}_{}.flo'.format(str(frame), str(prevframe)))
  flow = read_flow_file(path)

  warped_img = warp_image(prev_image, flow).astype(np.float32)
  img = preprocess(warped_img)

  return img

def get_content_weights(frame, prev_frame):
  # 'reliable_{}...' is the filename format for content optical flow files
  forward_fn = 'reliable_{}_{}.txt'.format(str(prev_frame), str(frame))
  backward_fn = 'reliable_{}_{}.txt'.format(str(frame), str(prev_frame))
  forward_path = os.path.join(args.video_input_dir, forward_fn)
  backward_path = os.path.join(args.video_input_dir, backward_fn)
  forward_weights = read_weights_file(forward_path)
  backward_weights = read_weights_file(backward_path)
  return forward_weights #, backward_weights

def warp_image(src, flow):
  _, h, w = flow.shape
  flow_map = np.zeros(flow.shape, dtype=np.float32)
  for y in range(h):
    flow_map[1,y,:] = float(y) + flow[1,y,:]
  for x in range(w):
    flow_map[0,:,x] = float(x) + flow[0,:,x]
  # remap pixels to optical flow
  dst = cv2.remap(
    src, flow_map[0], flow_map[1],
    interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
  return dst

def render_single_image():
  content_img = get_content_image(args.content_img)
  style_imgs = get_style_images(content_img)
  with tf.Graph().as_default():
    print('\n---- RENDERING SINGLE IMAGE ----\n')
    # first image type is 'content' by default)
    init_img = get_init_image('content', content_img, style_imgs)
    tick = time.time()
    stylize(content_img, style_imgs, init_img)
    tock = time.time()
    print('Single image elapsed time: {}'.format(tock - tick))

def render_video():
  for frame in range(1, args.end_frame+1):
    with tf.Graph().as_default():
      print('\n---- RENDERING VIDEO FRAME: {}/{} ----\n'.format(frame, args.end_frame))
      if frame == 1:
        content_frame = get_content_frame(frame)
        style_imgs = get_style_images(content_frame)
        # first frame type is 'content' by default
        init_img = get_init_image('content', content_frame, style_imgs, frame)
        # Default max number of optimizer iterations of the first frame
        max_iterations = 2000
        tick = time.time()
        stylize(content_frame, style_imgs, init_img, frame)
        tock = time.time()
        print('Frame {} elapsed time: {}'.format(frame, tock - tick))
      else:
        content_frame = get_content_frame(frame)
        style_imgs = get_style_images(content_frame)
        # initial frame type is 'prev_warped' by default
        init_img = get_init_image('prev_warped', content_frame, style_imgs, frame)
        # Default max number of optimizer iterations of the frames after first
        max_iterations = 800
        tick = time.time()
        stylize(content_frame, style_imgs, init_img, frame)
        tock = time.time()
        print('Frame {} elapsed time: {}'.format(frame, tock - tick))

def main():
  global args
  args = parse_args()
  if args.video: render_video()
  else: render_single_image()

if __name__ == '__main__':
  main()
