import argparse
import logging
import numpy as np
import os
import torch

from thirdparty.stylegan2_ada_pytorch import dnnlib, legacy
from thirdparty.stylegan2_ada_pytorch.experiment.util import image, latent
from thirdparty.stylegan2_ada_pytorch.training.training_loop_modified import save_image_grid
import forger.experimental.autoenc.factory as factory

logger = logging.getLogger(__name__)

def get_w_grid(w: np.ndarray,  # tensor of shape [1, 1, G.w_dim]
               dim0: int,
               dim1: int,
               step0=0.0001,
               step1=0.0001
               ) -> torch.Tensor:
    """
    @param w: A torch.Tensor of shape [1, 1, G.w_din]. w is used as the 'origin'
    @param dim0: The dimension of latent space that varies along the rows of the final visualization
    @param dim1: The dimension of latent space that varies along the columns of the final visualization
    @param step0: Values at dim0 along the rows of the final visualization is an arithmetic sequence of difference step0
    @param step1: Values at dim1 along the columns of the final visualization is an arithmetic sequence of difference step1
    @return T: A torch.Tensor of size [21, 21, G.w_dim]
    """
    n_step = 10
    grid_width = 2 * n_step + 1

    grid_w = np.tile(w, (grid_width, grid_width, 1))
    print("grid w shape", grid_w.shape)
    grid_w = torch.from_numpy(grid_w)
    meshgrid_y, meshgrid_x = torch.from_numpy(np.mgrid[-n_step:n_step+1, -n_step:n_step+1]).to(torch.float32)
    meshgrid_y *= step0
    meshgrid_x *= step1
    grid_w[:, :, dim0] += meshgrid_y
    grid_w[:, :, dim1] += meshgrid_x
    return grid_w


def get_center_w(mode: str, device='cuda', w_file=None, G=None):
    """
    @param mode: Determines how the w vector to be used at the center of the visualization is computed
    @param device: The device to use, in case G.mapping_network is used.
    @param w_file: The file. Needs to be provided to mode == 'npz'
    @param G: The Generator of which the mapping network will be used if mode == 'mean'
    """
    w_std = None
    if mode == 'npz':
        assert os.path.isfile(w_file)
        ws = np.load(w_file)['w']
        # ws = torch.tensor(ws, device=device)
    else:
        assert G is not None, "Need to pass in Generator to compute the avg of w vector"
        if mode == 'rand':
            n_sample = 1
        elif mode == 'mean':
            n_sample = 10000
        else:
            raise ValueError

        ws, w_std = latent.get_w_stats(num_samples=n_sample,
                                   z_dim=G.z_dim,
                                   mapping_network=G.mapping,
                                   device=device)

    assert ws.shape == (1, 1, G.w_dim)
    # Return w_std as it may be used to determine step0 & step1
    return {'w': ws, 'w_std': w_std}


def style_exploration(
    G,
    encoder,
    geom_tensor: torch.Tensor,
    dim0: int,
    dim1: int,
    outdir: str,
    w_dict={},
    step0 = None,
    step1 = None,
):
    """
    @param G: Pretrained Forger Generator to be used
    @param encoder: The geometry autoencoder that are used to train G
    @param geom_tensor: The input to encoder, converted from the geometry image
    @param dim0: The dimension of latent space that varies along the rows of the final visualization
    @param dim1: The dimension of latent space that varies along the columns of the final visualization
    @param outdir: The directory to save the output
    @param w_dict: The dict object that has two keys: 'w' and 'w_std',
                    where w_dict['w'] is a torch.Tensor of shape [1, 1, G.w_dim]
                    and w_dict['w_std'] is a scalar.
    @param step0: Values at dim0 along the rows of the final visualization is an arithmetic sequence of difference step0
    @param step1: Values at dim1 along the rows of the final visualization is an arithmetic sequence of difference step1
    """
    assert 0 <= dim0 < G.z_dim and 0 <= dim1 < G.z_dim, "The dimension(s) to explore is not valid."
    os.makedirs(outdir, exist_ok=True)

    geom_tensor = geom_tensor.to(device)
    geom_feature = encoder(geom_tensor)

    if w_dict['w_std'] is not None and w_dict['w_std'] > 0:
        logger.debug(f"w std: {w_dict['w_std']}")
        step0 = w_dict['w_std'] / 6
        step1 = w_dict['w_std'] / 6

    grid_w = get_w_grid(w=w_dict['w'],
                        dim0=dim0,
                        dim1=dim1,
                        step0=step0,
                        step1=step1
                        )
    grid_w = grid_w.unsqueeze(2).repeat(1, 1, G.num_ws, 1).to(device)
    row_geom_feature = geom_feature.repeat(grid_w.shape[1], 1, 1, 1)

    grid_size = (grid_w.shape[1], grid_w.shape[1])

    # NOTE: Currently this is meant to be used on the triad forger model
    grids = {
        'U': [],
        'V': [],
        'S': [],
        'final': [],
    }

    for r, row_w in enumerate(grid_w):
        img, debug_data = G.synthesis(row_w,
                            geom_feature=row_geom_feature,
                            return_debug_data=True)
        img = img.cpu()
        grids['final'].append(img)
        uvs = debug_data['uvs'].cpu()
        uvs = uvs * 2 - 1.0
        grids['U'].append(uvs[:, :1, :, :].repeat(1, 3, 1, 1))
        grids['V'].append(uvs[:, 1:2, :, :].repeat(1, 3, 1, 1))
        grids['S'].append(uvs[:, 2:, :, :].repeat(1, 3, 1, 1))

    for key, grid in grids.items():
        fname = os.path.join(outdir, key+".jpg")
        img_grid = torch.cat(grid).numpy()
        save_image_grid(img=img_grid, fname=fname, drange=(-1, 1), grid_size=grid_size)

    # Also save grid_w
    grid_w = grid_w.cpu().numpy()
    np.savez(file=os.path.join(outdir, 'grid_w.npz'),
             grid_w=grid_w,
             dim0=dim0,
             dim1=dim1,
             step0=step0,
             step1=step1)


if __name__ == '__main__' :
    # TODO: Allow visualizing a grid of z vectors.
    parser = argparse.ArgumentParser(description='Visualize a grid of w vectors')
    parser.add_argument('--gan_checkpt', type=str, required=True,
                        help='The .pkl file to load the Generator from.')
    parser.add_argument('--encoder_checkpt', type=str, default="",
                        help='The .pkl that contains the autoencoder for encoding the geometry data in Forger.')
    parser.add_argument('--forger', action='store_true')
    parser.add_argument('--geom_image', type=str,
                        help='The path to the geometry image')
    parser.add_argument('--w_mode', type=str, default='mean', choices=('mean', 'rand', 'npz'),
                        help='Mode to generate w vector. '
                             'If set to "mean", '
                             'If set to "rand"'
                             'If set to "npz"')
    parser.add_argument('--w_file', type=str, default=None,
                        help='The .npz file of the projected w')
    parser.add_argument('--dim0', type=int, default=0)
    parser.add_argument('--dim1', type=int, default=1)
    parser.add_argument('--step0', type=float, default=0.0001,
                        help='The ')
    parser.add_argument('--step1', type=float, default=0.0001,
                        help='')
    parser.add_argument('--noise_mode', choices=('const', 'random', 'none'), type=str, default='const',
                        help='Noise mode')
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    geom_tensor = None
    if args.forger:
        if args.geom_image is None:
            raise RuntimeError("geom image not provided for Art-forger model")
        if args.encoder_checkpt is None:
            raise RuntimeError("encoder checkpoint not provided for Art-forger model")
        geom_tensor = image.convert_geom_data(args.geom_image)

    if args.w_mode == 'npz' and not os.path.isdir(args.w_file):
        raise RuntimeError('The provided npz file for w is invalid')

    device = torch.device('cuda')
    print('Loading StyleGAN from "%s"...' % args.gan_checkpt)
    with dnnlib.util.open_url(args.gan_checkpt) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    w_dict = get_center_w(mode=args.w_mode, w_file=args.w_file, G=G)

    print("Loading autoencoder")
    autoencoder = factory.create_autoencoder_from_checkpoint(encoder_checkpt=args.encoder_checkpt)
    autoencoder.eval().requires_grad_(False).to(device)

    style_exploration(
        G=G,
        encoder=autoencoder.encoder,
        geom_tensor=geom_tensor,
        dim0=args.dim0,
        dim1=args.dim1,
        outdir=args.outdir,
        w_dict=w_dict,
        step0=args.step0,
        step1=args.step1,
    )