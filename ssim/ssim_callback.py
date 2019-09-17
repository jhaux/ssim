try:
    from skimage.metrics import structural_similarity as sk_ssim
except ImportError:
    from skimage.measure import compare_ssim as sk_ssim

import os
from tqdm import trange
import numpy as np

from edflow.util import retrieve


def ssim(root, data_in, data_out, config):

    data_range = retrieve(config, 'ssim_cb/data_range', default='None')
    if data_range == 'None':
        data_range = None
    im_in_key = retrieve(config, 'ssim_cb/im_in_key', default='image')
    im_out_key = retrieve(config, 'ssim_cb/im_out_key', default='image')

    im_shape = np.shape(data_in[0][im_in_key])
    multichannel = len(im_shape) == 3 and im_shape[-1] > 1

    ssims = []
    for i in trange(len(data_in), desc='SSIM'):
        im_targ = data_in[i][im_in_key]
        im_gen = data_in[i][im_out_key]

        similiarity = sk_ssim(im_targ, im_gen,
                              data_range=data_range,
                              multichannel=multichannel)

        ssims += [similiarity]

    ssims = np.array(ssims)

    mean_ssim = np.mean(ssims)
    std_ssim = np.std(ssims)

    print('\nSSIM: {:4.3f} +- {:4.3}'.format(mean_ssim, std_ssim))

    save_root = os.path.join(root, 'ssim')
    os.makedirs(save_root, exist_ok=True)

    save_name = os.path.join(save_root, 'vals.npz')

    np.savez(save_name, ssims=ssims, mean=mean_ssim, std=std_ssim)


if __name__ == '__main__':
    from edflow.debug import DebugDataset
    from edflow.data.dataset import ProcessedDataset

    D1 = DebugDataset(size=100)
    D2 = DebugDataset(size=100)

    P = lambda *args, **kwargs: {'image': np.ones([256, 256, 3])}

    D1 = ProcessedDataset(D1, P)
    D2 = ProcessedDataset(D2, P)

    print(D1[0])

    ssim('.', D1, D2, {})
