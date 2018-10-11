
"""
Authors: Gabriel Huang
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm

from scatwave.differentiable import scattering, cast, prepare_padding_size
from scatwave.filters_bank import filters_bank


def reshape_scattering(scat):
    '''
    Reshape scattering for visualization purposes
    '''
    assert type(scat) == np.ndarray and len(scat.shape) == 3
    n = int(np.ceil(np.sqrt(scat.shape[0])))

    filler = np.zeros((n*n-scat.shape[0], scat.shape[1], scat.shape[2]))
    padded = np.concatenate((scat, filler), axis=0)
    padded = padded.reshape((n, n, scat.shape[1], scat.shape[2]))
    padded = np.moveaxis(padded, 1, 2)
    padded = np.reshape(padded, (n * scat.shape[1], n * scat.shape[2]))
    return padded


class ScatteringTranscoder(object):
    def __init__(self, M, N, J, cache='.cache'):
        '''
        M : int
            height to resize

        N : int
            width to resize

        J : int
            scale

        cache : str or False
            filename of cache for filter banks
        '''
        assert M == N, 'for now must have M=N'

        self.M = M
        self.N = N
        self.J = J
        self.cache = cache
        self.M_padded, self.N_padded = prepare_padding_size(M, N, J)

        # Filter banks
        self.cache_file = '{}-Mpad-{}-Npad-{}-J-{}'.format(
            self.cache, self.M_padded, self.N_padded, self.J)
        filters = filters_bank(
            self.M_padded, self.N_padded, self.J, cache=self.cache_file)
        self.Psi = filters['psi']
        self.Phi = [filters['phi'][j] for j in range(self.J)]
        self.Psi, self.Phi = cast(self.Psi, self.Phi, torch.cuda.FloatTensor)

        # image preprocessor
        self.preprocessor = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Scale(M),
                transforms.CenterCrop(M),
                transforms.ToTensor(),
            ])

    def process_img(self, img):
        inputs = Variable(self.preprocessor(img).unsqueeze(0).cuda())
        return inputs

    def deprocess_img(self, img):
        img = img.data.cpu().numpy()
        img = np.moveaxis(img, 1, -1)  # move channel to end
        return img

    def encode_processed(self, processed_imgs):
        '''
        imgs : torch.Tensor
            shape (batch, channel. height, width)
        '''
        # processed_imgs = torch.Tensor(processed_imgs).cuda()
        assert (len(processed_imgs.size()) == 4
                and processed_imgs.size()[1] in (1, 3)),\
            'image shape must be (batch, channel, height, width)'

        S_imgs = scattering(processed_imgs, self.Psi, self.Phi, self.J)
        return S_imgs

    def decode(self, S_imgs, step=0.5, iterations=100):
        '''
        S_imgs : torch.Tensor
            shape (batch, channel, scat_channel, scat_height, scat_width)
        '''
        assert len(S_imgs.size()) == 5 and S_imgs.size()[1] in (1, 3),\
            'tensor shape must be (batch, channel, scat_channel, scat_height, scat_width)'

        noise = torch.Tensor(S_imgs.size()[0], S_imgs.size()[1], self.M, self.N).normal_(std=0.1).cuda()
        reconstructed = Variable(noise, requires_grad=True)
        abs_losses = []
        rel_losses = []

        optimizer = torch.optim.Adam([reconstructed], step)
        iterator = tqdm(xrange(iterations))

        try:
            for i in iterator:

                optimizer.zero_grad()
                S_reconstructed = scattering(reconstructed, self.Psi, self.Phi, self.J)
                loss = torch.abs(S_imgs - S_reconstructed).mean()
                rel_loss = (loss / S_imgs.abs().mean()).data[0]
                abs_losses.append(loss.data[0])
                rel_losses.append(rel_loss)
                loss.backward()
                optimizer.step()

                iterator.set_description('relative {:.4} absolute {:.4}'.format(rel_loss, loss.data[0]))

                '''
                if i % 5 == 0 or i == iterations-1:
                    I = (reconstructed.data.cpu()[0].permute(1,2,0).squeeze().numpy()).clip(0,1)
                    I = Image.fromarray(np.uint8(I*255.0),'YCbCr').convert('RGB')

                    # Save file
                    reconstructed_file = '{}.reconstructed.jpg'.format(base)
                    print 'Saving to {}'.format(reconstructed_file)
                    misc.imsave(reconstructed_file, I)

                    print('iteration%i'%(i))
                '''
        except KeyboardInterrupt:
            print 'Interrupted'
        return reconstructed, {'abs_losses': abs_losses, 'rel_losses': rel_losses}
