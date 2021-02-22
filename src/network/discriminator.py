import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


def get_thin_kernels(start=0, end=360, step=45):
    k_thin = 3  # actual size of the directional kernel
    # increase for a while to avoid interpolation when rotating
    k_increased = k_thin + 2

    # get 0° angle directional kernel
    thin_kernel_0 = np.zeros((k_increased, k_increased))
    thin_kernel_0[k_increased // 2, k_increased // 2] = 1
    thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

    # rotate the 0° angle directional kernel to get the other ones
    thin_kernels = []
    for angle in range(start, end, step):
        (h, w) = thin_kernel_0.shape
        # get the center to not rotate around the (0, 0) coord point
        center = (w // 2, h // 2)
        # apply rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

        # get the k=3 kerne
        kernel_angle = kernel_angle_increased[1:-1, 1:-1]
        is_diag = (abs(kernel_angle) == 1)  # because of the interpolation
        kernel_angle = kernel_angle * is_diag  # because of the interpolation
        thin_kernels.append(kernel_angle)
    return thin_kernels


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'

        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)

        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)

        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)

        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.hysteresis.weight[:] = torch.from_numpy(hysteresis)

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=True):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian

        for c in range(C):
            blurred[:, c:c + 1] = self.gaussian_filter(img[:, c:c + 1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c + 1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c + 1])

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180  # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges

        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1

        return thin_edges

torch.set_printoptions(threshold=10_000)


class Discriminator(nn.Module):
    def __init__(self, image_dims, context_dims, C, spectral_norm=True):
        """
        Convolutional patchGAN discriminator used in [1].
        Accepts as input generator output G(z) or x ~ p*(x) where
        p*(x) is the true data distribution.
        Contextual information provided is encoder output y = E(x)
        ========
        Arguments:
        image_dims:     Dimensions of input image, (C_in,H,W)
        context_dims:   Dimensions of contextual information, (C_in', H', W')
        C:              Bottleneck depth, controls bits-per-pixel
                        C = 220 used in [1], C = C_in' if encoder output used
                        as context.
        [1] Mentzer et. al., "High-Fidelity Generative Image Compression",
            arXiv:2006.09965 (2020).
        """
        super(Discriminator, self).__init__()

        self.image_dims = image_dims
        self.context_dims = context_dims
        im_channels = self.image_dims[0]
        kernel_dim = 4
        context_C_out = 12
        filters = (64, 128, 256, 512)

        # Upscale encoder output - (C, 16, 16) -> (12, 256, 256)
        self.context_conv = nn.Conv2d(C, context_C_out, kernel_size=3, padding=1, padding_mode='reflect')
        self.context_upsample = nn.Upsample(scale_factor=16, mode='nearest')

        # Images downscaled to 500 x 1000 + randomly cropped to 256 x 256
        # assert image_dims == (im_channels, 256, 256), 'Crop image to 256 x 256!'

        # Layer / normalization options
        # TODO: calculate padding properly
        cnn_kwargs = dict(stride=2, padding=1, padding_mode='reflect')
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        if spectral_norm is True:
            norm = nn.utils.spectral_norm
        else:
            norm = nn.utils.weight_norm

        # (C_in + C_in', 256,256) -> (64,128,128), with implicit padding
        # TODO: Check if removing spectral norm in first layer works
        self.conv1 = nn.Conv2d(im_channels, filters[0], kernel_dim, **cnn_kwargs)

        # (128,128) -> (64,64)
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs)

        # (64,64) -> (32,32)
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs)

        # (32,32) -> (16,16)
        self.conv4 = nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs)

        self.conv_out1 = nn.Conv2d(filters[3] + context_C_out, 256, kernel_dim, **cnn_kwargs)
        self.conv_out = nn.Conv2d(256, 1, kernel_size=1, stride=1)

    def forward(self, x, y):
        """
        x: Concatenated real/gen images
        y: Quantized latents
        """
        batch_size = x.size()[0]

        # Concatenate upscaled encoder output y as contextual information
        y = self.activation(self.context_conv(y))
        # y = self.context_upsample(y)

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = torch.cat((x, y), dim=1)

        out_logits = self.conv_out(self.conv_out1(x)).view(-1, 1)
        out = torch.sigmoid(out_logits)

        return out, out_logits


if __name__ == "__main__":
    B = 2
    C = 7
    print('Image 1')
    x = torch.randn((B, 3, 256, 256))
    x_dims = tuple(x.size())
    D = Discriminator(image_dims=x_dims[1:], context_dims=tuple(x.size())[1:], C=C)
    print('Discriminator output', x.size())
