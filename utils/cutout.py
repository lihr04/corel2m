import torch


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = torch.ones((h, w))

        for n in range(self.n_holes):
            y = torch.randint(low=0,high=h,size=(1,))
            x = torch.randint(low=0,high=w,size=(1,))

            y1 = torch.clamp(y - self.length // 2, 0, h)
            y2 = torch.clamp(y + self.length // 2, 0, h)
            x1 = torch.clamp(x - self.length // 2, 0, w)
            x2 = torch.clamp(x + self.length // 2, 0, w)

            mask[y1.item(): y2.item(), x1.item(): x2.item()] = 0.

        mask = mask.expand_as(img)
        img=img * mask

        return img
    
    def __repr__(self):
        return self.__class__.__name__ + '(n_holes={0}, length={1})'.format(self.n_holes, self.length)