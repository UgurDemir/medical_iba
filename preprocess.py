from torchvision import transforms

def get_xform(scale_size, crop_size, norm='01'):
    xform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(scale_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalization(norm)
    ])

    return xform

def normalization(norm):
    def apply(x):
        if norm is None:
            return x
        elif norm == '01':
            xmin, xmax = x.min(), x.max()
            x_norm = (x - xmin) / (xmax - xmin)
            return x_norm
        else:
            raise NotImplementedError('Normalization {} is not implemented yet'.format(norm))

    return apply

def pipeline(*ops):
    def apply(x):
        for o in ops:
            x = o(x)
        return x
    return apply
