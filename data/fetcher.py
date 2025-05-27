import torch
from munch import Munch


class Fetcher:
    def __init__(self, loader, args):
        self.loader = loader
        self.device = torch.device(args.device)
        # self.latent_dim = args.latent_dim # Removed as latent_dim is removed from args
        # self.generate_noise = args.mode == 'train' # Removed as noise generation is model specific

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # Attempt to get data and target, assuming loader provides tuples
            # If loader provides single items (e.g. DefaultDataset without labels), 
            # this will need adjustment or the loader should be wrapped to provide a dummy target.
            data_items = next(self.iter) 
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            data_items = next(self.iter)
        
        # Handle cases where loader might return single item or a tuple (data, label)
        if isinstance(data_items, (list, tuple)):
            x = data_items[0]
            if len(data_items) > 1:
                y = data_items[1]
                inputs = Munch(x=x, y=y)
            else:
                inputs = Munch(x=x) # Only data, no label
        else: # Assuming data_items is just x
            x = data_items
            inputs = Munch(x=x)

        return Munch({k: v.to(self.device) for k, v in inputs.items()})
