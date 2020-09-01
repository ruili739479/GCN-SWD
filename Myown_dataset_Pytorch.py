import torch
from torch_geometric.data import InMemoryDataset
from Loadin_connectivity_network import create_torch_data

class Myown_Dataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(Myown_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_torch.pt']

    def download(self):
        pass

    def process(self):
        self.data, self.slices = create_torch_data(self.name)
        torch.save((self.data, self.slices), self.processed_paths[0])

