from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, remote, local):
        if not local:
            self.download(remote, local)

    def preprocess(self):
        pass

    def download(self):
        pass

    def __getitem__(self):
        pass

    def __len__(self):
        pass
