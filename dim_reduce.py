import torch.nn as nn
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Mydata(Dataset):
    def __init__(self, feature):
        super(Mydata, self).__init__()
        self.feature = feature

    def __getitem__(self, idx):
        return torch.from_numpy(self.feature[idx]).float()

    def __len__(self):
        return self.feature.shape[0]


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim//2),
            nn.ReLU(),
            nn.Linear(self.input_dim//2, self.latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.input_dim//2),
            nn.ReLU(),
            nn.Linear(self.input_dim//2, self.input_dim),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        encode = self.encoder(inputs)
        decode = self.decoder(encode)

        return encode, decode


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def reduction(feature, num_epochs=80, lr=1e-4):
    dataset = Mydata(feature)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    model = Autoencoder(input_dim=feature.shape[1], latent_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        loop = tqdm(loader, leave=False, total=len(loader))
        loss_sum = 0
        step = 0
        for inputs in loop:
            # Forward
            inputs = inputs.to(device)
            encode, decode = model(inputs)

            # Backward
            optimizer.zero_grad()
            loss = criterion(decode, inputs)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            step += 1

            loop.set_description('Epoch[{}/{}]'.format(epoch+1, num_epochs))
            loop.set_postfix(loss=loss_sum/step)

    encoder_outputs, _ = model(torch.from_numpy(feature).float().to(device))
    return encoder_outputs


if __name__ == '__main__':
    df = pd.read_csv('data/mnist.csv', index_col=None)
    lbs = df.label
    feature = df.iloc[:, 1:]
    print('There are {} clusters in the true dataset'.format(len(lbs.unique())))

    feature = feature / 255.0

    rec = reduction(feature.values)

    print(rec)


