import os

import numpy as np
import torch.nn as nn
import torch
import wandb

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


class XSquaredApproximator(nn.Module):
    def __init__(self, epochs, learning_rate, depth, logdir, cfg, wandb_tags):
        super(XSquaredApproximator, self).__init__()
        if not (os.path.exists(logdir)):
            os.makedirs(logdir)
        np.save(os.path.join(logdir, "config.npy"), cfg)

        self.wandb_group = None
        self.cfg = cfg
        self.wandb_tags = wandb_tags
        self.logdir = logdir
        self.results = {}
        self.losses = []
        self.test_loss = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 256
        self.checkpoint = 1000
        self.nb_epoch = epochs
        self.input_size = 1
        self.mid_weight = 64
        # self.weight_size = 64
        self.learning_rate = learning_rate
        self.columns = ["epoch", "loss", "test_loss"]

        mid_layers = []
        for _ in range(depth - 2):
            mid_layers.append(nn.Linear(self.mid_weight, self.mid_weight))
            mid_layers.append(nn.ReLU())

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.mid_weight),
            nn.ReLU(),
            nn.Sequential(*mid_layers),
            nn.Linear(self.mid_weight, self.input_size),
        )

        self.model.apply(init_weights)
        self.model.double()

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=0.001,
                                          lr=learning_rate)  # , lr=self.learning_rate)#, weight_decay=1e-4)
        self.to(self.device)

    def fit(self, x, y, x_test, y_test, gif_info=None):
        if self.use_wandb:
            wandb.login(key="cfc00eee102a1e9647b244a40066bfc5f1a96610")
            name = f"Model "
            wb_resume = "allow"  # if args.model_file is None else "must"
            project = "X Squared"
            if self.wandb_group is not None:
                wandb.init(project=project, config=self.cfg, sync_tensorboard=True,
                           tags=self.wandb_tags, resume=wb_resume, name=name, group=self.wandb_group)
            else:
                wandb.init(project=project, config=self.cfg, sync_tensorboard=True,
                           tags=self.wandb_tags, resume=wb_resume, name=name)
        self.results = {}
        x = torch.from_numpy(x).to(self.device)
        y = torch.from_numpy(y).to(self.device)
        x_test = torch.from_numpy(x_test).to(self.device)
        y_test = torch.from_numpy(y_test).to(self.device)
        batch_size = self.batch_size
        for epoch in range(self.nb_epoch):
            permutation = torch.randperm(x.size()[0])
            for i in range(0, x.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = x[indices], y[indices]
                outputs = self.model(batch_x)
                self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            self.losses.append(loss.item())
            print(f"{loss:.2f}: loss at epoch {epoch}")

            # test section
            if epoch % self.checkpoint == 0:
                with torch.no_grad:
                    outputs = self.model(x_test)
                    loss = self.loss_fn(outputs, y_test)
                    self.results[epoch] = outputs.detach().numpy().squeeze()
                    self.test_loss.append(loss.item())
                print("Saving model.")
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           f'{self.logdir}/model_{epoch}.pth')
                if self.use_wandb:
                    log = [epoch, self.losses[-1], self.test_loss[-1]]
                    wandb.log({k: v for k, v in zip(self.columns, log)})
            # if epoch % 100 == 0 and epoch > 0:
            #     generate_results_gif(x_test, y_test, self.results, gif_info)

    # def forward(self, batch_x):
    #     batch_x = batch_x.reshape(batch_x.shape[0], 1)
    #     return self.model(batch_x)

    # def predict(self, state):
    #     return self.model(state)
