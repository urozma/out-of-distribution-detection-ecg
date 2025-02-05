import time
import numpy as np
from argparse import ArgumentParser
import torch
from sklearn.metrics import f1_score

class Learning_Appr:
    """Basic class for implementing learning approaches"""

    def __init__(self, model, device, data_type, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, momentum=0, wd=0):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.momentum = momentum
        self.wd = wd
        self.optimizer = None
        self.trn_loss_list = []
        self.val_loss_list = []
        self.data_type = data_type

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train(self, trn_loader, val_loader):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer()
    
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(trn_loader)
            clock1 = time.time()
            train_loss, train_acc, predictions, _ = self.eval(trn_loader)
            self.trn_loss_list.append(train_loss)
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, predictions, _ = self.eval(val_loader)
            self.val_loss_list.append(valid_loss)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        # if the lr decreases below minimum, stop the training session
                        print()
                        break
                    # reset patience and recover best model so far to continue training
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    self.model.set_state_dict(best_model)
            print()
        self.model.set_state_dict(best_model)

    def train_epoch(self, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        for signals, targets in trn_loader:
            # Forward current model
            outputs = self.model(signals.to(self.device))
            loss = self.criterion(outputs, targets.to(self.device))
            l1_loss = sum(p.abs().sum() for p in self.model.parameters())
            total_loss = loss + 0 * l1_loss

            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def eval(self, dataloader):
        self.model.eval()
        total_loss, correct_predictions, total_predictions = 0.0, 0, 0
        predictions_list, true_labels, latent_spaces = [], [], []

        with torch.no_grad():
            for inputs, targets in dataloader:

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)

                latent_space = self.model.extract_latent_space(inputs)
                latent_spaces.append(latent_space)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                predicted_classes = torch.argmax(outputs, dim=1, keepdim=True)
                predictions_list.append(predicted_classes)

                # Exclude samples with target value 0
                mask = targets != 0

                correct_predictions += (predicted_classes[mask] == targets[mask]).sum().item()
                total_predictions += mask.sum().item()


        average_loss = total_loss / len(dataloader)

        accuracy = correct_predictions / total_predictions if total_predictions != 0 else 0

        return average_loss, accuracy, predictions_list, latent_spaces

    def criterion(self, outputs, targets):
        """Returns the loss value"""

        weights = torch.tensor([0.04, 1.0, 1.0, 0.4, 1.0, 1.0]).to(self.device)
        targets = targets.squeeze(1).long()
        criterion= torch.nn.CrossEntropyLoss(weight=weights)
        loss = criterion(outputs, targets)

        return loss


