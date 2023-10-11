import time
import torch
import numpy as np
from argparse import ArgumentParser
import torch.nn.functional as F


class Learning_Appr:
    """Basic class for implementing learning approaches"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, momentum=0, wd=0):
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
            train_loss, train_acc, predictions = self.eval(trn_loader)
            self.trn_loss_list.append(train_loss)
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, met={:5.1f}% |'.format(
                e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, predictions = self.eval(val_loader)
            self.val_loss_list.append(valid_loss)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, met={:5.1f}% |'.format(
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
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # def eval(self, val_loader):
    #     """Contains the evaluation code"""
    #     with torch.no_grad():
    #         total_loss, total_num = 0, 0
    #         self.model.eval()
    #         for images, targets in val_loader:
    #             # Forward current model
    #             outputs = self.model(images.to(self.device))
    #             loss = self.criterion(outputs, targets.to(self.device))
    #             # Metric
    #             # hits = (outputs[0].argmax(1) == targets.to(self.device)).float()
    #             # Log
    #             total_loss += loss.item() * len(targets)
    #             # total_acc += hits.sum().item()
    #             total_num += len(targets)
        
    #     return total_loss / total_num, 0.0


    def eval(self, dataloader):
        self.model.eval()
        total_loss, correct_predictions, total_predictions = 0.0, 0, 0
        predictions_list = []
        with torch.no_grad():
            for inputs, targets in dataloader:

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                predicted_classes = torch.argmax(outputs, dim=1, keepdim=True)
                predictions_list.append(predicted_classes)

                # Exclude samples with target value 0
                mask = targets != 0

                correct_predictions += (predicted_classes[mask] == targets[mask]).sum().item()
                total_predictions += mask.sum().item()


                #correct_predictions += (predicted_classes == targets).sum().item()
                #total_predictions += targets.numel()
            

        average_loss = total_loss / len(dataloader)
        #accuracy = correct_predictions / total_predictions

        accuracy = correct_predictions / total_predictions if total_predictions != 0 else 0

        return average_loss, accuracy, predictions_list




    def criterion(self, outputs, targets):
        """Returns the loss value"""

    ## Cross Entropy Loss
        # Class 0 is dominant and classes 1-5 are of interest
        weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0, 1.0]).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        targets = targets.squeeze(1).long()

    ## Divergence Loss
        # criterion = torch.nn.KLDivLoss()
        # outputs = F.softmax(outputs, dim=1)
        # targets = torch.nn.functional.one_hot(targets.long(), num_classes=6)
        # targets = targets.transpose(1,3).squeeze().float()

    ## BCE Loss
        # criterion = torch.nn.BCELoss()
        # ouputs = torch.argmax(outputs, dim=1, keepdim=True)
        # outputs = torch.nn.functional.one_hot(outputs.long(), num_classes=6)
        # outputs = outputs.transpose(1,3).squeeze().float()
        # targets = torch.nn.functional.one_hot(targets.long(), num_classes=6)
        # targets = targets.transpose(1,3).squeeze().float()

    ## MCE Loss
        # outputs = F.softmax(outputs, dim=1)
        # targets = torch.nn.functional.one_hot(targets.long(), num_classes=6)
        # targets = targets.transpose(1,3).squeeze().float()
        # criterion = torch.nn.functional.mse_loss(outputs,targets)


        return criterion(outputs,targets)


