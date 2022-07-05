import logging

import torch
from torch import nn

from utils.model_trainer import ModelTrainer

class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.weight_out()

    def set_model_params(self, model_parameters1, model_parameters2):
        self.model.weight_in(model_parameters1, model_parameters2)

    def train(self, train_data, device, args, client_idx=0, cross_client=None, cross_pointer=None):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=args.wd)
        ptr = cross_pointer

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (images, _) in enumerate(train_data):
                images[0] = images[0].to(device)
                images[1] = images[1].to(device)
                output, target, ptr = model(images[0], images[1], cross_client, ptr)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            #logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format( client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))
        return ptr

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
