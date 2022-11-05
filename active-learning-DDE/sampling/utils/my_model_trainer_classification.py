import logging

import torch
from torch import nn

from utils.model_trainer import ModelTrainer

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)



def uncertainty_score(predictions, labels, classes, c):
    predictions = c(predictions)
    truth_p = torch.gather(predictions, 1, labels.reshape((-1,1))).reshape(-1)
    pp1 = 1/classes - (1-truth_p)/(classes-1)
    pp1 = (classes-1) * torch.pow(pp1, 2)
    pp2 = 1/classes - truth_p
    pp2 = torch.pow(pp2, 2)
    pp3 = (1/classes) * (pp1 + pp2)
    var = torch.var(predictions, 1)
    score = 1-truth_p*pp3/var

    return score


def LossPredLoss(input, logits, labels, margin=0.3, classes=100, c = None):
    input = (input - input.flip(0))[:len(input)//2]
    score = uncertainty_score(logits, labels, classes, c)
    score = (score - score.flip(0))[:len(score)//2]
    score = score.detach()
    one = 2 * torch.sign(torch.clamp(score, min=0)) - 1
    loss = torch.sum(torch.clamp(margin - one * input, min=0))
    loss = loss / input.size(0)
    return loss 

class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict(), self.statenet.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters[0])
        self.statenet.load_state_dict(model_parameters[1])

    def train(self, train_data, device, args, client_idx=0):
        model = self.model
        statenet = self.statenet
        softmaxm = nn.Softmax(dim=1)

        model.to(device)
        model.train()
        statenet.to(device)
        statenet.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay = args.wd)# 5e-4  momentum = 0.9
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,  weight_decay=args.wd) #
            
        optim_backbone = torch.optim.SGD(self.statenet.parameters(), lr=args.lr, momentum=0.8, weight_decay=args.wd)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            if epoch == 100:
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay = args.wd) # adam -> sgd, bp speeds up
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                statenet.zero_grad()
                log_probs, features = model(x, feature = True)
                loss = criterion(log_probs, labels)
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                pred_loss = statenet(features)
                pred_loss = pred_loss.view(pred_loss.size(0))

                m_module_loss   = LossPredLoss(pred_loss, log_probs, labels, margin=0.5, classes = self.class_num, c = softmaxm)
                loss            = loss +  m_module_loss

                loss.backward()

                # to avoid nan loss
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                optim_backbone.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            #logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format( client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))

    def get_uncertainty(self, unlabeled_loader, device):
        model = self.model
        statenet = self.statenet
        model.eval()
        statenet.eval()
        model.to(device)
        statenet.to(device)

        uncertainty = torch.tensor([]).cuda()

        with torch.no_grad():
            for (inputs, labels) in unlabeled_loader:
                inputs = inputs.cuda()
                # labels = labels.cuda()

                scores, features = model(inputs, feature = True)
                pred_loss = statenet(features) # pred_loss = criterion(scores, labels) # ground truth loss
                pred_loss = pred_loss.view(pred_loss.size(0))

                uncertainty = torch.cat((uncertainty, pred_loss), 0)
        
        return uncertainty.cpu()

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
