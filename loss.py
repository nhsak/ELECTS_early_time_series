import torch
from torch import nn

class EarlyRewardLoss(nn.Module):
    def __init__(self, alpha=0.5, epsilon=10, weight=None):
        super(EarlyRewardLoss, self).__init__()

        self.negative_log_likelihood = nn.NLLLoss(reduction="none", weight=weight)
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, log_class_probabilities, probability_stopping, y_true, return_stats=False):
        y_true = y_true.permute(1,0)
        T, N, C = log_class_probabilities.shape
        

        # equation 3
        Pt = calculate_probability_making_decision(probability_stopping)

        # equation 7 additive smoothing
        Pt = Pt + self.epsilon / T

        # equation 6, right term
        t = torch.ones(N, T, device=log_class_probabilities.device) * \
                  torch.arange(T).type(torch.FloatTensor).to(log_class_probabilities.device)

        y_haty = probability_correct_class(log_class_probabilities, y_true)
        y_haty = y_haty.permute(1,0)
        t = t.permute(1,0)
        earliness_reward = Pt * y_haty * (1 - t / T)
        earliness_reward = earliness_reward.sum(1).mean(0)
        # equation 6 left term
        y_true = y_true.reshape(T*N)
        log_class_probabilities = log_class_probabilities.view(T*N,C)
        
        
        cross_entropy = self.negative_log_likelihood(log_class_probabilities, y_true).view(T,N)
        classification_loss = (cross_entropy * Pt).sum(1).mean(0)

        # equation 6
        loss = self.alpha * classification_loss - (1-self.alpha) * earliness_reward

        if return_stats:
            stats = dict(
                classification_loss=classification_loss.cpu().detach().numpy(),
                earliness_reward=earliness_reward.cpu().detach().numpy(),
                probability_making_decision=Pt.cpu().detach().numpy()
            )
            return loss, stats
        else:
            return loss

def calculate_probability_making_decision(deltas):
    """
    Equation 3: probability of making a decision

    :param deltas: probability of stopping at each time t
    :return: comulative probability of having stopped
    """
    batchsize, sequencelength = deltas.shape
    

    pts = list()

    initial_budget = torch.ones(batchsize, device=deltas.device)

    budget = [initial_budget]
    for t in range(1, sequencelength):
        pt = deltas[:, t] * budget[-1]
        budget.append(budget[-1] - pt)
        pts.append(pt)

    # last time
    pt = budget[-1]
    pts.append(pt)

    return torch.stack(pts, dim=-1)

def probability_correct_class(logprobabilities, targets):
    
    seqquencelength, batchsize, nclasses = logprobabilities.shape
    

    eye = torch.eye(nclasses).type(torch.ByteTensor).to(logprobabilities.device)

    targets_one_hot = eye[targets]
   
    y_haty = torch.masked_select(logprobabilities, targets_one_hot.bool())
    return y_haty.view(batchsize, seqquencelength).exp()  
