from data import BavarianCrops, BugSenseData
from torch.utils.data import DataLoader
from earlyrnn import EarlyRNN
import torch
from tqdm import tqdm
from loss import EarlyRewardLoss
import numpy as np
from utils import VisdomLogger
import sklearn.metrics
import pandas as pd
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
    

writer = SummaryWriter()

def parse_args():
    parser = argparse.ArgumentParser(description='Run ELECTS Early Classification training on the BavarianCrops dataset.')
    parser.add_argument('--dataset', type=str, default="bugsense", choices=["bavariancrops","breizhcrops", "ghana", "southsudan","unitedstates", "bugsense"], help="dataset")
    parser.add_argument('--alpha', type=float, default=1, help="trade-off parameter of earliness and accuracy (eq 6): "
                                                                 "1=full weight on accuracy; 0=full weight on earliness")
    parser.add_argument('--epsilon', type=float, default=10, help="additive smoothing parameter that helps the "
                                                                  "model recover from too early classificaitons (eq 7)")
    parser.add_argument('--learning-rate', type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument('--weight-decay', type=float, default=0, help="weight_decay")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"], help="'cuda' (GPU) or 'cpu' device to run the code. "
                                                     "defaults to 'cuda' if GPU is available, otherwise 'cpu'")
    parser.add_argument('--epochs', type=int, default=100, help="number of training epochs")
    parser.add_argument('--sequencelength', type=int, default=80, help="sequencelength of the time series. If samples are shorter, "
                                                                "they are zero-padded until this length; "
                                                                "if samples are longer, they will be undersampled")
    parser.add_argument('--batchsize', type=int, default=8, help="number of samples per batch")
    parser.add_argument('--dataroot', type=str, default=os.path.join(os.environ["HOME"],"elects_data"), help="directory to download the "
                                                                                 "BavarianCrops dataset (400MB)."
                                                                                 "Defaults to home directory.")
    parser.add_argument('--snapshot', type=str, default="./snapshots/bugsense_model.pth",
                        help="pytorch state dict snapshot file")
    parser.add_argument('--resume', action='store_true')


    args = parser.parse_args()

    if args.patience < 0:
        args.patience = None

    return args
    

def main(args):
    if args.dataset == "bugsense":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(script_dir, "..", "..", "..",  "BugSenseData", "Usable")
        nclasses = 6
        input_dim = 3
        train_ds = BugSenseData(root_dir, partition="train", sequencelength=args.sequencelength)
        val_ds = BugSenseData(root_dir, partition="valid", sequencelength=args.sequencelength)
        test_ds = BugSenseData(root_dir, partition="eval", sequencelength=args.sequencelength)
    

    else:
        raise ValueError(f"dataset {args.dataset} not recognized")
    print("Train Data: ",len(train_ds))
    print("Validation Data: ", len(val_ds))
    print("Test Data: ", len(test_ds))


    traindataloader = DataLoader(
        train_ds,
        batch_size=args.batchsize,
        drop_last=True)
    valdataloader = DataLoader(
        val_ds,
        batch_size=args.batchsize,
        drop_last=True)

    model = EarlyRNN(nclasses=nclasses, input_dim=input_dim).to(args.device)

    decay, no_decay = list(), list()
    for name, param in model.named_parameters():
        if name == "stopping_decision_head.projection.0.bias":
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0, "lr": args.learning_rate}, {'params': decay}],
                                  lr=args.learning_rate, weight_decay=args.weight_decay)

    criterion = EarlyRewardLoss(alpha=args.alpha, epsilon=args.epsilon)

    if args.resume and os.path.exists(args.snapshot):
        model.load_state_dict(torch.load(args.snapshot, map_location=args.device))
        optimizer_snapshot = os.path.join(os.path.dirname(args.snapshot),
                                          os.path.basename(args.snapshot).replace(".pth", "_optimizer.pth")
                                          )
        optimizer.load_state_dict(torch.load(optimizer_snapshot, map_location=args.device))
        df = pd.read_csv(args.snapshot + ".csv")
        train_stats = df.to_dict("records")
        start_epoch = train_stats[-1]["epoch"]
        print(f"resuming from {args.snapshot} epoch {start_epoch}")
    else:
        train_stats = []
        start_epoch = 1

    visdom_logger = VisdomLogger()

    not_improved = 0
    with tqdm(range(start_epoch, args.epochs + 1)) as pbar:
        for epoch in pbar:
            trainloss = train_epoch(model, traindataloader, optimizer, criterion, device=args.device)
            testloss, stats = test_epoch(model, valdataloader, criterion, args.device)
            print("testloss: ", testloss)
            print("trainloss: ", trainloss)
            print("y_true shape: ", stats["targets"][:,0])
            print("predictions at t stop: ", stats["predictions_at_t_stop"][:,0])
            print("t_stop: ", stats["t_stop"][:,0])

            # statistic logging and visualization...
            precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(
                y_pred=stats["predictions_at_t_stop"][:, 0], y_true=stats["targets"][:,0], average="macro",
                zero_division=0)
            accuracy = sklearn.metrics.accuracy_score(
                y_pred=stats["predictions_at_t_stop"][:, 0], y_true=stats["targets"][:,0])
            kappa = sklearn.metrics.cohen_kappa_score(
                stats["predictions_at_t_stop"][:, 0], stats["targets"][:,0])

            classification_loss = stats["classification_loss"].mean()
            earliness_reward = stats["earliness_reward"].mean()
            earliness = 1 - (stats["t_stop"].mean() / (args.sequencelength - 1))

            stats["confusion_matrix"] = sklearn.metrics.confusion_matrix(y_pred=stats["predictions_at_t_stop"][:, 0],
                                                                         y_true=stats["targets"][:,0])

            train_stats.append(
                dict(
                    epoch=epoch,
                    trainloss=trainloss,
                    testloss=testloss,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    fscore=fscore,
                    kappa=kappa,
                    earliness=earliness,
                    classification_loss=classification_loss,
                    earliness_reward=earliness_reward
                )
            )

            writer.add_scalar('Loss/train', trainloss, epoch)
            writer.add_scalar('Loss/test', testloss, epoch)
            writer.add_scalar('Accuracy', accuracy, epoch)
            writer.add_scalar('Precision', precision, epoch)
            writer.add_scalar('Recall', recall, epoch)
            writer.add_scalar('F1', fscore, epoch)
            writer.add_scalar('Kappa', kappa, epoch)
            writer.add_scalar('Earliness', earliness, epoch)


            visdom_logger(stats)
            visdom_logger.plot_boxplot(stats["targets"][:,0], stats["t_stop"][:, 0], tmin=0, tmax=args.sequencelength)
            df = pd.DataFrame(train_stats).set_index("epoch")
            visdom_logger.plot_epochs(df[["precision", "recall", "fscore", "kappa"]], name="accuracy metrics")
            visdom_logger.plot_epochs(df[["trainloss", "testloss"]], name="losses")
            visdom_logger.plot_epochs(df[["accuracy", "earliness"]], name="accuracy, earliness")
            visdom_logger.plot_epochs(df[["classification_loss", "earliness_reward"]], name="loss components")

            savemsg = ""
            if len(df) > 2:
                if testloss < df.testloss[:-1].values.min():
                    savemsg = f"saving model to {args.snapshot}"
                    os.makedirs(os.path.dirname(args.snapshot), exist_ok=True)
                    torch.save(model.state_dict(), args.snapshot)

                    optimizer_snapshot = os.path.join(os.path.dirname(args.snapshot),
                                                      os.path.basename(args.snapshot).replace(".pth", "_optimizer.pth")
                                                      )
                    torch.save(optimizer.state_dict(), optimizer_snapshot)
                    

                    df.to_csv(args.snapshot + ".csv")
                    not_improved = 0 # reset early stopping counter
                else:
                    not_improved += 1 # increment early stopping counter
                    if args.patience is not None:
                        savemsg = f"early stopping in {args.patience - not_improved} epochs."
                    else:
                        savemsg = ""

            pbar.set_description(f"epoch {epoch}: trainloss {trainloss:.2f}, testloss {testloss:.2f}, "
                     f"accuracy {accuracy:.2f}, earliness {earliness:.2f}. "
                     f"classification loss {classification_loss:.2f}, earliness reward {earliness_reward:.2f}. {savemsg}")

            if args.patience is not None:
                if not_improved > args.patience:
                    print(f"stopping training. testloss {testloss:.2f} did not improve in {args.patience} epochs.")
                    break


    return train_ds, val_ds, test_ds

def train_epoch(model, dataloader, optimizer, criterion, device):
    losses = []
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        X, y_true = batch

        X, y_true = X.to(device), y_true.to(device)
        log_class_probabilities, probability_stopping = model(X)
        loss = criterion(log_class_probabilities, probability_stopping, y_true)
        if not torch.isnan(loss).any():
            loss.backward()

            optimizer.step()

            losses.append(loss.cpu().detach().numpy())
    return np.stack(losses).mean()

def test_epoch(model, dataloader, criterion, device):
    model.eval()

    stats = []
    losses = []
    for batch in dataloader:
        X, y_true = batch
        X, y_true = X.to(device), y_true.to(device)

        log_class_probabilities, probability_stopping, predictions_at_t_stop, t_stop = model.predict(X)
        loss, stat = criterion(log_class_probabilities, probability_stopping, y_true, return_stats=True)

        stat["loss"] = loss.cpu().detach().numpy()
        stat["probability_stopping"] = probability_stopping.cpu().detach().numpy()
        stat["class_probabilities"] = log_class_probabilities.exp().cpu().detach().numpy()
        stat["predictions_at_t_stop"] = predictions_at_t_stop.unsqueeze(-1).cpu().detach().numpy()
        stat["t_stop"] = t_stop.unsqueeze(-1).cpu().detach().numpy()
        stat["targets"] = y_true.cpu().detach().numpy()

        stats.append(stat)

        losses.append(loss.cpu().detach().numpy())

    # list of dicts to dict of lists
    stats = {k: np.vstack([dic[k] for dic in stats]) for k in stats[0]}

    return np.stack(losses).mean(), stats

if __name__ == '__main__':
    args = parse_args()
    train_ds, val_ds, test_ds = main(args)
    