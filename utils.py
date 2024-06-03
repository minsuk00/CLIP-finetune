import datetime
import time
import torch
import ignite.distributed as idist
from ignite.utils import convert_tensor


@torch.no_grad()
def evaluate_nn(model, trainloader, testloader):
    X_train, Y_train = [idist.all_gather(_) for _ in collect_features(model, trainloader)]
    X_test, Y_test = collect_features(model, testloader)

    BATCH_SIZE = 256
    corrects = []
    d_train = X_train.T.pow(2).sum(dim=0, keepdim=True)
    for i in range(0, X_test.shape[0], BATCH_SIZE):
        X_batch = X_test[i : i + BATCH_SIZE]
        Y_batch = Y_test[i : i + BATCH_SIZE]
        d_batch = X_batch.pow(2).sum(dim=1)[:, None]
        distance = d_batch - torch.mm(X_batch, X_train.T) * 2 + d_train
        corrects.append((Y_batch == Y_train[distance.argmin(dim=1)]).detach())
    corrects = idist.all_gather(torch.cat(corrects))
    acc = corrects.float().mean()
    return acc


@torch.no_grad()
def collect_features(model, loader):
    model.eval()
    device = idist.device()
    print(f"collecting features on device: {device}...")
    X, Y = [], []
    total_len = len(loader)
    start_time = time.time()
    avg_time_per_batch = 0
    for cnt, batch in enumerate(loader):
        temp_time = time.time()

        x, y = convert_tensor(batch, device=device)
        # x = model(x).image_embeds
        x = model(x).pooler_output
        X.append(x.detach())
        Y.append(y.detach())

        avg_time_per_batch = (avg_time_per_batch * cnt + (time.time() - temp_time)) / (cnt + 1)
        eta_seconds = (total_len - (cnt + 1)) * avg_time_per_batch
        print(
            f"collect done: {cnt+1} / {total_len}, eta: {datetime.timedelta(seconds=eta_seconds)}, time_passed: {datetime.timedelta(seconds=(time.time()-start_time))}",
            end="\r",
        )
    X = torch.cat(X).detach()
    Y = torch.cat(Y).detach()
    return X, Y
