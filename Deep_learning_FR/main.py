import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import argparse
import math
import time
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import pearsonr, entropy
from sklearn.metrics import mean_squared_error


from model import DeepGravity
from utils import Data_utility
from Optim import Optim

def make_dir(path, dic_name):
    path = os.path.join(path, dic_name)
    is_dir_exist = os.path.exists(path)
    if is_dir_exist:
        print("----Dic existed----")
    else:
        os.mkdir(path)
        print("----Dic created successfully----")
    return path

def jesen_shannon_divergence(generated_flow, real_flow):
    """
    The implementation of jesen_shannon_divergence
    """

    jsd = 0
    for i in range(len((generated_flow))):
        p = real_flow[i]
        q = generated_flow[i]
        M = (generated_flow + real_flow)/2
        jsd += 0.5*scipy.stats.entropy(generated_flow, M, base=2)+0.5*scipy.stats.entropy(real_flow, M, base=2)

    return jsd /len(generated_flow)

def common_part_of_commuters(generated_flow, real_flow, numerator_only=False):
    """
    The implementation of common_part_of_commuters
    :param generated_flow:
    :param real_flow:
    :param numerator_only:
    :return: The value of CPC
    """
    cpc = 0
    for i in range(len((generated_flow))):
        if numerator_only:
            tot = 1.0
        else:
            tot = (np.sum(generated_flow[i]) + np.sum(real_flow[i]))
        if tot > 0:
            cpc += 2.0 * np.sum(np.minimum(generated_flow[i], real_flow[i])) / tot
        else:
            return 0.0
    return cpc/len((generated_flow))

def Corr(generated_flow, real_flow,):
    """
    The implementation of pearsonr
    :param generated_flow:
    :param real_flow:
    :return: The value of corr
    """
    corr = 0
    warnings.filterwarnings('error')
    for i in range(len((generated_flow))):
        try:
            corr += pearsonr(generated_flow[i], real_flow[i])[0]
        except:
            r1 = random.randint(1,10)
            r2 = random.randint(10,20)
            corr += pearsonr(np.concatenate([generated_flow[i], np.array([r1]), np.array([r2])], axis = 0), np.concatenate([real_flow[i], np.array([r1]), np.array([r2])], axis = 0))[0]
    return corr / len((generated_flow))


def evaluate(data, X, Y, model, batch_size=1, mode="valid"):
    """
    Evaluating or testing function
    :param data:
    :param X:
    :param Y:
    :param model:
    :param batch_size:
    :return:
    """
    model.eval()
    generated_flow = []
    real_flow = []

    for x, y in data.get_batches(X, Y, batch_size):
        for u, v in zip(x, y):
            u = Variable(u).cuda().float()
            v = Variable(v).cuda().float() # (9, 1)
            # print(v.shape)
            sum = torch.sum(v)
            out = model(u) #(9)

            if len(out.shape)!=1:  #方式出现训练最后一个step时，出现v是一维的情况
                out=torch.unsqueeze(out,0)
            # torch.nn.CrossEntropyLoss()的input只需要是网络fc层的输出 y, 在torch.nn.CrossEntropyLoss()里它会自己把y转化成softmax(y), 然后再进行交叉熵loss的运算.
            output_prob = nn.functional.softmax(out, dim = -1)
            output = output_prob * sum

            generated_flow.append(output.cpu().detach().numpy())
            real_flow.append(v.cpu().numpy())

    cpc = common_part_of_commuters(generated_flow, real_flow)
    corr = Corr(generated_flow, real_flow)

    if mode == "test":
        o_index = data.test_o
        d_index = data.test_d
        o_index = np.concatenate([list(o_index[i]) for i in range(len(o_index))])
        d_index = np.concatenate([list(d_index[i]) for i in range(len(d_index))])
        # print(o_index.shape)
        generated_flow_flatten = np.concatenate([list(generated_flow[i]) for i in range(len(generated_flow))])
        generated_flow_flatten = generated_flow_flatten*(data.train_max_flow-data.train_min_flow) + data.train_min_flow
        real_flow_flatten = np.concatenate([list(real_flow[i]) for i in range(len(real_flow))])
        real_flow_flatten = real_flow_flatten*(data.train_max_flow-data.train_min_flow) + data.train_min_flow
        # print(real_flow_flatten.shape)
        test_results = pd.DataFrame.from_dict({'o':o_index, 'd':d_index, 
                                'Predict_flow': generated_flow_flatten, 'Actual_flow': real_flow_flatten})
    
        test_results.to_csv('/home/dizhu/ma000523/IIDS_intra_urban_scale_directional/Deep_learning_FR/without_weather_result/12345/results/without_weather_test.csv', index=False)
    return cpc, corr


def train(data, X, Y, model, criterion, optim, batch_size):
    """
    Training function
    :param data:
    :param X:
    :param Y:
    :param model:
    :param criterion:
    :param optim:
    :param batch_size:
    :return:
    """
    model.train()

    total_loss = 0.0
    generated_flow = []
    real_flow = []

    for x, y in data.get_batches(X, Y, batch_size):
        model.zero_grad()
        batch_loss = 0.0
        for u, v in zip(x, y):
            # print(u.shape)
            # print(v.shape)
            u = Variable(u).cuda().float()
            v = Variable(v).cuda().float() # (9, 1)
            sum = torch.sum(v)
            v_prob = v/sum
            out = model(u) #(9)
            # print(out)
            if len(out.shape)!=1:  #方式出现训练最后一个step时，出现v是一维的情况
                out=torch.unsqueeze(out,0)
            # torch.nn.CrossEntropyLoss()的input只需要是网络fc层的输出 y, 在torch.nn.CrossEntropyLoss()里它会自己把y转化成softmax(y), 然后再进行交叉熵loss的运算.
            batch_loss += criterion(out, v_prob)
            output_prob = nn.functional.softmax(out, dim = -1)

            output = output_prob * sum

            generated_flow.append(output.cpu().detach().numpy())
            real_flow.append(v.cpu().numpy())

        batch_loss.backward()
        grad_norm = optim.step()
        total_loss += batch_loss.cpu().data.numpy()
    cpc = common_part_of_commuters(generated_flow, real_flow)
    corr = Corr(generated_flow, real_flow)

    return total_loss, cpc, corr


parser = argparse.ArgumentParser(description='DeepGravity on Minneapolis')
parser.add_argument('--model', type=str, default='DeepGravity', help='The model of DG')

### hyper-parameters
parser.add_argument('--data', type=str, default="dg", help="the dataset that we want to predict")
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--seed', type=int, default=12345, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help="The Index of GPU where we want to run the code")
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument("--weather", type=bool, default=True, help="whether using weather data or not")

parser.add_argument('--normalize', type=int, default=2, help="The way of normalization during the data preprocessing")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip', type=float, default=10., help='gradient_visual clipping')
parser.add_argument('--optim', type=str, default='adam')

args = parser.parse_args()


# location, path, dictionary 


path = "/home/dizhu/ma000523/IIDS_intra_urban_scale_directional/Deep_learning_FR/without_weather_result"
path_result = make_dir(path, str(args.seed))
path_log = make_dir(path_result, "log")
path_model = make_dir(path_result, "pkl")
path_result = make_dir(path_result, "results")


print("----Spliting the training and testing data----")
Data = Data_utility("dg", 0.6, 0.2, 2)

# The setting of GPU
args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)



print("----Building models----")
model = eval(args.model).Model(input_dim = 19, hidden_dim = 128, dropout_p= 0.2)

if args.cuda:
    model.cuda()

nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('----number of parameters: %d----' % nParams)

criterion = nn.CrossEntropyLoss(reduction="mean")
optim = Optim(model.parameters(), args.optim, args.lr, args.clip)

best_valid_cpc = 0.0
writer = SummaryWriter(path_log)
try:
    print('----Traning begin----')
    last_update = 1 # 停止
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        select_att = [i for i in range(24)]
        select_att.append(27)
        select_att_no_weather=[3,4,5,6,7,8,9,10,11,15,16,17,18,19,20,21,22,23,27]
        train_loss, train_cpc, train_corr = train(Data, Data.train[:,:,select_att_no_weather], Data.train[:,:,-2], model, criterion, optim, args.batch_size)
        valid_cpc, valid_corr = evaluate(Data, Data.valid[:,:,select_att_no_weather], Data.valid[:,:,-2], model, 1, "valid")
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.4f} | train cpc  {:5.4f} | train corr {:5.4f} | valid cpc {:5.4f} | valid corr {:5.4f}'.format(
            epoch, (time.time() - epoch_start_time), train_loss, train_cpc, train_corr, valid_cpc, valid_corr))
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_cpc", train_cpc, epoch)
        writer.add_scalar("train_corr", train_corr, epoch)
        writer.add_scalar("valid_cpc", valid_cpc, epoch)
        writer.add_scalar("valid_corr", valid_corr, epoch)

        if best_valid_cpc < valid_cpc:
            print("----epoch:{}, save the model----".format(epoch))
            with open(os.path.join(path_model, "model.pkl"), 'wb') as f:
                torch.save(model, f)
            with open(os.path.join(path_log, "log.txt"), "a") as file:
                file.write("epoch:{}, save the model.\r\n".format(epoch))
            best_valid_cpc = valid_cpc
            last_update = epoch

        if epoch - last_update == 75:
            break

except KeyboardInterrupt:
    print('-' * 90)
    print('----Exiting from training early----')

print("----Testing begin----")
with open(os.path.join(path_model, "model_without_weather.pkl"), 'rb') as f:
    model = torch.load(f)
    select_att = [i for i in range(24)]
    select_att.append(27)
    select_att_no_weather=[3,4,5,6,7,8,9,10,11,15,16,17,18,19,20,21,22,23,27]
    test_cpc, test_corr= evaluate(Data, Data.test[:,:,select_att_no_weather], Data.test[:,:,-2], model, 1, "test")#, "test"

print("test cpc {:5.4f}| test cpc {:5.4f}".format(test_cpc, test_corr))
# time.sleep(60)


