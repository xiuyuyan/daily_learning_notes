from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from fastai.data.load import DataLoader
from numba.cuda.libdevice import frexp
from pyexpat import features

from statsmodels.sandbox.distributions.examples.matchdist import targetdist
from sympy.core.random import shuffle
from torch.utils.data import Dataset

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1 # 选择时间编码方式

    if flag == 'test':
        shuffle_flag = False
        drop_last = True # 是否丢弃最后一个不完整的批次（为了保证批次大小一致）
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path = args.root_path,
        data_path = args.data_path,
        flag = flag,
        size = [args.seq_len, args.label_len,args.pred_len], # 回视窗口，先验窗口，预测窗口
        features = args.features, # 预测任务类型，M:多预测多、S：单预测单、MS：多预测单
        target = args.target,
        timeenc = timeenc,
        freq = freq,
    )

    print(flag,len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size = batch_size,
        shuffle = shuffle_flag,
        num_workers = args.num_workers,
        drop_last = drop_last,
    )

    return data_set,data_loader