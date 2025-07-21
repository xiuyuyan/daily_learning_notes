import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status (1: training, 0: testing)')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model identifier (formatted as {model}_{seq_len}_{pred_len})')
    parser.add_argument('--model', type=str, required=False, default='PatchTST',
                        help='model name, options: [Autoformer, Informer, Transformer, PatchTST]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type (e.g., ETT-small, ETTh1)')
    parser.add_argument('--root_path', type=str, default='.\\dataset\\ETT-small', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='path to the data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task type: [M (multivariate predict multivariate), S (univariate predict univariate), MS (multivariate predict univariate)]')
    parser.add_argument('--target', type=str, default='OT', help='target feature (for S or MS tasks)')
    parser.add_argument('--freq', type=str, default='h',
                        help='frequency of time series (options: [s, t, h, d, b, w, m] or detailed like 15min, 3h)')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='directory to save model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length (history window)')
    parser.add_argument('--label_len', type=int, default=48, help='start token length (used in some models)')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length (forecast window)')

    # PatchTST-specific parameters
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='dropout rate for fully connected layers')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='dropout rate for prediction head')
    parser.add_argument('--patch_len', type=int, default=16, help='length of each patch (for PatchTST)')
    parser.add_argument('--stride', type=int, default=8, help='stride for patching (for PatchTST)')
    parser.add_argument('--padding_patch', default='end', help='padding strategy for patches (None/end)')
    parser.add_argument('--revin', type=int, default=1, help='whether to use RevIN (1: enable, 0: disable)')
    parser.add_argument('--affine', type=int, default=0, help='whether to use affine transformation in RevIN (1: enable, 0: disable)')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last value')
    parser.add_argument('--decomposition', type=int, default=0, help='whether to use decomposition (1: enable, 0: disable)')
    parser.add_argument('--kernel_size', type=int, default=25, help='kernel size for decomposition')
    parser.add_argument('--individual', type=int, default=0, help='whether to use individual head (1: enable, 0: disable)')

    # Transformer/Formers-specific parameters
    parser.add_argument('--embed_type', type=int, default=0,
                        help='type of time embedding: [0: default, 1: value+temporal+positional, 2: value+temporal, 3: value+positional, 4: value]')
    parser.add_argument('--enc_in', type=int, default=7, help='number of encoder input features')
    parser.add_argument('--dec_in', type=int, default=7, help='number of decoder input features')
    parser.add_argument('--c_out', type=int, default=7, help='number of output features')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model embeddings')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of feed-forward network')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size for moving average')
    parser.add_argument('--factor', type=int, default=1, help='attention factor (for Informer)')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use knowledge distillation (default: True)')
    parser.add_argument('--dropout', type=float, default=0.05, help='general dropout rate')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time feature embedding: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function (e.g., gelu, relu)')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention weights')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers for data loading')
    parser.add_argument('--itr', type=int, default=1, help='number of experiment iterations')
    parser.add_argument('--train_epochs', type=int, default=2, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--patience', type=int, default=1, help='patience for early stopping')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--des', type=str, default='test', help='description of the experiment')
    parser.add_argument('--loss', type=str, default='mse', help='loss function (e.g., mse, mae)')
    parser.add_argument('--lradj', type=str, default='type3', help='learning rate adjustment strategy')
    parser.add_argument('--pct_start', type=float, default=0.3, help='percentage of training steps for warmup')
    parser.add_argument('--use_amp', action='store_true', help='whether to use automatic mixed precision')

    # GPU configuration
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
    parser.add_argument('--use_multi_gpu', action='store_true', help='whether to use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='comma-separated list of GPU indices')
    parser.add_argument('--test_flop', action='store_true', default=False, help='whether to calculate FLOPs')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
