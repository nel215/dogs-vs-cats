# coding: utf-8
import dogs_vs_cats
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the id of gpu', type=int)
    parser.add_argument('--batchsize',
                        help='the batchsize id mini-batch', type=int)
    parser.add_argument('--cv', help='run cv prediction task or not',
                        action='store_true')
    parser.add_argument('--trial', help='the flag of trial run',
                        action='store_true')
    args = parser.parse_args()

    if args.batchsize is None:
        args.batchsize = 64

    task = dogs_vs_cats.PredictionTask(
        args.gpu,
        args.batchsize,
        args.cv,
    )
    output_path = 'vgg-cv-prediction.csv'
    n_epoch = 20
    if args.trial:
        n_epoch = 2
        output_path = 'vgg-cv-prediction-trial.csv'
    vgg_result = task.run(n_epoch, args.trial)
    vgg_result.to_csv(output_path, index=False)
