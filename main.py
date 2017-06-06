# coding: utf-8
import dogs_vs_cats
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the id of gpu', type=int)
    parser.add_argument('--trial', help='the flag of trial run',
                        action='store_true')
    args = parser.parse_args()

    output_path = 'vgg-cv-prediction.csv'
    n_epoch = 20
    if args.trial:
        n_epoch = 2
        output_path = 'vgg-cv-prediction-trial.csv'
    vgg_result = dogs_vs_cats.train_vgg16(n_epoch, args.gpu, args.trial)
    vgg_result.to_csv(output_path, index=False)
