import argparse
import pickle
import numpy as np
from sklearn.cluster import KMeans


def get_args():
    parser = argparse.ArgumentParser(description='Generating category exemplars with K-Means cluster.')
    parser.add_argument('--feature_file', default='./data/thumos_all_feature_val_Kinetics.pickle')
    parser.add_argument('--anno_file', default='./data/thumos_val_anno.pickle')
    parser.add_argument('--cluster_num', default=10)
    parser.add_argument('--category_num', default=21)
    parser.add_argument('--exemplar_file', default='./data/exemplar.pickle')
    args = parser.parse_args()
    return args


def select_feature(args):
    targets = pickle.load(open(args.anno_file, 'rb'))
    features = pickle.load(open(args.feature_file, 'rb'))

    # store all features
    features_all = [None for _ in range(args.category_num)]

    for vid_name, annos in targets.items():
        print(vid_name)
        vid_anno = annos['anno']
        feat_rgb = features[vid_name]['rgb']  # [N, 2048]
        feat_flow = features[vid_name]['flow']
        feature = np.concatenate([feat_rgb, feat_flow], axis=1)  # [N, 4096]
        print(feature.shape)

        category_indicator = np.sum(vid_anno, axis=0)  # [22]
        for i in range(args.category_num):
            if category_indicator[i] == 0:
                continue

            cate_flag = vid_anno[:, i]
            places = np.where(cate_flag == 1)
            cate_feat = feature[places[0], :]
            tmp = features_all[i]
            if tmp is None:
                tmp = cate_feat
            else:
                tmp = np.concatenate([tmp, cate_feat], axis=0)
            features_all[i] = tmp
            print("selct", i, cate_feat.shape)

    return features_all


# NOTICE: as there are 141300 background features, the cluster for background is slow, but others are fast.
def KMeans_cluster(args, features_all):
    exemplars = list()
    for cate_feat in features_all:
        print(cate_feat.shape)
        kmeans = KMeans(n_clusters=args.cluster_num).fit(cate_feat)
        distance = kmeans.transform(cate_feat)

        cate_exemplar = list()
        for i in range(args.cluster_num):
            d = distance[:, i]
            # select the one nearest to center
            idx = np.argsort(d)[0]
            cate_exemplar.append(cate_feat[idx, :])
        exemplars.append(cate_exemplar)

    # save exemplar
    pickle.dump(exemplars, open(args.exemplar_file, 'wb'))

    return


if __name__ == '__main__':
    args = get_args()
    features_all = select_feature(args)
    KMeans_cluster(args, features_all)
