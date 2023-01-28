import numpy as np
import os
from collections import defaultdict
import random
from typing import List

from .clusters import Member, Cluster
DATA_DIR = os.path.join(os.path.abspath(""), "data", "20news-bydate")

class KMeans:
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for i in range(self._num_clusters)]
        self._E: Member = []       # list of centroids
        self._S: float = 0.0        # overall similarity

    def load_data(self, path: str = DATA_DIR):
        def sparse_to_dense(sparse_r_d: str, vocab_size: int):
            r_d = [0.0 for _ in range(vocab_size)]
            indices_and_tfidfs = sparse_r_d.split()
            for index_and_tfidf in indices_and_tfidfs:
                index = int(index_and_tfidf.split(':')[0])
                tfidf = float(index_and_tfidf.split(':')[1])
                r_d[index] = tfidf
            return np.array(r_d)

        with open(os.path.join(path, "data_tf_idf.txt")) as f:
            data_lines = f.read().splitlines()
        with open(os.path.join(path, "words_idfs.txt")) as f:
            vocab_size = len(f.read().splitlines())
        
        # member store info of data points
        self._data: List[Member] = []
        # count number of label (newsgroup)
        self._label_count = defaultdict(int)
        
        for data_id, d in enumerate(data_lines):
            features = d.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)

            # append data with class Member
            self._data.append(Member(r_d=r_d, label=label, doc_id=doc_id))


    def random_init(self, seed_value: int):
        random.seed(seed_value)
        # Crawl list members
        members = [member._r_d for member in self._data]
        # random.choice: random sample from 1-D array
        # Same as np.arrange without same num in array
        pos = np.random.choice(len(self._data), self._num_clusters, replace=False)
        centroid = []
        for i in pos:
            centroid.append(members[i])
        # Update centroid
        self._E = centroid
        for i in range(self._num_clusters):
            self._clusters[i].set_centroid(centroid[i])

    def compute_similarity(self, member: Member, centroid: Member):
        # euclidean distance
        return np.sqrt(np.sum((member._r_d - centroid) ** 2))

    def select_cluster_for(self, member: Member):
        best_fit_cluster = None
        # cos = -1 <=> 180
        max_similarity = -1
        # check all clusters
        for cluster in self._clusters:
            # compute similarity for each member with centroid of each cluster
            similarity = self.compute_similarity(member, cluster._centroid)
            # peak max similarity locally
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        best_fit_cluster.add_members(member)
        return max_similarity

    def update_centroid_of(self, cluster: Cluster):
        # list all TF-IDF
        member_r_ds = [member._r_d for member in cluster._members]
        avg_r_d = np.mean(member_r_ds, axis=0)
        sqrt_sum_sqr = np.sqrt(np.sum(avg_r_d ** 2))
        new_centroid = avg_r_d / sqrt_sum_sqr

        # update centroid
        cluster._centroid = new_centroid

    def stopping_condition(self, criterion: str, threshold: float):
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria
        if criterion == 'max_iters':
            if self._iteration >= threshold:
                return True
            else:
                return False
        elif criterion == 'centroid':
            E_new = [list(cluster._centroid) for cluster in self._clusters]
            E_new_minus_E = [centroid for centroid in E_new if centroid not in self._E]
            self._E = E_new
            if len(E_new_minus_E) <= threshold:
                return True
            else:
                return False
        else:
            new_S_minus_S = self._new_S - self._S
            self._S = self._new_S
            if new_S_minus_S <= threshold:
                return True
            else:
                return False 

    def run(self, seed_value: int, criterion: str, threshold: float):
        self.random_init(seed_value)
        
        self._iteration = 0
        while True:
            # reset cluster (exclude centroids)
            for cluster in self._clusters:
                cluster.reset_members()
            self._new_S = 0
            for member in self._data:
                max_s = self.select_cluster_for(member)
                self._new_S += max_s
            for cluster in self._clusters:
                self.update_centroid_of(cluster)
            
            self._iteration += 1
            if self.stopping_condition(criterion, threshold):
                break

    def compute_purity(self):
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label) for label in range(20)])
            majority_sum += max_count
        return majority_sum * 1. / len(self._data)

    def compute_NMI(self):
        I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)
        for cluster in self._clusters:
            wk = len(cluster._members) * 1. 
            H_omega += - wk / N * np.log10(wk / N)
            member_labels = [member._label for member in cluster._members]
        for label in range(20):
            wk_cj = member_labels.count(label) * 1.
            cj = self._label_count[label]
            I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
        for label in range(20):
            cj = self._label_count[label] * 1.
            H_C += - cj / N * np.log10(cj / N)
        return I_value * 2. / (H_omega + H_C)