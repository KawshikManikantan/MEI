import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()


class F1Evaluator(object):
    def __init__(self):
        self.f1_macro_sum = 0.0
        self.f1_micro_sum = 0.0
        self.macro_support = 0
        self.micro_support = 0

    def update(self, predicted, gold):
        if gold:
            for cluster_ind, cluster in enumerate(gold):
                predicted_set = set(predicted[cluster_ind])
                correct = set(cluster).intersection(set(predicted_set))
                num_correct = len(correct)
                num_predicted = len(predicted_set)
                num_gt = len(cluster)
                precision = num_correct / num_predicted if num_predicted > 0 else 0
                recall = num_correct / num_gt if num_gt > 0 else 0
                f1_score = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall > 0
                    else 0
                )
                support_entity_micro = num_gt
                support_entity_macro = 1
                self.f1_macro_sum += f1_score * support_entity_macro
                self.f1_micro_sum += f1_score * support_entity_micro
                self.macro_support += support_entity_macro
                self.micro_support += support_entity_micro

    def get_numbers(self):
        f1_macro = (
            (self.f1_macro_sum / self.macro_support) * 100
            if self.macro_support > 0
            else 0
        )
        f1_micro = (
            (self.f1_micro_sum / self.micro_support) * 100
            if self.micro_support > 0
            else 0
        )
        return f1_macro, f1_micro


class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den

    def get_prf_str(self):
        perf_str = (
            f"Recall: {self.get_recall() * 100}, Precision: {self.get_precision() * 100}, "
            f"F-score: {self.get_f1() * 100}\n"
        )

        return perf_str


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_sum_assignment(-scores)
    matching = np.asarray(matching)
    matching = np.transpose(matching)

    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1 :]:
                    if (
                        m2 in mention_to_gold
                        and mention_to_gold[m] == mention_to_gold[m2]
                    ):
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem
