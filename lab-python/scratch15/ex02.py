import math
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, Counter, defaultdict
from typing import NamedTuple, Any


# Candidate = namedtuple('Candidate',
#                        ('level', 'lang', 'tweets', 'phd', 'result'))
class Candidate(NamedTuple):  # NamedTuple을 상속받는 클래스 선언
    level: str
    lang: str
    tweets: bool
    phd: bool
    result: bool = None  # 클래스 선언 방식의 NamedTuple은 field의 기본값을 설정할 수 있음.


def uncertainty(p):
    """ 0 <= p <= 1
    확률 p=0이면, 사건이 "항상" 발생하지 않는다 -> 불확실성 0
    확률 p=1이면, 사건이 "항상" 발생한다. -> 불확실성 0
    확률 0<p<1이면, 사건이 발생할 수도, 발생하지 않을 수도 있다. -> 불확실성이 있다.
    """
    return -p * math.log(p, 2)  # 2를 밑수로 하는 로그 함수


def entropy(class_probabilities):
    """ 엔트로피: 각 클래스의 불확실성의 정도를 모두 더한 값.
    주어진 확률들의 리스트에 대해서 엔트로피를 계산
    E = sum(i) [uncertainty(p_i)] = -p_1 * log(p_1) - p_2 * log(p_2) - ...
    """
    ent = 0
    for p in class_probabilities:
        if p != 0:
            ent += uncertainty(p)
            # 만약 p=0이면 log(p)를 계산할 때 Error가 발생하기 때문에
    return ent


def binary_entropy(p):
    """사건이 일어날 확률 p, 사건이 일어나지 않을 확률 (1-p)
    Entropy = -p * log(p) - (1-p) * log(1-p)
    """
    return uncertainty(p) + uncertainty(1-p)


def class_probabilities(labels):
    total_count = len(labels)
    counts = Counter(labels)  # {label_1: cnt_1, label_2: cnt_2, ...}
    print(counts)
    # probabilities = []
    # for count in counts.values():
    #     p = count / total_count  # 각 레이블의 확률
    #     probabilities.append(p)
    probabilities = [count/total_count for count in counts.values()]
    return probabilities


def partition_by(dataset, attr_name):
    """NamedTuple들의 리스트로 이루어진 dataset를
    NamedTuple의 특정 attribute로 partitioning"""
    partitions = defaultdict(list)  # list를 value로 갖는 dict
    for sample in dataset:
        # 해당 attr_name의 값을 찾아서
        key = getattr(sample, attr_name)
        # dict의 키로 사용해서 sample을 저장.
        partitions[key].append(sample)
    return partitions


def partition_entropy_by(dataset, by_partition, by_entropy):
    """by_partition(attr_name)으로 분리된 각 파티션에서
    by_entropy(label_name)의 엔트로피를 각각 계산하고,
    파티션 내에서의 엔트로피 * 파티션의 비율 들의 합을 리턴.
    """
    # 파티션을 나눔
    partitions = partition_by(dataset, by_partition)

    # 클래스(레이블) 별 확률을 계산하기 위해서 레이블들의 리스트를 생성
    labels = []
    for partition in partitions.values():  # 파티션 개수만큼 반복
        values = []
        for sample in partition:  # 파티션의 원소 개수만큼 반복
            values.append(getattr(sample, by_entropy))
        labels.append(values)
    print(labels)
    # 각 파티션이 차지하는 비율을 계산하고,
    # 각 파티션에서의 엔트로피에 그 비율을 곱해주기 위해서
    total_count = sum(len(label) for label in labels)
    ent = 0
    for label in labels:
        # 파티션이 가지고 있는 클래스들의 확률 리스트
        cls_prob = class_probabilities(label)  # [2/5, 3/5]
        part_ent = entropy(cls_prob)  # 파티션의 엔트로피
        # 파티션 엔트로피 * 파티션의 비율
        ent += part_ent * len(label) / total_count
    return ent


class Leaf(NamedTuple):  # Leaf는 NamedTuple을 상속받는 클래스
    value: Any


class Split(NamedTuple):
    attribute: str  # 트리에서 가지(branch)가 나눠지는 기준
    subtree: dict


def predict(model, sample):
    """sample을 model(의사 결정 나무)에 적용했을 때 예측 결과를 리턴"""
    if isinstance(model, Leaf):
        # model이 최종 노드인 Leaf 타입이면, Leaf가 가지고 있는 value(값)을 리턴.
        return model.value

    # model이 아닌 경우에는, 가지를 따라 내려가야하기 때문에
    # 샘플이 attribute로 가지고 있는 값을 찾아서, 해당 가지로 내려감.
    subtree_key = getattr(sample, model.attribute)
    print('subtree_key:', subtree_key)

    subtree = model.subtree[subtree_key]
    return predict(subtree, sample)


def build_tree(dataset, by_splits, target):
    print('\n>>> Building tree ...')
    print(f'dataset({len(dataset)})= {dataset}')
    print(f'by_splits={by_splits}, target={target}')

    # target의 개수를 셈 - Counter 객체 생성 {True: x, False: y}
    target_counts = Counter(getattr(sample, target) for sample in dataset)
    print('target_counts:', target_counts)
    # Counter의 length가 1이면
    # dataset의 모든 sample들이 하나의 타겟 값을 갖는다는 의미이므로,
    # Leaf를 생성하고 종료
    if len(target_counts) == 1:
        keys = list(target_counts.keys())
        # [k for k in target_counts.keys()]
        # dict의 keys()가 리턴하는 타입은 리스트가 아니어서 [] 연산자를 사용할 수 없다!
        result = keys[0]  # True 또는 False
        leaf = Leaf(result)
        print('leaf:', leaf)
        return leaf

    # 트리의 depth가 깊어져서 더 이상 서브 트리를 나눌 기준이 없을 때
    if not by_splits:  # if len(by_splits) == 0:
        return Leaf(list(target_counts.keys())[0])

    # Counter의 length가 1이 아니면, 파티션을 나눌 수 있음.
    # by_splits(가지 나누는 기준)의 각 변수로 파티션을 나눔.
    # 각 파티션별 엔트로피를 계산해서 가장 낮은 엔트로피 변수를 선택

    # partition_entropy_by(dataset, by_split, by_entropy)를 호출할 수 있는
    # wrapper 함수(helper 함수)를 작성
    def splitted_entropy(split_attr):
        print('split_attr:', split_attr)
        result = partition_entropy_by(dataset, split_attr, target)
        print('splitted entropy =', result)
        return result

    best_splitter = min(by_splits, key=splitted_entropy)
    print('best_splitter:', best_splitter)

    # 선택된 변수(엔트로피 최솟값을 주는 변수)로 파티션을 만듦.
    partitions = partition_by(dataset, best_splitter)
    print('partitions:', partitions)

    # 선택된 변수를 제외한 나머지 변수들 sub-tree를 만듦.
    # branch 기준 리스트에서 선택된 변수 제거.
    new_split = [x for x in by_splits if x != best_splitter]
    print(f'제거 후 by_splits: {by_splits}, new_split: {new_split}')
    # 서브 트리를 함수 재귀 호출을 통해서 생성
    subtree = {k: build_tree(subset, new_split, target)
               for k, subset in partitions.items()}

    # Split 객체를 생성해서 리턴
    return Split(best_splitter, subtree)


if __name__ == '__main__':
    candidates = [Candidate('Senior', 'Java', False, False, False),
                  Candidate('Senior', 'Java', False, True, False),
                  Candidate('Mid', 'Python', False, False, True),
                  Candidate('Junior', 'Python', False, False, True),
                  Candidate('Junior', 'R', True, False, True),
                  Candidate('Junior', 'R', True, True, False),
                  Candidate('Mid', 'R', True, True, True),
                  Candidate('Senior', 'Python', False, False, False),
                  Candidate('Senior', 'R', True, False, True),
                  Candidate('Junior', 'Python', True, False, True),
                  Candidate('Senior', 'Python', True, True, True),
                  Candidate('Mid', 'Python', False, True, True),
                  Candidate('Mid', 'Java', True, False, True),
                  Candidate('Junior', 'Python', False, True, False)]

    # uncertainty 함수 그래프
    x_pts = np.linspace(0.0001, 1, 100)
    y_pts = [uncertainty(p) for p in x_pts]
    plt.plot(x_pts, y_pts)
    plt.title('y = -p * log(p)')
    plt.xlim(0.0)  # 0 <= x
    plt.ylim(0.0)  # 0 <= y
    plt.show()

    # binary_entropy 함수 그래프
    x_pts = np.linspace(0.0001, 0.9999, 100)
    y_pts = [binary_entropy(p) for p in x_pts]
    plt.plot(x_pts, y_pts)
    plt.title('binary entropy')
    plt.axvline(x=0.5, color='0.75')
    plt.xlim(0)
    plt.ylim(0)
    plt.show()

    # entropy 함수 테스트
    rain_prob = [1, 0]  # 비가 올 확률 100%
    ent = entropy(rain_prob)
    print('entropy =', ent)  # 엔트로피 = 0 (최소 엔트로피/불확실성)

    rain_prob = [0.5, 0.5]  # 비가 올 확률 50%
    print('entropy =', entropy(rain_prob))  # 엔트로피 = 1.0 (최대 엔트로피/불확실성)

    rain_prob = [0.9, 0.1]  # 비가 올 확률 90%
    print('entropy =', entropy(rain_prob))  # 엔트로피 = 0.47

    # class_probabilities 함수 테스트
    level = ['junior', 'senior', 'mid', 'junior']
    # P(junior) = 2/4, P(senior) = 1/4, P(mid) = 1/4
    cls_prob = class_probabilities(level)
    print(cls_prob)

    # partition_by 함수 테스트
    partition_by_level = partition_by(candidates, 'level')
    print('partition_by_level:', partition_by_level)
    partition_by_tweet = partition_by(candidates, 'tweets')
    print('partition_by_tweets:', partition_by_tweet)

    # partition_entropy_by 함수 테스트
    # 전체 지원자들을 level로 파티션을 나눠서 result의 엔트로피를 계산
    ent_level = partition_entropy_by(candidates, 'level', 'result')
    print('entropy partitioned by level:', ent_level)
    ent_lang = partition_entropy_by(candidates, 'lang', 'result')
    print('entropy partitioned by lang', ent_lang)
    ent_tweets = partition_entropy_by(candidates, 'tweets', 'result')
    print('entropy partitioned by tweets', ent_tweets)
    ent_phd = partition_entropy_by(candidates, 'phd', 'result')
    print('entropy partitioned by phd', ent_phd)

    hire_tree = Split(
        'level',  # attribute: branch 나누는 기준
        # sub-tree
        {
            'Senior': Split(
                'tweets',
                {True: Leaf(True), False: Leaf(False)}),
            'Mid': Leaf(True),  # leaf(합격)
            'Junior': Split(
                'phd',
                {True: Leaf(False), False: Leaf(True)})
        }
    )
    print(hire_tree)

    candidate_1 = Candidate('Senior', 'Java', False, False, False)
    result = predict(hire_tree, candidate_1)
    print('합격 여부=', result)

    candidate_2 = Candidate('Mid', 'Python', False, False, True)
    result = predict(hire_tree, candidate_2)
    print('합격 여부=', result)

    # build_tree 함수 테스트
    tree = build_tree(candidates, ['level', 'lang', 'tweets', 'phd'], 'result')
    print(tree)



