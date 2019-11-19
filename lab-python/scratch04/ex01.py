"""
선형 대수(Linear Algebra)
"""


def add(v, w):
    """
    주어진 두 개의 n차원 벡터에서 성분별로 더하기를 해서,
    새로운 n차원 벡터를 리턴s

    :param v: n차원 벡터(성분이 n개인 벡터)
    :param w: n차원 벡터(성분이 n개인 벡터)
    :return: 각 성분별로 더하기 결과를 갖는 벡터
    """
    # result = []
    # for i in range(len(v)):
    #     result.append(v[i] + w[i])
    # return result
    if len(v) != len(w):
        raise ValueError('v와 w는 같은 length를 가져야 함.')
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def subtract(v, w):
    """
    주어진 두 개의 n차원 벡터에서 성분별로 뺄셈을 수행

    :param v: n차원 벡터
    :param w: n차원 벡터
    :return: n차원 벡터
    """
    if len(v) != len(w):
        raise ValueError('v와 w는 같은 length를 가져야 함.')
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors):
    """
    모든 벡터들에서 각 성분별 더하기를 수행
    vector_sum([[1, 2], [3, 4], [5, 6]]) = [9, 12]
    
    :param vectors: n차원 벡터들의 리스트(2차원 리스트)
    :return: n차원 벡터
    """
    num_of_elements = len(vectors[0])
    for vector in vectors[1:]:
        if num_of_elements != len(vector):
            raise ValueError('모든 벡터는 길이가 같아야 함.')

    # result = [0 for _ in range(num_of_elements)]  # [0, 0, 0 , ...]
    # for i in range(num_of_elements):
    #     for vector in vectors:
    #         result[i] += vector[i]
    # return result
    result = vectors[0]
    for vector in vectors[1:]:
        result = add(result, vector)
    return result


if __name__ == '__main__':
    v = [1, 2]
    w = [3, 4]
    result = add(v, w)
    print('add =', result)
    result = subtract(v, w)
    print('subtract =', result)

    v = [1, 2, 3]
    w = [4, 5, 6]
    result = add(v, w)
    print('add =', result)
    result = subtract(v, w)
    print('subtract =', result)

    vectors = [[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]]
    result = vector_sum(vectors)
    print(result)
