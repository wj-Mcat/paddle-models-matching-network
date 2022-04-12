import paddle
from paddle.nn import CosineSimilarity
import paddle.nn.functional as F
import pytest
from src import model


def test_cosine_similarity():
    x = paddle.randn([2, 3])
    y = paddle.randn([3, 3])

    similarity_fn = CosineSimilarity()
    similarities = []
    for vector in x:
        similarity = similarity_fn(paddle.unsqueeze(vector, 0), y)
        similarities.append(similarity)
    similarities = paddle.stack(similarities)
    assert similarities.shape == [2, 3]


def test_conv_layer():
    layer = model.conv_layer(10, 10, 0.4)
    assert issubclass(type(layer), paddle.nn.Layer)

    inputs = paddle.randn(shape=(100, 10, 100, 100))
    output = layer(inputs)
    assert paddle.is_tensor(output)


def test_image_encoder():
    encoder = model.ImageEncoder(10, 10, 0.4)
    assert issubclass(type(encoder), paddle.nn.Layer)

    inputs = paddle.randn(shape=(100, 10, 100, 100))
    output = encoder(inputs)
    assert paddle.is_tensor(output)


def test_distance_fn():
    distance_fn = model.DistanceNetwork()
    
    x = paddle.randn([2, 3])
    y = paddle.randn([3, 3])
    
    distance = distance_fn(x, y)
    assert distance.shape == [3, 2]


def test_bilstm():
    lstm = model.BidirectionalLSTM(14, 26, 1)

    inputs = paddle.randn(shape=[16, 128, 14])
    output = lstm(inputs)
    assert output.shape == [16, 128, 26 * 2]


def test_matching_network():
    batch_size = 32
    n_way, k_shot, q_query = 5, 1, 101

    matching_network = model.MatchingNetwork(
        keep_prob=0.1,
        batch_size=batch_size,
        num_channels=1,
    )
    assert issubclass(type(matching_network), paddle.nn.Layer)

    support_set = paddle.randn(shape=[n_way * k_shot, 1, 100, 100])
    query_set = paddle.randn(shape=[q_query, 1, 100, 100])
    output = matching_network(support_set, query_set) 