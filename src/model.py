"""
Authors:    Jingjing WU (吴京京) <https://github.com/wj-Mcat>


2022-now @ Copyright wj-Mcat

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import annotations
from tokenize import Intnumber
from typing import Optional, Union
import math
from django.forms import HiddenInput

from paddle import dtype, nn
import paddle.nn.functional as F
import paddle
from paddlenlp.transformers.auto.modeling import AutoModel
from src.config import Tensor



def conv_layer(in_channels: int, out_channels: int, keep_prob: float=0.0):
    """convLayer kernel for CNN.

    Args:
        in_channels (int): input channels.
        out_channels (int): output channels.
        keep_prob (float, optional): dropout probility. Defaults to 0.0.
    """
    cnn_seq = nn.Sequential(
        nn.Conv2D(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2D(out_channels),
        nn.MaxPool2D(kernel_size=2, stride=2),
        nn.Dropout(keep_prob)
    )
    return cnn_seq


class ImageEncoder(nn.Layer):
    def __init__(
        self,
        layer_size: int=64,
        num_channels: int=1,
        keep_prob: float=1.0,
        image_size: int=28
    ):
        """encoder for image which produce the embedding vector.

        Args:
            layer_size (int, optional): the size of conv layer. Defaults to 64.
            num_channels (int, optional): the num channels of layer. Defaults to 1.
            keep_prob (float, optional): the dropout probility of model. Defaults to 1.0.
            image_size (int, optional): the image size configuration. Defaults to 28.
        """
        super().__init__()
        self.layer1 = conv_layer(num_channels, layer_size, keep_prob)
        self.layer2 = conv_layer(layer_size, layer_size, keep_prob)
        self.layer3 = conv_layer(layer_size, layer_size, keep_prob)
        self.layer4 = conv_layer(layer_size, layer_size, keep_prob)

        final_size = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = final_size * final_size * layer_size

    def forward(self, image_input):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = paddle.reshape(x, shape=[x.shape[0], -1])
        return x


class DistanceNetwork(nn.Layer):
    """
    This model calculates the cosine distance between each of the support set embeddings and the target image embeddings.
    """

    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set: Tensor, query_set: Tensor):
        """
        forward implement
        :param support_set:the embeddings of the support set images.shape[sequence_length,batch_size,64]
        :param input_image: the embedding of the target image,shape[batch_size,64]
        :return:shape[batch_size,sequence_length]
        """
        similarity_fn = nn.CosineSimilarity()
        similarities = []
        for query_set_embedding in query_set:
            similarity = similarity_fn(paddle.unsqueeze(query_set_embedding, 0), support_set)
            similarities.append(similarity)
        similarities = paddle.stack(similarities)
        return similarities


class BidirectionalLSTM(nn.Layer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
    ):
        super(BidirectionalLSTM, self).__init__()
        """
        Initial a muti-layer Bidirectional LSTM
        :param layer_size: a list of each layer'size
        :param batch_size: 
        :param vector_dim: 
        """
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction='bidirectional'
        )
        # self.hidden = self.init_hidden()

    def init_hidden(self):
        prev_h = paddle.zeros(shape=[self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size])
        prev_c = paddle.zeros(shape=[self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size])
        return prev_h, prev_c

    def forward(self, inputs):
        output, self.hidden = self.lstm(inputs)
        return output


class MatchingNetwork(nn.Layer):
    def __init__(
        self,
        keep_prob: float,
        batch_size: int=32,
        num_channels: int=1,
        learning_rate: float=1e-3,
        fce: bool=False,
        n_way: int=20,
        k_shot=1,
        image_size: int=28,
        use_cuda: bool=True
    ):
        """
        This is our main network
        :param keep_prob: dropout rate
        :param batch_size:
        :param num_channels:
        :param learning_rate:
        :param fce: Flag indicating whether to use full context embeddings(i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set:
        :param num_samples_per_class:
        :param image_size:
        """
        super(MatchingNetwork, self).__init__()
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.fce = fce
        self.n_way = n_way
        self.k_shot = k_shot
        self.image_size = image_size
        self.encoder = ImageEncoder(layer_size=64, num_channels=num_channels, keep_prob=keep_prob, image_size=image_size)
        self.distance_fn = DistanceNetwork()
        if self.fce:
            self.lstm = BidirectionalLSTM(
                input_size=self.encoder.outSize,
                hidden_size=32,
                num_layers=1,
            )

    def forward(self, support_set: Tensor, support_set_labels: Tensor, query_set: Tensor, query_set_labels: Optional[Tensor] = None):
        """handl the matching network forward

        Args:
            support_set (Tensor): the source of the support set
            support_set_labels (Tensor): the labels of the support set
            query_set (Tensor): the source of the query set
            query_set_labels (Optional[Tensor], optional): the labels of the query set. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: the tuple of the [acc, cross_entropy loss]
        """
        # 1. compute the embedding of support set and query set
        support_set_embedding, query_set_embedding = self.encoder(support_set), self.encoder(query_set)
        
        # 2. get similarities between support set embeddings and target
        similarites = self.distance_fn(support_set_embedding, query_set_embedding)

        # 4. calculate the accuracy
        _, indices = similarites.max(1)
        accuracy = paddle.mean((indices.squeeze() == query_set_labels).float())
        crossentropy_loss = F.cross_entropy(similarites, query_set_labels.long())

        return accuracy, crossentropy_loss

# class TextEncoder(nn.Layer):
#     @abstractmethod
#     def forward(
#         self,
#         input_ids: Tensor,
#         attention_mask: Optional[Tensor] = None,
#         token_type_ids: Optional[Tensor] = None,
#     ):
#         raise NotImplementedError


# class WordEmbeddingTextEncoder(TextEncoder):
#     def __init__(
#         self,
#         embedding_or_file: Union[str, Tensor] = None,
#         vocab_size: Optional[int] = None,
#         embedding_size: Optional[int] = None,
#     ):
#         super().__init__()
#         if embedding_or_file:
#             if isinstance(embedding_or_file, str):
#                 if not os.path.exists(embedding_or_file):
#                     raise FileNotFoundError(
#                         f'embedding file {embedding_or_file} not found'
#                     )
#                 # TODO: to be tested under different embedding size
#                 embedding_or_file = Tensor.load(embedding_or_file)
#             elif not paddle.is_tensor(embedding_or_file):
#                 raise TypeError('embedding_or_file must be a Tensor or a str')
#         else:
#             if not vocab_size or not embedding_size:
#                 raise ValueError('when embedding_or_file is None, vocab_size and embedding_size should not be None ...')
#             embedding_or_file = nn.Embedding(
#                 shape=[vocab_size, embedding_size],
#                 default_initializer=
#             )   
                
#         self.embedding = nn.Embedding(
#             size=[vocab_size, embedding_size],
#             param_attr=nn.initializer.Xavier(uniform=False),
#             name='embedding'
#         )

#     def forward(
#         self,
#         input_ids: Tensor,
#         attention_mask: Optional[Tensor] = None,
#         token_type_ids: Optional[Tensor] = None,
#     ):
#         embedding_output = self.embedding(input_ids)
#         return embedding_output


# class PretrainedModelEncoder(TextEncoder):
#     def __init__(self, model_or_name, name_scope=None, dtype="float32"):
#         super().__init__(name_scope, dtype)

#         self.model = AutoModel.from_pretrained(model_or_name)
    
#     def forward(self, **inputs):
#         return self.model(**inputs)

# class MatchingNetworkSimple(nn.Layer):
#     def __init__(
#         self, encoder: TextEncoder,
#         n_way: int, k_shot: int,
#         name_scope=None, dtype="float32"
#     ):
#         super().__init__(name_scope, dtype)
#         self.encoder = encoder
#         self.n_way = n_way
#         self.k_shot = k_shot

#     def forward(self, inputs: Tensor):
#         # 1. compute the embedding of the input
#         embedding = self.encoder(**inputs)

#         # 2. split the support set & query set
#         support_set = embedding[:self.k_shot * self.n_way, :]   # (n_way * k_shot, embedding_size)
#         query_set = embedding[self.k_shot * self.n_way:, :]     # (n_way * n_query, embedding_size)

#         # 3. compute the similarity matrix
#         # TODO: 
#         logit = paddle
        

