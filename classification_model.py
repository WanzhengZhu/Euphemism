import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LR(nn.Module):  # Logsitic Regression
    def __init__(self, unique_vocab_dict, unique_vocab_list, num_class):
        super().__init__()
        self.unique_vocab_dict = unique_vocab_dict
        self.unique_vocab_list = unique_vocab_list
        self.vocab_size = len(unique_vocab_dict)
        self.num_class = num_class
        self.fc = nn.Linear(self.vocab_size, num_class)

    def forward(self, text):
        out = self.fc(text)
        return out
        # return torch.softmax(out, dim=1)

    def get_features(self, class_names, num=50):
        print('-----------------------')
        features = []
        for i in range(self.num_class):
            feature = []
            sorted_weight_index = self.fc.weight[i].argsort().tolist()[::-1]
            for j in range(min(num, len(sorted_weight_index))):
                feature.append(self.unique_vocab_list[sorted_weight_index[j]])
            print(class_names[i], end=': ')
            print(feature)
            features.append(feature)
        print('-----------------------')
        return features


class LR_embeddings(nn.Module):  # Logsitic Regression with embeddings
    def __init__(self, unique_vocab_dict, embedding_length, num_class):
        super().__init__()
        self.vocab_size = len(unique_vocab_dict)
        self.embedding = nn.EmbeddingBag(self.vocab_size, embedding_dim=embedding_length, sparse=True)
        self.fc = nn.Linear(embedding_length, num_class)

    def forward(self, text):
        out = self.embedding(text)
        out = self.fc(out)
        return torch.softmax(out, dim=1)


# class BiLSTM(nn.Module):  # BiLSTM
#     def __init__(self, unique_vocab_dict, num_class):
#         super().__init__()
#         self.embedding_size = 30
#         self.hidden_size = 50
#         self.num_layers = 2
#         self.vocab_size = len(unique_vocab_dict)
#         self.embedding = nn.EmbeddingBag(self.vocab_size, embedding_dim=self.embedding_size, sparse=True)
#         self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(self.hidden_size * 2, num_class)  # 2 for bidirection
#
#     def forward(self, text):
#         # Set initial states
#         h0 = torch.zeros(self.num_layers * 2, text.size(0), self.hidden_size).to(device)  # 2 for bidirection
#         c0 = torch.zeros(self.num_layers * 2, text.size(0), self.hidden_size).to(device)
#
#         # Forward propagate LSTM
#         out = self.embedding(text)
#         out, _ = self.lstm(out, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
#
#         # Decode the hidden state of the last time step
#         out = self.fc(out[:, -1, :])
#         return torch.softmax(out, dim=1)


class CNN(nn.Module):
    def __init__(self, batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab,
                 vocab_size, embedding_length, weights, pre_train, embedding_tune):
        super(CNN, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        out_channels : Number of output channels after convolution operation performed on the input matrix
        kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
        keep_probab : Probability of retaining an activation node during dropout operation
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embedding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        --------

        """
        self.batch_size = batch_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        if pre_train:
            self.word_embeddings.weight = nn.Parameter(weights, requires_grad=embedding_tune)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
        self.dropout = nn.Dropout(keep_probab)
        self.label = nn.Linear(len(kernel_heights) * out_channels, output_size)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, input_sentences, batch_size=None):
        """
        The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix
        whose shape for each batch is (num_seq, embedding_length) with kernel of varying height but constant width which is same as the embedding_length.
        We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor
        and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
        to the output layers consisting two units which basically gives us the logits for both positive and negative classes.

        Parameters
        ----------
        input_sentences: input_sentences of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.
        logits.size() = (batch_size, output_size)

        """

        input = self.word_embeddings(input_sentences)
        # input.size() = (batch_size, num_seq, embedding_length)
        input = input.unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        logits = self.label(fc_in)
        return logits


class LSTM(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights, pre_train, embedding_tune):
        super(LSTM, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)  # Initializing the look-up table.
        if pre_train:
            self.word_embeddings.weight = nn.Parameter(weights, requires_grad=embedding_tune)  # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)

        """

        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
        input = self.word_embeddings(input_sentence)  # embedded input of shape = (batch_size, num_sequences, embedding_length)
        input = input.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device))  # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device))  # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1])  # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        return final_output


class LSTM_AttentionModel(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights, pre_train, embedding_tune):
        super(LSTM_AttentionModel, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        --------

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        if pre_train:
            self.word_embeddings.weights = nn.Parameter(weights, requires_grad=embedding_tune)
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output, final_state):

        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, input_sentences, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)

        """

        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device))
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device))
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))  # final_hidden_state.size() = (1, batch_size, hidden_size)
        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits


class RCNN(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights, pre_train, embedding_tune):
        super(RCNN, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embedding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)  # Initializing the look-up table.
        if pre_train:
            self.word_embeddings.weight = nn.Parameter(weights, requires_grad=embedding_tune)  # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.dropout = 0.8
        self.lstm = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True)
        self.W2 = nn.Linear(2 * hidden_size + embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)

        """

        """

        The idea of the paper "Recurrent Convolutional Neural Networks for Text Classification" is that we pass the embedding vector
        of the text sequences through a bidirectional LSTM and then for each sequence, our final embedding vector is the concatenation of 
        its own GloVe embedding and the left and right contextual embedding which in bidirectional LSTM is same as the corresponding hidden
        state. This final embedding is passed through a linear layer which maps this long concatenated encoding vector back to the hidden_size
        vector. After this step, we use a max pooling layer across all sequences of texts. This converts any varying length text into a fixed
        dimension tensor of size (batch_size, hidden_size) and finally we map this to the output layer.

        """
        input = self.word_embeddings(input_sentence)  # embedded input of shape = (batch_size, num_sequences, embedding_length)
        input = input.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).to(device))  # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).to(device))  # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).to(device))
            c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).to(device))

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))

        final_encoding = torch.cat((output, input), 2).permute(1, 0, 2)
        y = self.W2(final_encoding)  # y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0, 2, 1)  # y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y, y.size()[2])  # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)
        logits = self.label(y)

        return logits


class RNN(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights, pre_train, embedding_tune):
        super(RNN, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        if pre_train:
            self.word_embeddings.weight = nn.Parameter(weights, requires_grad=embedding_tune)
        self.rnn = nn.RNN(embedding_length, hidden_size, num_layers=2, bidirectional=True)
        self.label = nn.Linear(4 * hidden_size, output_size)

    def forward(self, input_sentences, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the final_hidden_state of RNN.
        logits.size() = (batch_size, output_size)

        """

        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(4, self.batch_size, self.hidden_size).to(device))  # 4 = num_layers*num_directions
        else:
            h_0 = Variable(torch.zeros(4, batch_size, self.hidden_size).to(device))
        output, h_n = self.rnn(input, h_0)
        # h_n.size() = (4, batch_size, hidden_size)
        h_n = h_n.permute(1, 0, 2)  # h_n.size() = (batch_size, 4, hidden_size)
        h_n = h_n.contiguous().view(h_n.size()[0], h_n.size()[1] * h_n.size()[2])
        # h_n.size() = (batch_size, 4*hidden_size)
        logits = self.label(h_n)  # logits.size() = (batch_size, output_size)

        return logits


class SelfAttention(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights, pre_train, embedding_tune):
        super(SelfAttention, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        --------

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.weights = weights

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        if pre_train:
            self.word_embeddings.weights = nn.Parameter(weights, requires_grad=embedding_tune)
        self.dropout = 0.8
        self.bilstm = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True)
        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
        self.W_s1 = nn.Linear(2 * hidden_size, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fc_layer = nn.Linear(30 * 2 * hidden_size, 2000)
        self.label = nn.Linear(2000, output_size)

    def attention_net(self, lstm_output):

        """
        Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
        encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of
        the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully
        connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e.,
        pos & neg.

        Arguments
        ---------

        lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
        ---------

        Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
                  attention to different parts of the input sentence.

        Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
                      attn_weight_matrix.size() = (batch_size, 30, num_seq)

        """
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, input_sentences, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.

        """

        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).to(device))
            c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).to(device))
        else:
            h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).to(device))
            c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).to(device))

        output, (h_n, c_n) = self.bilstm(input, (h_0, c_0))
        output = output.permute(1, 0, 2)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        # h_n.size() = (1, batch_size, hidden_size)
        # c_n.size() = (1, batch_size, hidden_size)
        attn_weight_matrix = self.attention_net(output)
        # attn_weight_matrix.size() = (batch_size, r, num_seq)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        # hidden_matrix.size() = (batch_size, r, 2*hidden_size)
        # Let's now concatenate the hidden_matrix and connect it to the fully connected layer.
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        logits = self.label(fc_out)
        # logits.size() = (batch_size, output_size)

        return logits

