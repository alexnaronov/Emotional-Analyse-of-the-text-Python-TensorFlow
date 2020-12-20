from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, LSTM, Input


def build_text_cnn(maxlen,
                   max_features,
                   embedding_dims,
                   class_num=1,
                   last_activation='sigmoid'):
    inputs = Input(shape=(maxlen,))

    embedding = Embedding(max_features, embedding_dims, input_length=maxlen)
    rnn = LSTM(128)
    classifier = Dense(class_num, activation=last_activation)

    embedding = embedding(inputs)
    x = rnn(embedding)
    outputs = classifier(x)

    return Model(inputs=inputs, outputs=outputs)
