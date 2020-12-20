from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Input


def build_fast_text(maxlen,
                    max_features,
                    embedding_dims,
                    class_num=1,
                    last_activation='sigmoid'):
    inputs = Input(shape=(maxlen,))

    embedding = Embedding(max_features, embedding_dims, input_length=maxlen)
    avg_pooling = GlobalAveragePooling1D()
    classifier = Dense(class_num, activation=last_activation)

    embedding = embedding(inputs)
    x = avg_pooling(embedding)
    outputs = classifier(x)

    return Model(inputs=inputs, outputs=outputs)
