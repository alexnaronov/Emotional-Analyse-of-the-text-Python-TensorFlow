from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Input


def build_text_cnn(maxlen,
                   max_features,
                   embedding_dims,
                   kernel_sizes=[3, 4, 5],
                   class_num=1,
                   last_activation='sigmoid'):
    inputs = Input(shape=(maxlen,))

    embedding = Embedding(max_features, embedding_dims, input_length=maxlen)
    convs = []
    max_poolings = []
    for kernel_size in kernel_sizes:
        convs.append(Conv1D(128, kernel_size, activation='relu'))
        max_poolings.append(GlobalMaxPooling1D())
    classifier = Dense(class_num, activation=last_activation)

    embedding = embedding(inputs)
    conv_outputs = []
    for i in range(len(kernel_sizes)):
        c = convs[i](embedding)
        c = max_poolings[i](c)
        conv_outputs.append(c)
    x = Concatenate()(conv_outputs)
    outputs = classifier(x)

    return Model(inputs=inputs, outputs=outputs)
