import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, Input

def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # Lenght of series
    N = series.shape[0]
    
    # Generate inputs and outputs
    for i in range(N-window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    return X,y

def example_to_io(example, input_window_size, input_step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    # This is the number of iterations taking into acount the step_size and the window_size
    N = int((len(example) - input_window_size) / input_step_size)

    # Get inputs and outputs
    for k in range(N):
        i = k * input_step_size
        inputs.append(example[i:i + input_window_size])
        outputs.append(example[i + input_window_size])

    return inputs, outputs


def example_to_io_pairs(example, possible_elements, input_window_size, input_step_size):
    num_chars = len(possible_elements)
    chars_to_indices = dict((c, i) for i, c in enumerate(possible_elements))

    # cut up text into character input/output pairs
    inputs, outputs = example_to_io(example, input_window_size, input_step_size)

    # create empty vessels for one-hot encoded input/output
    X = np.zeros((len(inputs), input_window_size, num_chars), dtype=np.int)
    y = np.zeros((len(inputs), num_chars), dtype=np.int)

    # loop over inputs/outputs and tranform and store in X/y
    for i, sentence in enumerate(inputs):
        for t, char in enumerate(sentence):
            if char not in chars_to_indices:
                char = ' '
            X[i, t, chars_to_indices[char]] = 1
        out_char = outputs[i]
        if out_char not in chars_to_indices:
            out_char = ' '
        y[i, chars_to_indices[out_char]] = 1

    return X, y
def get_deep_rnn(input_shape, dense_units = 80, LSTM_units_1=200, LSTM_units_2=200, dropout_p=0.2, stateful=False, verbose=True):
    # Model definition
    if verbose:
        print("input shape = ",input_shape)
    model = Sequential()
    if (stateful):
        model.add(LSTM(LSTM_units_1, 
                       batch_input_shape=(1,input_shape[0],input_shape[1]), 
                       return_sequences=True, 
                       name='lstm_1',
                       dropout=dropout_p, 
                       recurrent_dropout=dropout_p, 
                       stateful=stateful))
    else:    
        model.add(LSTM(LSTM_units_1, input_shape=input_shape, return_sequences=True, name='lstm_1',
                       dropout=dropout_p, recurrent_dropout=dropout_p, stateful=stateful))
    model.add(LSTM(LSTM_units_2, dropout=dropout_p, recurrent_dropout=dropout_p, name='lstm_2', stateful=stateful))
    model.add(Dense(dense_units, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    if verbose:
        model.summary()
    return model

def sample(a, temperature=1.0, verbose = False, return_dist=False):
    a = np.array(a)
    a = a/a.sum()
    a = a**(1/temperature)
    p_sum = a.sum()
    sample_temp = a/p_sum 
    if verbose:
        print(sample_temp)
    #sample_temp = sample_temp*(sample_temp>1e-4)
    choices = range(len(a))
    if return_dist:
        return np.random.choice(choices, p=sample_temp), sample_temp
    else:
        return np.random.choice(choices, p=sample_temp)
    #return np.argmax(np.random.multinomial(1, sample_temp, 1))

def to_one_hot_vector(example, possible_elements, element_to_indices, window_size):
    sequence_len = get_sequence_len(example, window_size)

    one_hot_vector = create_one_hot_vector(
        sequence_len=sequence_len,
        possible_elements_count=len(possible_elements)
    )

    for index, element in enumerate(example):
        if element in element_to_indices:
            one_hot_vector[0, index, element_to_indices[element]] = 1

    return one_hot_vector


def create_one_hot_vector(sequence_len, possible_elements_count):
    return np.zeros((1, sequence_len, possible_elements_count), dtype=np.int)


def get_sequence_len(example, window_size): return max(len(example), window_size)


def show_results(
    outputs,
    chars_to_indices,
    indices_to_chars,
    input_sequence_resolver
):
    print('\n --------------------------------------------------------------')
    print('| Sequence len (T) => ',  len(input_sequence_resolver(0)))
    print('| Output Shape: ', outputs.shape)
    print(' --------------------------------------------------------------')

    for char in chars_to_indices.keys(): print('   _' if ' ' == char else char, end='    ')
    print()
    for index, row in enumerate(outputs):
        input_sequence = input_sequence_resolver(index)
        classes_probabilities = (row * 100).astype(int) / 100
        predicted_char = indices_to_chars[np.argmax(row)]
        print(input_sequence, classes_probabilities, predicted_char)
