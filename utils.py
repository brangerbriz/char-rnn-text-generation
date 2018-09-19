import os
import random
import string
import numpy as np

###
# data processing
###

def create_dictionary():
    """
    create char2id, id2char and vocab_size
    from printable ascii characters.
    """
    chars = sorted(ch for ch in string.printable if ch not in ("\x0b", "\x0c", "\r"))
    char2id = dict((ch, i + 1) for i, ch in enumerate(chars))
    char2id.update({"": 0})
    id2char = dict((char2id[ch], ch) for ch in char2id)
    vocab_size = len(char2id)
    return char2id, id2char, vocab_size

CHAR2ID, ID2CHAR, VOCAB_SIZE = create_dictionary()


def encode_text(text, char2id=CHAR2ID):
    """
    encode text to array of integers with CHAR2ID
    """
    return np.fromiter((char2id.get(ch, 0) for ch in text), int)


def decode_text(int_array, id2char=ID2CHAR):
    """
    decode array of integers to text with ID2CHAR
    """
    return "".join((id2char[ch] for ch in int_array))


def one_hot_encode(indices, num_classes):
    """
    one-hot encoding
    """
    return np.eye(num_classes)[indices]

# NOTE: there is no rolling in this generator, so RNN states must be
# reset after each Epoch
def io_batch_generator(text_path, max_bytes_in_ram=1000000, batch_size=64, seq_len=64, one_hot_features=False, one_hot_labels=False):
    """
    batch generator for sequence
    ensures that batches generated are continuous along axis 1
    so that hidden states can be kept across batches and epochs
    """

    total_bytes = os.path.getsize(text_path)
    effective_file_end = total_bytes - total_bytes % max_bytes_in_ram
    
    print('total_bytes: {}'.format(total_bytes))
    print('max_bytes_in_ram: {}'.format(max_bytes_in_ram))
    print('effective_file_end: {}'.format(effective_file_end))

    with open(text_path, 'r') as file:
        epoch = 0
        while True:

            # once we are back at the beginning of the file we have 
            # entered a new epoch. Epoch is also initialized to zero so
            # that it will be set to one here at the beginning.
            if file.tell() == 0: 
                epoch += 1
                print('debug: now in epoch {}'.format(epoch))

            # load max_bytes_in_ram into ram
            io_batch = file.read(max_bytes_in_ram)
            print('debug: new io_batch of {} bytes'.format(max_bytes_in_ram))
            
            # if we are within max_bytes_in_ram of the effecitive
            # end of the file, set the file read playhead back to
            # the beginning, which will increase the epoch next loop
            if file.tell() + max_bytes_in_ram > effective_file_end:
                file.seek(0)

            # print('debug: encoding {} bytes of text from io_batch'.format(len(io_batch)))
            # encode this batch of text
            encoded = encode_text(io_batch)

            # the number of data batches for this io batch of bytes in RAM
            num_batches = (len(encoded) - 1) // (batch_size * seq_len)
            
            if num_batches == 0:
                raise ValueError("No batches created. Use smaller batch_size or seq_len or larger value for max_bytes_in_ram.")
            
            # print("debug: number of batches in io_batch: {}".format(num_batches))
            
            rounded_len = num_batches * batch_size * seq_len
            # print("debug: effective text length in io_batch: {}".format(rounded_len))

            x = np.reshape(encoded[: rounded_len], [batch_size, num_batches * seq_len])
            if one_hot_features:
                x = one_hot_encode(x, VOCAB_SIZE)
            # print("debug: x shape: {}".format(x.shape))

            y = np.reshape(encoded[1: rounded_len + 1], [batch_size, num_batches * seq_len])
            if one_hot_labels:
                y = one_hot_encode(y, VOCAB_SIZE)
            # print("debug: y shape: {}".format(y.shape))

            x_batches = np.split(x, num_batches, axis=1)
            y_batches = np.split(y, num_batches, axis=1)

            for batch in range(num_batches):
                yield x_batches[batch], y_batches[batch], epoch
            
            # free the mem
            del x
            del y
            del x_batches
            del y_batches
            del encoded

###
# text generation
###

def generate_seed(text, seq_lens=(2, 4, 8, 16, 32)):
    """
    select subsequence randomly from input text
    """
    # randomly choose sequence length
    seq_len = random.choice(seq_lens)
    # randomly choose start index
    start_index = random.randint(0, len(text) - seq_len - 1)
    seed = text[start_index: start_index + seq_len]
    return seed


def sample_from_probs(probs, top_n=10):
    """
    truncated weighted random choice.
    """
    # need 64 floating point precision
    probs = np.array(probs, dtype=np.float64)
    # set probabilities after top_n to 0
    probs[np.argsort(probs)[:-top_n]] = 0
    # renormalise probabilities
    probs /= np.sum(probs)
    sampled_index = np.random.choice(len(probs), p=probs)
    return sampled_index
