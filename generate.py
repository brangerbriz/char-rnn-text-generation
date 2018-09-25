import os
import random
import utils
import numpy as np
from argparse import ArgumentParser
from keras.models import load_model, Sequential


def main():
    dsc = "generate synthetic text from a pre-trained LSTM text generation model"
    arg_parser = ArgumentParser(description=dsc)

    # generate args
    arg_parser.add_argument("--checkpoint-path", required=True,
                            help="path to load model checkpoints (required)")
    group = arg_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text-path", help="path of text file to generate seed")
    group.add_argument("--seed", default=None, help="seed character sequence")
    arg_parser.add_argument("--length", type=int, default=1024,
                            help="length of character sequence to generate (default: %(default)s)")
    arg_parser.add_argument("--top-n", type=int, default=3,
                            help="number of top choices to sample (default: %(default)s)")

    args = arg_parser.parse_args()
    generate(args)


def generate(args):
    """
    generates text from trained model specified in args.
    main method for generate subcommand.
    """
    # load learning model for config and weights
    model = load_model(args.checkpoint_path)
    # build inference model and transfer weights
    inference_model = build_inference_model(model)
    inference_model.set_weights(model.get_weights())
    print("model loaded: {}.".format(args.checkpoint_path))
    # create seed if not specified
    if args.seed is None:
        with open(args.text_path) as f:
            text = f.read()
        seed = generate_seed(text)
        print("seed sequence generated from {}".format(args.text_path))
    else:
        seed = args.seed

    return generate_text(inference_model, seed, args.length, args.top_n)


def generate_text(model, seed, length=512, top_n=10):
    """
    generates text of specified length from trained model
    with given seed character sequence.
    """
    print("generating {} characters from top {} choices.".format(length, top_n))
    print('generating with seed: "{}".'.format(seed))
    generated = seed
    encoded = utils.encode_text(seed)
    model.reset_states()

    for idx in encoded[:-1]:
        x = np.array([[idx]])
        # input shape: (1, 1)
        # set internal states
        model.predict(x)

    next_index = encoded[-1]
    for i in range(length):
        x = np.array([[next_index]])
        # input shape: (1, 1)
        probs = model.predict(x)
        # output shape: (1, 1, vocab_size)
        next_index = sample_from_probs(probs.squeeze(), top_n)
        # append to sequence
        generated += utils.ID2CHAR[next_index]

    print("generated text: \n{}\n".format(generated))
    return generated


def build_inference_model(model, batch_size=1, seq_len=1):
    """
    build inference model from model config
    input shape modified to (1, 1)
    """
    print("building inference model.")
    config = model.get_config()
    # edit batch_size and seq_len
    config[0]["config"]["batch_input_shape"] = (batch_size, seq_len)
    inference_model = Sequential.from_config(config)
    inference_model.trainable = False
    return inference_model

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

if __name__ == '__main__':
    main()
