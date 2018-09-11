#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from tqdm import tqdm

from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--save', type=str, default="model.pt")
argparser.add_argument('--model', type=str, default="rnn")
argparser.add_argument('--n_epochs', type=int, default=20000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=256)
argparser.add_argument('--n_layers', type=int, default=1)
argparser.add_argument('--learning_rate', type=float, default=0.001)
argparser.add_argument('--chunk_len', type=int, default=250)
argparser.add_argument('--batch_size', type=int, default=150)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
argparser.add_argument('--stabilize', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

file, file_len = read_file(args.filename)

def random_training_set(chunk_len, batch_size):
    tries = 0
    while True:
        try:
            inp = torch.LongTensor(batch_size, chunk_len)
            target = torch.LongTensor(batch_size, chunk_len)
            for bi in range(batch_size):
                start_index = random.randint(0, file_len - chunk_len)
                end_index = start_index + chunk_len + 1
                chunk = file[start_index:end_index]
                inp[bi] = char_tensor(chunk[:-1])
                target[bi] = char_tensor(chunk[1:])
            break
        except:
            tries += 1
            if tries > 10:
                raise
            continue 
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        if args.model == "lstm":
            hidden = (hidden[0].cuda(), hidden[1].cuda())
        else:
            hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    # Project based on the stable set
    if args.stabilize and args.model == "lstm":
        # Ensure word vectors are normalized in l-infinity.
        wvecs = decoder.encoder.weight.data
        ones = torch.ones_like(wvecs)
        trimmed_wvecs = torch.min(torch.max(wvecs, -ones), ones)
        decoder.encoder.weight.data.set_(trimmed_wvecs)

        # One set of weights satisfying stability requirement
        recur_weights = decoder.rnn.weight_hh_l0.data
        wi, wf, wz, wo = recur_weights.chunk(4, 0)

        trimmed_wi =  wi * 0.395  / torch.sum(torch.abs(wi), 0)
        trimmed_wf =  wf * 0.155  / torch.sum(torch.abs(wf), 0)
        trimmed_wz =  wz * 0.099  / torch.sum(torch.abs(wz), 0)
        trimmed_wo =  wo * 0.395 / torch.sum(torch.abs(wo), 0)
        new_recur_weights = torch.cat([trimmed_wi, trimmed_wf, trimmed_wz, trimmed_wo], 0)
        decoder.rnn.weight_hh_l0.data.set_(new_recur_weights)

        # Also trim the input to hidden weight for the forget gate
        ih_weights = decoder.rnn.weight_ih_l0.data
        ui, uf, uz, uo = ih_weights.chunk(4, 0)
        trimmed_uf =  uf * 0.25  / torch.sum(torch.abs(uf), 0)
        new_ih_weights = torch.cat([ui, trimmed_uf, uz, uo], 0)
        decoder.rnn.weight_ih_l0.data.set_(new_ih_weights)

        decoder.rnn.flatten_parameters()

    elif args.stabilize and args.model == "rnn":
        # If the projection fails, raise error
        try:
            U, s, V = torch.svd(decoder.rnn.weight_hh_l0.data)
            s = torch.min(s, torch.ones_like(s))
            projected = torch.mm(torch.mm(U, torch.diag(s)), V.t())
            decoder.rnn.weight_hh_l0.data.set_(projected)
        except:
            print("Projection failed...")
            raise

    return loss.data[0] / args.chunk_len

def save():
    torch.save(decoder, args.save)
    print('Saved as %s' % args.save)

# Initialize models and start training

decoder = CharRNN(
    n_characters,
    args.hidden_size,
    n_characters,
    model=args.model,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0

try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in range(1, args.n_epochs + 1):
        loss = train(*random_training_set(args.chunk_len, args.batch_size))
        loss_avg += loss
        
        msg = "Epoch {}, Loss {:.4f}, Elapsed {}".format(epoch, loss, time_since(start))
        print(msg)
        with open(args.save + "_log", "a") as handle:
            handle.write(msg + "\n")

        if epoch % args.print_every == 0:
            print(generate(decoder, 'Wh', 100, cuda=args.cuda), '\n')

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

