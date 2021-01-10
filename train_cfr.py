import torch
import copy
import multiprocessing as mp
import numpy as np
import os
from memory import Buffer
from master_memory import MasterBuffer
from deep_cfr_net import AdvantageNet, PolicyNet, loss_fn
from strategy_networks import StrategyNetworks
from full_game_sampling import full_game_traversal


def generate_master_buffers(buffer_size=int(1e8)):

    advantage_buffer_p0 = MasterBuffer(int(buffer_size))
    advantage_buffer_p1 = MasterBuffer(int(buffer_size))

    advantage_buff_arr = [advantage_buffer_p0, advantage_buffer_p1]

    policy_buffer = MasterBuffer(int(buffer_size))

    return advantage_buff_arr, policy_buffer


def generate_local_buffers(buffer_size=int(1e7)):

    advantage_buffer_p0 = Buffer(int(buffer_size))
    advantage_buffer_p1 = Buffer(int(buffer_size))

    advantage_buff_arr = [advantage_buffer_p0, advantage_buffer_p1]

    policy_buffer = Buffer(2 * int(buffer_size))

    return advantage_buff_arr, policy_buffer


def initialize_networks():

    strategy_networks = StrategyNetworks()
    policy_net = PolicyNet()

    return strategy_networks, policy_net


def local_game_traversal(k, p, net_arr, t, process_id, initial_strat_num, initial_policy_num):

    (advantage_buffer_arr, policy_buffer) = generate_local_buffers(3e5)

    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    for _ in range(k):

        full_game_traversal(p, net_arr, advantage_buffer_arr, policy_buffer, t)

    n_strat_samples_added = len(advantage_buffer_arr[p])
    n_policy_samples_added = len(policy_buffer)

    strat_prefix = 'strategy/p_' + str(p) + '_'
    policy_prefix = 'policy/'

    advantage_buffer_arr[p].save(strat_prefix, process_id, initial_strat_num)
    policy_buffer.save(policy_prefix, process_id, initial_policy_num)

    return (n_strat_samples_added, n_policy_samples_added)


def train(K=int(5e4), max_t=100):

    (Advantage_buffer_arr, Policy_buffer) = generate_master_buffers()
    strategy_networks = StrategyNetworks()
    trained_nets = [None, None]

    #n_proc = 10
    n_proc = 12
    games_per_proc = int(K / n_proc)

    initial_strat_nums = np.zeros(n_proc)
    initial_policy_nums = np.zeros(n_proc)

    for t in range(1, max_t+1):
        for p in range(2):
            pool = mp.Pool(n_proc)

            samples_added = [pool.apply_async(local_game_traversal, args=(games_per_proc, p, strategy_networks.deepcopy(), t)) for _ in range(n_proc)]

            samples_added = [data.get() for data in samples_added]

            pool.close()
            pool.join()

            for i in range(n_proc):
                initial_strat_nums[i] += samples_added[0]
                initial_policy_nums[i] += samples_added[1]

                print('local buffer size: ', len(local_advantage_buffer))

                Advantage_buffer_arr[p].add(local_advantage_buffer)
                Policy_buffer.add(local_policy_buffer)

            trained_nets[p] = train_advantage_net(Advantage_buffer_arr, p)

            if (t % 5 == 4):
                torch.save(trained_nets[p].state_dict(), './checkpoints/strategy' + str(p) + '.model')
                Policy_buffer.save('policy', t)

        print('FINISHED ITERATION ', t)
        print('Number of nodes sampled: ', len(Advantage_buffer_arr[0]))
        strategy_networks.net_arr = copy.deepcopy(trained_nets)



def test():

    n_proc = 6
    total_games = 600

    games_per_proc = int(total_games / n_proc)

    strategy_networks = StrategyNetworks()
    (Advantage_buffer_arr, Policy_buffer) = generate_master_buffers()

    t = 1
    p = 0

    pool = mp.Pool(n_proc)

    local_buffers_arr = [pool.apply_async(local_game_traversal, args=(games_per_proc, p, strategy_networks.deepcopy(), t)) for _ in range(n_proc)]

    local_buffers_arr = [p.get() for p in local_buffers_arr]

    pool.close()
    pool.join()

    print('All games traversed')

    for i in range(n_proc):
        local_advantage_buffer = local_buffers_arr[i][0][p]
        local_policy_buffer = local_buffers_arr[i][1]

        print('local buffer size: ', len(local_policy_buffer))

        Advantage_buffer_arr[p].add(local_advantage_buffer)
        Policy_buffer.add(local_policy_buffer)

    print('Master buffers updated.')

    print(len(Policy_buffer))



def train_advantage_net(advantage_buffer_arr, p):

    batch_size = 2000
    n_minibatch = int(len(advantage_buffer_arr[p]) / batch_size)
    n_epoch = 5

    net = AdvantageNet()
    #net.to('cuda:0')

    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    for epoch in range(n_epoch):
        running_loss = 0.0

        for minibatch in range(n_minibatch):
            (x_train, t, advantage) = advantage_buffer_arr[p].recast(batch_size)

            #x_train.to('cuda:0')
            #t.to('cuda:0')
            #advantage.to('cuda:0')

            opt.zero_grad()

            pred_advantage = net.forward(x_train)
            loss = loss_fn(pred_advantage, t, advantage)
            loss.backward()
            opt.step()

            running_loss += loss.item()

            #del x_train
            #del t
            #del advantage

            #print('Running loss after minibatch %d: %.3f' % (minibatch, running_loss / ((minibatch + 1) * batch_size)))

            if minibatch % 500 == 499:
                print('Epoch: %d, minibatch_id: %d. Loss: %.3f' % (epoch+1, minibatch, running_loss / 500))
                running_loss = 0.0

    return net


def train_policy_net(policy_buffer):

    batch_size = 2000
    n_minibatch = int(len(policy_buffer) / batch_size)
    n_epoch = 12

    policy_net = PolicyNet()

    opt = torch.optim.Adam(policy_net.parameters(), lr=1e-4)

    for epoch in range(n_epoch):
        running_loss = 0.0

        for minibatch in range(n_minibatch):
            (x_train, t, strategy) = policy_buffer.recast(batch_size)

            opt.zero_grad()

            pred_strategy = policy_net.forward(x_train)
            loss = loss_fn(pred_strategy, t, strategy)
            loss.backward()
            opt.step()

            running_loss += loss.item()

#            print('Running loss after minibatch %d: %.3f' % (minibatch, running_loss / ((minibatch + 1) * batch_size)))

            if minibatch % 500 == 499:
                print('Epoch: %d, minibatch_id: %d. Loss: %.3f' % (epoch+1, minibatch, running_loss / 500))
                running_loss = 0.0

    torch.save(policy_net.state_dict(), './checkpoints/policy_net')
    return policy_net


#test()
train()
