import random
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from fedlab.utils.dataset.partition import CIFAR100Partitioner
from fedlab.utils.functional import partition_report
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.transforms import transforms


def get_cifar100_transforms(TF=3):
    print(TF)
    if TF == 0:  # simple
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
    elif TF == 1:  # add TrivialAugmentWide and RandomErasing
        train_transform = transforms.Compose([
            torchvision.transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.RandomErasing(p=0.1)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
    # test transform
    test_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float)
    ])
    return train_transform, test_transform

def get_sets(TF=3):
    train_transform, test_transform = get_cifar100_transforms(TF)
    TRAIN_SET = torchvision.datasets.CIFAR100(root="data/CIFAR100/", train=True, download=True, transform=train_transform)
    TEST_SET = torchvision.datasets.CIFAR100(root="data/CIFAR100/", train=False, download=True, transform=test_transform)
    return TRAIN_SET,TEST_SET


def analyze_dataloaders(dataloaders_dict, num_classes):
    # 初始化一个字典来存储每个类别被分配到各个客户端的情况
    label_distribution = {i: {} for i in range(num_classes)}

    for client_id, dataloader in dataloaders_dict.items():
        label_counts = np.zeros(num_classes, dtype=int)
        total_samples = 0  # Counter for the total number of samples in the client
        for _, labels in dataloader:
            total_samples += len(labels)  # Update the total count
            for label in labels:
                label_counts[label] += 1

        print(f"Client {client_id}, Total samples: {total_samples}")
        for i, count in enumerate(label_counts):
            if count > 0:  # Only print labels with a count greater than 0
                print(f"Label {i}: {count} samples, Percentage: {count / total_samples * 100:.2f}%")
                label_distribution[i][client_id] = count  # 更新每个类别的客户端分布

    # 打印每个类别被分配到各个客户端的情况
    for label, distribution in label_distribution.items():
        print(f"Label {label}: {distribution}")




def get_label_distribution(dataloaders_dict, num_classes):
    label_distribution = {i: {} for i in range(num_classes)}
    for client_id, dataloader in dataloaders_dict.items():
        label_counts = np.zeros(num_classes)

        for _, labels in dataloader:

            for label in labels:
                label_counts[label] += 1

        for i, count in enumerate(label_counts):
            if count > 0:
                label_distribution[i][client_id] = count

    return label_distribution

def split_testset(testset, label_distribution, num_clients, num_classes, batch_size=64, shuffle=True, num_workers=1):

    total_train_samples = sum([sum(distribution.values()) for distribution in label_distribution.values()])

    # 先按标签把测试集分成100份
    testset_per_label = [[] for _ in range(num_classes)]
    for i, (data, label) in enumerate(testset):
        testset_per_label[label].append(i)
    total = len(testset)
    # 然后按照训练集的标签分布来分割测试集
    test_indices_dict = {client_id: [] for client_id in range(num_clients)}
    for label, distribution in label_distribution.items():
        for client_id, count in distribution.items():
            # 根据训练集的样本数来计算应该分配给每个客户端的测试样本数
            percent = count / total_train_samples
            required_samples = round(total * percent)

            if required_samples > len(testset_per_label[label]):
                # 如果需要的样本数大于可用的样本数，则需要从其它客户端复制样本
                extra_samples_needed = required_samples - len(testset_per_label[label])

                # 找出有足够样本的客户端
                clients_with_enough_samples = [id for id in test_indices_dict.keys() if
                                               len(test_indices_dict[id]) >= 1 and id != client_id]
                if clients_with_enough_samples:
                    # 随机选择一个客户端来复制样本
                    random_client_id = random.choice(clients_with_enough_samples)
                    # 取其它客户端的样本索引
                    random_client_samples = [index for index in test_indices_dict[random_client_id] if
                                             testset[index][1] == label]
                    # 从随机选择的客户端随机选择样本
                    extra_samples = random.sample(random_client_samples,
                                                  min(extra_samples_needed, len(random_client_samples)))
                    testset_per_label[label].extend(extra_samples)
                    # Remove the copied samples from the selected client
                    for sample in extra_samples:
                        random_client_samples.remove(sample)
                else:
                    print(f"Warning: No clients with enough samples for client {client_id}.")
                    break

            indices = testset_per_label[label][:required_samples]
            testset_per_label[label] = testset_per_label[label][required_samples:]
            test_indices_dict[client_id].extend(indices)

    # 检查每个客户端是否有足够的样本，如果没有，从尚未分配的样本中随机添加
    for client_id in test_indices_dict:
        if len(test_indices_dict[client_id]) < num_clients:
            extra_samples_needed = num_clients - len(test_indices_dict[client_id])
            # 合并所有未分配的样本
            unallocated_samples = sum(testset_per_label, [])
            if len(unallocated_samples) >= extra_samples_needed:
                # 如果有足够的未分配样本，从中随机选择
                extra_samples = random.sample(unallocated_samples, extra_samples_needed)
            else:
                # 如果未分配的样本不够，从整个测试集中随机选择
                all_indices = list(range(len(testset)))
                extra_samples = random.sample(all_indices, extra_samples_needed)
            test_indices_dict[client_id].extend(extra_samples)

    # 最后，创建每个客户端的DataLoader
    test_dataloaders_dict = {}
    test_datasets = []
    for client_id, indices in test_indices_dict.items():
        subset_data = Subset(testset, indices)
        test_datasets.append(subset_data)
        test_dataloaders_dict[client_id] = DataLoader(subset_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)

    return test_dataloaders_dict, test_datasets
def plot_all_train_distribution(trainset, hetero_dir_part,num_classes,display_classes=10):
    col_names = [f"class{i}" for i in range(num_classes)]
    display_col_names = [f"class{i}" for i in range(display_classes)]
    hist_color = '#4169E1'
    plt.rcParams['figure.facecolor'] = 'white'
    csv_file = "report_of_trainset.csv"
    partition_report(trainset.targets, hetero_dir_part.client_dict,
                     class_num=num_classes,
                     verbose=False, file=csv_file)

    hetero_dir_part_df = pd.read_csv(csv_file, header=1)
    hetero_dir_part_df = hetero_dir_part_df.set_index('client')
    for col in col_names:
        hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)

    # select first 10 clients and first 10 classes for bar plot
    hetero_dir_part_df[display_col_names].iloc[:10].plot.barh(stacked=True)
    # plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('sample num')
    plt.show()
def partition_train_and_test_dataloaders(trainset, testset, num_clients, num_classes,partition="dirichlet",
                                         dir_alpha=9999.9,balance=None, batch_size=10, seed=808,num_workers=1,need_plot_and_report=False):
    hetero_dir_part = CIFAR100Partitioner(trainset.targets,
                                          num_clients=num_clients,
                                          balance=balance,
                                          partition=partition,
                                          dir_alpha=dir_alpha,
                                          seed=seed)

    train_dataloaders_dict = {}
    train_datasets = []  # 新增一个字典，用于保存每个客户端的训练集数据
    # 得到客户端训练集
    for client_id, indices in hetero_dir_part.client_dict.items():
        subset_data = Subset(trainset, indices)
        train_datasets.append(subset_data)  # 将子集数据保存到新的字典中
        train_dataloaders_dict[client_id] = DataLoader(subset_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)

    # 使用函数来获取训练集的标签分布
    label_distribution = get_label_distribution(train_dataloaders_dict, num_classes)

    # 使用函数来分割测试集
    test_dataloaders_dict, test_datasets= split_testset(testset, label_distribution, num_clients, num_classes, batch_size=batch_size,num_workers=num_workers)
    if need_plot_and_report:
        plot_all_train_distribution(trainset, hetero_dir_part,num_classes,display_classes=10)

    return train_dataloaders_dict, test_dataloaders_dict,train_datasets,test_datasets #返回字典


def plot_client_train_and_test_dataset_distribution(train_data_loadars, test_data_loaders,num_classes, clients_id):
    # 为客户端的训练集计数
    train_dataloader = train_data_loadars[clients_id]
    train_labels = [label.item() for batch in train_dataloader for label in batch[1]]
    train_label_distribution = Counter(train_labels)

    # 为客户端的测试集计数
    test_dataloader = test_data_loaders[clients_id]
    test_labels = [label.item() for batch in test_dataloader for label in batch[1]]
    test_label_distribution = Counter(test_labels)
    print("Train label distribution:")
    # 按键排序，输出标签计数
    for label in sorted(train_label_distribution.keys()):
        print(f"{label}: {round(train_label_distribution[label])}", end=" ")
    print()
    print("Test label distribution:")
    # 按键排序，输出标签计数
    for label in sorted(test_label_distribution.keys()):
        print(f"{label}: {test_label_distribution[label]}", end=" ")
    print()

    # 获取存在的标签
    existing_train_labels = [label for label in range(num_classes) if train_label_distribution.get(label, 0) > 0]
    existing_test_labels = [label for label in range(num_classes) if train_label_distribution.get(label, 0) > 0]

    # 获取训练集和测试集的标签分布列表
    train_label_distribution_list = [train_label_distribution[label] / 5 for label in existing_train_labels]
    test_label_distribution_list = [test_label_distribution[label] for label in existing_test_labels]

    # 使用这些列表创建柱状图
    index_train = np.arange(len(existing_train_labels))
    index_test = np.arange(len(existing_test_labels))

    # 设置宽度
    width = 0.35

    # 创建画布
    fig, ax = plt.subplots(figsize=(20, 10))

    # 绘制训练集标签分布图
    rects1 = ax.bar(index_train - width / 2, train_label_distribution_list, width, label='Train', color='b')

    # 绘制测试集标签分布图
    rects2 = ax.bar(index_test + width / 2, test_label_distribution_list, width, label='Test', color='g')

    ax.set_xlabel('Label')
    ax.set_ylabel('Percentage of Samples')
    ax.set_title('Distribution of Labels for Client 4 in Training and Test Set')
    ax.legend()

    fig.tight_layout()
    plt.show()
def class_name_sample(sampling_counts, class_num, num_client, replace,seed=1):
    random.seed(seed)
    if len(sampling_counts) != num_client:
        raise ValueError("Length of sampling_counts must be equal to num_client.")

    if not replace and max(sampling_counts) > class_num:
        raise ValueError(
            "The maximum value in sampling_counts must be less than or equal to class_num when replace is False.")

    labels = list(range(class_num))
    remaining_labels = labels.copy()
    clients = [[] for _ in range(num_client)]
    clients_tags = {}
    label_clients = {label: [] for label in labels}  # 创建标签到客户端的映射

    for i, client in enumerate(clients):
        if replace:
            samples = random.sample(remaining_labels, k=sampling_counts[i])
            client.extend(samples)
            for sample in samples:
                label_clients[sample].append(i)  # 更新标签对应的客户端列表
        else:
            samples = random.sample(remaining_labels, k=sampling_counts[i])
            client.extend(samples)
            for sample in samples:
                remaining_labels.remove(sample)
                label_clients[sample].append(i)  # 更新标签对应的客户端列表
        clients_tags[i] = client

    return clients_tags, label_clients

def distribute_data(label_clients, dataset, data_alpha):
    client_datasets = []

    # 获取数据集中的所有标签
    all_labels = [label for _, label in dataset]

    # 计算每个标签在数据集中的索引
    label_indices = {label: [] for label in set(all_labels)}
    for idx, label in enumerate(all_labels):
        label_indices[label].append(idx)

    # 分配数据给客户端
    clients_data = {}
    for label, clients in label_clients.items():
        num_clients = len(clients)
        label_data_indices = label_indices[label]

        # 计算权重
        weights = [data_alpha ** i for i in range(num_clients)]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        # 计算每个客户端应接收的数据量
        data_per_client = [round(len(label_data_indices) * w) for w in normalized_weights]

        # 随机打乱客户端顺序
        random.shuffle(clients)

        # 根据计算出的每个客户端的数据量进行分配
        start_index = 0
        for client_idx, client in enumerate(clients):
            end_index = start_index + data_per_client[client_idx]
            client_data_indices = label_data_indices[start_index:end_index]
            start_index = end_index

            if client in clients_data:
                clients_data[client].extend(client_data_indices)
            else:
                clients_data[client] = client_data_indices

    # 将索引列表转换为数据集对象列表
    for client_id, data_indices in clients_data.items():
        client_datasets.append(Subset(dataset, data_indices))

    return client_datasets

def print_client_data_info(dataloaders, num_classes=100):
    print("{dataset information:")
    label_clients = {i: set() for i in range(num_classes)}

    for client_id, dataloader in enumerate(dataloaders):
        client_labels = []
        for _, labels in dataloader:
            client_labels.extend([int(label.item()) for label in labels])
        label_count = {label: client_labels.count(label) for label in set(client_labels)}
        print(f"Client {client_id}: {len(client_labels)} samples, class distribution: {label_count}")

        for label in label_count:
            label_clients[label].add(client_id)

    print(f"\nLabel distribution for dataset:")
    for label, clients in label_clients.items():
        total_samples = 0
        client_samples = {}
        for client_id in clients:
            client_labels = []
            for _, labels in dataloaders[client_id]:
                client_labels.extend([int(label.item()) for label in labels])
            num_samples = client_labels.count(label)
            total_samples += num_samples
            client_samples[f"Client {client_id}"] = num_samples
        print(f"Label {label}: {total_samples} samples, client distribution: {client_samples}")



def get_simple_dataloaders(SAMPLING_COUNTS=None, REPLACE = False, ALPHA_DATA = 1,NUM_CLIENTS=10,BATCH_SIZE=10,NUM_WORKERS=1,NUM_CLASSES=100,SEED=1,TF=3):
    if SAMPLING_COUNTS is None:
        SAMPLING_COUNTS = [5 for i in range(NUM_CLIENTS)]
        print(SAMPLING_COUNTS)
    TRAIN_SET, TEST_SET=get_sets(TF)
    test_loader_all = DataLoader(TEST_SET, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    train_loader_all = DataLoader(TRAIN_SET, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


    clients_tags, label_clients = class_name_sample(SAMPLING_COUNTS, NUM_CLASSES, NUM_CLIENTS, REPLACE, SEED)
    client_train_datasets = distribute_data(label_clients, TRAIN_SET, ALPHA_DATA)
    client_test_datasets = distribute_data(label_clients, TEST_SET, 1)

    train_loaders = [torch.utils.data.DataLoader(client_dset, batch_size=BATCH_SIZE, shuffle=True,
                                                 num_workers=NUM_WORKERS, pin_memory=True)
                     for client_dset in client_train_datasets]

    test_loaders = [torch.utils.data.DataLoader(client_dset, batch_size=BATCH_SIZE, shuffle=False,
                                                num_workers=NUM_WORKERS, pin_memory=True)
                    for client_dset in client_test_datasets]
    return train_loaders, test_loaders, train_loader_all, test_loader_all,client_train_datasets,client_test_datasets
def get_dir_dataloaders(BATCH_SIZE=10,NUM_WORKERS=1,NUM_CLIENTS=20,NUM_CLASSES=100,DIR_alpha=0.1,SEED=1,TF=3):
    # CIFAR100数据集的预处理
    TRAIN_SET, TEST_SET = get_sets(TF)
    train_loader_all = DataLoader(TRAIN_SET, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader_all = DataLoader(TEST_SET, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


    # 分割数据集到每个客户端
    train_loaders, test_loaders,traindatasets,test_datasets = partition_train_and_test_dataloaders(TRAIN_SET, TEST_SET, num_clients=NUM_CLIENTS,
                                                                       num_classes=NUM_CLASSES, partition="dirichlet",
                                                                       dir_alpha=DIR_alpha, balance=None,
                                                                       batch_size=BATCH_SIZE,
                                                                       seed=SEED, num_workers=NUM_WORKERS,
                                                                       need_plot_and_report=False)

    return train_loaders, test_loaders, train_loader_all, test_loader_all,traindatasets,test_datasets
