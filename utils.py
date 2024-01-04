import numpy as np
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
#from ignite.utils import convert_tensor



#训练过程函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

#测试过程函数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = running_loss / len(test_loader)
    test_accuracy = 100.0 * correct / total
    return test_loss, test_accuracy


#聚合前一个模型和当前模型的函数
def aggr_without_last_module(models, client, train_loaders):
    print("#" * 50)
    if client == 0:
        temp_model = models[-1]
        temp_count = len(train_loaders[max(train_loaders.keys())].dataset)
    else:
        temp_model = models[client - 1]
        temp_count = len(train_loaders[client - 1].dataset)
    # 获取当前模型和临时模型的子模块列表
    children_current = list(models[client].children())
    children_temp = list(temp_model.children())

    client_data_count = len(train_loaders[client].dataset)
    total_size = client_data_count + temp_count

    print("Aggregating parameters...")

    # 遍历每个子模块，除了最后一个
    for i in range(len(children_current) - 1):
        # 获取子模块的参数字典
        params_current = dict(children_current[i].named_parameters())
        params_temp = dict(children_temp[i].named_parameters())

        # 对每个参数进行加权平均
        for name in params_current:
            param_current = params_current[name]
            param_temp = params_temp[name]

            param_current.data = param_current.data * client_data_count / total_size + \
                                 param_temp.data * temp_count / total_size
    print("#" * 50)


#只转移最后一层的函数
def transfer_weights_without_last_layer(model1, model2):
    # 这个函数假设 model1 和 model2 具有完全相同的架构

    # 获取模型的子模块
    children1 = list(model1.children())
    children2 = list(model2.children())

    # 除最后一层外，复制所有层的权重
    for i in range(len(children1)):
        # 如果不是最后一个模块，复制所有权重
        if i != len(children1) - 1:
            children2[i].load_state_dict(children1[i].state_dict())
        else:
            # 如果是最后一个模块，只复制除了最后一层的权重
            for name, module in children1[i].named_children():
                if name != list(children1[i].named_children())[-1][0]:  # 如果不是最后一层
                    getattr(children2[i], name).load_state_dict(module.state_dict())

#只转移最后一个模块的函数
def transfer_weights_without_last_module(model1, model2): #copy model1 to model2
    # 这个函数假设 model1 和 model2 具有完全相同的架构

    # 获取模型的子模块
    children1 = list(model1.children())
    children2 = list(model2.children())

    # 除最后一层外，复制所有层的权重
    for i in range(len(children1)):
        # 如果不是最后一个模块，复制所有权重
        if i != len(children1) - 1:
            children2[i].load_state_dict(children1[i].state_dict())

#冻结模型共享部分函数
def frozen_without_last_module(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model[-1].parameters():
        param.requires_grad = True


#冻结模型最后一层
def frozen_last_module(model):
    for param in model.parameters():
        param.requires_grad = True
    for param in model[-1].parameters():
        param.requires_grad = False

#解冻模型函数
def unfronzen(model):
    for param in model.parameters():
        param.requires_grad = True
def prt_trainable_param(model):
    for name, param in model.named_parameters():
        print(f'Layer: {name} | Training: {param.requires_grad}')

#初始化全局模型函数
def init_global_model_from_clients_models(mhc_model, models,num_heads):
    # 复制主干前5层
    for i in range(5):
        mhc_model[i].load_state_dict(models[-1][i].state_dict())
    # 复制多头，堆叠分类头
    for client in range(num_heads):
        (list(mhc_model[-2].children())[0][client]).load_state_dict((models[client][-1]).state_dict())

#训练步骤函数
def training_step(models, train_loaders, test_loaders,criterion,device, client, STEP_EPOCHS,step,optimizer=None):
    if optimizer is None:
        raise ValueError("optimizer is None")
    if step == 0:
        print(f"Head Fintuning")
        for i in range(STEP_EPOCHS[step]):
            train(models[client], train_loaders[client], criterion, optimizer, device)
            test_loss, accuracy = evaluate(models[client], test_loaders[client], criterion, device)
            print(f"Head Test Loss: {test_loss:.4f}, Head Test Accuracy: {accuracy:.2f}%")
        test_loss, accuracy = evaluate(models[client], test_loaders[client], criterion, device)
        return test_loss, accuracy
    if step == 1:
        print(f"Backbone Training")
        for i in range(STEP_EPOCHS[step]):
            train(models[client], train_loaders[client], criterion, optimizer, device)
            test_loss, accuracy = evaluate(models[client], test_loaders[client], criterion, device)
            print(f"Backbone Test Loss: {test_loss:.4f}, Backbone Test Accuracy: {accuracy:.2f}%")
        test_loss, accuracy = evaluate(models[client], test_loaders[client], criterion, device)
        return test_loss, accuracy
    if step == 2:
        print(f"All Layers Training")
        for i in range(STEP_EPOCHS[step]):
            train(models[client], train_loaders[client], criterion, optimizer, device)
            test_loss, accuracy = evaluate(models[client], test_loaders[client], criterion, device)
            print(f"All Test Loss: {test_loss:.4f}, All Test Accuracy: {accuracy:.2f}%")
        test_loss, accuracy = evaluate(models[client], test_loaders[client], criterion, device)
        return test_loss, accuracy

#一个最新的优化器
class Lion(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss





