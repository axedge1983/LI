from my_dataloaders import *
from utils import *
from MHC_CoAt_model import CoAtNet,MHC_CoAtNet

TF=1
BATCH_SIZE=100
NUM_CLASSES=100
NUM_CLIENTS=20
SEED = 1
ROUND = 60
IMAGE_SIZE = 32
NUM_WORKERS = 8
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-1
STEP_EPOCHS = [2,2,2] #训练阶段，每个元素表示每个阶段的迭代次数，如[4,2,2]表示4轮头部训练、2轮主干训练、2轮全部训练
HEAD_FINE_TUNE_EPOCHS = 6 #微调1，头部微调轮次
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DIR_alpha=0.1 #
PAT=True

if not PAT:
    train_loaders, test_loaders, train_loader_all, test_loader_all,\
        client_train_datasets,client_test_datasets= get_dir_dataloaders(BATCH_SIZE=BATCH_SIZE,NUM_WORKERS=NUM_WORKERS,NUM_CLIENTS=NUM_CLIENTS,NUM_CLASSES=NUM_CLASSES,DIR_alpha=DIR_alpha,SEED=SEED,TF=TF)
else:
    train_loaders, test_loaders, train_loader_all, test_loader_all,client_train_datasets,client_test_datasets= (
        get_simple_dataloaders(SAMPLING_COUNTS=None, REPLACE = False, ALPHA_DATA = 1,NUM_CLIENTS=NUM_CLIENTS,BATCH_SIZE=BATCH_SIZE,NUM_WORKERS=NUM_WORKERS,NUM_CLASSES=NUM_CLASSES,SEED=SEED,TF=TF))
   
def initialize_client(index): #初始化客户端函数

    model = CoAtNet(NUM_CLASSES, IMAGE_SIZE, head_channels=32, channel_list=[64, 64, 128, 256, 512],
                num_blocks=[2, 2, 2, 2, 2], strides=[1, 1, 2, 2, 2],
                trans_p_drop=0.3, head_p_drop=0.3) #客户端模型
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    frozen_without_last_module(model)
    optimizer_head = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE/5, weight_decay=WEIGHT_DECAY)
    scheduler_head = optim.lr_scheduler.StepLR(optimizer_head, step_size=10, gamma=0.5)
    frozen_last_module(model)
    optimizer_backbone = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE/5*4, weight_decay=WEIGHT_DECAY)
    scheduler_backbone = optim.lr_scheduler.StepLR(optimizer_backbone, step_size=10, gamma=0.5)
    unfronzen(model)
    return model, optimizer, optimizer_head, optimizer_backbone, scheduler, scheduler_head, scheduler_backbone

#初始化客户端
criterion = nn.CrossEntropyLoss()
models, optimizers, optimizers_head, optimizers_backbone, schedulers, schedulers_head, schedulers_backbone = [], [], [], [], [], [], []
for i in range(NUM_CLIENTS):
    model, optimizer, optimizer_head, optimizer_backbone, scheduler, scheduler_head, scheduler_backbone = initialize_client(i)
    models.append(model)
    optimizers.append(optimizer)
    schedulers.append(scheduler)
    optimizers_head.append(optimizer_head)
    schedulers_head.append(scheduler_head)
    optimizers_backbone.append(optimizer_backbone)
    schedulers_backbone.append(scheduler_backbone)

#打印模型参数量
print("Number of parameters: {:,}".format(sum(p.numel() for p in models[0].parameters())))

#定义记录每个客户端训练损失和准确率的dataframe
accuracy_df = pd.DataFrame(columns=['round', 'client',"bfaccuracy", 'haccuracy','bbaccuracy',"Gacuracy"])

#训练过程
for round in range(ROUND):
    print("starting round", round)
    for client in range(NUM_CLIENTS):
        print("#" * 60)
        print("training client", client)

        #before training, the test accuracy
        bftest_loss,bfaccuracy = evaluate(models[client], test_loaders[client], criterion,DEVICE)
        print(f"before training Test Loss: {bftest_loss:.4f}, before training Test Accuracy: {bfaccuracy:.2f}%")

        #step 1: head finetune
        frozen_without_last_module(models[client])
        _,haccuracy=training_step(models, train_loaders, test_loaders,criterion,device=DEVICE,
                                  client=client, STEP_EPOCHS=STEP_EPOCHS,step=0,optimizer=optimizers_head[client])
        schedulers_head[client].step()

        #step 2: backbone finetune
        frozen_last_module(models[client])
        _,bbaccuracy=training_step(models, train_loaders, test_loaders, criterion, device=DEVICE,
                                   client=client, STEP_EPOCHS=STEP_EPOCHS, step=1,optimizer=optimizers_backbone[client])
        schedulers_backbone[client].step()

        #step 3: all training
        unfronzen(models[client])
        _,Aaccuracy=training_step(models, train_loaders, test_loaders, criterion, device=DEVICE,
                                  client=client, STEP_EPOCHS=STEP_EPOCHS, step=2,optimizer=optimizers[client])
        schedulers[client].step()

        #全局测试,对整体测试数据集的测试
        test_Gloss, Gaccuracy = evaluate(models[client], test_loader_all, criterion, DEVICE)
        print(f"Global Test Loss: {test_Gloss:.4f}, Global Test Accuracy: {Gaccuracy:.2f}%")

        # 将精度数据添加到 DataFrame中
        new_row = pd.DataFrame({'round': [round], 'client': [client], 'bfaccuracy': [bfaccuracy],
                                'haccuracy': [haccuracy],'bbaccuracy':[bbaccuracy],"Aaccuracy":[Aaccuracy],
                                "Gacuracy":[Gaccuracy]})
        accuracy_df = pd.concat([accuracy_df, new_row], ignore_index=True)


        print("#" * 50)
        if client < NUM_CLIENTS - 1:
            print("transfer weights to next client")
            transfer_weights_without_last_module(models[client], models[client + 1])
        elif client == NUM_CLIENTS - 1 and round < ROUND - 1:
            print("transfer weights to first client")
            transfer_weights_without_last_module(models[client], models[0])
        print("#" * 50)

#打印精度数据
print(accuracy_df)

accuracy_ft_df = pd.DataFrame(columns=['round','client',"shots","shot_accuracy","fthaccuracy", 'ftaaccuracy','global_model_accuracy'])

print("*****all clients use the last client's model****")
for client in range(NUM_CLIENTS-1):
    transfer_weights_without_last_module(models[-1], models[client])

# #微调客户端模型头部的过程
print("fine-tuning client models")

round=-1 #微调不计入轮数
for client in range(NUM_CLIENTS):
    #微调头部
    print("fine-tuning client head", client)
    frozen_without_last_module(models[client])
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, models[client].parameters()),
                            lr=LEARNING_RATE / 25, weight_decay=WEIGHT_DECAY)
    for epoch in range(HEAD_FINE_TUNE_EPOCHS):
        train(models[client], train_loaders[client], criterion, optimizer, DEVICE)
        print("test:",evaluate(models[client], test_loaders[client], criterion, DEVICE))
    fthtest_loss, fthaccuracy = evaluate(models[client], test_loaders[client], criterion, DEVICE)
    new_row = pd.DataFrame({'round':[round],'client': [client], 'fthaccuracy': [fthaccuracy]})
    accuracy_ft_df = pd.concat([accuracy_ft_df, new_row], ignore_index=True)

#保存客户端模型
# for client in range(NUM_CLIENTS):
#     torch.save(models[client].state_dict(), f"save_model/C{NUM_CLASSES}R{ROUND}DIR{DIR_alpha}{STEP_EPOCHS[0]}{STEP_EPOCHS[1]}{STEP_EPOCHS[2]}Seed{SEED}client_{client}.pth")

#定义全局模型
mhc_model=MHC_CoAtNet(NUM_CLASSES, IMAGE_SIZE, head_channels=32, channel_list=[64, 64, 128, 256, 512],
                num_blocks=[2, 2, 2, 2, 2], strides=[1, 1, 2, 2, 2],
                trans_p_drop=0.3, head_p_drop=0.3)
mhc_model.to(DEVICE)
print("Number of global model's parameters: {:,}".format(sum(p.numel() for p in mhc_model.parameters())))

#从客户端模型初始化全局模型
init_global_model_from_clients_models(mhc_model, models, num_heads=NUM_CLIENTS)

#打印微调精度数据
print(accuracy_ft_df)

#得到全局模型
print("get global model")

#冻结层和设置优化器、学习率调度器
frozen_without_last_module(mhc_model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, mhc_model.parameters()), lr=LEARNING_RATE,
                         weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)

#prt_trainable_param(mhc_model) #打印可训练参数

#训练全局模型
for round in range(30):
    print("starting round", round)
    train(mhc_model, train_loader_all, criterion, optimizer, DEVICE)
    test_loss, test_accuracy = evaluate(mhc_model, test_loader_all, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    scheduler.step()
    new_row = pd.DataFrame({'round': [round], 'global_model_accuracy': [test_accuracy]})
    accuracy_ft_df = pd.concat([accuracy_ft_df, new_row], ignore_index=True)

#测试共享层在所有数据上的能力
frozen_without_last_module(models[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, models[1].parameters()), lr=LEARNING_RATE/8,
                         weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.5)
for round in range(20):
    print("Test shared layer finetuning ability at all dataset starting round", round)
    train(models[1], train_loader_all, criterion, optimizer, DEVICE)
    ggtest_loss, ggtest_accuracy = evaluate(models[1], test_loader_all, criterion, DEVICE)
    scheduler.step()
    print(f"gg Test Loss: {ggtest_loss:.4f}, Test Accuracy: {ggtest_accuracy:.2f}%")
    new_row = pd.DataFrame({'round': [round], 'client': [999], "Gacuracy":[ggtest_accuracy]})
    accuracy_df = pd.concat([accuracy_df, new_row], ignore_index=True)

#保存精度数据
print(accuracy_ft_df)
print(accuracy_df)
accuracy_df.to_csv(f"pat{PAT}_CIFAR100_df_C{NUM_CLIENTS}_R{ROUND}_STEP_EPOCHS{STEP_EPOCHS}_DIR{DIR_alpha}_Seed{SEED}_TF{TF}_BS{BATCH_SIZE}_Coat.csv", index=False)
accuracy_ft_df.to_csv(f"pat{PAT}_CIFAR100_df_C{NUM_CLIENTS}_R{ROUND}_STEP_EPOCHS{STEP_EPOCHS}_DIR{DIR_alpha}_Seed{SEED}_TF{TF}_BS{BATCH_SIZE}_Coat_ft.csv", index=False)
print("ALL DONE")

