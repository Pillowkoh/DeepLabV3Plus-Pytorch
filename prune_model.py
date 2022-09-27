import torch
import torch.nn as nn
import torch_pruning as tp

def prune_model(model, amount=0.4, crop_size=513):
    # model.cpu()
    # print(model)
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, crop_size, crop_size) )

    def prune_conv(conv, amount=amount):
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv_out_channel, pruning_index)
        plan.exec()

    conv2d_list =[]
    for m in model.modules():
        if isinstance( m, nn.Conv2d ):
            conv2d_list.append(m)

    for conv in conv2d_list[:-1]:
        prune_conv(conv)

    return model