import torch
import numpy as np
# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR = SR.type(torch.float)
    GT = GT.type(torch.float)

    # TP : True Positive
    # FN : False Negative
    tempResultSR_1 = torch.eq(SR, 1)
    tempResultGT_1 = torch.eq(GT, 1)
    tempResultSR_0 = torch.eq(SR, 0)
    tempResultGT_0 = torch.eq(GT, 0)

    TP = tempResultSR_1+tempResultGT_1
    TP = TP.type(torch.float)*2.0

    FN = tempResultSR_0+tempResultGT_0
    FN = FN.type(torch.float) * 2.0

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR = SR.type(torch.float)
    GT = GT.type(torch.float)

    # TN : True Negative
    # FP : False Positive
    tempResultSR_1 = torch.eq(SR, 1)
    tempResultGT_1 = torch.eq(GT, 1)
    tempResultSR_0 = torch.eq(SR, 0)
    tempResultGT_0 = torch.eq(GT, 0)

    TN =  tempResultSR_0 + tempResultGT_0
    TN = TN.type(torch.float)*2.0

    FP = tempResultSR_1 + tempResultGT_0
    FP = FP.type(torch.float)*2.0

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR = SR.type(torch.float)
    GT = GT.type(torch.float)

    # TP : True Positive
    # FP : False Positive
    tempResultSR_1 = torch.eq(SR, 1)
    tempResultGT_1 = torch.eq(GT, 1)
    tempResultSR_0 = torch.eq(SR, 0)
    tempResultGT_0 = torch.eq(GT, 0)

    TP = tempResultSR_1 +tempResultGT_1
    TP = TP.type(torch.float)*2.0
    FP = tempResultSR_1 +tempResultGT_0
    FP = FP.type(torch.float)*2.0

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR = SR.type(torch.float)
    GT = GT.type(torch.float)

    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR = SR.type(torch.float)
    GT = GT.type(torch.float)

    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC

def get_MSE(SR,GT):
    # DC : Dice Coefficient

    #SR = SR.type(torch.float)
    #GT = GT.type(torch.float)

    #MSE = float(torch.sum(torch.nn.MSELoss(SR,GT)))
    MSE = torch.mean(torch.pow(SR-GT, 2))

    return MSE


