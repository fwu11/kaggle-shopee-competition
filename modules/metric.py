def accuracy(output, target, topk=(1,)):
    """
        parameters:
        -----------
        output: Tensor
            output from the model with the size of (batch_size,class_num)
        target: Tensor
            label with the size of (batch_size,1)
        topk: tuple
            choose top k classification accuracy to return (k1,k2,...)
        return:
        -------
        res: list 
            A list of tensors containing topk accuracy info
    """

    maxk = max(topk)
    batch_size = target.size(0)

    # dim=1表示按行取值
    # output的值是精度，选top5是选这一行精度最大的五个对应的列，也就是属于哪一类
    # pred是(batch_size, maxk) 值为类别号，0，1，...,9
    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    # 转置
    pred = pred.t()

    # eq返回pred和target对应位置值相等返回1，不等返回0
    # target原来是64行1列，值为类别；target.view(1, -1)把target拉成一行64列。
    # expand_as(pred)又把target变成5行64列
    # print("pred",pred)
    # print("target",target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0/batch_size))
    return res