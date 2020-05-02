class DefaultConfigs(object):
    # set default configs, if you don't understand, don't modify
    seed = 666            # set random seed
    workers = 4           # set number of data loading workers (default: 4)
    beta1 = 0.9           # adam parameters beta1
    beta2 = 0.999         # adam parameters beta2
    mom = 0.9             # momentum parameters
    wd = 1e-4             # weight-decay
    resume = None         # path to latest checkpoint (default: none),should endswith ".pth" or ".tar" if used
    evaluate = False      # just do evaluate
    start_epoch = 0       # deault start epoch is zero,if use resume change it
    split_online = False  # split dataset to train and val online or offline

    # set changeable configs, you can change one during your experiment
    dataset = "/data/zcj/webank/classification/dataset/df/cloud/data/dataset/"  # dataset folder with train and val
    test_folder =  "/data/zcj/webank/classification/dataset/df/cloud/data/test/"      # test images' folder
    submit_example =  "/data/zcj/webank/classification/dataset/df/cloud/data/submit_example.csv"    # submit example file
    checkpoints = "./checkpoints/"        # path to save checkpoints
    log_dir = "./logs/"                   # path to save log files
    submits = "./submits/"                # path to save submission files
    bs = 32               # batch size
    lr = 2e-3             # learning rate
    epochs = 40           # train epochs
    input_size = 512      # model input size or image resied
    num_classes = 9       # num of classes
    gpu_id = "0"          # default gpu id
    model_name = "se_resnext50_32x4d-model-sgd-512"      # model name to use
    optim = "sgd"        # "adam","radam","novograd",sgd","ranger","ralamb","over9000","lookahead","lamb"
    fp16 = True          # use float16 to train the model
    opt_level = "O1"      # if use fp16, "O0" means fp32，"O1" means mixed，"O2" means except BN，"O3" means only fp16
    keep_batchnorm_fp32 = False  # if use fp16,keep BN layer as fp32
    loss_func = "CrossEntropy" # "CrossEntropy"、"FocalLoss"、"LabelSmoothCE"
    lr_scheduler = "step"  # lr scheduler method,"adjust","on_loss","on_acc","step"

    
configs = DefaultConfigs()