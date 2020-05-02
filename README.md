### pytorch 图像分类竞赛框架

### 更新日志
- (2020年5月2日) 基础版本上线

### 依赖库
- pretrainedmodels
- progress
- efficientnet-pytorch
- apex

### 支持功能

- [x] pytorch官网模型
- [x] [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) 复现的部分模型
- [x] [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) 
- [x] fp16混合精度训练
- [x] TTA
- [x] 固定验证集/随机划分验证集
- [x] 多种优化器：adam、radam、novograd、sgd、ranger、ralamb、over9000、lookahead、lamb
- [x] OneCycle训练策略
- [x] LabelSmoothLoss
- [x] Focal Loss
- [ ] AotuAgument
  
### 使用方法
更改`config.py`中的参数，训练执行 `python main.py`，预测执行`python test.py`

### TODO

- [ ] 优化模型融合策略
- [ ] 优化online数据增强
- [ ] 优化pytorch官方模型调用接口
- [ ] 增加模型全连接层初始化
- [ ] 增加更多学习率衰减策略
- [ ] 增加find lr
- [ ] 增加dali
- [ ] 增加wsl模型
- [ ] 增加tensorboardX
- [ ] 优化文件夹创建