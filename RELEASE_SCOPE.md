# 发布范围分析

| 内容 | 处理 | 原因 |
| --- | --- | --- |
| my_test.py | 保留 | 完整增强入口，修正参考图评测开关 |
| cldm/cldm.py、model.py、hack.py | 保留 | 模型构建、权重加载、注意力工具 |
| ldm 中实际导入的模块 | 保留 | 扩散模型、采样器、编码器及公共工具 |
| my_vae/autoencoder.py、models.py | 保留 | 旁路解码器；移除直接运行的数据集调试入口 |
| new_ciconv2d_2.py | 保留 | cldm 当前实际使用的先验网络 |
| extract_invirant/net.py | 保留 | 推理所需分解网络，目录中的训练脚本不复制 |
| annotator/util.py、img_name.py、models/cldm_v15.yaml | 保留 | 预处理、共享状态、模型配置 |
| empty_embedding.pkl | 保留 | 推理读取的空文本条件，不是训练数据集 |
| paired-metrics.py | 保留 | PSNR/SSIM/LPIPS/LOE 辅助函数 |
| train.py、coco_dataset.py、cldm/logger.py | 排除 | 训练启动、数据加载、训练回调 |
| 其他测试变体、复杂度实验、特征提取脚本 | 排除 | 不属于选定增强入口的依赖 |
| 权重、数据集、图片、实验结果、日志、IDE 缓存 | 排除 | 不属于代码发布范围 |

采用逐文件依赖白名单复制，不对原项目执行删除或修改。
共享模型文件中的 training_step、configure_optimizers 等方法按要求保留。
推理版另将增强网络切换到 eval 模式，并创建现有调试图输出目录；没有修改网络结构或采样参数。

验证边界：Python 语法编译、项目内导入闭包、Git 暂存清单和文件大小检查。
当前执行 Python 环境没有 torch/cv2/omegaconf/pytorch_lightning，未执行 GPU 推理或验证输出质量。
