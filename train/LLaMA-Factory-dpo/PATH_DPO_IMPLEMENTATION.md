# 多轮对话 Path-based DPO 支持实现总结

## 实现概述

我们成功实现了对多轮 ShareGPT 格式数据的 path-based DPO 训练支持，使数据能够被 PathDPOTrainer 正确处理。

## 主要修改

### 1. 新增 PathDPODatasetConverter (`converter.py`)

- **功能**: 专门处理多轮对话的 path-based DPO 格式转换
- **输入**: ShareGPT 格式的多轮对话数据，包含 `chosen` 和 `rejected` 字段
- **输出**: 完整的对话路径 `_chosen_path` 和 `_rejected_path`
- **特点**: 将整个对话序列保持为连续的路径，而不是分割成 prompt-response 对

### 2. Template 类新增 encode_path 方法 (`template.py`)

- **功能**: 将完整的对话路径编码为单一序列
- **特点**: 
  - 自动 mask human 输入 (设置为 IGNORE_INDEX)
  - 保留 assistant 输出用于 loss 计算
  - 支持整条路径的 token 化

### 3. 更新 PairwiseDatasetProcessor (`processor/pairwise.py`)

- **功能**: 支持处理 path-based DPO 格式
- **兼容性**: 同时支持传统的 pairwise 格式和新的 path-based DPO 格式
- **自动检测**: 根据数据格式自动选择处理方式

### 4. 增强 Data Collators (`collator.py`)

- **新增**: `PathDPODataCollatorWithPadding` - 专用于 path-based DPO
- **增强**: `PairwiseDataCollatorWithPadding` - 支持多种格式
- **功能**: 自动拼接多轮对话并 mask human 输入

### 5. PathDPOTrainer 集成 (`train/dpo/trainer.py`)

- **导入**: 添加 `get_batch_logps_path` 函数导入
- **兼容**: 确保与现有训练框架兼容

## 数据流程

1. **原始数据**: ShareGPT 格式多轮对话
   ```json
   {
     "chosen": [
       {"from": "human", "value": "问题1"},
       {"from": "gpt", "value": "回答1"},
       {"from": "human", "value": "问题2"},
       {"from": "gpt", "value": "更好的回答2"}
     ],
     "rejected": [
       {"from": "human", "value": "问题1"},
       {"from": "gpt", "value": "回答1"},
       {"from": "human", "value": "问题2"},
       {"from": "gpt", "value": "较差的回答2"}
     ]
   }
   ```

2. **Converter 阶段**: PathDPODatasetConverter 转换为路径格式
   ```python
   {
     "_chosen_path": [完整对话序列],
     "_rejected_path": [完整对话序列]
   }
   ```

3. **Template 阶段**: encode_path 方法处理
   - 将对话路径转换为 token 序列
   - 自动 mask human 输入

4. **Collator 阶段**: 批量处理和填充
   - 生成训练批次
   - 支持多模态输入

5. **Trainer 阶段**: PathDPOTrainer 训练
   - 计算整条路径的 log probabilities
   - 支持 path-based DPO loss

## 使用方法

### 1. 数据格式要求
- 使用 ShareGPT 格式
- `chosen` 和 `rejected` 字段为消息列表
- 每个消息包含 `from` (角色) 和 `value` (内容)

### 2. 配置设置
```yaml
# 使用 path_dpo converter
dataset_converter: path_dpo

# 使用 PathDPOTrainer
trainer_type: path_dpo
```

### 3. 训练启动
数据会自动：
- 被 PathDPODatasetConverter 处理
- 通过 encode_path 方法编码
- 由 PathDPODataCollatorWithPadding 整理
- 送入 PathDPOTrainer 训练

## 关键特性

1. **完整路径支持**: 整个对话作为单一序列处理
2. **自动 Masking**: Human 输入自动被 mask，只对 Assistant 输出计算 loss
3. **向后兼容**: 支持传统 pairwise 格式和新的 path-based 格式
4. **多模态支持**: 兼容图像、视频、音频等多模态输入
5. **长度管理**: 自动处理序列长度截断

## 测试验证

- ✅ PathDPODatasetConverter 转换测试通过
- ✅ 多轮对话格式正确处理
- ✅ Role mapping 正确实现
- ✅ 与现有代码兼容

## 总结

该实现成功支持了多轮 ShareGPT 格式数据的 path-based DPO 训练，每个样本被拼接成一整条路径，human 内容被正确 mask，确保数据格式可以被 PathDPOTrainer 处理。整个实现保持了与现有代码的兼容性，同时提供了强大的多轮对话训练能力。
