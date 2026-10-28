# NTL 重要模型归档

三个 Name That LLM 分类器检查点、共用 NF4 基础模型及训练资料。

**[下载完整归档 Release](https://github.com/purplellllll/ntl/releases/tag/important-models-20261028)**

|检查点|训练来源|本地验证准确率|Public LB|
|---|---|---:|---:|
|step9579|153261条外部数据阶段结束|55.74074%|未提交|
|step9849|9579之后官方2159条数据微调2轮|59.25926%|0.573|
|official_epoch05_step675|从9579重新初始化官方微调任务的第5轮|60.18519%|0.572|

官方第5轮step675是新任务的局部步数，与外部池轮换任务的step675无关。9849与官方第5轮是从9579出发的两条训练分支。本地分数使用固定540条、30个prompt组验证集，不能视为独立测试或线上分数。

## 下载与恢复

安装GitHub CLI与Python 3.12或更高版本，预留约20GB磁盘空间：

```shell
gh release download important-models-20261028 --repo purplellllll/ntl --dir ntl-assets
python ntl-assets/restore_release.py --assets-dir ntl-assets --output ntl-restored
```

恢复脚本验证每个附件的SHA256，合并4个基础模型分片，并解压检查点和资料包，最后与服务器逐文件清单核对。基础模型文件共6073558092字节，SHA256为`2d833ae29600b913e13d08851f14ce980d3450a3a1809ea0ad4faf5e94bac4c7`。

Release附件包括：三版完整LoRA/LayerNorm/分类头/tokenizer、完整共用基础模型、155420条原训练数据、单独2159条官方训练集、固定540条验证集、验证预测、训练日志、原始run_manifest、代码快照、软件版本、六个既有离线依赖及Kaggle提交记录。`metadata/`可直接浏览主要参数与核对报告；`docs/`保存解释；`code/`为可获得的代码快照。

## 模型与恢复边界

基础模型为`unsloth/Qwen3-8B-Base-bnb-4bit`，36层，NF4；LoRA r16、alpha32、dropout0.05，448tokens、微批4/有效批量16。读取层[2,4,6,8,12,16,24,36]，每层mean pooling后LayerNorm，拼接32768维，由18类线性头分类。输入为回答文本，标签顺序保存在readouts.npz中。

推理需同时加载基础模型、LoRA、LayerNorm和分类头。三个历史检查点没有完整优化器/scheduler/RNG状态，可用于推理或权重初始化，不能宣称精确断点续训。现存trainer代码快照与历史manifest记录的源码哈希不完全相同，核对报告明确保留这一限制。已核对文件哈希、权重结构及保存的验证预测，本次归档没有重新执行GPU推理。

权重通过Release分发，避免普通Git仓库的大文件限制，见[GitHub文档](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。底座来源、文件版本和哈希记录在归档内的source_manifest.json；上游模型说明随底座保留。本项目没有将第三方模型或数据重新授权为自有作品。
