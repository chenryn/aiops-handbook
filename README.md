# AIOps 手册

AIOps 的论文、演讲、开源库的汇总手册。按照[《企业AIOps实施建议白皮书》](https://www.rizhiyi.com/assets/docs/AIOps.pdf)中的场景分类进行收集和展示。

## 异常检测

### 指标

#### 单指标

* 清华裴丹团队举办的 KPI 异常检测比赛用数据(需注册登录) <http://iops.ai/competition_detail/?competition_id=5&flag=1>
** 比赛前 5 名的分享：<http://workshop.aiops.org/>
** 某博士生发的程序(未在参赛的 40 支队伍上看到名字哈)：<https://github.com/chengqianghuang/exp-anomaly-detector-AIOps>
* 阿里巴巴用 VAE 做 KPI 异常检测：<http://netman.ai/wp-content/uploads/2018/03/www2018.pdf>
* skyline
** etsy 开源版(9 个检测器，简单投票)：<https://github.com/etsy/skyline>
** etsy 未开源版的介绍(小波分解后分别过 KS 和广发 ESD 检验)：<https://vimeo.com/131581331>
** lytics 公司用 golang 重写的：<https://github.com/lytics/anomalyzer>
** 社区版(加入Ionosphere模块做反馈修正，使用了 tsfresh 库)：<https://github.com/earthgecko/skyline>
* 开源的时序特征值提取库 tsfresh：<http://tsfresh.readthedocs.io/en/latest/>
* netflix基于 PCA 算法的异常检测，跑在 Pig 上：<https://github.com/netflix/surus>
* twitter 的异常检测库，R 语言：<https://github.com/twitter/anomalydetection>
* numenta 公司，HTM 算法：<https://github.com/numenta/nupic>
** 顺带还做了一个项目专门用来比较效果(和裴丹比赛的评价标准不太一样，裴的标准是异常点往后 7 个都算；NAB 标准是异常区间内前一半算满分，后一半衰减)：<https://github.com/numenta/NAB>
* 雅虎开源的时序预测和异常检测项目 EGADS：<https://github.com/yahoo/egads>
* 百度的 opprentice系统(14个检测器，平均取参数值，随机森林)：<http://netman.cs.tsinghua.edu.cn/wp-content/uploads/2015/11/liu_imc15_Opprentice.pdf>

#### 多指标

* CMU 做多指标模式提取和异常检测的 SPIRIT 系统，论文：<https://bitquill.net/pdf/spirit_vldb05.pdf>

### 日志
* 香港中文大学团队收集的多篇日志异常检测相关论文和数据集(共87GB)：<https://github.com/logpai/loghub>
* DeepLog 论文(包含模式检测、参数检测、工作流检测三部分)：<https://acmccs.github.io/papers/p1285-duA.pdf>
* NEC 美国实验室 LogMine 系统：<http://www.cs.unm.edu/~mueen/Papers/LogMine.pdf>

### 磁盘

## 预测

### 单指标

* 脸书开源，时序预测：<https://github.com/facebook/prophet>
* 红帽开源，是 hawkular(已被 jaeger 取代)项目的一部分，在 ARIMA 基础上做了自动调参：<https://github.com/hawkular/hawkular-datamining>

### 容量规划

* 谷歌 某 12.5k 规模集群的任务资源和机器数据：<https://github.com/google/cluster-data>
* 微软 Azure 云某区的云主机启停、 CPU 和内存时序数据：<https://github.com/Azure/AzurePublicDataset>
* 阿里云某 1.3k 规模集群的云主机、容器和后台任务的时序数据：<https://github.com/alibaba/clusterdata>
* Trulia 开源，根据查询语句预测 Solr 集群的查询性能并调度：<https://github.com/trulia/thoth>

### 网络

* 南开张圣林，交换机 syslog 故障预测：<http://workshop.aiops.org/files/shenglinzhang2018prefix.pdf>

### 事件关联挖掘

* 微软亚研的 Log3C，日志和 KPI 的关联挖掘：<https://github.com/logpai/Log3C>

## 根因分析

### 调用链

这块没啥好说的，属于很有用，但比拼的不是 AI：

* zipkin
* skywalking
* pinpoint

### 瓶颈分析

* 多维属性的 KPI 瓶颈分析：<http://netman.ai/wp-content/uploads/2018/03/sunyq_IEEEAccess_HotSpot.pdf>

### 时序相关性

* etsy 开源版，基于 elasticsearch 实现的 fastDTW 时序相关性排序：<https://github.com/etsy/oculus>
* linkedin 开源的基于 SAX 的异常检测和相关性计算库：<https://github.com/linkedin/luminol>
** 此外，还有一个完整系统的介绍分享：<https://docs.google.com/presentation/d/1DWMNgoAtxuK8ZbFJOpq5vt3dEz5_4ptMuLtoTUjQ_ro/pub?start=false&loop=false&delayms=3000&slide=id.p>
* netflix 公司的 Argos 系统，只有介绍文章：<https://eng.uber.com/argos/>

## 告警归并

* anodot 公司论文，第三部分，利用 SAE 和 LDA 做 KPI 和告警的拓扑：<http://proceedings.mlr.press/v71/toledano18a/toledano18a.pdf>

## 图谱

* 清华大学徐葳团队，状态图谱解决 OpenStack 问题：<http://iiis.tsinghua.edu.cn/~weixu/files/apsys-yong-slides.pdf>
* 徐葳早年论文，用状态图来辅助开源项目更好的修改 logging 代码：<http://iiis.tsinghua.edu.cn/~weixu/files/slaml10-rabkin.pdf>

## 感谢

* 感谢林锦进的 [awesome-AIOps](https://github.com/linjinjin123/awesome-AIOps) 库，但我认为运维人员可能更需要的是一个从实用场景出发的归类。
