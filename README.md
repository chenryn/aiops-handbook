# AIOps 手册

英文版见 <README_en.md>。

AIOps 的论文、演讲、开源库的汇总手册。按照[《企业AIOps实施建议白皮书》](https://pic.huodongjia.com/ganhuodocs/2018-04-16/1523873064.74.pdf)中的场景分类进行收集和展示。

对于同一个场景，尽量提供比较新的链接。因为新论文里一般会引用和对比旧论文。

## 异常检测

### 指标

#### 单指标

* 清华裴丹团队举办的 KPI 异常检测比赛用数据(需注册登录) <http://iops.ai/competition_detail/?competition_id=5&flag=1>
    * 比赛前 5 名的分享：<http://workshop.aiops.org/>
    * 某博士生发的程序(未在参赛的 40 支队伍上看到名字哈)：<https://github.com/chengqianghuang/exp-anomaly-detector-AIOps>
    * 北邮某硕士论文(KPI 部分考虑标记异常区间，并加异常过滤)：<https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFD202101&filename=1021025248.nh&v=4bo62xCRybUZ6jmdkuL2wRQvfR0LRDN2TNkCZ1Og3VbUglRzjmact7Ot3k2Yf2vT>
* 清华/阿里巴巴开源的 Donut(基于 VAE 算法)：<https://github.com/haowen-xu/donut>
    * 清华开源的 Bagel(Donut改进型，基于 CVAE 算法)：<https://github.com/lizeyan/Bagel>
    * 基于 Donut 封装的开源项目 LoudML(支持从不同数据源自动获取数据做异常检测，RESTful 接口配置)：<https://github.com/regel/loudml>
* 清华/百度的 opprentice 系统(14 个检测器，平均取参数值，随机森林)：<http://netman.cs.tsinghua.edu.cn/wp-content/uploads/2015/11/liu_imc15_Opprentice.pdf>
* 清华/南开/腾讯的 ADS 系统(ROCKA+opprentice+CPLE)，论文：<https://netman.aiops.org/wp-content/uploads/2018/12/bujiahao.pdf>
    * github上一个开源的CPLE实现：<https://github.com/tmadl/semisup-learn>
* 腾讯开源的 metis 系统(参考了 opprentice 实现)：<https://github.com/tencent/metis>
* 阿里巴巴开源的Time2Graph(基于序列片段的图迁移路径做异常检测)：<https://github.com/petecheng/Time2Graph>
* skyline
    * etsy 开源版(9 个检测器，简单投票)：<https://github.com/etsy/skyline>
    * etsy 未开源版的介绍(小波分解后分别过 KS 和广义 ESD 检验)：<https://vimeo.com/131581331>
    * lytics 公司用 golang 重写的：<https://github.com/lytics/anomalyzer>
    * 社区版(加入 Ionosphere 模块做反馈修正，使用 tsfresh 库)：<https://github.com/earthgecko/skyline>
    * 360 公司开源的异常检测，和skyline一样简单投票，不过自己另写了几个EWMA、iForest、同环比等检测器：<https://github.com/jixinpu/aiopstools/tree/master/aiopstools/anomaly_detection>
* 开源的时序特征值提取库 tsfresh：<http://tsfresh.readthedocs.io/en/latest/>
* facebook 开源的时序数据处理库 kats，包括时序特征提取、模式检测、预测等功能：<https://github.com/facebookresearch/Kats>
* arundo 开源的 adtk 时序异常检测 python 库：<https://github.com/arundo/adtk>
* netflix基于 PCA 算法的异常检测，跑在 Pig 上：<https://github.com/netflix/surus>
* twitter 的异常检测库，R 语言：<https://github.com/twitter/anomalydetection>
* numenta 公司，HTM 算法：<https://github.com/numenta/nupic>
    * 顺带还做了一个项目专门用来比较效果(和裴丹比赛的评价标准不太一样，裴的标准是异常点往后 7 个都算；NAB 标准是异常区间内前一半算满分，后一半衰减)：<https://github.com/numenta/NAB>
    * 南京大学一位硕士论文的 windowKDE 实现(文中还实现了另外两个叫 RDE 和 TEDA 的检测器)，代码很简陋：<https://github.com/lyzhang0614/windowKDEdetector>
* 微软开源的 anomalydetector 项目(基于Spectral Residual算法，辅以 CNN，不过验证评估是对整个数据集所有指标训练一个大模型)：<https://github.com/microsoft/anomalydetector>
* 雅虎开源的时序预测和异常检测项目 EGADS：<https://github.com/yahoo/egads>
    * 对应解释论文的中文翻译版：<http://www.infoq.com/cn/articles/automated-time-series-anomaly-detection>
* 亿客行Expedia开源的异常检测项目 adaptive-alerting：<https://github.com/ExpediaDotCom/adaptive-alerting>
* RedHat公司CTO办公室开源的prometheus anomaly detector项目(基于傅里叶变换和Facebook的prophet预测算法)：<https://github.com/AICoE/prometheus-anomaly-detector>
* AWS 开源的 gluon-ts 项目，基于 MXNet 进行时序指标的概率模型训练，以此做预测和异常检测：<https://github.com/awslabs/gluon-ts/>
    * 以及 AWS 自己用 gluon-ts 实现云资源性能指标异常检测的论文：<http://export.arxiv.org/pdf/2007.15541>
* 华为爱尔兰研究中心发的 SLMAD 论文(对周期性数据直接做 Robust BoxPlot，非周期性的做 Matrix Profile，但文中没说 MP 的 window size 如何定)：<https://www.researchgate.net/publication/344378625_SLMAD_Statistical_Learning-Based_Metric_Anomaly_Detection>
    * Matrix Profile 本身也是一个比较新的时序分析算法，支持流式更新，国内介绍较少，对应的开源项目 STUMPY 官方文档见：<https://stumpy.readthedocs.io/en/latest/Tutorial_The_Matrix_Profile.html> 

#### 多指标

* CMU 做多指标模式提取和异常检测的 SPIRIT 系统，论文：<https://bitquill.net/pdf/spirit_vldb05.pdf>
* 清华裴丹团队做的多指标聚类 ROCKA 系统，论文：<https://netman.aiops.org/~peidan/ANM2018/8.DependencyDiscovery/LectureCoverage/2018IWQOS_ROCKA.pdf>
* 清华裴丹团队做的多指标异常检测 OmniAnomaly 开源实现，主要是对同一对象的多个指标，采用和单指标Donut类似的方法：<https://github.com/NetManAIOps/OmniAnomaly>
* 微软亚研做的多指标聚类 YADING 系统，论文：<https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/p457-ding.pdf>
* 北卡顾晓晖团队做云主机异常检测和根因定位的 UBL 系统(基于 SOM 算法)，论文：<http://dance.csc.ncsu.edu/papers/UBL.pdf> 
* 新加坡国立大学做传感器多变量指标异常检测的开源项目(基于 GAN 算法)：<https://github.com/LiDan456/MAD-GANs>

### 日志

* 国防科大的日志领域研究综述(日志监测部分比我前面列的老，但还提了基于源码的静态分析和基于虚拟机增强的日志内容改进两个方向，基本都是袁丁教授团队做的)：<http://www.jos.org.cn/1000-9825/4936.htm>
    * morningpaper 博客关于静态分析的 lprof 论文解析：<https://blog.acolyer.org/2015/10/08/lprof-a-non-intrusive-request-flow-profiler-for-distributed-systems/>
    * morningpaper 博客关于日志增强的 log20 论文解析：<https://blog.acolyer.org/2017/11/03/log20-fully-automated-optimal-placement-of-log-printing-statements-under-specified-overhead-threshold/>
* 香港中文大学的日志领域研究综述(比国防科大的新，加入了关于日志压缩、人机交互、语义等新方向)：<https://arxiv.org/pdf/2009.07237.pdf>
* 荷兰代尔夫特理工大学的日志领域研究综述(2021 年，统计了不同方向的研究趋势)：<https://pdfs.semanticscholar.org/b3c1/e91f3f73ff1d63504fb8d522558baa7334d4.pdf?_ga=2.256964171.1591127296.1641452870-511869175.1640757218>
* 加拿大滑铁卢大学的日志领域研究综述(2022 年，总结了各方向各算法的优劣)：<https://arxiv.org/pdf/2110.12489.pdf>
* 香港中文大学团队收集的多篇日志异常检测相关论文和数据集(共87GB)：<https://github.com/logpai/loghub>
    * 他们也做了各种现有算法的开源实现和自己的 Drain 算法进行横向测试对比，报告见：<https://arxiv.org/pdf/1811.03509.pdf>
    * 华为开源的 [NuLog](https://jorge-cardoso.github.io/publications/Papers/CP-2020-094-ICDM_Self_Attentive_Classification_Based_Anomaly_Detection.pdf) 项目(采用MLM掩码语言模型，并复现了上一篇论文一样的对比)：<https://github.com/nulog/nulog>
    * IBM云数据中心团队改进和开源的 Drain3 包，加强了持久化，自定义参数替换等：<https://github.com/IBM/Drain3>
    * IBM 在 Drain3 基础上，通过公开文档爬虫获取事件 ID 的关键字描述，然后走语义分析相似度，来提取复杂变量类型(即除了常量、变量以外，新定义了sequential、optional 和 single-select 类型)：<https://arxiv.org/pdf/2202.07169.pdf>
    * 上海交通大学采用日志中的 punct 部分作为日志模式学习的来源，实现了一个 logpunk 系统，在 loghub 下对比，效果居然也好过其他算法：<https://www.mdpi.com/2076-3417/11/24/11974/pdf>
    * 微软的 UniParser 论文，通过语义分析，识别训练集中某些常量为变量：<https://arxiv.org/pdf/2202.06569.pdf>
* 斯里兰卡莫拉图瓦大学/WSO2 公司的 vue4logs-parser 开源实现，直接利用倒排索引搜索相关性来完成模式过滤：<https://github.com/IsuruBoyagane15/vue4logs-parser>
* IBM 研究院基于语言模型做的日志异常检测模型，对比了 fasttext 和 BERT 的效果：<https://www.researchgate.net/publication/344693315_Using_Language_Models_to_Pre-train_Features_for_Optimizing_Information_Technology_Operations_Management_Tasks>
* 香港中文大学的 LogZip 开源实现：<https://github.com/logpai/logzip>
    * 清华/阿里的 LogReducer 系统(用 C/C++ 重写了 logzip，并加上对特定数值型参数值的差分、关联和变长压缩优化)，论文：<https://www.usenix.org/system/files/fast21-wei.pdf>
    * 匈牙利罗兰大学的改进，主要在内存消耗上领先，论文：<https://www.mdpi.com/2076-3417/12/4/2044/pdf>
* 香港中文大学的 SemParser 论文，尝试用语义分析来命名模式中的参数位：<https://arxiv.org/pdf/2112.12636.pdf>
* DeepLog 论文(包含模式检测、参数检测、工作流检测三部分)：<https://acmccs.github.io/papers/p1285-duA.pdf>
    * 开源实现：<https://github.com/wuyifan18/DeepLog>
    * 另一个开源实现，还实现了另外两种算法[LogAnomaly](https://www.ijcai.org/Proceedings/2019/658)和[RobustLog](https://dl.acm.org/doi/10.1145/3338906.3338931)，可切换：<https://github.com/donglee-afar/logdeep>
* 中山大学的 SwissLog 论文，和 RobustLog 一样关注模型的鲁棒性问题：<https://www.researchgate.net/publication/346867203_SwissLog_Robust_and_Unified_Deep_Learning_Based_Log_Anomaly_Detection_for_Diverse_Faults>
* 清华/南开/腾讯的 FT-tree 开源实现：<https://github.com/WeibinMeng/ft-tree>
* 清华/南开/百度的 LogClass 开源实现：<https://github.com/NetManAIOps/LogClass>
* 北卡顾晓晖团队做日志异常检测的 ELT 系统(拆分为粗粒度的 MAV 和细粒度的 MFG 两层)：<http://dance.csc.ncsu.edu/papers/srds11.pdf>
* NEC 美国实验室/北卡做云系统工作流监控的 CloudSeer 系统：<https://people.engr.ncsu.edu/gjin2/Classes/591/Spring2017/case-cloudseer.pdf>
* NEC 美国实验室 LogMine 系统：<http://www.cs.unm.edu/~mueen/Papers/LogMine.pdf>
   * 开源实现：<https://github.com/trungdq88/logmine>
* NEC 美国实验室/蚂蚁金服做的 LogLens 系统(在 LogMine 基础上，和 ELK 的 Grok 设计结合；并加上了对 traceid 的判断处理，支持序列异常检测)，论文：<http://120.52.51.14/www.cs.ucsb.edu/~bzong/doc/icdcs-18.pdf>
* 香港中文大学/华为的 POP 系统(和 LogMine 思路比较类似，在 Spark 上运行)：<http://www.cse.cuhk.edu.hk/lyu/_media/journal/pjhe_tdsc18.pdf>
* 康考迪亚大学发表的 logram 论文(用 n-gram 来做日志解析)：<https://petertsehsun.github.io/papers/HetongTSE2020.pdf>
* 康考迪亚大学发表的 LogAssist 论文，用 n-gram 对已提取的日志模板序列做二次压缩，并用案例研究法对比效果：<https://petertsehsun.github.io/papers/TSE2021_LogAssist.pdf>
* 加拿大麦克马斯特大学的日志序列异常检测开源实现，加上序列每一步 duration 子序列做神经网络特征：<https://github.com/hfyxin/Ts-models-log-data-analysis>
* RedHat公司CTO办公室开源的Log Anomaly Detector项目(基于word2vec和SOM算法)：<https://github.com/AICoE/log-anomaly-detector>
* 其他商业公司：
    * Loomsystems(已被 serviceNow 收购，其对参数类型的 meter/gauge/timeless-gauge/histogram/invalid/root-cause 分类值得借鉴)：<https://www.loomsystems.com/hubfs/SophieTechnicalOverview.pdf>
    * coralogix(有基础的无关顺序的关联模式检测，对 XML/JSON 类型进行对象参数检测)：<https://coralogix.com/tutorials/what-is-coralogix-pattern-anomaly/>
    * zebrium(存 newsql，参数名称的自动识别值得借鉴，最后用 GPT-3 生成告警描述也很有趣)：<https://www.zebrium.com/blog/using-ml-to-auto-learn-changing-log-structures>

## 标注

### 指标异常标注

* 百度开源的指标异常标注工具：<https://github.com/baidu/Curve>
* 微软开源的指标异常工具：<https://github.com/Microsoft/TagAnomaly>
* 清华/建行做的指标批量标注 Label-less 项目(基于 iForest 异常检测和 DTW 相似度学习)：<https://netman.aiops.org/wp-content/uploads/2019/10/Label-less-v2.pdf>

## 预测

### 单指标

* 脸书开源，时序预测：<https://github.com/facebook/prophet>
* 红帽开源，是 hawkular(已被 jaeger 取代)项目的一部分，在 ARIMA 基础上做了自动调参：<https://github.com/hawkular/hawkular-datamining>
* 360 开源，封装了LR、ARIMA、LSTM等通用算法做预测：<https://github.com/jixinpu/aiopstools/tree/master/aiopstools/timeseries_predict>
* 北卡顾晓晖团队做监控系统数据传输压缩的论文(先聚类并下发预测模型，agent上预测无偏差就不上报了)：<http://dance.csc.ncsu.edu/papers/ICAC09.pdf>

### 容量规划

* 谷歌 某 12.5k 规模集群的任务资源和机器数据：<https://github.com/google/cluster-data>
* 微软 Azure 云某区的云主机启停、 CPU 和内存时序数据：<https://github.com/Azure/AzurePublicDataset>
* 阿里云某 1.3k 规模集群的云主机、容器和后台任务的时序数据：<https://github.com/alibaba/clusterdata>
   * 使用该数据集做的天池调度算法大赛第6名"地球漫步"的开源版本：<https://github.com/NeuronEmpire/aliyun_schedule_semi>
* Trulia 开源，根据查询语句预测 Solr 集群的查询性能并调度：<https://github.com/trulia/thoth>
* CMU 开源的关系型数据库自动调参工具 ottertune：<https://github.com/cmu-db/ottertune>
* PingCAP 仿作的 TiKV 自动调参工具：https://github.com/tikv/auto-tikv
* 中山大学发的 Elasticsearch 自动调参研究：<https://ieeexplore.ieee.org/ielx7/6287639/8948470/09079492.pdf>
* 康考迪亚大学发的 Kafka 容量预测研究(前半段过程和自动调参类似，但是目的是得到模型后做参数变化下的预测，所以主要对比的是 XGBoost、LR、RF、MLP 的区别)：<https://github.com/SPEAR-SE/mlasp>

### 网络

* 南开张圣林，交换机 syslog 故障预测：<http://workshop.aiops.org/files/shenglinzhang2018prefix.pdf>

### 事件关联挖掘

* 微软亚研的 Log3C，日志和 KPI 的关联挖掘：<https://github.com/logpai/Log3C>
* 微软和吉林大学的论文：<http://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/SIGKDD-2014-Correlating-Events-with-Time-Series-for-Incident-Diagnosis.pdf>
    * 360 公司按照该论文的开源实现：<https://github.com/jixinpu/aiopstools/tree/master/aiopstools/association_analysis>
* IBM研究院的论文，利用微服务错误指标和错误日志，采用因果推理算法和个性化 PageRank 算法，进行故障定位(文章主要目的是引入个性化 PR，因果推理这方面没区分 PC 和回归有什么差异)：<https://www.researchgate.net/publication/344435606_Localization_of_Operational_Faults_in_Cloud_Applications_by_Mining_Causal_Dependencies_in_Logs_using_Golden_Signals>


## 根因分析

### 调用链

* 开源 APM/tracing 实现：
   * zipkin/brave：<https://github.com/openzipkin/brave>
   * springcloud/sleuth：<https://github.com/spring-cloud/spring-cloud-sleuth>
   * skywalking：<https://skywalking.apache.org/>
   * jaeger：<https://github.com/jaegertracing/jaeger>
   * pinpoint：<https://github.com/pinpoint-apm/pinpoint>
   * elastic apm：<https://github.com/elastic/apm>
   * datadog apm：<https://github.com/DataDog/dd-trace-java>
   * opencensus/opentelemetry：<https://opentelemetry.io/>
   * cilium/hubble：<https://github.com/cilium/hubble>
   * pixie/stirling：<https://github.com/pixie-labs/pixie/tree/main/src/stirling>
* 萨尔布吕肯大学Jonathan Mace，利用层次聚类尽量避免采样时丢失罕见个例：<https://people.mpi-sws.org/~jcmace/papers/lascasas2018weighted.pdf>
* 谷歌开源的微服务测试床：<https://github.com/GoogleCloudPlatform/microservices-demo>
* 复旦大学开源的大规模微服务测试床：<https://github.com/FudanSELab/train-ticket/>
* 清华裴丹团队举办的调用链根因定位比赛：<http://iops.ai/competition_detail/?competition_id=15&flag=1>
    * 华南理工针对该比赛发的论文：<https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9293310>
* 意大利比萨大学关于微服务环境下异常检测和根因分析的综述论文：<https://arxiv.org/pdf/2105.12378.pdf>

### 多维定位

* 清华/百度的HotSpot论文，有多维属性的 KPI 故障范围分析：<http://netman.ai/wp-content/uploads/2018/03/sunyq_IEEEAccess_HotSpot.pdf>
* 清华/建行的FluxRank论文，在服务级故障时，快速定位到小范围的主机层问题并重启：<https://netman.aiops.org/wp-content/uploads/2019/08/liuping-camera-ready.pdf>
* 里昂国立应用科学学院的论文，采用子群发现算法，综合 SQL 解析、环境版本、告警、指标进行 SQL 慢查询定位：<https://www.researchgate.net/publication/353776691_What_makes_my_queries_slow_Subgroup_Discovery_for_SQL_Workload_Analysis>

### 时序相关性分析

* etsy 开源版，基于 elasticsearch 实现的 fastDTW 时序相关性排序：<https://github.com/etsy/oculus>
* linkedin 开源的基于 SAX 的异常检测和相关性计算库：<https://github.com/linkedin/luminol>
    * 此外，还有一个完整系统的介绍分享：<https://docs.google.com/presentation/d/1DWMNgoAtxuK8ZbFJOpq5vt3dEz5_4ptMuLtoTUjQ_ro/pub?start=false&loop=false&delayms=3000&slide=id.p>
* Uber 公司的 Argos 系统，只有介绍文章：<https://eng.uber.com/argos/>

### 解决方案相关性推荐

* 佛罗里达国际大学/IBM的论文，基于 CNN 做工单的关联和推荐，重点在如何过滤和提取工单文本的有效特征值：<https://www.researchgate.net/publication/318373831_STAR_A_System_for_Ticket_Analysis_and_Resolution>

## 告警归并

* 360 开源，基于 Apriori 算法：<https://github.com/jixinpu/aiopstools/blob/master/examples/alarm_convergence.py>
* 里昂国立应用科学学院的 SplitSD4X，采用子群发现算法，配合 NLP 解析事故报告文本，生成同类故障描述：<https://github.com/RemilYoucef/split-sd4x>
* anodot 公司论文，第三部分，利用 SAE 和 LDA 做 KPI 和告警的拓扑：<http://proceedings.mlr.press/v71/toledano18a/toledano18a.pdf>
* 其他商业公司：
    * [moogsoft](https://docs.moogsoft.com/) 公司(专门做告警处理的AIOps创业公司)有关技术实现的演讲：<https://docs.google.com/presentation/d/1F-8eop-9ffCpX4trOJS28FXATAFlQkgkfnDV_Zqnkuo/edit?pref=2&pli=1#slide=id.g990524d96_0_0>

## 图谱

* 清华大学徐葳团队，状态图谱解决 OpenStack 问题：<http://iiis.tsinghua.edu.cn/~weixu/files/apsys-yong-slides.pdf>
* 徐葳早年论文，用状态图来辅助开源项目更好的修改 logging 代码：<http://iiis.tsinghua.edu.cn/~weixu/files/slaml10-rabkin.pdf>
* 华为开源的 [OpenStack 可观测性数据](https://jorge-cardoso.github.io/publications/Papers/CP-2020-093-Multi-source_Distributed_System_Data.pdf)样本集，trace 部分用的osprofiler 采集：<https://zenodo.org/record/3549604#.YE8Q-eHithE>
* 宜信张真的演讲：[WOT2018 -张真-运维机器人之任务决策系统演讲](https://pan.baidu.com/s/1gSjJZIXswOPoeQzZ6cJT1g?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=)
* CA/加泰罗尼亚理工大学，用日志和指标构建的基于图谱的微服务根因分析系统：<https://www.researchgate.net/publication/336585890_Graph-based_Root_Cause_Analysis_for_Service-Oriented_and_Microservice_Architectures>
* 微众银行的智能运维系列分享第八篇(给了非常详细的节点属性和边属性设计，比 CA 的更细节)：[事件指纹库：构建异常案例的“博物馆”](https://mp.weixin.qq.com/s/M8tcS8q6sPPRRebAJkrb7Q)
* 中山大学融合了拓扑的指标异常检测系统 TopoMAD，开源的数据样本：<https://github.com/QAZASDEDC/TopoMAD>
    * 网络另一篇针对 TopoMAD 论文的解析和评论文章，比较犀利：<https://dreamhomes.top/posts/202103111131.html>

## 行为异常

* 北卡顾晓晖团队做程序行为异常分析的 PerfScope 系统(利用 LTTng 在线采集系统调用，配合 Apriori 算法)，论文：<http://dance.csc.ncsu.edu/papers/socc14.pdf>
* ee-outliers 开源项目，利用 word2vec 做 osquery 输出的事件异常检测：<https://github.com/NVISO-BE/ee-outliers>
* ubad 开源项目，利用 LSTM 和 OCSVM 做 osquery 输出的用户行为事件异常检测：<https://github.com/morrigan/user-behavior-anomaly-detector>
* 阿里云天池算法大赛扫描爆破部分，第10名的开源介绍：<https://github.com/wufanyou/aliyun_safety>
* Security Repo数据集，Mike Sconzo收集的各种和安全运维有关的数据：<https://www.secrepo.com/>
* 奥地利 AIT 的安全日志数据集：<https://zenodo.org/record/5789064#.YkFnZWJBxhE> 及其数据生成的测试床项目：<https://github.com/ait-aecid/kyoushi-environment>
* 其他商业公司：
    * ExtraHop：<https://docs.extrahop.com/current/>

## 扩展阅读

* @linjinjin123 的 [awesome-AIOps](https://github.com/linjinjin123/awesome-AIOps) 库，但我认为运维人员可能更需要的是一个从实用场景出发的归类。
* @zhuyiche 的 [awesome-anomaly-detection](https://github.com/zhuyiche/awesome-anomaly-detection) 库，专精在异常检测领域的论文收集。
* @logpai 的 [log-survey](https://github.com/logpai/log-survey) 库，专精在日志领域的论文和研究者收集，不光包括分析，还包括生成增强、压缩等。
* @jorge-cardoso 对 AIOps 领域的系统映射研究：<https://jorge-cardoso.github.io/publications/Papers/WP-2020-079-AIOPS2020_A_Systematic_Mapping_Study_in_AIOps.pdf>
* @jorge-cardoso 对 AIOps 故障大类别的研究综述：<https://jorge-cardoso.github.io/publications/Papers/JA-2021-025-Survey_AIOps_Methods_for_Failure_Management.pdf>
* 中国移动信息技术有限公司关于自动化运维和智能运维的成熟度评估模型介绍：<https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2021&filename=DGJB202101009&v=4TGdKbAr7slYCfXAgxJfw2rHqvDI%25mmd2FLQ8iARm627FUgNvt7c9lvfHVIbYjAz8eR2x>
