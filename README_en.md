# aiops-handbook

Collection of slides, repositories, papers about AIOps, sort by the scenes which were advanced by the [<WhitePaper of Enterprise AIOps Recommendation>](https://pic.huodongjia.com/ganhuodocs/2018-04-16/1523873064.74.pdf).

Chinese edition: <README.md>.

## Anomaly Detection

### Metric

#### single

* competiton of KPI anomalydetection(need register), hosted by peidan, THU: <http://iops.ai/competition_detail/?competition_id=5&flag=1>
    * shared slides of top 5 teams: <http://workshop.aiops.org/>
    * another program by an unknown PhD student: <https://github.com/chengqianghuang/exp-anomaly-detector-AIOps>
    * Master's thesis from Beijing University of Posts and Telecommunications (considering labeling anomaly intervals and anomaly filtering for KPIs): <https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFD202101&filename=1021025248.nh&v=4bo62xCRybUZ6jmdkuL2wRQvfR0LRDN2TNkCZ1Og3VbUglRzjmact7Ot3k2Yf2vT>
* Donut(VAE-based algorithm), opensourced by THU/alibaba: <https://github.com/haowen-xu/donut>
    * Bagel (an improved version of Donut based on CVAE algorithm) from Tsinghua University: <https://github.com/lizeyan/Bagel>
    * LoudML (an open-source project encapsulating Donut, supporting automatic data acquisition from different sources, RESTful API configuration): <https://github.com/regel/loudml>
* Opprentice(14 detector+randomforest), published by THU/baidu: <http://netman.cs.tsinghua.edu.cn/wp-content/uploads/2015/11/liu_imc15_Opprentice.pdf>
* ADS(ROCKA+opprentice+CPLE), published by THU/NKU/baidu: <https://netman.aiops.org/wp-content/uploads/2018/12/bujiahao.pdf>
    * a CPLE implement opensourced in github: <https://github.com/tmadl/semisup-learn>
* metis(inspired by opprentice), opensourced by tencent: <https://github.com/tencent/metis>
* Time2Graph (anomaly detection based on sequence fragment graph transition paths) from Alibaba: <https://github.com/petecheng/Time2Graph>
* skyline
    * version 1(9 detector+vote threshold), opensourced(but archived) by etsy: <https://github.com/etsy/skyline>
    * version 2(wavelet+ks test+generalized ESD test) slide, published by etsy: <https://vimeo.com/131581331>
    * golang version, opensourced by lytics: <https://github.com/lytics/anomalyzer>
    * community version(a new Ionosphere module use tsfresh to learn what's no anomalous) opensourced by earthgecko: <https://github.com/earthgecko/skyline>
    * Anomaly detection open-sourced by 360 Company, similar to Skyline with simple voting, but with additional detectors like EWMA, iForest, and year-over-year:<https://github.com/jixinpu/aiopstools/tree/master/aiopstools/anomaly_detection>
* tsfresh, an open source tool for calculating a large number of time series features: <http://tsfresh.readthedocs.io/en/latest/>
* Kats, an open-source time series processing library from Facebook, including time series feature extraction, pattern detection, forecasting, etc.: <https://github.com/facebookresearch/Kats>
* Adtk, an open-source Python library for time series anomaly detection from arundo: <https://github.com/arundo/adtk>
* Surus(PCA-based, running on Pig), opensourced by netflix: <https://github.com/netflix/surus>
* anomalydetection(generalized ESD, written in R), opensourced by twitter: <https://github.com/twitter/anomalydetection>
* NuPIC(HTM algorithm), opensourced by numenta: <https://github.com/numenta/nupic>
    * NAB competiton(with a curved scoring, by contrast, peidan's competiton give a static score within 7 points after the anmaly window start), hosted by numenta: <https://github.com/numenta/NAB>
    * A crude implementation of windowKDE (along with two other detectors called RDE and TEDA) from a master's thesis at Nanjing University: <https://github.com/lyzhang0614/windowKDEdetector>
* Anomalydetector open-source project from Microsoft (based on Spectral Residual algorithm, aided by CNN, but validation and evaluation are done by training a single large model on the entire dataset of all metrics): <https://github.com/microsoft/anomalydetector>
* EGADS(adboost-based algorithm), opensourced by Yahoo!: <https://github.com/yahoo/egads>
* adaptive-alerting(with some very good wiki articles), opensourced by Expedia: <https://github.com/ExpediaDotCom/adaptive-alerting>
* Prometheus anomaly detector open-source project from RedHat CTO Office (based on Fourier transform and Facebook's prophet forecasting algorithm): <https://github.com/AICoE/prometheus-anomaly-detector>
* Gluon-ts, an open-source project from AWS, using MXNet for probabilistic model training on time series metrics for forecasting and anomaly detection: <https://github.com/awslabs/gluon-ts/>
* A paper on using gluon-ts for anomaly detection on cloud resource performance metrics at AWS: <http://export.arxiv.org/pdf/2007.15541>
* SLMAD paper (applying Robust BoxPlot directly to periodic data, and Matrix Profile for non-periodic data, but not mentioning how to determine the window size for MP) from Huawei Irish Research Center: <https://www.researchgate.net/publication/344378625_SLMAD_Statistical_Learning-Based_Metric_Anomaly_Detection>
* Matrix Profile itself is a relatively new time series analysis algorithm that supports streaming updates, with limited introduction in China. The official documentation for the corresponding open-source project STUMPY can be found at: <https://stumpy.readthedocs.io/en/latest/Tutorial_The_Matrix_Profile.html>
* Anomaly.io Blog: <https://anomaly.io/blog/>

#### multiple series

* SPIRIT(KPIs' pattern discovery, need similiarity score>0.9 ), published by CMU: <https://bitquill.net/pdf/spirit_vldb05.pdf>
* ROCKA, published by THU: <https://netman.aiops.org/~peidan/ANM2018/8.DependencyDiscovery/LectureCoverage/2018IWQOS_ROCKA.pdf>
* OmniAnomaly, an open-source implementation of multi-metric anomaly detection from Tsinghua University's Dan Pei team, mainly for multiple metrics of the same object, using a similar approach to the single-metric Donut: <https://github.com/NetManAIOps/OmniAnomaly>
* YADING system for multi-metric clustering from Microsoft Research Asia, paper: <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/p457-ding.pdf>
* UBL system for cloud host anomaly detection and root cause localization (based on SOM algorithm) from North Carolina State University's Xiaohui Gu team, paper: <http://dance.csc.ncsu.edu/papers/UBL.pdf>
* Open-source project for multi-variate sensor anomaly detection (based on GAN algorithm) from National University of Singapore: <https://github.com/LiDan456/MAD-GANs>
* Comparison of 71 different algorithms for multi/single metric anomaly detection (RUC performs poorly in most worst-case scenarios) from University of Potsdam, Germany: <https://hpi-information-systems.github.io/timeeval-evaluation-paper/>

#### Large Model Methods

* GPT4TS large model based on GPT2 from Tsinghua University: <https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All>. A pioneering work in this field, serving as a benchmark for future progress.
    * anomalyLLM from Zhejiang University: <https://arxiv.org/html/2401.15123v1>, distilling the GPT4TS large model into a smaller model.
    * aLLM4TS from HKPU/Tongji University: <https://arxiv.org/pdf/2402.04852.pdf>, modifying the structure of GPT4TS.
* MOMENT model and Time-Series Pile dataset from Carnegie Mellon University: <https://arxiv.org/pdf/2402.03885.pdf>. Analogous to the Pile dataset for large language models, it collects the 5 most commonly used metric datasets, covering single and multi-dimensional metrics for tasks like classification, long and short-term forecasting, and anomaly detection. Pre-trained moment-base, large, and small metric models in a manner similar to T5. The paper mainly compares baselines such as TimesNet and GPT4TS.
* TimesFM, an open-source time series forecasting base model from Google: <https://github.com/google-research/timesfm>

### Logdata

#### Traditional Methods

* A research review on log analysis (the log monitoring part is older than the ones listed earlier, but it also proposes two new directions: static analysis based on source code and log content enhancement based on virtual machines, mostly done by Professor Yuan Ding's team) from National University of Defense Technology: <http://www.jos.org.cn/1000-9825/4936.htm>
    * morningpaper blog's analysis of the lprof paper on static analysis: <https://blog.acolyer.org/2015/10/08/lprof-a-non-intrusive-request-flow-profiler-for-distributed-systems/>
    * morningpaper blog's analysis of the log20 paper on log enhancement: <https://blog.acolyer.org/2017/11/03/log20-fully-automated-optimal-placement-of-log-printing-statements-under-specified-overhead-threshold/>
* A research review on log analysis (newer than the one from National University of Defense Technology, incorporating new directions such as log compression, human-computer interaction, and semantics) from The Chinese University of Hong Kong: <https://arxiv.org/pdf/2009.07237.pdf>
* A research review on log analysis (2021, statistics on research trends in different directions) from Delft University of Technology, Netherlands: <https://pdfs.semanticscholar.org/b3c1/e91f3f73ff1d63504fb8d522558baa7334d4.pdf>
* A research review on log analysis (2022, summarizing the pros and cons of various algorithms in different directions) from University of Waterloo, Canada: <https://arxiv.org/pdf/2110.12489.pdf>
* loghub(a collection of system log datasets for intelligent log analysis, 77GB in total), opensourced by CUHK: <https://github.com/logpai/loghub>
    * They also provide open-source implementations of various existing algorithms and their own Drain algorithm for horizontal comparison, report: <https://arxiv.org/pdf/1811.03509.pdf>
For multi-line logs, especially tabular and key-value multi-line logs from middleware applications, they upgraded the dataset and introduced new algorithms, corresponding project: <https://github.com/logpai/hybridlog>
    * NuLog, an open-source project from Huawei (adopting the MLM masked language model, and reproducing the comparison from the previous paper): <https://github.com/nulog/nulog>
    * ADLILog, an open-source project from Huawei German Research Center, crawling static text from the source code of the 1000 most active open-source projects on GitHub that output logs, to serve as a corpus: <https://github.com/ADLILog/ADLILog>
    * Drain3 package improved and open-sourced by the IBM Cloud Data Center team, enhancing persistence and custom parameter replacement: <https://github.com/IBM/Drain3>. IBM's work on top of Drain3, using a public document crawler to obtain keyword descriptions of event IDs, then performing semantic similarity analysis to extract complex variable types (i.e., defining sequential, optional, and single-select types in addition to constants and variables): <https://arxiv.org/pdf/2202.07169.pdf>
    * LogPunk system from Shanghai Jiao Tong University, using the "punct" part of logs as the source for log pattern learning, outperforming other algorithms on the loghub dataset: <https://www.mdpi.com/2076-3417/11/24/11974/pdf>
* Updated log anomaly detection dataset and related implementation evaluation from The Chinese University of Hong Kong: <https://github.com/logpai/LogPub>. Unlike loghub, where only 2k logs per type are labeled, LogPub has all logs manually labeled. Additionally, due to the time gap, the evaluation also includes GPU-dependent solutions like UniParser and LogPPT. The evaluation criteria also consider factors such as frequency bias in templates and log volumes, training time, etc.
* UniParser paper from Microsoft, using semantic analysis to identify constants as variables in the training set: <https://arxiv.org/pdf/2202.06569.pdf>
* SemParser paper from The Chinese University of Hong Kong, attempting to use semantic analysis to name parameter positions in patterns: <https://arxiv.org/pdf/2112.12636.pdf>
* Open-source implementation of categorize_text aggregation from Elasticsearch, using the open-source dictionary SCOWL for part-of-speech analysis and weighting verbs: <https://github.com/elastic/elasticsearch/pull/80867>
* vue4logs-parser open-source implementation from University of Moratuwa, Sri Lanka/WSO2 Company, directly using inverted index search for relevance to complete pattern filtering: <https://github.com/IsuruBoyagane15/vue4logs-parser>
* LoganMeta from Ericsson Research, adopting meta learning algorithm, but in a supervised manner: <https://arxiv.org/pdf/2212.10992.pdf>
* LogReducer open-source project from Sun Yat-sen University/Tencent, using eBPF technology to analyze hot spots in log output and encourage developers to improve logging: <https://github.com/IntelligentDDS/LogReducer>
* Open-source implementation of CLP from University of Toronto: <https://github.com/y-scope/clp, Uber has already used this technology to process its Spark platform logs, official blog post: <https://www.uber.com/en-US/blog/reducing-logging-cost-by-two-orders-of-magnitude-using-clp>
    *  LogZip open-source implementation from The Chinese University of Hong Kong: <https://github.com/logpai/logzip>
* LogReducer system from Tsinghua University/Alibaba (a C/C++ rewrite of LogZip with additional optimizations like differencing, association, and variable-length compression for specific numeric parameter values), paper: <https://www.usenix.org/system/files/fast21-wei.pdf>
    * open-source implementation from Sun Yat-sen University/Tencent WeChat, providing feedback to online eBPF filters through offline analysis of inefficient logs. The paper also offers interesting insights into WeChat's current status, such as incorrect logging and forgotten test log deletion accounting for half of invalid logs, amounting to several PBs per day. WeChat currently uses promtail+loki+clickhouse for logging: <https://github.com/IntelligentDDS/LogReducer>
* LogGrep open-source implementation from Tsinghua University/Alibaba, redesigning the compression of parameter values based on LogReducer and CLP, enabling querying without decompression. In summary, it sacrifices some write CPU for disk space and query CPU. Paper: <https://web.cse.ohio-state.edu/~wang.7564/papers/eurosys23-final39.pdf>
* An improvement from University of Roland, Hungary, excelling in memory consumption, paper: <https://www.mdpi.com/2076-3417/12/4/2044/pdf>
* DeepLog(DL, anomalydetection for log keys, parameters, workflows), published by CUHK/MSRA: <https://acmccs.github.io/papers/p1285-duA.pdf>
    * Open-source implementation: <https://github.com/wuyifan18/DeepLog>
    * Another open-source implementation, also implementing two other algorithms LogAnomaly and RobustLog, with switching capability: <https://github.com/donglee-afar/logdeep>
* SwissLog paper from Sun Yat-sen University, focusing on model robustness like RobustLog: <https://www.researchgate.net/publication/346867203_SwissLog_Robust_and_Unified_Deep_Learning_Based_Log_Anomaly_Detection_for_Diverse_Faults>
* FT-tree(frequent words as log template), published by THU/NKU/tecent: <http://nkcs.iops.ai/wp-content/uploads/2018/06/paper-iwqos17-Syslog-Processing-for-Switch-Failure-Diagnosis-and-Prediction-in-Datacenter-Networks.pdf>
* LogClass(bag-of-word+PU learning, anomaly detection and classification), published by THU/NKU/baidu: <http://nkcs.iops.ai/wp-content/uploads/2018/06/paper-IWQOS2018-Device_Agnostic_Log_Anomaly_Classification.pdf>
* ELT system for log anomaly detection (divided into coarse-grained MAV and fine-grained MFG layers) from North Carolina State University's Xiaohui Gu team: <http://dance.csc.ncsu.edu/papers/srds11.pdf>
* CloudSeer system for cloud system workflow monitoring from NEC American Laboratory/North Carolina State University: <https://people.engr.ncsu.edu/gjin2/Classes/591/Spring2017/case-cloudseer.pdf>
* LogMine, published by NEC American Lab: <http://www.cs.unm.edu/~mueen/Papers/LogMine.pdf>
* LogLens system from NEC American Laboratory/Ant Financial (based on LogMine, combined with the design of Grok in ELK; added judgment and processing of traceids, supporting sequence anomaly detection), paper: <http://120.52.51.14/www.cs.ucsb.edu/~bzong/doc/icdcs-18.pdf>
* POP system from The Chinese University of Hong Kong/Huawei (similar approach to LogMine, running on Spark): <http://www.cse.cuhk.edu.hk/lyu/_media/journal/pjhe_tdsc18.pdf>
* Logram paper from Concordia University (using n-grams for log parsing): <https://petertsehsun.github.io/papers/HetongTSE2020.pdf>
* LogAssist paper from Concordia University, performing secondary compression on extracted log template sequences using n-grams, and using case studies to compare the effects: <https://petertsehsun.github.io/papers/TSE2021_LogAssist.pdf>
* Open-source implementation of log sequence anomaly detection from McMaster University, Canada, adding duration subsequences for each step as neural network features: <https://github.com/hfyxin/Ts-models-log-data-analysis>
* Log Anomaly Detector open-source project from RedHat CTO Office (based on word2vec and SOM algorithm): <https://github.com/AICoE/log-anomaly-detector>
* some other documents: 
    * Loomsystems User Guide: <http://support.loomsystems.com/loom-guides>
    * Coralogix (basic unordered pattern detection, object parameter detection for XML/JSON types): <https://coralogix.com/tutorials/what-is-coralogix-pattern-anomaly/>
    * Zebrium (using NewSQL, automatic parameter name identification is worth noting, and using GPT-3 to generate alert descriptions is also interesting): <https://www.zebrium.com/blog/using-ml-to-auto-learn-changing-log-structures>

#### Large Model Methods

* LogQA paper from Beihang University, using the T5 large model and manually labeled [training data](https https://github.com/LogQA-dataset/LogQA/tree/main/data) to enable natural language question answering for logs: <https://arxiv.org/pdf/2303.11715.pdf>
* LogPPT open-source project from the University of Newcastle, Australia, using the RoBERTa large model and loghub dataset. The most interesting point is that although the loghub dataset contains 80G of logs, only 2k logs per class are labeled. This paper takes a reverse approach and uses the 2k labeled logs as prompts: <https://github.com/LogIntelligence/LogPPT>
* DivLog paper from The Chinese University of Hong Kong, using the GPT3 large model, comprehensively outperforming LogPPT. It also explores the ICL method, where 5-shot may be optimal: <https://arxiv.org/pdf/2307.09950v3.pdf>
    *  The subsequent LILAC open-source project, through designed sampling methods and caching, approaches Drain's template inference speed! Additionally, in the comparison with LogPPT/LogDiv, it verifies that as the base model grows from the 110MB RoBerta to the 13B Curie to the 176B ChatGPT, the improvement is not substantial. For template recognition tasks, the language understanding ability of mid-sized LMs may already be decent: <https://github.com/logpai/LILAC>
* BERTOps open-source project from IBM, using the BERT large model and some manually labeled data, attempting three classification tasks in the log domain: log format classification, golden signal classification, and fault classification (however, this library is just a demonstration and cannot run; the pretrain.txt file in train.sh is missing, and only the cleaned Excel annotation file is provided): <https://github.com/BertOps/bertops>
* Log anomaly detection model based on language models from IBM Research, comparing the effects of fasttext and BERT: <https://www.researchgate.net/publication/344693315_Using_Language_Models_to_Pre-train_Features_for_Optimizing_Information_Technology_Operations_Management_Tasks>
* KTeleBERT open-source project from Zhejiang University/Huawei, integrating knowledge graphs and the BERT large model, and utilizing product manuals, device alert logs, and KPIs for fault analysis in the telecommunications domain: <https://github.com/hackerchenzhuo/KTeleBERT>
* Biglog large model from Huawei/USTC, based on Bert and unsupervised pre-training on 450 million logs from 16 projects: <https://github.com/BiglogOpenSource/PretrainedModel>.Corresponding paper for Biglog: <https://ieeexplore.ieee.org/document/10188759/>
* LogPrompt paper from Huawei/Beijing University of Posts and Telecommunications, using ChatGPT and Vicuna-13B to verify the effects of zero-shot, CoT, and ICL prompt strategies for log template extraction and anomaly detection: <https://arxiv.org/pdf/2308.07610.pdf>. The baselines for comparison are the aforementioned LogPPT. The conclusion is that even in the zero-shot setting, ChatGPT slightly outperforms LogPPT, while the open-source Vicuna-13B performs poorly in the zero-shot setting but greatly improves with the ICL approach, approaching a usable level.
* "Recommending Root-Cause and Mitigation Steps for Cloud Incidents using Large Language Models" paper from Microsoft, studying whether GPT models have an advantage over BERT models in fault diagnosis by analyzing 40,000 internal fault incidents at Microsoft. The rough conclusion is that there is an advantage, but it is still not very useful: <https://arxiv.org/pdf/2301.03797.pdf>
* "Assess and Summarize: Improve Outage Understanding with Large Language Models" paper from Microsoft Asia Research/Nankai University, comparing GPT2 (local single-GPU fine-tuning), GPT3 (6.7b), and GPT3.5 (175b) in generating alert summaries. The difference between 3 and 2 is indeed very significant, but the improvement from 6.7b to 175b is not substantial: <https://arxiv.org/pdf/2305.18084.pdf>
* Owl Operations Large Model Dataset from Beihang University/Yunzhihu, including question-answering and multiple-choice questions: <https://github.com/HC-Guo/Owl>. The corresponding paper also evaluates the differences in MoA fine-tuning, NBCE long context support, and log pattern recognition on the loghub dataset, although the advantages are very marginal.
* OpsEval paper from Tsinghua University/Mustshowme, with a similar scenario to Owl, but only comparing the performance of open-source models and distinguishing between Chinese and English. Practice has shown that the quality of Chinese question answering is much poorer: <https://arxiv.org/pdf/2310.07637.pdf>.
* CodeFuse-DevOpsEval evaluation dataset from Peking University/Ant Financial, covering 12 scenarios in DevOps and AIOps: <https://github.com/codefuse-ai/codefuse-devops-eval/blob/main/README_zh.md>. However, the scores for the root cause analysis scenario "qwen" in AIOps are abnormally high, leading to suspicion that the pretraining may have used internal data from Alibaba.
* UniLog paper from The Chinese University of Hong Kong/Microsoft, applying the ICL method of LLMs to log enhancement: <https://www.computer.org/csdl/proceedings-article/icse/2024/021700a129/1RLIWpCelqg>
* KnowLog open-source project from Fudan University, crawling descriptions of log templates from public documentation of Cisco, New H3C, and Huawei network devices, and creating pre-trained models based on Bert and RoBerta: <https://github.com/LeaperOvO/KnowLog>
* Xpert paper from Microsoft, generating Microsoft Azure's proprietary Kusto Query Language based on alert messages as context: <https://arxiv.org/pdf/2312.11988.pdf>. The paper proposes an Xcore evaluation method, comprehensively evaluating text, symbol, and field name matching. However, the error examples given in the paper show no overlap between the alert context and the correct output, making it impossible to generate the correct query - a suggestion that at the current stage, purely relying on Chat to generate query languages from prompts is too challenging due to the lack of context information.
* RCACopilot paper from Microsoft: <https://yinfangchen.github.io/assets/pdf/rcacopilot_paper.pdf>. It first summarizes the alert information, then uses a pre-trained fasttext embedding model to perform vector search on historical faults, and includes the summary and fault classification and description in the final prompt for the LLM to determine if it is an old fault and how to handle it if so. The paper provides a fair amount of evaluation data, but it has strong dependencies on the team and business environment being evaluated, making it difficult to judge its applicability.
* Another technical report from Microsoft on using the ReAct framework for RCA: <https://arxiv.org/pdf/2403.04123.pdf>. The rough conclusion is that without developing a specific Tool, relying on a generic document retrieval tool, ReAct performs worse than directly using RAG or CoT. Even with a specific Tool developed, the quality of the analysis plans written in the knowledge base is the most influential factor. Once multiple knowledge base documents are involved, ReAct tends to fail continuously from the second or third round onwards.
* A technical report from Flip.AI, a company that developed its own DevOps large model. It adopts a 1 encoder -> N decoder MoE architecture, with incremental pre-training on 80B tokens; the fine-tuning training data is mainly from simulated data based on RAG, supplemented by 18 months of human double-blind filtering; the reinforcement learning stage is RLHAIF, building a fault injection environment for the model to generate RCA reports: <https://assets-global.website-files.com/65379657a6e8b5a6ad9463ed/65a6ec298f8b53c8ddb87408_System%20of%20Intelligent%20Actors_FlipAI.pdf>

## Label

### Label tools for timeseries

* Curve(An Integrated Experimental Platform for time series data anomaly detection.), opensourced by baidu: <https://github.com/baidu/Curve>
* TagAnomaly(Anomaly detection analysis and labeling tool, specifically for multiple time series), opensourced by Microsoft: <https://github.com/Microsoft/TagAnomaly>
* Label-less project (based on iForest anomaly detection and DTW similarity learning) for batch labeling of metrics from Tsinghua University/Construction Bank of China: <https://netman.aiops.org/wp-content/uploads/2019/10/Label-less-v2.pdf>

## Prediction

### KPI

* Prophet, opensourced by facebook: <https://github.com/facebook/prophet>
* hawkular-datamining(autotunning for ARIMA), opensourced by redhat: <https://github.com/hawkular/hawkular-datamining>
* Open-source project from 360, encapsulating general algorithms like LR, ARIMA, LSTM for forecasting: <https://github.com/jixinpu/aiopstools/tree/master/aiopstools/timeseries_predict>
* A paper from North Carolina State University's Xiaohui Gu team on data transmission compression for monitoring systems (clustering and distributing prediction models, agents only report when predictions deviate): <http://dance.csc.ncsu.edu/papers/ICAC09.pdf>
* Pyraformer, an open-source project from Ant Financial, also open-sourcing the company's internal AppFlow dataset: <https://github.com/ant-research/Pyraformer>

### Capacity Planning

* cluster-data(a 12.5k Borg cluster traces), opensourced by google: <https://github.com/google/cluster-data>
* AzurePublicDataset(a sanitized subset of the Azure VM workload), opensourced by Microsoft: <https://github.com/Azure/AzurePublicDataset>
* clusterdata(4000 machines in a period of 8 days), opensourced by aliyun: <https://github.com/alibaba/clusterdata>
   * aliyun\_schedule\_semi(alibaba 'tianchi' schedule competiton), opensourced by the top 6 winner of competiton: <https://github.com/NeuronEmpire/aliyun_schedule_semi>
* thoth(predict the performance of SolrCloud using the querystring features), opensourced by Trulia: <https://github.com/trulia/thoth>
* Ottertune, an open-source automatic tuning tool for relational databases from CMU: <https://github.com/cmu-db/ottertune>
* Auto-tikv, an imitation of ottertune by PingCAP for automatically tuning TiKV: <https://github.com/tikv/auto-tikv>
* A research paper from Sun Yat-sen University on automatic tuning of Elasticsearch: <https://ieeexplore.ieee.org/ielx7/6287639/8948470/09079492.pdf>
* A research paper from Concordia University on Kafka capacity prediction (the first half of the process is similar to auto-tuning, but the purpose is to obtain a model for predicting parameter changes, so the main comparison is between XGBoost, LR, RF, and MLP): <https://github.com/SPEAR-SE/mlasp>
* Unicorn, an open-source project from the University of Southern California, adopting causal inference algorithms to discover the most important parameters. The most interesting part is that it also posts the review and modification comments from previous submissions on the wiki: <https://github.com/softsys4ai/unicorn>
* Chroma paper from AT&T and the University of Texas at Austin, recommending the best 4G/5G cellular network configurations based on network KPIs, network topology, and attributes: <https://dl.acm.org/doi/pdf/10.1145/3570361.3613256>

### Network

* PreFix(LCS-based), published by NKU: <http://workshop.aiops.org/files/shenglinzhang2018prefix.pdf>

### Event Correlation

* Log3C, opensourced by MSRA: <https://github.com/logpai/Log3C>
* A paper from Microsoft and Jilin University: <http://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/SIGKDD-2014-Correlating-Events-with-Time-Series-for-Incident-Diagnosis.pdf>
    * Open-source implementation based on the above paper from 360: <https://github.com/jixinpu/aiopstools/tree/master/aiopstools/association_analysis>
* A paper from IBM Research, using microservice error metrics and error logs, adopting causal inference algorithms and personalized PageRank algorithms for fault localization (the main purpose is to introduce personalized PR, without distinguishing between PC and regression for causal inference): <https://www.researchgate.net/publication/344435606_Localization_of_Operational_Faults_in_Cloud_Applications_by_Mining_Causal_Dependencies_in_Logs_using_Golden_Signals>

## Root Cause Analysis

### tracing

* Open-source APM/tracing implementations:
     * zipkin/brave: <https://github.com/openzipkin/brave>
     * springcloud/sleuth: <https://github.com/spring-cloud/spring-cloud-sleuth>
     * skywalking: <https://skywalking.apache.org/>
     * jaeger: <https://github.com/jaegertracing/jaeger>
     * pinpoint: <https://github.com/pinpoint-apm/pinpoint>
     * elastic apm: <https://github.com/elastic/apm>
     * datadog apm: <https://github.com/DataDog/dd-trace-java>
     * opencensus/opentelemetry: <https://opentelemetry.io/>
     * cilium/hubble: <https://github.com/cilium/hubble>
     * pixie/stirling: <https://github.com/pixie-labs/pixie/tree/main/src/stirling>
* Jonathan Mace from Saarland University, using hierarchical clustering to avoid losing rare instances during sampling: <https://people.mpi-sws.org/~jcmace/papers/lascasas2018weighted.pdf>
* Open-source microservice test bed from Google: <https://github.com/GoogleCloudPlatform/microservices-demo>
* Open-source large-scale microservice test bed from Fudan University: <https://github.com/FudanSELab/train-ticket/>
* Root cause localization competition for call traces from Tsinghua University's Dan Pei team: <http://iops.ai/competition_detail/?competition_id=15&flag=1>
    * A paper from South China University of Technology targeting the above competition: <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9293310>
* A review paper on anomaly detection and root cause analysis in microservice environments from the University of Pisa, Italy: <https://arxiv.org/pdf/2105.12378.pdf>
* CRISP, an open-source key path analysis tool for call traces from Uber, including features like clock skew compensation and anomaly detection: <https://github.com/uber-research/CRISP>

### bottleneck localization

* competiton of bottleneck Localization(need register), hosted by peidan, THU: <http://iops.ai/competition_detail/?competition_id=8&flag=1>
* HotSpot(Anomaly Localization for Additive KPIs With Multi-Dimensional Attributes), published by THU/NKU/baidu: <http://netman.ai/wp-content/uploads/2018/03/sunyq_IEEEAccess_HotSpot.pdf>
* A paper from the National Institute of Applied Sciences of Lyon, using subgroup discovery algorithms to locate slow SQL queries by integrating SQL parsing, environment versions, alerts, and metrics: <https://www.researchgate.net/publication/353776691_What_makes_my_queries_slow_Subgroup_Discovery_for_SQL_Workload_Analysis>


### timeseries correlation

* oculus(fastDTW+elasticsearch), opensourced by etsy: <https://github.com/etsy/oculus>
* luminol(SAX-based, anomalydetection and correlation), opensourced by linkedin: <https://github.com/linkedin/luminol>
    * full system introduction: <https://docs.google.com/presentation/d/1DWMNgoAtxuK8ZbFJOpq5vt3dEz5_4ptMuLtoTUjQ_ro/pub?start=false&loop=false&delayms=3000&slide=id.p>
* Argos, introduced by Uber: <https://eng.uber.com/argos/>

### Solution Relevance Recommendation

* A paper from Florida International University/IBM, using CNN for ticket correlation and recommendation, focusing on how to filter and extract effective text features from tickets: <https://www.researchgate.net/publication/318373831_STAR_A_System_for_Ticket_Analysis_and_Resolution>

## Alert Grouping

* Open-source implementation based on the Apriori algorithm from 360: <https://github.com/jixinpu/aiopstools/blob/master/examples/alarm_convergence.py>
* SplitSD4X from the National Institute of Applied Sciences of Lyon, using subgroup discovery algorithms and NLP to parse incident reports and generate fault descriptions: <https://github.com/RemilYoucef/split-sd4x>
* Real-time anomaly detection system for time series at scale(SAE+LDA, Behavior Topology learning), published by anodot: <http://proceedings.mlr.press/v71/toledano18a/toledano18a.pdf>
* some other documents: 
    * alert clustering principle introduced by [moogsoft](https://docs.moogsoft.com/): <https://docs.google.com/presentation/d/1F-8eop-9ffCpX4trOJS28FXATAFlQkgkfnDV_Zqnkuo/edit?pref=2&pli=1#slide=id.g990524d96_0_0>

## Knowledge Graph

* SOSG(Debugging Openstack Problems Using A State Graph Approach), published by xuwei, THU: <http://iiis.tsinghua.edu.cn/~weixu/files/apsys-yong-slides.pdf>
* A graphical representation for identifier structure in logs ,published by xuwei, UC Berkeley: <http://iiis.tsinghua.edu.cn/~weixu/files/slaml10-rabkin.pdf>
* [WOT2018 -张真-运维机器人之任务决策系统演讲](https://pan.baidu.com/s/1gSjJZIXswOPoeQzZ6cJT1g?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=), published by CreditEase
* Huawei's open-source [OpenStack Observability Data](https://jorge-cardoso.github.io/publications/Papers/CP-2020-093-Multi-source\_Distributed\_System\_Data.pdf) sample set, with the trace part collected using osprofiler: <https://zenodo.org/record/3549604#.YE8Q-eHithE>
* CA/Catalonia Polytechnic University, a graph-based microservice root cause analysis system built using logs and metrics: <https://www.researchgate.net/publication/336585890\_Graph-based\_Root\_Cause\_Analysis\_for\_Service-Oriented\_and\_Microservice\_Architectures>
*  Wezhuangyuan Bank's intelligent operation and maintenance series sharing, the eighth article (providing very detailed node attributes and edge attributes design, more detailed than CA's): [Event Fingerprint Library: Building a "Museum" of Anomaly Cases](https://mp.weixin.qq.com/s/M8tcS8q6sPPRRebAJkrb7Q)
* Sun Yat-sen University's TopoMAD system for metric anomaly detection by fusing topology information, with open-source data samples: <https://github.com/QAZASDEDC/TopoMAD>
* Another article analyzing and commenting on the TopoMAD paper, quite critical: <https://dreamhomes.top/posts/202103111131.html>
* Sun Yat-sen University's "Nezha" system for root cause localization by fusing observability data (metrics, logs, and traces), converting them into events, building a topology graph, and comparing abnormal and normal topologies triggered by business alerts to identify root cause events (e.g., call changes or metric/log anomalies). Open-source link: <https://github.com/IntelligentDDS/Nezha>. The paper validated the system on two open-source microservice demos (e-commerce and train ticket booking, with otel instrumentation and traceid/spanid added to logs) and evaluated the impact of missing certain data types, finding that one demo was more metric-heavy while the other was more log-heavy. However, the reported 95+ accuracy seems a bit too good to be true.
* Vienna University of Technology's [VloGraph](https://www.mdpi.com/2504-4990/4/2/16) project, using NLP and graph analysis techniques for log parsing, storage, and visualization in security scenarios: <https://github.com/sepses>
* Tsinghua University/eBay's AlertRCA, an enhanced version of the previous Peking University/eBay's [Groot](https://arxiv.org/pdf/2108.00344). It uses Bert vector representation for alerts and graph attention to learn causality, without manual rule configuration: <https://netman.aiops.org/wp-content/uploads/2024/03/AlertRCA\_CCGRID2024\_CameraReady.pdf>

## Behavior Anomaly Detection

* ee-outliers(word2vec+osquery, for elasticsearch events), opensourced by NVISO: <https://github.com/NVISO-BE/ee-outliers>
* aliyun\_safety(alibaba 'tianchi' portscan competiton), opensourced by the top 10 winner of competiton: <https://github.com/wufanyou/aliyun_safety>
* some other documents: 
    * ExtraHop User Guide: <https://docs.extrahop.com/current/>
* North Carolina State University's Xiaohui Gu's team's PerfScope system for program behavior anomaly analysis (using LTTng for online system call tracing and Apriori algorithm), paper: <http://dance.csc.ncsu.edu/papers/socc14.pdf>
* The ubad open-source project, using LSTM and OCSVM for anomaly detection on osquery output user behavior events: <https://github.com/morrigan/user-behavior-anomaly-detector>
* Security Repo dataset, a collection of various security and operations-related data by Mike Sconzo: <https://www.secrepo.com/>
* Austrian AIT's security log dataset: <https://zenodo.org/record/5789064#.YkFnZWJBxhE> and its data generation testbed project: <https://github.com/ait-aecid/kyoushi-environment>

## Further Reading

* [awesome-AIOps](https://github.com/linjinjin123/awesome-AIOps) from @linjinjin123, slimilar to my repository but sort by resource type.
* [awesome-anomaly-detection](https://github.com/zhuyiche/awesome-anomaly-detection) from @zhuyiche, focus on papers about anomalydetection.
* Salesforce AI Research's comprehensive survey paper on AIOps: <https://arxiv.org/pdf/2304.04661.pdf>, covering 237 papers in areas such as metrics, logs, traces, root cause analysis, failure prediction, and self-healing strategies.
* @logpai's [log-survey](https://github.com/logpai/log-survey) repository, focusing on log-related papers and researchers, including not only analysis but also log generation, augmentation, compression, etc.
* @jorge-cardoso's mapping study of AIOps systems: <https://jorge-cardoso.github.io/publications/Papers/WP-2020-079-AIOPS2020\_A\_Systematic\_Mapping\_Study\_in\_AIOps.pdf>
* @jorge-cardoso's survey on AIOps methods for failure management: <https://jorge-cardoso.github.io/publications/Papers/JA-2021-025-Survey\_AIOps\_Methods\_for\_Failure\_Management.pdf>
* China Mobile Information Technology Company's introduction to a maturity assessment model for automated and intelligent operations and maintenance: <https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2021&filename=DGJB202101009&v=4TGdKbAr7slYCfXAgxJfw2rHqvDI%25mmd2FLQ8iARm627FUgNvt7c9lvfHVIbYjAz8eR2x>

