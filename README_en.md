# aiops-handbook

Collection of slides, repositories, papers about AIOps, sort by the scenes which were advanced by the [<WhitePaper of Enterprise AIOps Recommendation>](https://pic.huodongjia.com/ganhuodocs/2018-04-16/1523873064.74.pdf).

Chinese edition: <README.md>.

## Anomaly Detection

### Metric

#### single

* competiton of KPI anomalydetection(need register), hosted by peidan, THU: <http://iops.ai/competition_detail/?competition_id=5&flag=1>
    * shared slides of top 5 teams: <http://workshop.aiops.org/>
    * another program by an unknown PhD student: <https://github.com/chengqianghuang/exp-anomaly-detector-AIOps>
* Donut(VAE-based algorithm), opensourced by THU/alibaba: <https://github.com/haowen-xu/donut>
* Opprentice(14 detector+randomforest), published by THU/baidu: <http://netman.cs.tsinghua.edu.cn/wp-content/uploads/2015/11/liu_imc15_Opprentice.pdf>
* ADS(ROCKA+opprentice+CPLE), published by THU/NKU/baidu: <https://netman.aiops.org/wp-content/uploads/2018/12/bujiahao.pdf>
    * a CPLE implement opensourced in github: <https://github.com/tmadl/semisup-learn>
* metis(inspired by opprentice), opensourced by tencent: <https://github.com/tencent/metis>
* skyline
    * version 1(9 detector+vote threshold), opensourced(but archived) by etsy: <https://github.com/etsy/skyline>
    * version 2(wavelet+ks test+generalized ESD test) slide, published by etsy: <https://vimeo.com/131581331>
    * golang version, opensourced by lytics: <https://github.com/lytics/anomalyzer>
    * community version(a new Ionosphere module use tsfresh to learn what's no anomalous) opensourced by earthgecko: <https://github.com/earthgecko/skyline>
* tsfresh, an open source tool for calculating a large number of time series features: <http://tsfresh.readthedocs.io/en/latest/>
* Surus(PCA-based, running on Pig), opensourced by netflix: <https://github.com/netflix/surus>
* anomalydetection(generalized ESD, written in R), opensourced by twitter: <https://github.com/twitter/anomalydetection>
* NuPIC(HTM algorithm), opensourced by numenta: <https://github.com/numenta/nupic>
    * NAB competiton(with a curved scoring, by contrast, peidan's competiton give a static score within 7 points after the anmaly window start), hosted by numenta: <https://github.com/numenta/NAB>
* EGADS(adboost-based algorithm), opensourced by Yahoo!: <https://github.com/yahoo/egads>
* adaptive-alerting(with some very good wiki articles), opensourced by Expedia: <https://github.com/ExpediaDotCom/adaptive-alerting>
* Anomaly.io Blog: <https://anomaly.io/blog/>

#### cluster

* SPIRIT(KPIs' pattern discovery, need similiarity score>0.9 ), published by CMU: <https://bitquill.net/pdf/spirit_vldb05.pdf>
* ROCKA, published by THU: <https://netman.aiops.org/~peidan/ANM2018/8.DependencyDiscovery/LectureCoverage/2018IWQOS_ROCKA.pdf>

### Logdata

* loghub(a collection of system log datasets for intelligent log analysis, 77GB in total), opensourced by CUHK: <https://github.com/logpai/loghub>
* DeepLog(DL, anomalydetection for log keys, parameters, workflows), published by CUHK/MSRA: <https://acmccs.github.io/papers/p1285-duA.pdf>
* FT-tree(frequent words as log template), published by THU/NKU/tecent: <http://nkcs.iops.ai/wp-content/uploads/2018/06/paper-iwqos17-Syslog-Processing-for-Switch-Failure-Diagnosis-and-Prediction-in-Datacenter-Networks.pdf>
* LogClass(bag-of-word+PU learning, anomaly detection and classification), published by THU/NKU/baidu: <http://nkcs.iops.ai/wp-content/uploads/2018/06/paper-IWQOS2018-Device_Agnostic_Log_Anomaly_Classification.pdf>
* LogMine, published by NEC American Lab: <http://www.cs.unm.edu/~mueen/Papers/LogMine.pdf>
* some other documents: 
    * Loomsystems User Guide: <http://support.loomsystems.com/loom-guides>

## Label

### Label tools for timeseries

* Curve(An Integrated Experimental Platform for time series data anomaly detection.), opensourced by baidu: <https://github.com/baidu/Curve>
* TagAnomaly(Anomaly detection analysis and labeling tool, specifically for multiple time series), opensourced by Microsoft: <https://github.com/Microsoft/TagAnomaly>


## Prediction

### KPI

* Prophet, opensourced by facebook: <https://github.com/facebook/prophet>
* hawkular-datamining(autotunning for ARIMA), opensourced by redhat: <https://github.com/hawkular/hawkular-datamining>

### Capacity Planning

* cluster-data(a 12.5k Borg cluster traces), opensourced by google: <https://github.com/google/cluster-data>
* AzurePublicDataset(a sanitized subset of the Azure VM workload), opensourced by Microsoft: <https://github.com/Azure/AzurePublicDataset>
* clusterdata(4000 machines in a period of 8 days), opensourced by aliyun: <https://github.com/alibaba/clusterdata>
   * aliyun\_schedule\_semi(alibaba 'tianchi' schedule competiton), opensourced by the top 6 winner of competiton: <https://github.com/NeuronEmpire/aliyun_schedule_semi>
* thoth(predict the performance of SolrCloud using the querystring features), opensourced by Trulia: <https://github.com/trulia/thoth>


### Network

* PreFix(LCS-based), published by NKU: <http://workshop.aiops.org/files/shenglinzhang2018prefix.pdf>

### Event Correlation

* Log3C(), opensourced by MSRA: <https://github.com/logpai/Log3C>

## Root Cause Analysis

### tracing

Tracing is very useful for RCA, but donot need AI:

* <https://github.com/openzipkin/zipkin>
* <https://github.com/apache/incubator-skywalking>
* <https://github.com/naver/pinpoint>

### bottleneck localization

* competiton of bottleneck Localization(need register), hosted by peidan, THU: <http://iops.ai/competition_detail/?competition_id=8&flag=1>
* HotSpot(Anomaly Localization for Additive KPIs With Multi-Dimensional Attributes), published by THU/NKU/baidu: <http://netman.ai/wp-content/uploads/2018/03/sunyq_IEEEAccess_HotSpot.pdf>

### timeseries correlation

* oculus(fastDTW+elasticsearch), opensourced by etsy: <https://github.com/etsy/oculus>
* luminol(SAX-based, anomalydetection and correlation), opensourced by linkedin: <https://github.com/linkedin/luminol>
    * full system introduction: <https://docs.google.com/presentation/d/1DWMNgoAtxuK8ZbFJOpq5vt3dEz5_4ptMuLtoTUjQ_ro/pub?start=false&loop=false&delayms=3000&slide=id.p>
* Argos, introduced by Uber: <https://eng.uber.com/argos/>

## Alert Grouping

* Real-time anomaly detection system for time series at scale(SAE+LDA, Behavior Topology learning), published by anodot: <http://proceedings.mlr.press/v71/toledano18a/toledano18a.pdf>
* some other documents: 
    * alert clustering principle introduced by [moogsoft](https://docs.moogsoft.com/): <https://docs.google.com/presentation/d/1F-8eop-9ffCpX4trOJS28FXATAFlQkgkfnDV_Zqnkuo/edit?pref=2&pli=1#slide=id.g990524d96_0_0>

## Knowledge Graph

* SOSG(Debugging Openstack Problems Using A State Graph Approach), published by xuwei, THU: <http://iiis.tsinghua.edu.cn/~weixu/files/apsys-yong-slides.pdf>
* A graphical representation for identifier structure in logs ,published by xuwei, UC Berkeley: <http://iiis.tsinghua.edu.cn/~weixu/files/slaml10-rabkin.pdf>
* [WOT2018 -张真-运维机器人之任务决策系统演讲](https://pan.baidu.com/s/1gSjJZIXswOPoeQzZ6cJT1g?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=), published by CreditEase

## Behavior Anomaly Detection

* ee-outliers(word2vec+osquery, for elasticsearch events), opensourced by NVISO: <https://github.com/NVISO-BE/ee-outliers>
* aliyun\_safety(alibaba 'tianchi' portscan competiton), opensourced by the top 10 winner of competiton: <https://github.com/wufanyou/aliyun_safety>
* some other documents: 
    * ExtraHop User Guide: <https://docs.extrahop.com/current/>


## More

* [awesome-AIOps](https://github.com/linjinjin123/awesome-AIOps) from @linjinjin123, slimilar to my repository but sort by resource type.
* [awesome-anomaly-detection](https://github.com/zhuyiche/awesome-anomaly-detection) from @zhuyiche, focus on papers about anomalydetection.

