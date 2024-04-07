# GPT-IE
这是北京师范大学系统科学学院18级博士生毕业设计中的一部分工作，主要探讨了在开放信息抽取中使用GPT-4的提示设计尝试。其中zero-shot文件夹中是指没有样本参与的基础提示设计和思维链提示设计的实验代码，few-shot文件夹是指有样本加入的提示设计的实验代码，result文件夹存储着各个数据集的结果，结果我们使用.carb的文件格式来标注，这是为了使用在[CaRB](https://github.com/dair-iitd/CaRB)上的评价标准。我们的代码只有评价部分需要在python命令行中运行，具体的代码为：  
```shell 
python carb/carb.py --allennlp result/zero-shot/CaRB/task1.carb --gold result/gold/carb_50.tsv --out /dev/null
```
data文件夹中保存着句子和树的数据，其中树的标记我们参考[SMiLe_OIE](https://github.com/daviddongkc/smile_oie)的工作，使用StanfordCoreNLP包进行标注。  
另外，原始的[LSOIE](https://github.com/Jacobsolawetz/large-scale-oie)数据集和[CaRB](https://github.com/dair-iitd/CaRB)数据集都可以在网页中获得，这里不做过多赘述



