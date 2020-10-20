需要预先下载的模型及数据： <Br/>
training_data 训练集 <Br/>
testing_data  测试集  数据格式详见这两个文件中的例子  <Br/>
elmo模型 <Br/>
word2vec模型 <Br/> 
bert模型 <Br/>
生成模型的文件夹需要预先建立，位置和代码并列 <Br/>

=========预处理数据================ <Br/>
processing.py：将预训练数据进行清洗转换处理，生成所需要的字典数据 <Br/>
=========使用word2vec进行计算======= <Br/>
core_word2vec.py:将测试集结果写入本地文件 <Br/>
=========使用weightedAvg模型进行计算======= <Br/>
buile_weightedAvg_model.py：训练型保存模型 <Br/>
core_elmo_weightedAvg.py：使用模型将测试集结果写入本地文件 <Br/>
=========使用snli模型进行计算======= <Br/>
build_snli_model.py：训练型保存模型 <Br/>
core_elmo_snli.py：使用模型将测试集结果写入本地文件 <Br/>
=========使用bert模型进行计算======= <Br/>
buile_bert_model.py：训练型保存模型 <Br/>
core_bert.py：使用模型将测试集结果写入本地文件 <Br/>
==========定义三种模型============== <Br/>
models.py <Br/>
==========验证结果文件的F1值============== <Br/>
calF1.py <Br/>

word2vec和weightedAvg效果较好，且相差很小，选择对这两种方法进行了单条数据测试比较，代码见core_word2vec.py和core_elmo_weightedAvg.py最下端测试参考代码
