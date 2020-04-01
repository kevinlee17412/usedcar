Task4 建模与调参

前面将数据的清洗，特征的处理等方面，接下来就是比较常听到的模型训练。

1. 学习目标

​	了解常用的机器学习模型，并掌握机器学习模型的建模和调参流程。

2. 学习内容

   2.1 简单建模

   2.2 模型性能验证方法

   - 评价函数与目标函数
   - 交叉验证方法
   - 留一验证方法
   - 针对时间序列问题的验证
   - 绘制学习率曲线
   - 绘制验证曲线

   2.3 模型调参

   - 贪心调参
   - 网格调参
   - 贝叶斯调参

3. 代码演示

   3.1 常用模型导入方法

   ```python
   #线性模型
   from sklearn.linear_model import LinearRegression
   
   model = LinearRegression(normalize=True)
   
   model = model.fit(train_X, train_y)
   ```

   ```python
   #非线性模型
   from sklearn.linear_model import LinearRegression
   from sklearn.svm import SVC
   from sklearn.tree import DecisionTreeRegressor
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.ensemble import GradientBoostingRegressor
   from sklearn.neural_network import MLPRegressor
   from xgboost.sklearn import XGBRegressor
   from lightgbm.sklearn import LGBMRegressor
   ```

   ```pythoN
   models = [LinearRegression(),
             DecisionTreeRegressor(),
             RandomForestRegressor(),
             GradientBoostingRegressor(),
             MLPRegressor(solver='lbfgs', max_iter=100), 
             XGBRegressor(n_estimators = 100, objective='reg:squarederror'), 
             LGBMRegressor(n_estimators = 100)]
             
   result = dict()
   for model in models:
       model_name = str(model).split('(')[0]
       scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error))
       result[model_name] = scores
       print(model_name + ' is finished')
   ```

   3.2 模型性能评估

   ​	（1）评价函数与目标函数

   ​	很多模型都假设数据误差项都服从正太分布。像数据呈长尾分布，可以使用log(x+1)，做一个转换。

   ​	（2）五折交叉验证

   ​	简而言之，将数据集分为训练集和测试集（有的还有验证集）

   ​	（3）绘制学习率曲线和验证曲线

   ```python
   from sklearn.model_selection imort learning_curve, validation_curve
   def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1,train_size=np.linspace(.1, 1.0, 5 )):  
           plt.figure()  
           plt.title(title)  
           if ylim is not None:  
               plt.ylim(*ylim)  
           plt.xlabel('Training example')  
           plt.ylabel('score')  
           train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_size, scoring = make_scorer(mean_absolute_error))  
           train_scores_mean = np.mean(train_scores, axis=1)  
           train_scores_std = np.std(train_scores, axis=1)  
           test_scores_mean = np.mean(test_scores, axis=1)  
           test_scores_std = np.std(test_scores, axis=1)  
           plt.grid()#区域  
           plt.fill_between(train_sizes, train_scores_mean - train_scores_std,  
                            train_scores_mean + train_scores_std, alpha=0.1,  
                            color="r")  
           plt.fill_between(train_sizes, test_scores_mean - test_scores_std,  
                            test_scores_mean + test_scores_std, alpha=0.1,  
                            color="g")  
           plt.plot(train_sizes, train_scores_mean, 'o-', color='r',  
                    label="Training score")  
           plt.plot(train_sizes, test_scores_mean,'o-',color="g",  
                    label="Cross-validation score")  
           plt.legend(loc="best")  
           return plt  
      
   ```

   ​	(4) 模型调参

   - 贪心算法 https://www.jianshu.com/p/ab89df9759c8
   - 网格调参 https://blog.csdn.net/weixin_43172660/article/details/83032029
   - 贝叶斯调参 https://blog.csdn.net/linxid/article/details/81189154

   参考自

   [链接]: https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.1cd866c229EthJ&amp;amp;postId=95460

    

