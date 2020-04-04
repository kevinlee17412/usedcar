Task5 模型融合

模型融合即为融合多个模型，提升机器学习的性能，最简单的就是加权平均。

１．常用加权融合方法

（１）简单加权融合

- ​	回归：算数平均融合，几何平均融合
- 　分类：投票
- 　综合：排序融合，log融合

（２）stacking/blending

​	构建多层模型，并且利用预测结果再拟合预测

（３）boosting/bagging

​	多树的提升方法

![Image](http://jupter-oss.oss-cn-hangzhou.aliyuncs.com/public/files/image/2326541042/1584448806789_1ElRtHaacw.jpg)

​                                                                           Stacking 算法

```python
#Stacking融合回归
from sklearn import linear_model

def Stacking_method(train_reg1,train_reg2,train_reg3,y_train_true,test_pre1,test_pre2,test_pre3,model_L2= linear_model.LinearRegression()):
    model_L2.fit(pd.concat([pd.Series(train_reg1),pd.Series(train_reg2),pd.Series(train_reg3)],axis=1).values,y_train_true)
    Stacking_result = model_L2.predict(pd.concat([pd.Series(test_pre1),pd.Series(test_pre2),pd.Series(test_pre3)],axis=1).values)
    return Stacking_result


## 生成一些简单的样本数据，test_prei 代表第i个模型的预测值
train_reg1 = [3.2, 8.2, 9.1, 5.2]
train_reg2 = [2.9, 8.1, 9.0, 4.9]
train_reg3 = [3.1, 7.9, 9.2, 5.0]
# y_test_true 代表第模型的真实值
y_train_true = [3, 8, 9, 5] 

test_pre1 = [1.2, 3.2, 2.1, 6.2]
test_pre2 = [0.9, 3.1, 2.0, 5.9]
test_pre3 = [1.1, 2.9, 2.2, 6.0]

# y_test_true 代表第模型的真实值
y_test_true = [1, 3, 2, 6] 


model_L2= linear_model.LinearRegression()
Stacking_pre = Stacking_method(train_reg1,train_reg2,train_reg3,y_train_true,
                               test_pre1,test_pre2,test_pre3,model_L2)
print('Stacking_pre MAE:',metrics.mean_absolute_error(y_test_true, Stacking_pre))
```

这是最后一节课了，如果没有这件事，我可能坚持不下来，时间紧，只能先这样了。

[参考链接]: https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.15.1cd81b43dGWTiP&amp;postId=95535

