ML模型：Logistic Regression, L2 正则

OUTPUT：
1. preds_sentence.csv（或 preds_chunk.csv）
这是公司级预测结果表。
关键列：
company_id：公司 ID（来自 x_mij 的 company 字段）
y_true：真实标签（现在用你提供的默认/字典标签；当前都是 1）
y_prob：模型对“被收购”的概率预测
现在因为是恒等模型，所以通常是常数 1.0（或接近 1）。
以后有了 0/1 混合公司后，会是 0~1 的非平凡概率。
y_pred：0/1 预测结果（按 0.5 阈值）
这张表以后可加上 fold、split 等列用于交叉验证/时间外验证。
怎么用：看整体预测分布与真值的对齐，等我们有 0/1 混合数据时，可计算 Accuracy、AUC、F1 等指标判断公司层面的风险预测能力。

2) feat_importance_sentence.csv（或 feat_importance_chunk.csv）
这是细粒度特征重要性排行。
列：
feature：特征名
绝大多数是形如 ROLE|RISK 的列（例如 CFO|Liquidity, CEO|External_Shock）。
还有一列 prior_z：表示基于 f_ijz 对所有 (role,risk) 先验加权得到的“先验得分”。
abs_importance：特征的绝对系数（|coef|），表示边际影响强度
在“恒等模型”阶段，它只是一个占位解释（没有真正训练出的系数）；
用上 Logistic Regression 后，它就是真实学到的权重大小（越大代表对预测“被收购”越重要）。
怎么用：
1. 正常训练时，排序前几名就是最能区分被收购与否的“角色×风险”组合；
2. prior_z 的高权重说明“先验统计（f_ijz）”本身就很有信息量。

3) role_importance_sentence.csv（或 role_importance_chunk.csv）
这是把细粒度特征按角色维度聚合后的重要性，回答“谁的发言更有用？”
列：
role：角色（如 CEO、CFO、ANALYST、IR、OPERATOR），另外有一项 PRIOR 表示“先验得分”的贡献。
abs_importance：把该角色下所有 ROLE|RISK 特征的 |coef| 求和，越大表示这个角色总体更关键。
怎么用？
观察哪个角色的 abs_importance 最大，就能回答**“哪类说话者的表达对预测更有帮助”**。
后续有更多公司时，这个排名会更稳定、更有解释力。
Importance计算方式：
我们先做细粒度特征：
把公司级特征矩阵的每一列视作一个“(role|risk)”特征（再加一列 prior_z）。
训练 逻辑回归（L2） 后，取系数的绝对值作为该特征的影响强度
然后把列名里形如 ROLE|risk 的ROLE 部分提取出来，按角色把所有该角色的特征重要性求和
总结：某个角色在不同风险维度上对应的系数越大（无论正负），说明“该角色说的这类话”对模型判别越有力量；把它们加总，就是这个角色的总体重要性。