wordbag/
  data/
    Loughran-McDonald_MasterDictionary_1993-2024.csv
    combined_politicaltbb_npb_finalbigrams.csv
  standardized_lexicons/
    lm_standardized.csv
    flr_standardized.csv
  lexicon_standardizer.py
  build_risk_lexicons.py
  risk_lexicons.json   ← 合并后的总词袋（运行后生成）

一、两个输入文件分别是什么？
1) Loughran–McDonald 金融情感/不确定性词典（简称 LM）
典型用途：金融文本的情感（Positive/Negative）、不确定性（Uncertainty）、诉讼（Litigious）、强/弱情态（Strong/Weak Modal）、约束性（Constraining）等高频维度识别。
你会看到的列（常见）：
Word：词条（大写或首字母大写，建议统一转小写使用）
Word Count / Word Proportion / Average Proportion / Std Dev / Doc Count：构建词典时的统计信息，可忽略
Negative / Positive / Uncertainty / Litigious / Strong_Modal / Weak_Modal / Constraining / Complexity：关键二进制或数值列，非 0 表示该词属于该类
其他辅助列（如 Source、Syllables 等）
我们怎么用：把每个“非 0 的类别”映射到一个风险桶（bucket），例如：
negative → lm_negative
uncertainty → lm_uncertainty
litigious → lm_litigious
strong_modal → lm_modal_strong
weak_modal → lm_modal_weak
constraining → lm_constraining
这样 LM 词表会被拆成多个通用风险桶，用于衡量普适的金融语气/不确定性维度。
小结：LM = “通用金融语气/不确定性的权威词典”，覆盖面广、可复现，非常适合作为基线。

2) Firm-level Risk（简称 FLR）的主题型外部风险 n-grams
典型用途：识别主题型外部风险，比如政治/政策、贸易、健康/疫情、环境、机构/制度、安全、税收、科技等类别下的专名词组（bigrams/n-grams）。
你会看到的列（不同版本名字会略有差异）：
一列表示 n-gram 本体（例如 bigram/ngram/term）
一列或多列表示所属主题（例如 politics/policy/health/trade/...），或一个 category/topic 字段给出分类
可能还会有权重列（如频次/概率/评分），如果有，我们会保留到 description 里当备注
我们怎么用：把 n-gram 归入主题风险桶，比如：
flr_political_policy（政治/政策）
flr_trade
flr_health_epidemic
flr_environment
flr_institutions
flr_security
flr_tax
flr_technology
…（按文件里实际的主题来映射，脚本里可扩展）
小结：FLR = “主题型外部风险的细分词表”，能捕捉电话会里经常出现的政策/外部冲击相关术语（如疫情、制裁、关税、合规等）。


输出risk_lexicons.json：
结构：一个 Python/JSON 字典，{ bucket_name: [term1, term2, ...] }
例子（示意）：
{
  "lm_uncertainty": ["uncertain", "unknown", "unpredictable", "..."],
  "lm_litigious":   ["lawsuit", "litigation", "complaint", "..."],
  "flr_political_policy": ["executive order", "regulatory review", "tariff hike", "..."],
  "flr_health_epidemic":  ["covid outbreak", "pandemic response", "contact tracing", "..."]
}