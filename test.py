import scipy.stats as stats

# 样本数据
mean1, std1, n1 = 92.86, 0.9455, 10  # Anomaly Transformer
mean2, std2, n2 = 95.23, 1.016, 10  # MEMTO

# 计算合并标准差
sp = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
sp = sp**0.5

# 计算t值
t_value = (mean1 - mean2) / (sp * ((1/n1 + 1/n2)**0.5))

# 计算自由度
df = n1 + n2 - 2

# 计算p值
p_value = 2 * stats.t.cdf(t_value, df)

print("t值:", t_value)
print("p值:", p_value)