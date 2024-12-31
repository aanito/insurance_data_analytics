# Trends in CoverType and Premium by Province
province_trends = data.groupby(['Province', 'CoverType'])['TotalPremium'].mean().unstack()
province_trends.plot(kind='bar', stacked=True, figsize=(10, 6), cmap='tab20c')
plt.title('Premium Trends by Province and CoverType')
plt.ylabel('Average TotalPremium')
plt.xlabel('Province')
plt.show()
