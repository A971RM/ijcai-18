"""
    Deal the sampled file
    instance_id,

    item_id,item_category_list,item_property_list,item_brand_id,item_city_id,
    item_price_level,item_sales_level,item_collected_level,item_pv_level,

    user_id,user_gender_id,user_age_level,user_occupation_id,user_star_level,

    context_id,context_timestamp,context_page_id,predict_category_property,

    shop_id,shop_review_num_level,shop_review_positive_rate,shop_star_level,
    shop_score_service,shop_score_delivery,shop_score_description,

    is_trade
"""
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from datetime import datetime

SAMPLE_TRAIN_FILE = 'sample_train.csv'
df = pd.read_csv(SAMPLE_TRAIN_FILE)
print("To examine the nan:\n", df.isna().sum())

# all the figures
fig, ax = plt.subplots(2,2, figsize=(10.24, 6), dpi=100)

# the cunt
countFeatures = ['instance_id', 'item_id', 'user_id', 'shop_id', "context_id", "shop_id"]
countFeatureNums = {}
for feat in countFeatures:
    countFeatureNums[feat] = len(set(df[feat].values))

x = countFeatureNums.keys()
y = countFeatureNums.values()
sbn.barplot(list(x), list(y), ax=ax[0,0])

# the days counts context_timestamp
def dealtimestamp(ts):
    dt = datetime.fromtimestamp(ts)
    return 0 if dt.day == 31 else dt.day, dt.hour, dt.hour * 2 + dt.minute // 30

df[['day', 'hour', 'hour2']] = pd.DataFrame(list(df.context_timestamp.apply(lambda x: dealtimestamp(x))))
daydf = df.groupby('day').is_trade.count().reset_index()
sbn.barplot(x = daydf.day, y = daydf.is_trade, ax=ax[0, 1])

daycvrdf = df.groupby('day').is_trade.mean().reset_index()
sbn.barplot(x = daycvrdf.day, y = daycvrdf.is_trade, ax=ax[1, 1], )

dayhourcvrdf = df.groupby(['day', 'hour']).is_trade.mean().reset_index()
dayhourcvrdf = dayhourcvrdf.pivot(index='day', columns='hour', values='is_trade')
sbn.heatmap(dayhourcvrdf, ax=ax[1, 0])

plt.show()